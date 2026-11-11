//! The Gemma 3 model family (ADR 0006).

use std::sync::RwLock;
use std::time::Instant;

use peregrine_core::arch::{ActLayout, Architecture, Family, Loaded};
use peregrine_core::blocks::linear::LinJob;
use peregrine_core::blocks::par::{Disjoint, OutPtr};
use peregrine_core::gguf::{Format, Gguf, TensorInfo};
use peregrine_core::model::{Caches, Config, Linear, State, StepStats, TileStore, Weights};
use peregrine_core::pool::Pool;
use peregrine_core::tokenizer::Tokenizer;

pub(crate) struct Gemma3Layer {
    pub(crate) attn_norm: TensorInfo,
    pub(crate) wq: Linear,
    pub(crate) wk: Linear,
    pub(crate) wv: Linear,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) wo: Linear,
    pub(crate) attn_post_norm: TensorInfo,
    pub(crate) ffn_norm: TensorInfo,
    pub(crate) gate: Linear,
    pub(crate) up: Linear,
    pub(crate) down: Linear,
    pub(crate) ffn_post_norm: TensorInfo,
}

/// Gemma 3's attention scale, as llama.cpp v0.9.0 sets `f_attention_scale` (`src/models/gemma3.cpp`): the 27B
/// (the only size with 62 layers) uses `1/sqrt(n_embd / n_head)` with an *integer* division
/// (`query_pre_attn_scalar = d_model / n_head`), every other size `1/sqrt(head_dim)`. llama.cpp loads any layer
/// count and takes the second branch for an unknown one; so does this.
pub fn attention_scale(n_layer: usize, d_model: usize, n_head: usize, head_dim: usize) -> f32 {
    if n_layer == 62 {
        1.0 / ((d_model / n_head) as f32).sqrt()
    } else {
        1.0 / (head_dim as f32).sqrt()
    }
}

/// One projection over many rows, split into matmul-sized chunks.
///
/// The shared GEMM accepts at most `MAX_BATCH` rows per call (`blocks::linear::MAX_BATCH`);
/// an encoder forwards a whole text (up to the model's context) at once, so a layer's
/// projection runs in as many calls as it takes, one weight pass each. Rows of the batch
/// are contiguous, so the chunk is a plain row range and a row's bits do not depend on
/// where a chunk boundary fell.
#[allow(clippy::too_many_arguments)]
fn linear_rows(
    w: &Weights,
    pool: &Pool,
    stats: &mut StepStats,
    lin: &Linear,
    acts: &mut [peregrine_core::quant::QuantAct],
    lb_x: &mut [f32],
    lb_t: &mut [f32],
    input: &[f32],
    d_in: usize,
    nb: usize,
    out: &mut [f32],
    out_dim: usize,
    kq: &mut peregrine_kernels::kq::KqActs,
) {
    let step = peregrine_core::blocks::linear::MAX_BATCH;
    let mut r0 = 0;
    while r0 < nb {
        let r1 = (r0 + step).min(nb);
        let take = r1 - r0;
        for (i, a) in acts[r0..r1].iter_mut().enumerate() {
            a.fill(&input[(r0 + i) * d_in..(r0 + i + 1) * d_in]);
        }
        let lbx = Disjoint::new(lb_x);
        let lbt = Disjoint::new(lb_t);
        let jobs = [LinJob {
            lin,
            out: Disjoint::new(&mut out[r0 * out_dim..r1 * out_dim]),
        }];
        peregrine_core::blocks::linear::linear_group(
            w,
            w.lb(),
            &jobs,
            &acts[r0..r1],
            &input[r0 * d_in..r1 * d_in],
            d_in,
            take,
            lbx,
            lbt,
            pool,
            stats,
            kq,
        );
        r0 = r1;
    }
}

/// One or more projections that read the **same** activation rows, in matmul-sized
/// chunks, without quantising anything: the rows are already in `acts`, filled once by
/// the norm or activation that produced them (S4.9a).
///
/// The causal families group q/k/v and gate/up this way: several matrices read the same
/// normed row, so one fill and one fork-join serve them all. An encoder had the float
/// rows in hand and re-quantised them per projection — six of the seven fills of a layer
/// were the same bytes again, serially, while the matmuls ran in parallel.
#[allow(clippy::too_many_arguments)]
fn linear_rows_shared(
    w: &Weights,
    pool: &Pool,
    stats: &mut StepStats,
    jobs: &mut [RowsOut<'_>],
    acts: &[peregrine_core::quant::QuantAct],
    lb_x: &mut [f32],
    lb_t: &mut [f32],
    input: &[f32],
    d_in: usize,
    nb: usize,
    kq: &mut peregrine_kernels::kq::KqActs,
) {
    debug_assert!(!jobs.is_empty() && jobs.len() <= 4);
    let step = peregrine_core::blocks::linear::MAX_BATCH;
    let empty: &mut [f32] = &mut [];
    let mut lj = [LinJob {
        lin: jobs[0].lin,
        out: Disjoint::new(empty),
    }; 4];
    let mut r0 = 0;
    let njobs = jobs.len();
    while r0 < nb {
        let r1 = (r0 + step).min(nb);
        let take = r1 - r0;
        for i in 0..njobs {
            lj[i].lin = jobs[i].lin;
            let width = jobs[i].out_dim;
            // SAFETY: each job's output is a different buffer (q, k, v; gate, up; one
            // projection), and this chunk of it belongs to this call alone.
            lj[i].out = Disjoint::new(unsafe { jobs[i].out.rows(r0..r1, width) });
        }
        let lbx = Disjoint::new(lb_x);
        let lbt = Disjoint::new(lb_t);
        peregrine_core::blocks::linear::linear_group(
            w,
            w.lb(),
            &lj[..njobs],
            &acts[r0..r1],
            &input[r0 * d_in..r1 * d_in],
            d_in,
            take,
            lbx,
            lbt,
            pool,
            stats,
            kq,
        );
        r0 = r1;
    }
}

/// One projection of [`linear_rows_shared`]: the matrix, its output buffer, and the
/// width of one output row. Every job's buffer must be a different one — the group
/// writes them in one fork-join.
struct RowsOut<'a> {
    lin: &'a Linear,
    out: OutPtr<'a>,
    out_dim: usize,
}

pub struct Gemma3 {
    pub(crate) tok_embd: TensorInfo,
    pub(crate) out_norm: TensorInfo,
    pub(crate) head: Linear,
    pub(crate) layers: Vec<Gemma3Layer>,
    pub(crate) sliding_window: usize,
    pub(crate) is_swa: Vec<bool>,
    pub(crate) rope_swa: RwLock<Vec<f32>>,
    pub(crate) rope_base_swa: f32,
    pub(crate) final_logit_softcapping: Option<f32>,
    pub(crate) attn_scale: f32,
    /// EmbeddingGemma's sentence-transformers dense modules (`dense_2`, `dense_3`), applied
    /// after pooling and before normalisation. `None` for a `gemma3` generation model.
    pub(crate) dense_2: Option<Linear>,
    pub(crate) dense_3: Option<Linear>,
    /// The width of `dense_2`'s output and `dense_3`'s input (3072 on EmbeddingGemma).
    pub(crate) dense_width: usize,
    /// A bidirectional encoder (`gemma-embedding`): no KV cache and no causal mask.
    pub(crate) embedding: bool,
    /// Whether any matrix this model multiplies reads the per-256 activation layout
    /// (`Format::reads_k_layout`). A Q8_0/F16 model does not, and the layout is then
    /// never built (S4.9a).
    pub(crate) k_layout: bool,
}

#[cfg(feature = "arch-gemma-embedding")]
pub struct GemmaEmbeddingFamily;

#[cfg(feature = "arch-gemma3")]
pub struct Gemma3Family;

#[cfg(feature = "arch-gemma3")]
impl Family for Gemma3Family {
    fn names(&self) -> &'static [&'static str] {
        &["gemma3"]
    }

    fn load(&self, g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        Gemma3::load_impl(g, tk, false)
    }
}

/// EmbeddingGemma 300M: the `gemma-embedding` architecture (roadmap S4.4).
///
/// The same Gemma 3 blocks, read through the same loader, plus llama.cpp's
/// `--sentence-transformers-dense-modules` pair (`dense_2`, `dense_3`) and the
/// bidirectional forward (design §2.2). The file declares `pooling_type = 1` (mean).
#[cfg(feature = "arch-gemma-embedding")]
impl Family for GemmaEmbeddingFamily {
    fn names(&self) -> &'static [&'static str] {
        &["gemma-embedding"]
    }

    fn load(&self, g: &Gguf, tk: &Tokenizer) -> Result<Loaded, String> {
        Gemma3::load_impl(g, tk, true)
    }

    fn embedding_only(&self) -> bool {
        true
    }
}

impl Gemma3 {
    fn ensure_rope_swa_len(&self, cfg: &Config, needed_ctx: usize) {
        let half = cfg.rope_dim / 2;
        let needed_floats = needed_ctx * half * 2;
        {
            let current = self.rope_swa.read().unwrap();
            if current.len() >= needed_floats {
                return;
            }
        }
        let mut current = self.rope_swa.write().unwrap();
        if current.len() < needed_floats {
            let target_ctx = needed_ctx.max(current.len() / (half * 2) * 2);
            *current = peregrine_core::blocks::rope::build_rope_table(
                cfg.rope_dim,
                self.rope_base_swa,
                target_ctx,
                None,
            );
        }
    }

    pub(crate) fn load_impl(g: &Gguf, tk: &Tokenizer, embedding: bool) -> Result<Loaded, String> {
        let want_arch = if embedding {
            "gemma-embedding"
        } else {
            "gemma3"
        };
        let arch_name = g.str("general.architecture")?;
        if arch_name != want_arch {
            return Err(format!(
                "expected architecture {want_arch:?}, found {arch_name:?}"
            ));
        }
        let k = |suffix: &str| format!("{want_arch}.{suffix}");
        let n_layer = g.u32(&k("block_count"))? as usize;
        let d_model = g.u32(&k("embedding_length"))? as usize;
        let n_head = g.u32(&k("attention.head_count"))? as usize;
        let n_kv_head = g
            .u32(&k("attention.head_count_kv"))
            .unwrap_or(n_head as u32) as usize;
        let n_ff = g.u32(&k("feed_forward_length"))? as usize;
        if n_head == 0 || n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) {
            return Err(format!(
                "implausible head configuration: {n_head} query heads over {n_kv_head} kv heads"
            ));
        }
        let head_dim = g
            .u32(&k("attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(d_model / n_head);
        let rope_dim = g.u32(&k("rope.dimension_count")).unwrap_or(head_dim as u32) as usize;
        if rope_dim > head_dim || !rope_dim.is_multiple_of(2) {
            return Err(format!(
                "implausible rope dimension {rope_dim} for head dim {head_dim}"
            ));
        }
        for (what, n) in [("embedding length", d_model), ("feed-forward length", n_ff)] {
            if !n.is_multiple_of(32) {
                return Err(format!("{what} {n} is not a multiple of 32"));
            }
        }
        if !(head_dim * n_head).is_multiple_of(32) {
            return Err(format!(
                "query width {} is not a multiple of 32",
                head_dim * n_head
            ));
        }

        let attn_scale = attention_scale(n_layer, d_model, n_head, head_dim);

        let rope_scaling_type = g.str(&k("rope.scaling.type")).ok();
        let rope_scaling_factor = g.f32(&k("rope.scaling.factor")).ok();
        let mut rope_factors: Option<Vec<f32>> = match (rope_scaling_type, rope_scaling_factor) {
            (Some("none"), _) => None,
            (Some("linear"), Some(factor)) if factor > 0.0 => Some(vec![factor; rope_dim / 2]),
            (None, Some(factor)) if factor > 0.0 => Some(vec![factor; rope_dim / 2]),
            (Some(other), _) => {
                return Err(format!("unsupported Gemma 3 rope scaling type {other:?}"));
            }
            _ => None,
        };
        if let Ok(t) = g.tensor("rope_freqs.weight") {
            let n = t.numel() as usize;
            if n != rope_dim / 2 {
                return Err(format!(
                    "rope_freqs.weight has {n} entries, expected {}",
                    rope_dim / 2
                ));
            }
            let mut f = vec![0.0f32; n];
            peregrine_kernels::quant::dequant_row(t.kind, g.bytes(t), &mut f);
            if let Some(base) = &rope_factors {
                for (x, b) in f.iter_mut().zip(base) {
                    *x *= b;
                }
            }
            rope_factors = Some(f);
        }

        let sliding_window = g.u32(&k("attention.sliding_window")).unwrap_or(4096) as usize;

        // Sliding window pattern:
        // In llama.cpp, if sliding_window is found and > 0, default pattern is 6 (5 local, 1 global).
        // It can also be specified as an explicit bool/u32 array or scalar pattern.
        let is_swa: Vec<bool> = if sliding_window > 0 {
            if let Ok(pattern) = g.u32(&k("attention.sliding_window_pattern")) {
                let p = pattern as usize;
                if p == 0 {
                    vec![true; n_layer]
                } else {
                    (0..n_layer).map(|il| (il % p) < (p - 1)).collect()
                }
            } else {
                // Default pattern 6 in Gemma 3: il % 6 < 5 is sliding-window; il % 6 == 5 is global
                (0..n_layer).map(|il| (il % 6) < 5).collect()
            }
        } else {
            vec![false; n_layer]
        };

        let rope_base = g.f32(&k("rope.freq_base")).unwrap_or(1_000_000.0);
        let rope_base_swa = g.f32(&k("rope.freq_base_swa")).unwrap_or(10_000.0);
        let ctx_train = g.u32(&k("context_length")).unwrap_or(2048) as usize;
        let eps = g
            .f32(&k("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);

        let final_logit_softcapping = g
            .f32(&k("final_logit_softcapping"))
            .ok()
            .filter(|&c| c > 0.0);

        let cfg = Config {
            arch_name: if embedding {
                "gemma-embedding"
            } else {
                "gemma3"
            },
            rope_neox: true,
            n_layer,
            n_kv_layer: n_layer,
            n_head,
            n_kv_head,
            head_dim,
            d_model,
            n_ff,
            n_vocab: g.u32(&k("vocab_size")).unwrap_or(0) as usize,
            rope_dim,
            eps,
            rope_base,
            ctx_train,
            rope_yarn: None,
        };

        let get = |n: &str| -> Result<TensorInfo, String> { g.tensor(n).cloned() };
        let head_norm = |n: &str| -> Result<Vec<f32>, String> {
            let t = get(n)?;
            if t.numel() as usize != head_dim {
                return Err(format!(
                    "{n} has {} entries, expected the head dim {head_dim}",
                    t.numel()
                ));
            }
            let mut v = vec![0.0f32; head_dim];
            peregrine_kernels::quant::dequant_row(t.kind, g.bytes(&t), &mut v);
            Ok(v)
        };

        let get_norm = |name1: &str, name2: &str| -> Result<TensorInfo, String> {
            g.tensor(name1)
                .or_else(|_| g.tensor(name2))
                .cloned()
                .map_err(|e| e.to_string())
        };

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            layers.push(Gemma3Layer {
                attn_norm: get(&format!("blk.{i}.attn_norm.weight"))?,
                wq: Linear::Dense(get(&format!("blk.{i}.attn_q.weight"))?),
                wk: Linear::Dense(get(&format!("blk.{i}.attn_k.weight"))?),
                wv: Linear::Dense(get(&format!("blk.{i}.attn_v.weight"))?),
                q_norm: head_norm(&format!("blk.{i}.attn_q_norm.weight"))?,
                k_norm: head_norm(&format!("blk.{i}.attn_k_norm.weight"))?,
                wo: Linear::Dense(get(&format!("blk.{i}.attn_output.weight"))?),
                attn_post_norm: get_norm(
                    &format!("blk.{i}.post_attention_norm.weight"),
                    &format!("blk.{i}.attn_post_norm.weight"),
                )?,
                ffn_norm: get(&format!("blk.{i}.ffn_norm.weight"))?,
                gate: Linear::Dense(get(&format!("blk.{i}.ffn_gate.weight"))?),
                up: Linear::Dense(get(&format!("blk.{i}.ffn_up.weight"))?),
                down: Linear::Dense(get(&format!("blk.{i}.ffn_down.weight"))?),
                ffn_post_norm: get_norm(
                    &format!("blk.{i}.post_ffw_norm.weight"),
                    &format!("blk.{i}.ffn_post_norm.weight"),
                )?,
            });
        }

        let tok_embd = get("token_embd.weight")?;
        let out_norm = get("output_norm.weight")?;
        let head = Linear::Dense(
            g.tensor("output.weight")
                .ok()
                .cloned()
                .unwrap_or_else(|| tok_embd.clone()),
        );

        // EmbeddingGemma's sentence-transformers dense modules, read exactly as llama.cpp
        // creates them (`src/models/gemma-embedding.cpp`): `dense_2 {n_embd, feat_out}` and
        // `dense_3 {feat_in, n_embd}`, both optional and both without a bias tensor. The
        // metadata keys are `%s.dense_2_feat_in/out` and `%s.dense_3_feat_in/out`; llama.cpp
        // asserts the two fixed ends (`feat_in` of dense_2 and `feat_out` of dense_3) are
        // `n_embd`, and this loader refuses anything else by name.
        let mut dense_2 = None;
        let mut dense_3 = None;
        let mut dense_width = 0usize;
        if embedding {
            let key =
                |suffix: &str| -> Option<usize> { g.u32(&k(suffix)).ok().map(|v| v as usize) };
            let d2_in = key("dense_2_feat_in");
            let d2_out = key("dense_2_feat_out");
            match g.tensor("dense_2.weight") {
                Ok(t) => {
                    let (d_in, d_out) = (
                        d2_in.ok_or("dense_2.weight is present but dense_2_feat_in is not")?,
                        d2_out.ok_or("dense_2.weight is present but dense_2_feat_out is not")?,
                    );
                    if d_in != d_model {
                        return Err(format!(
                            "dense_2_feat_in is {d_in}, but llama.cpp requires the embedding length {d_model}"
                        ));
                    }
                    let (a, b) = (
                        t.dims.first().copied().unwrap_or(0) as usize,
                        t.dims.get(1).copied().unwrap_or(1) as usize,
                    );
                    if t.dims.len() > 2 || a != d_in || b != d_out {
                        return Err(format!(
                            "dense_2.weight is {:?}, the metadata implies [{d_in}, {d_out}]",
                            t.dims
                        ));
                    }
                    dense_2 = Some(Linear::Dense(t.clone()));
                    dense_width = dense_width.max(d_out);
                }
                Err(_) if d2_in.is_some() || d2_out.is_some() => {
                    return Err("dense_2_feat_in/out are set but dense_2.weight is missing".into())
                }
                Err(_) => {}
            }
            let d3_in = key("dense_3_feat_in");
            let d3_out = key("dense_3_feat_out");
            match g.tensor("dense_3.weight") {
                Ok(t) => {
                    let (d_in, d_out) = (
                        d3_in.ok_or("dense_3.weight is present but dense_3_feat_in is not")?,
                        d3_out.ok_or("dense_3.weight is present but dense_3_feat_out is not")?,
                    );
                    if d_out != d_model {
                        return Err(format!(
                            "dense_3_feat_out is {d_out}, but llama.cpp requires the embedding length {d_model}"
                        ));
                    }
                    let (a, b) = (
                        t.dims.first().copied().unwrap_or(0) as usize,
                        t.dims.get(1).copied().unwrap_or(1) as usize,
                    );
                    if t.dims.len() > 2 || a != d_in || b != d_out {
                        return Err(format!(
                            "dense_3.weight is {:?}, the metadata implies [{d_in}, {d_out}]",
                            t.dims
                        ));
                    }
                    dense_3 = Some(Linear::Dense(t.clone()));
                    dense_width = dense_width.max(d_in);
                }
                Err(_) if d3_in.is_some() || d3_out.is_some() => {
                    return Err("dense_3_feat_in/out are set but dense_3.weight is missing".into())
                }
                Err(_) => {}
            }
        }

        if dense_2.is_some() != dense_3.is_some() {
            return Err(
                "gemma-embedding has only one of dense_2/dense_3; both are applied together \
                 (llama.cpp's --sentence-transformers-dense-modules writes both)"
                    .into(),
            );
        }

        let mut cfg = cfg;
        if cfg.n_vocab == 0 {
            cfg.n_vocab = tk.n_vocab;
        }
        if cfg.n_vocab != tk.n_vocab {
            return Err(format!(
                "metadata says {} vocabulary entries but the tokenizer has {}",
                cfg.n_vocab, tk.n_vocab
            ));
        }

        let shape = |t: &TensorInfo, d_in: usize, d_out: usize| -> Result<(), String> {
            let (a, b) = (
                t.dims.first().copied().unwrap_or(0) as usize,
                t.dims.get(1).copied().unwrap_or(1) as usize,
            );
            if t.dims.len() > 2 || a != d_in || b != d_out {
                return Err(format!(
                    "{} is {:?}, the metadata implies [{d_in}, {d_out}]",
                    t.name, t.dims
                ));
            }
            Ok(())
        };
        let (dm, qd, kd, ff) = (cfg.d_model, cfg.q_dim(), cfg.kv_dim(), cfg.n_ff);
        for l in &layers {
            for (lin, d_in, d_out) in [
                (&l.wq, dm, qd),
                (&l.wk, dm, kd),
                (&l.wv, dm, kd),
                (&l.wo, qd, dm),
                (&l.gate, dm, ff),
                (&l.up, dm, ff),
                (&l.down, ff, dm),
            ] {
                if let Some(t) = lin.dense() {
                    shape(t, d_in, d_out)?;
                }
            }
            shape(&l.attn_norm, dm, 1)?;
            shape(&l.attn_post_norm, dm, 1)?;
            shape(&l.ffn_norm, dm, 1)?;
            shape(&l.ffn_post_norm, dm, 1)?;
        }
        shape(&tok_embd, dm, cfg.n_vocab)?;
        shape(&out_norm, dm, 1)?;
        if let Some(t) = head.dense() {
            shape(t, dm, cfg.n_vocab)?;
        }

        // The activation layout the matrices read: a model of 32- and 128-weight
        // formats never reads the per-256 one, so it is not built (S4.9a).
        let mut linears: Vec<&Linear> = Vec::new();
        for l in &layers {
            linears.extend([&l.wq, &l.wk, &l.wv, &l.wo, &l.gate, &l.up, &l.down]);
        }
        linears.push(&head);
        linears.extend([&dense_2, &dense_3].into_iter().flatten());
        let k_layout = linears
            .iter()
            .filter_map(|lin| lin.dense())
            .any(|t| t.kind.reads_k_layout());

        let initial_swa_ctx = ctx_train.max(2048);
        let rope_swa = peregrine_core::blocks::rope::build_rope_table(
            rope_dim,
            rope_base_swa,
            initial_swa_ctx,
            None,
        );

        Ok(Loaded {
            cfg,
            arch: Box::new(Gemma3 {
                tok_embd,
                out_norm,
                head,
                layers,
                sliding_window,
                is_swa,
                rope_swa: RwLock::new(rope_swa),
                rope_base_swa,
                final_logit_softcapping,
                attn_scale,
                dense_2,
                dense_3,
                dense_width,
                embedding,
                k_layout,
            }),
            rope_factors,
            tiles: TileStore::default(),
        })
    }
}

impl Architecture for Gemma3 {
    fn session_kv_rotation(&self) -> bool {
        true
    }

    fn linears_mut(
        &mut self,
        f: &mut dyn FnMut(&mut Linear) -> Result<(), String>,
    ) -> Result<(), String> {
        for l in &mut self.layers {
            for lin in [
                &mut l.wq,
                &mut l.wk,
                &mut l.wv,
                &mut l.wo,
                &mut l.gate,
                &mut l.up,
                &mut l.down,
            ] {
                f(lin)?;
            }
        }
        if let Some(ti) = self.head.dense() {
            if ti.name == "output.weight" {
                f(&mut self.head)?;
            }
        }
        if let Some(d) = &mut self.dense_2 {
            f(d)?;
        }
        if let Some(d) = &mut self.dense_3 {
            f(d)?;
        }
        Ok(())
    }

    fn linear_groups(&self, f: &mut dyn FnMut(&[&Linear])) {
        for l in &self.layers {
            f(&[&l.wq, &l.wk, &l.wv]);
            f(&[&l.wo]);
            f(&[&l.gate, &l.up]);
            f(&[&l.down]);
        }
        f(&[&self.head]);
        if let Some(d) = &self.dense_2 {
            f(&[d]);
        }
        if let Some(d) = &self.dense_3 {
            f(&[d]);
        }
    }

    fn act_layout(&self) -> ActLayout {
        let mut keep_f32 = false;
        self.linear_groups(&mut |lins| {
            for lin in lins {
                if let Some(ti) = lin.dense() {
                    if matches!(ti.kind, Format::F32 | Format::F16 | Format::BF16) {
                        keep_f32 = true;
                    }
                }
            }
        });
        ActLayout {
            keep_f32,
            act_width: self.dense_width,
            ..ActLayout::default()
        }
    }

    fn causality(&self) -> peregrine_core::arch::Causality {
        if self.embedding {
            peregrine_core::arch::Causality::Bidirectional
        } else {
            peregrine_core::arch::Causality::Causal
        }
    }

    fn forward_rows(
        &self,
        w: &Weights,
        cfg: &Config,
        st: &mut State,
        caches: &mut Caches<'_>,
        pool: &Pool,
        tokens: &[u32],
        positions: &[usize],
        want_logits: &[usize],
        stats: &mut StepStats,
    ) -> Result<(), String> {
        if self.embedding {
            return Err(
                "a gemma-embedding model has no causal forward; the engine must use forward_texts"
                    .into(),
            );
        }
        let nb = tokens.len();
        let t_fwd = Instant::now();
        // The per-256 activation layout is built only when a matrix reads it (S4.9a).
        for a in st.acts.iter_mut() {
            a.set_k_layout(self.k_layout);
        }

        // 1. Embedding lookup scaled by sqrt(d_model)
        for (i, &t) in tokens.iter().enumerate() {
            peregrine_core::blocks::embed::embed(
                &self.tok_embd,
                w,
                t,
                &mut st.x[i * cfg.d_model..(i + 1) * cfg.d_model],
                stats,
            );
        }
        let embed_scale = (cfg.d_model as f32).sqrt();
        for x in &mut st.x[..nb * cfg.d_model] {
            *x *= embed_scale;
        }

        // Ensure rope_swa is long enough for all requested positions
        let max_pos = positions.iter().copied().max().unwrap_or(0);
        self.ensure_rope_swa_len(cfg, max_pos + 1);
        let rope_swa_guard = self.rope_swa.read().unwrap();
        let rope_swa: &[f32] = &rope_swa_guard;

        let qd = cfg.q_dim();
        let dm = cfg.d_model;
        let ff = cfg.n_ff;

        for (li, l) in self.layers.iter().enumerate() {
            let is_swa = self.is_swa[li];
            let rope_table = if is_swa { rope_swa } else { &st.rope };
            // KV cache for local layers stays full-length for now (memory, not correctness)
            let window = if is_swa {
                Some(self.sliding_window)
            } else {
                None
            };

            // ---- attention pre-norm ----
            peregrine_core::blocks::norm::norm_rows(
                w,
                &l.attn_norm,
                dm,
                cfg.eps,
                &st.x,
                &mut st.xb,
                &mut st.acts,
                nb,
                pool,
                stats,
            );

            // ---- attention QKV projections ----
            {
                let lbx = Disjoint::new(&mut st.lb_x);
                let lbt = Disjoint::new(&mut st.lb_t);
                let acts = &st.acts[..nb];
                let xs: &[f32] = &st.xb;
                let jobs = [
                    LinJob {
                        lin: &l.wq,
                        out: Disjoint::new(&mut st.q),
                    },
                    LinJob {
                        lin: &l.wk,
                        out: Disjoint::new(&mut st.k),
                    },
                    LinJob {
                        lin: &l.wv,
                        out: Disjoint::new(&mut st.v),
                    },
                ];
                peregrine_core::blocks::linear::linear_group(
                    w,
                    w.lb(),
                    &jobs,
                    acts,
                    xs,
                    dm,
                    nb,
                    lbx,
                    lbt,
                    pool,
                    stats,
                    &mut st.kq,
                );
            }

            // ---- prepare rows (per-head q/k RMSNorm + RoPE) ----
            peregrine_core::blocks::attn::prepare_rows(
                cfg,
                None,
                None,
                None,
                Some(&l.q_norm),
                Some(&l.k_norm),
                rope_table,
                &mut st.q,
                &mut st.k,
                &mut st.v,
                nb,
                positions,
                pool,
                stats,
            );

            // ---- KV cache write ----
            peregrine_core::blocks::attn::write_kv_rows(
                caches,
                li,
                &st.k,
                &st.v,
                nb,
                positions,
                cfg.n_kv_head,
                cfg.head_dim,
            );

            // ---- Attend ----
            let t_attn = Instant::now();
            for (r0, len, c) in caches.runs(nb) {
                peregrine_core::blocks::attn::attend_scaled_window(
                    cfg,
                    &mut st.attn,
                    &st.q,
                    &mut st.xb,
                    caches.cache(c),
                    pool,
                    li,
                    len,
                    positions[r0],
                    r0,
                    window,
                    Some(self.attn_scale),
                    None,
                    stats,
                );
            }
            stats.ns_attn += t_attn.elapsed().as_nanos() as u64;

            // ---- Output projection wo ----
            for b in 0..nb {
                st.acts[b].fill(&st.xb[b * qd..(b + 1) * qd]);
            }
            {
                let lbx = Disjoint::new(&mut st.lb_x);
                let lbt = Disjoint::new(&mut st.lb_t);
                let acts = &st.acts[..nb];
                let jobs = [LinJob {
                    lin: &l.wo,
                    out: Disjoint::new(&mut st.xb2),
                }];
                peregrine_core::blocks::linear::linear_group(
                    w,
                    w.lb(),
                    &jobs,
                    acts,
                    &st.xb,
                    qd,
                    nb,
                    lbx,
                    lbt,
                    pool,
                    stats,
                    &mut st.kq,
                );
            }

            // ---- Attention post-norm and residual add ----
            // Nothing multiplies this norm's output: it is added to the residual, so its
            // quantisation is skipped (S4.9a).
            peregrine_core::blocks::norm::norm_rows_f32(
                w,
                &l.attn_post_norm,
                dm,
                cfg.eps,
                &st.xb2,
                &mut st.xb,
                nb,
                pool,
                stats,
            );
            for i in 0..nb * dm {
                st.x[i] += st.xb[i];
            }

            // ---- feed-forward pre-norm ----
            peregrine_core::blocks::norm::norm_rows(
                w,
                &l.ffn_norm,
                dm,
                cfg.eps,
                &st.x,
                &mut st.xb,
                &mut st.acts,
                nb,
                pool,
                stats,
            );

            // ---- Gate and Up projections ----
            {
                let lbx = Disjoint::new(&mut st.lb_x);
                let lbt = Disjoint::new(&mut st.lb_t);
                let acts = &st.acts[..nb];
                let jobs = [
                    LinJob {
                        lin: &l.gate,
                        out: Disjoint::new(&mut st.hb),
                    },
                    LinJob {
                        lin: &l.up,
                        out: Disjoint::new(&mut st.hb2),
                    },
                ];
                peregrine_core::blocks::linear::linear_group(
                    w,
                    w.lb(),
                    &jobs,
                    acts,
                    &st.xb,
                    dm,
                    nb,
                    lbx,
                    lbt,
                    pool,
                    stats,
                    &mut st.kq,
                );
            }

            // ---- GeGLU activation ----
            peregrine_core::blocks::ffn::geglu_rows(
                ff,
                &mut st.hb,
                &st.hb2,
                &mut st.acts,
                nb,
                pool,
                stats,
            );

            // ---- Down projection ----
            {
                let lbx = Disjoint::new(&mut st.lb_x);
                let lbt = Disjoint::new(&mut st.lb_t);
                let acts = &st.acts[..nb];
                let jobs = [LinJob {
                    lin: &l.down,
                    out: Disjoint::new(&mut st.xb2),
                }];
                peregrine_core::blocks::linear::linear_group(
                    w,
                    w.lb(),
                    &jobs,
                    acts,
                    &st.hb,
                    ff,
                    nb,
                    lbx,
                    lbt,
                    pool,
                    stats,
                    &mut st.kq,
                );
            }

            // ---- Feed-forward post-norm and residual add ----
            peregrine_core::blocks::norm::norm_rows_f32(
                w,
                &l.ffn_post_norm,
                dm,
                cfg.eps,
                &st.xb2,
                &mut st.xb,
                nb,
                pool,
                stats,
            );
            for i in 0..nb * dm {
                st.x[i] += st.xb[i];
            }

            if let Some(t) = st.tap_layers.iter().position(|&x| x == li) {
                let nt = st.tap_layers.len();
                for b in 0..nb {
                    let dst = (b * nt + t) * dm;
                    st.tap[dst..dst + dm].copy_from_slice(&st.x[b * dm..(b + 1) * dm]);
                }
            }
        }

        // ---- output head ----
        crate::layer::write_hidden(w, cfg, st, &self.out_norm, nb);
        if want_logits.is_empty() {
            stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
            return Ok(());
        }
        let nl = want_logits.len();
        // The same head as the other families (`layer::head_rows`, which reads a
        // label row only at its token ids, S4.9d).
        crate::layer::head_rows(
            w,
            cfg,
            st,
            &self.out_norm,
            &self.head,
            nb,
            want_logits,
            pool,
            stats,
        )?;

        if let Some(cap) = self.final_logit_softcapping {
            let inv_cap = 1.0 / cap;
            for logit in &mut st.logits[..nl * cfg.n_vocab] {
                *logit = cap * peregrine_core::vmath::det_tanh(*logit * inv_cap);
            }
            // A label entry is the full row's entry after the same softcap.
            for v in &mut st.label_out {
                *v = cap * peregrine_core::vmath::det_tanh(*v * inv_cap);
            }
        }

        if let Some(cap) = self.final_logit_softcapping {
            let inv_cap = 1.0 / cap;
            for logit in &mut st.logits[..nl * cfg.n_vocab] {
                *logit = cap * peregrine_core::vmath::det_tanh(*logit * inv_cap);
            }
        }

        stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
        Ok(())
    }

    /// One bidirectional pass over whole texts (design §2.2, roadmap S4.4).
    ///
    /// The transcription of llama.cpp v0.9.0's `gemma-embedding` graph
    /// (`src/models/gemma-embedding.cpp`) with its no-cache attention
    /// (`llm_graph_input_attn_no_cache::set_input`, `src/llama-graph.cpp`):
    ///
    /// - the token embedding is scaled by `sqrt(n_embd)`;
    /// - every layer is `attn_norm` → q/k/v → per-head q/k RMSNorm → RoPE (the sliding
    ///   layers use `rope.freq_base_swa`) → `Q · 1/sqrt(head_dim)` → attention →
    ///   `wo` → `attn_post_norm` → residual → `ffn_norm` → gate/up → GeGLU → down →
    ///   `ffn_post_norm` → residual;
    /// - causal attention is off (`hparams.causal_attn = false`), so every row sees the
    ///   whole text; a sliding-window layer masks `|p1 - p0| > n_swa / 2`
    ///   (`LLAMA_SWA_TYPE_SYMMETRIC`), a global layer masks nothing;
    /// - `output_norm` is `result_norm`, which is pooled per text;
    /// - `dense_2` then `dense_3` are applied after pooling, pure matmuls with no bias
    ///   and no activation (`llm_graph_context::build_dense_out`).
    ///
    /// Pooling and the two dense layers happen here, text by text, before the engine's
    /// `embed::finish` truncates and normalises.
    fn forward_texts(
        &self,
        w: &Weights,
        cfg: &Config,
        st: &mut State,
        pool: &Pool,
        texts: &[&[u32]],
        poolings: &[peregrine_core::embed::Pooling],
        stats: &mut StepStats,
    ) -> Result<(), String> {
        let t_fwd = Instant::now();
        let dm = cfg.d_model;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let ff = cfg.n_ff;
        let nb: usize = texts.iter().map(|t| t.len()).sum();
        // The per-256 activation layout is built only when a matrix reads it (S4.9a).
        for a in st.acts.iter_mut() {
            a.set_k_layout(self.k_layout);
        }

        // The batch's positions: every text starts at 0. Both buffers are taken out of
        // the state (and put back at the end) so the loop can borrow the rest of `st`
        // mutably: a bidirectional pass must not allocate.
        let mut pos_buf = std::mem::take(&mut st.positions);
        let mut spans = std::mem::take(&mut st.spans);
        if pos_buf.len() < nb {
            pos_buf.resize(nb, 0);
        }
        spans.clear();
        let mut row = 0usize;
        for t in texts {
            spans.push(peregrine_core::blocks::bidir::TextSpan {
                row0: row,
                len: t.len(),
            });
            for i in 0..t.len() {
                pos_buf[row] = i;
                row += 1;
            }
        }
        let positions = &pos_buf[..nb];

        // The embedding lookup, scaled by sqrt(d_model) like llama.cpp's `build_inp_embd`.
        let mut row = 0usize;
        for t in texts {
            for &tok in t.iter() {
                peregrine_core::blocks::embed::embed(
                    &self.tok_embd,
                    w,
                    tok,
                    &mut st.x[row * dm..(row + 1) * dm],
                    stats,
                );
                row += 1;
            }
        }
        let embed_scale = (dm as f32).sqrt();
        for x in &mut st.x[..nb * dm] {
            *x *= embed_scale;
        }

        let max_pos = positions.iter().copied().max().unwrap_or(0);
        self.ensure_rope_swa_len(cfg, max_pos + 1);
        let rope_swa_guard = self.rope_swa.read().unwrap();
        let rope_swa: &[f32] = &rope_swa_guard;

        for (li, l) in self.layers.iter().enumerate() {
            let is_swa = self.is_swa[li];
            let rope_table = if is_swa { rope_swa } else { &st.rope };
            let half_window = if is_swa {
                Some(self.sliding_window / 2)
            } else {
                None
            };

            // ---- attention pre-norm ----
            peregrine_core::blocks::norm::norm_rows(
                w,
                &l.attn_norm,
                dm,
                cfg.eps,
                &st.x,
                &mut st.xb,
                &mut st.acts,
                nb,
                pool,
                stats,
            );

            // ---- QKV projections: one quantisation and one fork-join for the three
            // matrices, which read the same normed rows (S4.9a) ----
            {
                let mut jobs = [
                    RowsOut {
                        lin: &l.wq,
                        out: Disjoint::new(&mut st.q),
                        out_dim: qd,
                    },
                    RowsOut {
                        lin: &l.wk,
                        out: Disjoint::new(&mut st.k),
                        out_dim: kvd,
                    },
                    RowsOut {
                        lin: &l.wv,
                        out: Disjoint::new(&mut st.v),
                        out_dim: kvd,
                    },
                ];
                linear_rows_shared(
                    w,
                    pool,
                    stats,
                    &mut jobs,
                    &st.acts,
                    &mut st.lb_x,
                    &mut st.lb_t,
                    &st.xb,
                    dm,
                    nb,
                    &mut st.kq,
                );
            }

            // ---- per-head q/k RMSNorm and RoPE ----
            peregrine_core::blocks::attn::prepare_rows(
                cfg,
                None,
                None,
                None,
                Some(&l.q_norm),
                Some(&l.k_norm),
                rope_table,
                &mut st.q,
                &mut st.k,
                &mut st.v,
                nb,
                positions,
                pool,
                stats,
            );

            // llama.cpp scales Q by `f_attention_scale` after RoPE and passes kq_scale 1.
            for q in &mut st.q[..nb * qd] {
                *q *= self.attn_scale;
            }

            // ---- attention: one n × n block per text, no cache ----
            peregrine_core::blocks::bidir::attend_texts(
                cfg,
                &st.q,
                &st.k,
                &st.v,
                &mut st.xb,
                &spans,
                half_window,
                1.0,
                &mut st.attn.bidir,
                pool,
                stats,
            );

            // ---- output projection ----
            linear_rows(
                w,
                pool,
                stats,
                &l.wo,
                &mut st.acts,
                &mut st.lb_x,
                &mut st.lb_t,
                &st.xb,
                qd,
                nb,
                &mut st.xb2,
                dm,
                &mut st.kq,
            );
            // The post-attention norm feeds only the residual add: no quantisation.
            peregrine_core::blocks::norm::norm_rows_f32(
                w,
                &l.attn_post_norm,
                dm,
                cfg.eps,
                &st.xb2,
                &mut st.xb,
                nb,
                pool,
                stats,
            );
            for i in 0..nb * dm {
                st.x[i] += st.xb[i];
            }

            // ---- feed-forward ----
            peregrine_core::blocks::norm::norm_rows(
                w,
                &l.ffn_norm,
                dm,
                cfg.eps,
                &st.x,
                &mut st.xb,
                &mut st.acts,
                nb,
                pool,
                stats,
            );
            // Gate and up read the same normed rows: one quantisation, one fork-join.
            {
                let mut jobs = [
                    RowsOut {
                        lin: &l.gate,
                        out: Disjoint::new(&mut st.hb),
                        out_dim: ff,
                    },
                    RowsOut {
                        lin: &l.up,
                        out: Disjoint::new(&mut st.hb2),
                        out_dim: ff,
                    },
                ];
                linear_rows_shared(
                    w,
                    pool,
                    stats,
                    &mut jobs,
                    &st.acts,
                    &mut st.lb_x,
                    &mut st.lb_t,
                    &st.xb,
                    dm,
                    nb,
                    &mut st.kq,
                );
            }
            peregrine_core::blocks::ffn::geglu_rows(
                ff,
                &mut st.hb,
                &st.hb2,
                &mut st.acts,
                nb,
                pool,
                stats,
            );
            // The activation of `geglu_rows` is already quantised: the down projection
            // reads it as it is instead of filling the same rows again (S4.9a).
            {
                let mut jobs = [RowsOut {
                    lin: &l.down,
                    out: Disjoint::new(&mut st.xb2),
                    out_dim: dm,
                }];
                linear_rows_shared(
                    w,
                    pool,
                    stats,
                    &mut jobs,
                    &st.acts,
                    &mut st.lb_x,
                    &mut st.lb_t,
                    &st.hb,
                    ff,
                    nb,
                    &mut st.kq,
                );
            }
            // The post-FFN norm feeds only the residual add: no quantisation.
            peregrine_core::blocks::norm::norm_rows_f32(
                w,
                &l.ffn_post_norm,
                dm,
                cfg.eps,
                &st.xb2,
                &mut st.xb,
                nb,
                pool,
                stats,
            );
            for i in 0..nb * dm {
                st.x[i] += st.xb[i];
            }
        }

        // ---- output_norm: llama.cpp's `result_norm`, the rows pooling reads ----
        let saved_hidden = st.want_hidden;
        st.want_hidden = true;
        crate::layer::write_hidden(w, cfg, st, &self.out_norm, nb);
        st.want_hidden = saved_hidden;

        // ---- pooling, then dense_2 / dense_3, text by text ----
        if st.pool_sum.len() < dm {
            st.pool_sum.resize(dm, 0.0);
        }
        for (t, text) in texts.iter().enumerate() {
            let row0 = spans[t].row0;
            let n = text.len();
            peregrine_core::embed::pool_rows_into(
                &st.hidden[row0 * dm..(row0 + n) * dm],
                dm,
                poolings[t],
                &mut st.pool_sum,
                &mut st.xb2[..dm],
            );
            // The loader requires both dense modules or neither, so the vector is back at
            // `d_model` width after the second one.
            if let (Some(d2), Some(d3)) = (&self.dense_2, &self.dense_3) {
                let d_out = d2
                    .dense()
                    .expect("dense_2 is a dense tensor")
                    .dims
                    .get(1)
                    .copied()
                    .unwrap_or(0) as usize;
                if st.emb.len() < d_out {
                    st.emb.resize(d_out, 0.0);
                }
                linear_rows(
                    w,
                    pool,
                    stats,
                    d2,
                    &mut st.acts,
                    &mut st.lb_x,
                    &mut st.lb_t,
                    &st.xb2[..dm],
                    dm,
                    1,
                    &mut st.emb,
                    d_out,
                    &mut st.kq,
                );
                linear_rows(
                    w,
                    pool,
                    stats,
                    d3,
                    &mut st.acts,
                    &mut st.lb_x,
                    &mut st.lb_t,
                    &st.emb[..d_out],
                    d_out,
                    1,
                    &mut st.xb2,
                    dm,
                    &mut st.kq,
                );
            }
            // The vector this text's slot reads: the first row of the text, so the
            // engine needs no buffer of its own.
            st.hidden[row0 * dm..row0 * dm + dm].copy_from_slice(&st.xb2[..dm]);
        }

        // Put the caller's buffers back with the capacity they had.
        st.positions = pos_buf;
        st.spans = spans;
        stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
        Ok(())
    }
}
