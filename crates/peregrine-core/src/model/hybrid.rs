//! Qwen3.5-style hybrid models: PrismML's Bonsai 2 (GGUF architecture `qwen35`).
//!
//! Three of every four layers are Gated DeltaNet (linear attention with a recurrent state per sequence), the
//! fourth is gated full attention; every layer has a SwiGLU feed-forward. The ternary weights were rotated by a
//! block Hadamard transform with explicit signs, which the engine applies to each matmul input (`rotate_rows`)
//! and, inverted, to each embedding row. The specification this follows, op by op, with citations into
//! PrismML's llama.cpp fork, is `docs/reference/bonsai2_spec.md`.
//!
//! What is shared with the dense path: the GGUF reader, the tokenizer, the activation quantiser and the fused
//! matmul (`matmul_group`, here with the ternary kernels of `tern.rs`), the KV cache and grouped attention for the
//! full-attention layers, and the pool. What is new: the conv and the delta rule, whose state lives in the
//! sequence's `KvCache` (so several users batched in one pass each carry their own), and the rotation.

use super::*;
use crate::hadamard::fwht;
use std::sync::atomic::{AtomicU64, Ordering as AO};

// Profiling buckets (`PEREGRINE_PROF=1`): wall nanoseconds per part of the hybrid pass, read by `profile()`.
const PROF_NAMES: [&str; 16] = [
    "embed",
    "norm",
    "rotate",
    "matmul",
    "gdn_ab",
    "gdn_conv",
    "gdn_delta",
    "gdn_gnorm",
    "attn",
    "ffn_act",
    "mm<8MB ns",
    "mm<8MB bytes",
    "mm8-20MB ns",
    "mm8-20MB bytes",
    "mm>20MB ns",
    "mm>20MB bytes",
];
static PROF: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
fn prof_on() -> bool {
    use std::sync::OnceLock;
    static P: OnceLock<bool> = OnceLock::new();
    *P.get_or_init(|| crate::options::get().prof)
}
struct Tick(Option<Instant>, usize);
impl Tick {
    fn new(b: usize) -> Tick {
        Tick(prof_on().then(Instant::now), b)
    }
}
impl Drop for Tick {
    fn drop(&mut self) {
        if let Some(t) = self.0 {
            PROF[self.1].fetch_add(t.elapsed().as_nanos() as u64, AO::Relaxed);
        }
    }
}
/// The profiling buckets since the last call, `(name, ns)`, and reset them.
pub fn profile() -> Vec<(&'static str, u64)> {
    PROF_NAMES
        .iter()
        .zip(PROF.iter())
        .map(|(n, v)| (*n, v.swap(0, AO::Relaxed)))
        .collect()
}

#[derive(Debug, Clone, Copy)]
pub struct HybridCfg {
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub dk: usize,
    pub dv: usize,
    pub d_conv: usize,
    pub interval: usize,
    pub had_block: usize,
    pub n_gdn: usize,
    pub n_attn: usize,
    /// Every ternary matmul is PTQ1_0 (else every one is PQ2_0): selects the activation layout.
    pub ptq1: bool,
    /// PQ2_0 matrices run from 16-row tiles (`tern::dot_tiles_pq2`) with one activation scale per `tb_span`
    /// values; 0 = the per-row kernels with one scale per 16.
    pub tb_span: usize,
}

impl HybridCfg {
    pub fn key_dim(&self) -> usize {
        self.n_k_heads * self.dk
    }
    pub fn value_dim(&self) -> usize {
        self.n_v_heads * self.dv
    }
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
    /// Floats of delta-rule state per sequence per linear-attention layer.
    pub fn state_floats(&self) -> usize {
        self.n_v_heads * self.dv * self.dk
    }
    /// Floats of conv input history per sequence per linear-attention layer: `(d_conv − 1)` taps per channel.
    pub fn conv_floats(&self) -> usize {
        (self.d_conv - 1) * self.conv_dim()
    }
    fn is_attn(&self, layer: usize) -> bool {
        (layer + 1) % self.interval == 0
    }
}

pub(super) enum Mixer {
    Gdn {
        qkv: TensorInfo,
        gate: TensorInfo,
        /// `ssm_beta` then `ssm_alpha`, `[2·n_v][d_model]` f32: small, unrotated, applied in f32.
        ab: Vec<f32>,
        /// Depthwise conv taps, `[d_conv][conv_dim]`; tap `d_conv − 1` multiplies the current token.
        conv: Vec<f32>,
        dt_bias: Vec<f32>,
        /// `−exp(A_log)` as stored.
        a: Vec<f32>,
        norm: Vec<f32>,
        out: TensorInfo,
        /// Index of this layer's recurrent state in the cache.
        idx: usize,
    },
    Attn {
        q: TensorInfo,
        k: TensorInfo,
        v: TensorInfo,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        o: TensorInfo,
        /// Index of this layer in the KV cache.
        idx: usize,
    },
}

pub(super) struct HLayer {
    attn_norm: TensorInfo,
    mix: Mixer,
    post_norm: TensorInfo,
    gate: TensorInfo,
    up: TensorInfo,
    down: TensorInfo,
}

pub(super) struct Hybrid {
    pub cfg: HybridCfg,
    layers: Vec<HLayer>,
    /// `signs / √block` per input width.
    signs: Vec<(usize, Vec<f32>)>,
    /// `ssm_out`'s input is permuted from tiled to grouped v-head order before its rotation.
    v_grouped: bool,
}

/// Per-row raw pointers into the caches' recurrent state, and the snapshots a speculative verification asks
/// for, kept beside the other hybrid scratch so a decode step reuses them instead of allocating four vectors
/// per Gated DeltaNet layer.
struct RecScratch {
    /// Row `b`'s `(state, conv)` pointers for the recurrent layer being processed.
    rec: Vec<(*mut f32, *mut f32)>,
    /// Row `b`'s snapshot request `(position, state, conv)`, if its cache asked for one.
    snap: Vec<Option<(usize, *mut f32, *mut f32)>>,
}

// SAFETY: the vectors only hold pointers into the `KvCache`s passed to the same `forward_hybrid` call; they are
// rewritten by every `gdn`. `State` owns them and only one thread uses it at a time; the pool closures see copies
// through `RowPtrs`/`SnapPtrs`, whose `Send`/`Sync` impls carry the disjoint-write argument. The `Send` impl keeps
// `State` (and the `Engine` holding it) movable between threads like the rest of the scratch.
unsafe impl Send for RecScratch {}
// SAFETY: as `Send`; shared access only reads the vector's values (pointer copies) and never dereferences them.
unsafe impl Sync for RecScratch {}

/// Scratch for the hybrid pass, beside `State`'s own buffers.
pub struct HyScratch {
    /// `attn_q` output (q and gate interleaved per head) or the GDN `attn_qkv` output, per row.
    big: Vec<f32>,
    big_w: usize,
    /// GDN `z`, or the attention gate, per row.
    z: Vec<f32>,
    z_w: usize,
    /// GDN per row: `[β (n_v) | α (n_v)]`, then `[β | decay]` after the gate maths.
    ab: Vec<f32>,
    /// Rotated matmul inputs, per row.
    rot: Vec<f32>,
    rot_w: usize,
    /// GDN output per row, `value_dim`.
    o: Vec<f32>,
    /// Runs of rows sharing one cache, and the caches already seen, reused every call.
    runs: Vec<(usize, usize, usize)>,
    seen: Vec<usize>,
    /// Recurrent-state pointers for the delta-rule and conv workers, reused every layer.
    ptrs: RecScratch,
}

impl HyScratch {
    pub fn new(c: &Config, h: &HybridCfg, b: usize) -> HyScratch {
        let big_w = (2 * c.q_dim()).max(h.conv_dim());
        let z_w = h.value_dim().max(c.q_dim());
        let rot_w = c.n_ff.max(c.d_model).max(h.value_dim()).max(c.q_dim());
        HyScratch {
            big: vec![0.0; b * big_w],
            big_w,
            z: vec![0.0; b * z_w],
            z_w,
            ab: vec![0.0; b * 2 * h.n_v_heads],
            rot: vec![0.0; b * rot_w],
            rot_w,
            o: vec![0.0; b * h.value_dim()],
            runs: Vec::with_capacity(b),
            seen: Vec::with_capacity(b),
            ptrs: RecScratch {
                rec: vec![(std::ptr::null_mut(), std::ptr::null_mut()); b],
                snap: vec![None; b],
            },
        }
    }
}

/// A per-row step fused into a rotation's pass: `(row index, row)`.
type RowPrep<'a> = dyn Fn(usize, &mut [f32]) + Sync + 'a;

/// Raw per-row state pointers for the delta-rule workers.
#[derive(Clone, Copy)]
struct RowPtrs(*const (*mut f32, *mut f32));
// SAFETY: each (row's state, head) region is written by exactly one task; rows of a shared cache are processed in
// order by the task that owns the head.
unsafe impl Send for RowPtrs {}
unsafe impl Sync for RowPtrs {}

impl RowPtrs {
    /// Row `b`'s (state, conv) pointers. A method, so a closure captures the wrapper (Send + Sync), not its field.
    ///
    /// # Safety
    /// `b` must be inside the pointer array, which must outlive the call.
    #[inline(always)]
    unsafe fn at(self, b: usize) -> (*mut f32, *mut f32) {
        *self.0.add(b)
    }
}

fn f32_vec(g: &Gguf, t: &TensorInfo) -> Vec<f32> {
    let mut v = vec![0.0f32; t.numel() as usize];
    crate::quant::dequant_row(t.kind, g.bytes(t), &mut v);
    v
}

pub(super) fn load(g: Gguf) -> Result<Model, String> {
    let k = |s: &str| format!("qwen35.{s}");
    // A file with an MTP block counts it in block_count; the main pass runs the layers before it.
    let n_nextn = g.u32(&k("nextn_predict_layers")).unwrap_or(0) as usize;
    let n_layer = (g.u32(&k("block_count"))? as usize)
        .checked_sub(n_nextn)
        .ok_or("nextn_predict_layers exceeds block_count")?;
    let d_model = g.u32(&k("embedding_length"))? as usize;
    let n_head = g.u32(&k("attention.head_count"))? as usize;
    let n_kv_head = g.u32(&k("attention.head_count_kv"))? as usize;
    let head_dim = g.u32(&k("attention.key_length"))? as usize;
    let n_ff = g.u32(&k("feed_forward_length"))? as usize;
    let rope_dim = g.u32(&k("rope.dimension_count"))? as usize;
    let interval = g.u32(&k("full_attention_interval"))? as usize;
    let hc0 = HybridCfg {
        n_k_heads: g.u32(&k("ssm.group_count"))? as usize,
        n_v_heads: g.u32(&k("ssm.time_step_rank"))? as usize,
        dk: g.u32(&k("ssm.state_size"))? as usize,
        dv: g.u32(&k("ssm.state_size"))? as usize,
        d_conv: g.u32(&k("ssm.conv_kernel"))? as usize,
        interval,
        had_block: g.u32("prism.hadamard.block_size")? as usize,
        n_gdn: 0,
        n_attn: 0,
        ptq1: false,
        tb_span: 0,
    };
    if hc0.value_dim() != g.u32(&k("ssm.inner_size"))? as usize {
        return Err("ssm.inner_size ≠ time_step_rank × state_size".into());
    }
    if g.str("prism.hadamard.transform")? != "normalized-sylvester-walsh-hadamard"
        || g.str("prism.hadamard.sign_mode")? != "explicit"
    {
        return Err("unsupported prism.hadamard transform or sign mode".into());
    }
    if !hc0.had_block.is_power_of_two()
        || n_head % n_kv_head != 0
        || hc0.n_v_heads % hc0.n_k_heads != 0
    {
        return Err("implausible hybrid geometry".into());
    }
    let cfg0 = Config {
        arch: Arch::Qwen35,
        rope_neox: true,
        n_layer,
        n_head,
        n_kv_head,
        head_dim,
        d_model,
        n_ff,
        n_vocab: 0,
        rope_dim,
        eps: g
            .f32(&k("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6),
        rope_base: g.f32(&k("rope.freq_base")).unwrap_or(1e7),
        ctx_train: g.u32(&k("context_length")).unwrap_or(4096) as usize,
        hybrid: None,
    };
    // Sign vectors, one per input width, pre-divided by √block.
    let widths = match g.meta.get("prism.hadamard.sign_widths") {
        Some(crate::gguf::Value::ArrI32(v)) => v.iter().map(|&x| x as usize).collect::<Vec<_>>(),
        _ => return Err("prism.hadamard.sign_widths missing".into()),
    };
    let vals = g.arr_i32("prism.hadamard.sign_values")?;
    if vals.len() != widths.iter().sum::<usize>() {
        return Err("prism.hadamard.sign_values length does not match sign_widths".into());
    }
    let scale = 1.0 / (hc0.had_block as f32).sqrt();
    let mut signs = Vec::new();
    let mut at = 0;
    for &w in &widths {
        if w % hc0.had_block != 0 {
            return Err(format!(
                "sign width {w} is not a multiple of the Hadamard block"
            ));
        }
        signs.push((
            w,
            vals[at..at + w].iter().map(|&s| s as f32 * scale).collect(),
        ));
        at += w;
    }
    let v_grouped = g.bool("prism.hadamard.gdn_v_grouped").unwrap_or(false);

    let get = |n: &str| -> Result<TensorInfo, String> { g.tensor(n).cloned() };
    let vecf = |n: &str| -> Result<Vec<f32>, String> { Ok(f32_vec(&g, g.tensor(n)?)) };
    let ternary = |t: &TensorInfo| matches!(t.kind, Format::PQ2_0 | Format::PTQ1_0);
    let mut layers = Vec::with_capacity(n_layer);
    let (mut n_gdn, mut n_attn) = (0usize, 0usize);
    for i in 0..n_layer {
        let p = |s: &str| format!("blk.{i}.{s}");
        let mix = if hc0.is_attn(i) {
            n_attn += 1;
            Mixer::Attn {
                q: get(&p("attn_q.weight"))?,
                k: get(&p("attn_k.weight"))?,
                v: get(&p("attn_v.weight"))?,
                q_norm: vecf(&p("attn_q_norm.weight"))?,
                k_norm: vecf(&p("attn_k_norm.weight"))?,
                o: get(&p("attn_output.weight"))?,
                idx: n_attn - 1,
            }
        } else {
            n_gdn += 1;
            let mut ab = vecf(&p("ssm_beta.weight"))?;
            ab.extend(vecf(&p("ssm_alpha.weight"))?);
            Mixer::Gdn {
                qkv: get(&p("attn_qkv.weight"))?,
                gate: get(&p("attn_gate.weight"))?,
                ab,
                conv: {
                    // Stored `[ch][tap]`; kept `[tap][ch]` so the conv vectorises across channels.
                    let w = vecf(&p("ssm_conv1d.weight"))?;
                    let (kk, cd) = (hc0.d_conv, hc0.conv_dim());
                    let mut t = vec![0.0f32; kk * cd];
                    for ch in 0..cd {
                        for tap in 0..kk {
                            t[tap * cd + ch] = w[ch * kk + tap];
                        }
                    }
                    t
                },
                dt_bias: vecf(&p("ssm_dt.bias"))?,
                a: vecf(&p("ssm_a"))?,
                norm: vecf(&p("ssm_norm.weight"))?,
                out: get(&p("ssm_out.weight"))?,
                idx: n_gdn - 1,
            }
        };
        let l = HLayer {
            attn_norm: get(&p("attn_norm.weight"))?,
            mix,
            post_norm: get(&p("post_attention_norm.weight"))?,
            gate: get(&p("ffn_gate.weight"))?,
            up: get(&p("ffn_up.weight"))?,
            down: get(&p("ffn_down.weight"))?,
        };
        for t in [&l.gate, &l.up, &l.down] {
            if !ternary(t) {
                return Err(format!(
                    "{} is {}; the hybrid path runs ternary matmuls",
                    t.name,
                    t.kind.label()
                ));
            }
        }
        layers.push(l);
    }
    // One activation layout per model: every ternary matmul must share a format.
    let mut kinds: Vec<Format> = Vec::new();
    for l in &layers {
        let mut ts: Vec<&TensorInfo> = vec![&l.gate, &l.up, &l.down];
        match &l.mix {
            Mixer::Gdn { qkv, gate, out, .. } => ts.extend([qkv, gate, out]),
            Mixer::Attn { q, k, v, o, .. } => ts.extend([q, k, v, o]),
        }
        for t in ts {
            if !kinds.contains(&t.kind) {
                kinds.push(t.kind);
            }
        }
    }
    // The output head: `output.weight`, or with `prism.hadamard.tied_output` (format version 2: the 2B and 4B) the
    // latent token embedding itself, which the fork then treats as a forward-rotated weight like any other.
    let tied = g.bool("prism.hadamard.tied_output").unwrap_or(false);
    let head_name = if tied {
        "token_embd.weight"
    } else {
        "output.weight"
    };
    if tied && g.tensor("output.weight").is_ok() {
        return Err("prism.hadamard.tied_output with an output.weight present".into());
    }
    let output_kind = g.tensor(head_name)?.kind;
    if !kinds.contains(&output_kind) {
        kinds.push(output_kind);
    }
    let ptq1 = match kinds.as_slice() {
        [Format::PQ2_0] => false,
        [Format::PTQ1_0] => true,
        other => {
            return Err(format!(
            "the hybrid path needs one ternary format for every matmul; this file mixes {other:?}"
        ))
        }
    };
    // Repack every PQ2_0 matrix into 16-row tiles when the tile kernel can run (PEREGRINE_TILES=0 keeps the
    // per-row kernels). PEREGRINE_TB_SPAN picks the activation scale span: 32 (default, the reference fork's
    // granularity), 64 or 128 (faster, coarser).
    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
    // mutated only by the x86 repack below
    let mut output = g.tensor(head_name)?.clone();
    #[allow(unused_mut)]
    let mut tiles = TileStore::default();
    #[allow(unused_mut)]
    let mut tb_span = 0usize;
    #[cfg(target_arch = "x86_64")]
    {
        let want = !ptq1 && crate::table::MatKernels::get().tiles() && crate::options::get().tiles;
        let mut mats: Vec<&mut TensorInfo> = Vec::new();
        for l in layers.iter_mut() {
            mats.extend([&mut l.gate, &mut l.up, &mut l.down]);
            match &mut l.mix {
                Mixer::Gdn { qkv, gate, out, .. } => mats.extend([qkv, gate, out]),
                Mixer::Attn { q, k, v, o, .. } => mats.extend([q, k, v, o]),
            }
        }
        mats.push(&mut output);
        let ok = mats
            .iter()
            .all(|t| t.dims.get(1).is_some_and(|r| r % 16 == 0) && t.dims[0] % 128 == 0);
        if want && ok {
            let total: usize = mats.iter().map(|t| t.bytes() as usize).sum();
            tiles = TileStore::new(total);
            let mut at = 0usize;
            for t in mats {
                let (n, rows) = (t.dims[0] as usize, t.dims[1] as usize);
                let rb = row_bytes(t.kind, n);
                let len = rows * rb;
                crate::tern::repack_tiles(
                    g.bytes(t),
                    rows,
                    rb,
                    &mut tiles.bytes_mut()[at..at + len],
                );
                t.tile_off = Some(at);
                at += len;
            }
            tb_span = crate::options::get().tb_span;
        }
    }
    let hc = HybridCfg {
        n_gdn,
        n_attn,
        ptq1,
        tb_span,
        ..hc0
    };
    let tok_embd = get("token_embd.weight")?;
    let out_norm = get("output_norm.weight")?;
    let tk = Tokenizer::from_gguf(&g)?;
    let cfg = Config {
        n_vocab: tk.n_vocab,
        hybrid: Some(hc),
        ..cfg0
    };
    if tok_embd.dims.get(1).copied().unwrap_or(0) as usize != cfg.n_vocab {
        return Err("token_embd rows ≠ vocabulary".into());
    }
    // The activation quantisers work in groups of 128 (pairs of groups, 256, for the PTQ1_0 layout): refuse a file
    // whose widths are not, here rather than as a panic at the first token.
    let unit = if ptq1 { 256 } else { 128 };
    for (what, w) in [
        ("embedding", d_model),
        ("feed-forward", n_ff),
        ("value", hc.value_dim()),
        ("query", cfg.q_dim()),
    ] {
        if w % unit != 0 {
            return Err(format!("{what} width {w} is not a multiple of {unit}"));
        }
    }
    let hy = Hybrid {
        cfg: hc,
        layers,
        signs,
        v_grouped,
    };
    for w in [d_model, n_ff, hc.value_dim(), cfg.q_dim()] {
        hy.signs_for(w)?;
    }
    let mtp = if n_nextn == 1 {
        Some(mtp::Mtp::load(&g, n_layer)?)
    } else {
        None
    };
    Ok(Model {
        g,
        cfg,
        tk,
        tok_embd,
        out_norm,
        head: Linear::Dense(output),
        lb: None,
        lb_replaced: 0,
        lb_count: 0,
        rope_factors: None,
        layers: Vec::new(),
        mtp,
        hy: Some(hy),
        tiles,
    })
}

impl Hybrid {
    fn signs_for(&self, width: usize) -> Result<&[f32], String> {
        self.signs
            .iter()
            .find(|(w, _)| *w == width)
            .map(|(_, s)| s.as_slice())
            .ok_or_else(|| format!("no Hadamard sign vector for input width {width}"))
    }
}

/// Runs of at most this many rows of one cache attend one row at a time (see the attention call).
const SHORT_RUN: usize = 4;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

impl Model {
    /// Row `token` of the (latent) embedding table in the primal basis: dequantised, then `s ⊙ H·row`.
    pub(super) fn embed_primal(
        &self,
        token: u32,
        row: &mut [f32],
        stats: &mut StepStats,
    ) -> Result<(), String> {
        let hy = self.hy();
        let se = hy.signs_for(self.cfg.d_model)?;
        self.embed(token, row, stats);
        for ch in row.chunks_exact_mut(hy.cfg.had_block) {
            fwht(ch);
        }
        for (v, s) in row.iter_mut().zip(se) {
            *v *= s;
        }
        Ok(())
    }

    /// The output head over its first `rows` rows only, from normed rows `y` (`[nb][d]`), into `logits`
    /// (`[nb][rows]`): the rotated input, then the tiled ternary matmul (the MTP draft head).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn head_rows(
        &self,
        hs: &mut HyScratch,
        acts: &mut [QuantAct],
        y: &[f32],
        nb: usize,
        rows: usize,
        logits: &mut [f32],
        pool: &Pool,
        stats: &mut StepStats,
    ) -> Result<(), String> {
        let d = self.cfg.d_model;
        let Linear::Dense(h) = &self.head else {
            return Err("the hybrid head is dense".into());
        };
        let rows = rows.min(h.dims[1] as usize) / 16 * 16;
        self.rotate_rows(hs, acts, y, d, nb, false, pool, stats);
        // The head reads only its first `rows` output rows; narrowing the job keeps the full `TensorInfo`
        // borrowed instead of cloning it, which would allocate on every draft step.
        self.mm_jobs(
            &[MatJob {
                w: h,
                out: OutPtr(logits.as_mut_ptr()),
                rows: Some(rows),
            }],
            &acts[..nb],
            nb,
            pool,
            stats,
        );
        Ok(())
    }

    pub(super) fn hy_cfg(&self) -> HybridCfg {
        self.hy().cfg
    }

    fn hy(&self) -> &Hybrid {
        self.hy.as_ref().expect("hybrid model")
    }

    /// `x` (row `b` of `src`, width `w`) → optional tiled→grouped v-head permutation → signs → blockwise FWHT, into
    /// `rot`, then quantised into `acts[b]`. Rows are independent and split across the pool.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rotate_rows(
        &self,
        hs: &mut HyScratch,
        acts: &mut [QuantAct],
        src: &[f32],
        w: usize,
        nb: usize,
        perm: bool,
        pool: &Pool,
        stats: &mut StepStats,
    ) {
        self.rotate_rows_pre(
            hs,
            acts,
            SharedBuf(src.as_ptr() as *mut f32),
            w,
            nb,
            perm,
            None,
            pool,
            stats,
        );
    }

    /// `rotate_rows` with `pre(b, row)` run on row `b` of `src` first, in the same pass. Fusing a row's norm,
    /// activation or gating into its rotation saves a fork–join per use, which on a VM under load costs tens of µs.
    ///
    /// `src` must hold `nb` rows of `w` floats that nothing else touches during the call (only `pre` writes them).
    #[allow(clippy::too_many_arguments)]
    fn rotate_rows_pre(
        &self,
        hs: &mut HyScratch,
        acts: &mut [QuantAct],
        src: SharedBuf,
        w: usize,
        nb: usize,
        perm: bool,
        pre: Option<&RowPrep>,
        pool: &Pool,
        stats: &mut StepStats,
    ) {
        let hy = self.hy();
        let hc = hy.cfg;
        let _t = Tick::new(2);
        let ss = hy.signs_for(w).expect("sign vector checked at load");
        let blk = hc.had_block;
        let rw = hs.rot_w;
        let rot = SharedBuf(hs.rot.as_mut_ptr());
        let ap = SharedActs(acts.as_mut_ptr());
        let next = std::sync::atomic::AtomicUsize::new(0);
        let body = |_wk: usize, _nw: usize| loop {
            let b = next.fetch_add(1, AO::Relaxed);
            if b >= nb {
                break;
            }
            // SAFETY: row `b` of `src`, of `rot` and `acts[b]` is claimed by this worker alone.
            let x = unsafe { src.slice(b * w, w) };
            if let Some(f) = pre {
                f(b, x);
            }
            let r = unsafe { rot.slice(b * rw, w) };
            if perm {
                let (hd, nk, rep) = (hc.dv, hc.n_k_heads, hc.n_v_heads / hc.n_k_heads);
                for rr in 0..rep {
                    for gg in 0..nk {
                        let s = (rr * nk + gg) * hd;
                        let d = (gg * rep + rr) * hd;
                        r[d..d + hd].copy_from_slice(&x[s..s + hd]);
                    }
                }
                for (v, s) in r.iter_mut().zip(ss) {
                    *v *= s;
                }
            } else {
                for ((v, &xi), s) in r.iter_mut().zip(x.iter()).zip(ss) {
                    *v = xi * s;
                }
            }
            for c in r.chunks_exact_mut(blk) {
                fwht(c);
            }
            unsafe { ap.get(b) }.fill(r);
        };
        if nb >= ROW_PARALLEL && pool.width() > 1 {
            pool.run(body);
            stats.barriers += 1;
        } else {
            body(0, 1);
        }
    }

    pub(super) fn mm(
        &self,
        jobs: &[(&TensorInfo, *mut f32)],
        acts: &[QuantAct],
        nb: usize,
        pool: &Pool,
        stats: &mut StepStats,
    ) {
        assert!(!jobs.is_empty() && jobs.len() <= 4);
        let mj: [MatJob; 4] = std::array::from_fn(|i| {
            let (w, o) = jobs[i.min(jobs.len() - 1)];
            MatJob {
                w,
                out: OutPtr(o),
                rows: None,
            }
        });
        let mj = &mj[..jobs.len()];
        self.mm_jobs(mj, acts, nb, pool, stats);
    }

    /// One matmul group with its jobs already described, so a caller can narrow a matrix's output rows (the MTP
    /// draft head) without cloning its `TensorInfo` — that clone allocated a `String` and a `Vec` per draft step.
    fn mm_jobs(
        &self,
        mj: &[MatJob],
        acts: &[QuantAct],
        nb: usize,
        pool: &Pool,
        stats: &mut StepStats,
    ) {
        let _t = Tick::new(3);
        let t0 = prof_on().then(Instant::now);
        self.matmul_group(mj, acts, nb, pool, stats);
        if let Some(t0) = t0 {
            let bytes: u64 = mj.iter().map(|j| j.w.bytes()).sum();
            let k = if bytes < 8 << 20 {
                10
            } else if bytes < 20 << 20 {
                12
            } else {
                14
            };
            PROF[k].fetch_add(t0.elapsed().as_nanos() as u64, AO::Relaxed);
            PROF[k + 1].fetch_add(bytes, AO::Relaxed);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_hybrid(
        &self,
        st8: &mut State,
        caches: &mut Caches,
        pool: &Pool,
        tokens: &[u32],
        positions: &[usize],
        want_logits: &[usize],
        stats: &mut StepStats,
    ) -> Result<(), String> {
        let t_fwd = Instant::now();
        let c = self.cfg;
        let hy = self.hy();
        let hc = hy.cfg;
        let nb = tokens.len();
        let d = c.d_model;
        let mut hs = st8.hy.take().ok_or("state has no hybrid scratch")?;
        for b in 0..nb {
            if !caches.get(b).has_recurrent() {
                st8.hy = Some(hs);
                return Err("the cache has no recurrent state (KvCache::with_recurrent)".into());
            }
        }
        // A recurrent state advances one token at a time: each cache's rows must be one run of consecutive
        // positions starting where the cache ends, or the state would silently describe another sequence.
        {
            caches.runs_into(nb, &mut hs.runs);
            let mut err = None;
            {
                let seen = &mut hs.seen;
                seen.clear();
                for &(r0, len, ci) in &hs.runs {
                    let ok = !seen.contains(&ci)
                        && positions[r0] == caches.cache(ci).filled()
                        && (1..len).all(|k| positions[r0 + k] == positions[r0] + k);
                    if !ok {
                        err = Some(format!(
                            "rows {r0}..{} do not continue their recurrent cache in order (cache holds {}, row starts at {})",
                            r0 + len,
                            caches.cache(ci).filled(),
                            positions[r0]
                        ));
                        break;
                    }
                    seen.push(ci);
                }
            }
            if let Some(e) = err {
                st8.hy = Some(hs);
                return Err(e);
            }
        }

        // ---- embedding: dequantised row, then the inverse rotation `s ⊙ H·row` ----
        let te = Tick::new(0);
        let se = hy.signs_for(d)?;
        for (i, &t) in tokens.iter().enumerate() {
            let row = &mut st8.x[i * d..(i + 1) * d];
            self.embed(t, row, stats);
            for ch in row.chunks_exact_mut(hc.had_block) {
                fwht(ch);
            }
            for (v, s) in row.iter_mut().zip(se) {
                *v *= s;
            }
        }

        drop(te);
        for l in &hy.layers {
            // ---- mixer ----
            {
                // The norm runs inside the rotation's pass (one fork–join); `xb` keeps the unrotated normed rows.
                let (xs, nw): (&[f32], &TensorInfo) = (&st8.x, &l.attn_norm);
                let pre =
                    |b: usize, row: &mut [f32]| self.rmsnorm_into(&xs[b * d..(b + 1) * d], nw, row);
                let xbp = SharedBuf(st8.xb.as_mut_ptr());
                self.rotate_rows_pre(
                    &mut hs,
                    &mut st8.acts,
                    xbp,
                    d,
                    nb,
                    false,
                    Some(&pre),
                    pool,
                    stats,
                );
                stats.bytes_weights += l.attn_norm.bytes();
            }
            match &l.mix {
                Mixer::Gdn {
                    qkv,
                    gate,
                    ab,
                    conv,
                    dt_bias,
                    a,
                    norm,
                    out,
                    idx,
                } => {
                    self.mm(
                        &[(qkv, hs.big.as_mut_ptr()), (gate, hs.z.as_mut_ptr())],
                        &st8.acts[..nb],
                        nb,
                        pool,
                        stats,
                    );
                    // `qkv` writes rows of `conv_dim`, `z` rows of `value_dim`: restride to the scratch widths.
                    restride(&mut hs.big, hc.conv_dim(), hs.big_w, nb);
                    restride(&mut hs.z, hc.value_dim(), hs.z_w, nb);
                    self.gdn(
                        &mut hs, st8, caches, ab, conv, dt_bias, a, norm, *idx, positions, nb,
                        pool, stats,
                    );
                    let vd = hc.value_dim();
                    let mut o = std::mem::take(&mut hs.o);
                    let z = std::mem::take(&mut hs.z);
                    {
                        let zw = hs.z_w;
                        let zr: &[f32] = &z;
                        let pre = |b: usize, row: &mut [f32]| {
                            self.gated_norm_row(row, &zr[b * zw..b * zw + vd], norm)
                        };
                        let op = SharedBuf(o.as_mut_ptr());
                        self.rotate_rows_pre(
                            &mut hs,
                            &mut st8.acts,
                            op,
                            vd,
                            nb,
                            hy.v_grouped,
                            Some(&pre),
                            pool,
                            stats,
                        );
                    }
                    hs.o = o;
                    hs.z = z;
                    self.mm(
                        &[(out, st8.xb2.as_mut_ptr())],
                        &st8.acts[..nb],
                        nb,
                        pool,
                        stats,
                    );
                }
                Mixer::Attn {
                    q,
                    k,
                    v,
                    q_norm,
                    k_norm,
                    o,
                    idx,
                } => {
                    let (qd, kvd, hd) = (c.q_dim(), c.kv_dim(), c.head_dim);
                    self.mm(
                        &[
                            (q, hs.big.as_mut_ptr()),
                            (k, st8.k.as_mut_ptr()),
                            (v, st8.v.as_mut_ptr()),
                        ],
                        &st8.acts[..nb],
                        nb,
                        pool,
                        stats,
                    );
                    // q and gate are interleaved per head: [q_h (hd) | gate_h (hd)].
                    let ta = Tick::new(8);
                    {
                        let (qp, kp, zp) = (
                            SharedBuf(st8.q.as_mut_ptr()),
                            SharedBuf(st8.k.as_mut_ptr()),
                            SharedBuf(hs.z.as_mut_ptr()),
                        );
                        let (big, rope, zw): (&[f32], &[f32], usize) = (&hs.big, &st8.rope, hs.z_w);
                        let next = std::sync::atomic::AtomicUsize::new(0);
                        let body = |_wk: usize, _nw: usize| loop {
                            let b = next.fetch_add(1, AO::Relaxed);
                            if b >= nb {
                                break;
                            }
                            // SAFETY: row b of q, k and z is claimed by this worker.
                            let (q, k, z) = unsafe {
                                (
                                    qp.slice(b * qd, qd),
                                    kp.slice(b * kvd, kvd),
                                    zp.slice(b * zw, qd),
                                )
                            };
                            for h in 0..c.n_head {
                                let src = b * 2 * qd + h * 2 * hd;
                                q[h * hd..(h + 1) * hd].copy_from_slice(&big[src..src + hd]);
                                z[h * hd..(h + 1) * hd]
                                    .copy_from_slice(&big[src + hd..src + 2 * hd]);
                            }
                            for row in q.chunks_exact_mut(hd) {
                                head_rmsnorm(row, q_norm, c.eps);
                            }
                            for row in k.chunks_exact_mut(hd) {
                                head_rmsnorm(row, k_norm, c.eps);
                            }
                            self.rope(rope, q, c.n_head, positions[b]);
                            self.rope(rope, k, c.n_kv_head, positions[b]);
                        };
                        if nb >= ROW_PARALLEL && pool.width() > 1 {
                            pool.run(body);
                            stats.barriers += 1;
                        } else {
                            body(0, 1);
                        }
                    }
                    for b in 0..nb {
                        let kv = caches.get_mut(b);
                        for h in 0..c.n_kv_head {
                            let ks = b * kvd + h * hd;
                            kv.put(
                                *idx,
                                h,
                                positions[b],
                                &st8.k[ks..ks + hd],
                                &st8.v[ks..ks + hd],
                            );
                        }
                    }
                    for b in 0..nb {
                        caches.get_mut(b).advance(positions[b] + 1);
                    }
                    let t_attn = Instant::now();
                    for (r0, len, ci) in caches.runs(nb) {
                        // A short run is a speculative verification: one row per call, the same call shape as a
                        // one-row decode step, so a verified row's attention equals the step it replaces bit for
                        // bit. Prompt chunks keep the grouped call.
                        if len <= SHORT_RUN {
                            for k in 0..len {
                                self.attend(
                                    st8,
                                    caches.cache(ci),
                                    pool,
                                    *idx,
                                    1,
                                    positions[r0 + k],
                                    r0 + k,
                                    stats,
                                );
                            }
                        } else {
                            self.attend(
                                st8,
                                caches.cache(ci),
                                pool,
                                *idx,
                                len,
                                positions[r0],
                                r0,
                                stats,
                            );
                        }
                    }
                    stats.ns_attn += t_attn.elapsed().as_nanos() as u64;
                    drop(ta);
                    {
                        // The sigmoid gate inside the attn_output rotation's pass.
                        let z = std::mem::take(&mut hs.z);
                        let zw = hs.z_w;
                        let zr: &[f32] = &z;
                        let pre = |b: usize, row: &mut [f32]| {
                            crate::vmath::mul_sigmoid(row, &zr[b * zw..b * zw + qd])
                        };
                        let xbp = SharedBuf(st8.xb.as_mut_ptr());
                        self.rotate_rows_pre(
                            &mut hs,
                            &mut st8.acts,
                            xbp,
                            qd,
                            nb,
                            false,
                            Some(&pre),
                            pool,
                            stats,
                        );
                        hs.z = z;
                    }
                    self.mm(
                        &[(o, st8.xb2.as_mut_ptr())],
                        &st8.acts[..nb],
                        nb,
                        pool,
                        stats,
                    );
                }
            }
            for i in 0..nb * d {
                st8.x[i] += st8.xb2[i];
            }

            // ---- feed-forward ----
            {
                // The norm runs inside the rotation's pass (one fork–join); `xb` keeps the unrotated normed rows.
                let (xs, nw): (&[f32], &TensorInfo) = (&st8.x, &l.post_norm);
                let pre =
                    |b: usize, row: &mut [f32]| self.rmsnorm_into(&xs[b * d..(b + 1) * d], nw, row);
                let xbp = SharedBuf(st8.xb.as_mut_ptr());
                self.rotate_rows_pre(
                    &mut hs,
                    &mut st8.acts,
                    xbp,
                    d,
                    nb,
                    false,
                    Some(&pre),
                    pool,
                    stats,
                );
                stats.bytes_weights += l.post_norm.bytes();
            }
            self.mm(
                &[
                    (&l.gate, st8.hb.as_mut_ptr()),
                    (&l.up, st8.hb2.as_mut_ptr()),
                ],
                &st8.acts[..nb],
                nb,
                pool,
                stats,
            );
            let f = c.n_ff;
            if nb >= ROW_PARALLEL {
                // Several rows: each row's SwiGLU inside its rotation's pass.
                let up: &[f32] = &st8.hb2;
                let pre =
                    |b: usize, row: &mut [f32]| crate::vmath::swiglu(row, &up[b * f..(b + 1) * f]);
                let hbp = SharedBuf(st8.hb.as_mut_ptr());
                self.rotate_rows_pre(
                    &mut hs,
                    &mut st8.acts,
                    hbp,
                    f,
                    nb,
                    false,
                    Some(&pre),
                    pool,
                    stats,
                );
            } else {
                let tf = Tick::new(9);
                {
                    // One row: its 12 288 exps split across the pool, then the rotation.
                    let total = nb * f;
                    let g = SharedBuf(st8.hb.as_mut_ptr());
                    let up: &[f32] = &st8.hb2;
                    pool.run(|wk, nw| {
                        let (lo, hi) = share(total / 8, wk, nw);
                        let (lo, hi) = (lo * 8, if wk + 1 == nw { total } else { hi * 8 });
                        // SAFETY: `share` gives each worker a disjoint range.
                        let gs = unsafe { g.slice(lo, hi - lo) };
                        crate::vmath::swiglu(gs, &up[lo..hi]);
                    });
                    stats.barriers += 1;
                }
                drop(tf);
                let hb = std::mem::take(&mut st8.hb);
                self.rotate_rows(&mut hs, &mut st8.acts, &hb, f, nb, false, pool, stats);
                st8.hb = hb;
            }
            self.mm(
                &[(&l.down, st8.xb2.as_mut_ptr())],
                &st8.acts[..nb],
                nb,
                pool,
                stats,
            );
            for i in 0..nb * d {
                st8.x[i] += st8.xb2[i];
            }
        }

        if st8.want_hidden {
            st8.hidden.resize(nb * d, 0.0);
            for b in 0..nb {
                let (xv, hv) = (&st8.x, &mut st8.hidden);
                self.rmsnorm_into(
                    &xv[b * d..(b + 1) * d],
                    &self.out_norm,
                    &mut hv[b * d..(b + 1) * d],
                );
            }
        }
        // ---- output head ----
        if !want_logits.is_empty() {
            let nl = want_logits.len();
            if nl * c.n_vocab > st8.logits.len() {
                st8.hy = Some(hs);
                return Err(format!(
                    "logits requested for {nl} rows; scratch holds fewer"
                ));
            }
            for (i, &b) in want_logits.iter().enumerate() {
                if b >= nb {
                    st8.hy = Some(hs);
                    return Err(format!("logits requested for row {b} of a {nb}-row batch"));
                }
                let (xv, xbv) = (&st8.x, &mut st8.xb);
                self.rmsnorm_into(
                    &xv[b * d..(b + 1) * d],
                    &self.out_norm,
                    &mut xbv[i * d..(i + 1) * d],
                );
            }
            stats.bytes_weights += self.out_norm.bytes();
            let xb = std::mem::take(&mut st8.xb);
            self.rotate_rows(&mut hs, &mut st8.acts, &xb, d, nl, false, pool, stats);
            st8.xb = xb;
            let before = stats.ns_matmul;
            if let Linear::Dense(h) = &self.head {
                self.mm(
                    &[(h, st8.logits.as_mut_ptr())],
                    &st8.acts[..nl],
                    nl,
                    pool,
                    stats,
                );
            }
            stats.ns_head += stats.ns_matmul - before;
            stats.ns_matmul = before;
        }
        st8.hy = Some(hs);
        stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
        Ok(())
    }

    /// The Gated DeltaNet core for one layer: conv, gates, delta rule and gated norm, from `hs.big` (qkv), `hs.z`
    /// and the unrotated normed input `st8.xb`, into `hs.o`.
    #[allow(clippy::too_many_arguments)]
    fn gdn(
        &self,
        hs: &mut HyScratch,
        st8: &mut State,
        caches: &mut Caches,
        ab_w: &[f32],
        conv_w: &[f32],
        dt_bias: &[f32],
        a_log: &[f32],
        norm_w: &[f32],
        idx: usize,
        positions: &[usize],
        nb: usize,
        pool: &Pool,
        stats: &mut StepStats,
    ) {
        let c = self.cfg;
        let hc = self.hy().cfg;
        let (d, nv, nk, dk, dv) = (c.d_model, hc.n_v_heads, hc.n_k_heads, hc.dk, hc.dv);
        let (kd, vd, cd, taps) = (hc.key_dim(), hc.value_dim(), hc.conv_dim(), hc.d_conv - 1);

        // β and α: `[2·n_v][d]` f32 against the unrotated normed rows, split across the pool by output.
        {
            let _t = Tick::new(4);
            let xb: &[f32] = &st8.xb;
            let out = SharedBuf(hs.ab.as_mut_ptr());
            let n_out = 2 * nv;
            let next = std::sync::atomic::AtomicUsize::new(0);
            pool.run(|_wk, _nw| loop {
                let lo = next.fetch_add(4, AO::Relaxed);
                if lo >= n_out {
                    break;
                }
                for r in lo..(lo + 4).min(n_out) {
                    let w = &ab_w[r * d..(r + 1) * d];
                    for b in 0..nb {
                        // SAFETY: output (b, r) belongs to the worker claiming r.
                        let v = crate::vmath::dot_f32(w, &xb[b * d..(b + 1) * d]);
                        unsafe { out.slice(b * n_out + r, 1)[0] = v };
                    }
                }
            });
            stats.barriers += 1;
            stats.bytes_weights += (ab_w.len() * 4) as u64;
            stats.macs += (ab_w.len() * nb) as u64;
        }

        // Conv, SiLU, L2 norms and the gate maths, row by row in order: a shared cache's conv history advances
        // one token per row.
        let q_scale = 1.0 / (dk as f32).sqrt();
        let tc = Tick::new(5);
        // Conv: channels are independent, so a worker takes a channel range and runs every row through it in
        // order (a shared cache's history advances one token per row). The history is a ring of `taps` planes of
        // `cd` channels: the input at position p sits in plane p mod taps, so the oldest of the window ending at
        // `pos` is plane pos mod taps, the one this token overwrites after the sum.
        {
            let (rp, spp) = {
                rec_ptrs(caches, idx, nb, &mut hs.runs, &mut hs.ptrs.rec);
                snap_ptrs(caches, idx, nb, &mut hs.runs, &mut hs.ptrs.snap);
                (
                    RowPtrs(hs.ptrs.rec.as_ptr()),
                    SnapPtrs(hs.ptrs.snap.as_ptr()),
                )
            };
            let bigp = SharedBuf(hs.big.as_mut_ptr());
            let bw = hs.big_w;
            let wcur = &conv_w[taps * cd..(taps + 1) * cd];
            const C: usize = 256;
            let next = std::sync::atomic::AtomicUsize::new(0);
            pool.run(|_wk, _nw| {
                let mut acc = [0.0f32; C];
                loop {
                    let c0 = next.fetch_add(C, AO::Relaxed);
                    if c0 >= cd {
                        break;
                    }
                    let c1 = (c0 + C).min(cd);
                    for b in 0..nb {
                        // SAFETY: channels c0..c1 of every row's buffer and history are this worker's.
                        let (_, hp) = unsafe { rp.at(b) };
                        let pos = positions[b];
                        let y = unsafe { bigp.slice(b * bw + c0, c1 - c0) };
                        for (i, ch) in (c0..c1).enumerate() {
                            acc[i] = y[i] * wcur[ch];
                        }
                        for t in 0..taps {
                            let plane = (pos + t) % taps;
                            let hist = unsafe {
                                std::slice::from_raw_parts(hp.add(plane * cd + c0), c1 - c0)
                            };
                            let wt = &conv_w[t * cd + c0..t * cd + c1];
                            for i in 0..c1 - c0 {
                                acc[i] += hist[i] * wt[i];
                            }
                        }
                        let slot = pos % taps;
                        let dst = unsafe {
                            std::slice::from_raw_parts_mut(hp.add(slot * cd + c0), c1 - c0)
                        };
                        dst.copy_from_slice(y);
                        y.copy_from_slice(&acc[..c1 - c0]);
                        // A speculative verification's snapshot: this range of the history after its row.
                        if let Some((sp, _, sconv)) = unsafe { spp.at(b) } {
                            if pos == sp {
                                for t in 0..taps {
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(
                                            hp.add(t * cd + c0),
                                            sconv.add(t * cd + c0),
                                            c1 - c0,
                                        )
                                    };
                                }
                            }
                        }
                    }
                }
            });
            stats.barriers += 1;
        }
        // SiLU, the q/k L2 norms and the gate maths, per row (rows are independent now).
        {
            let bigp = SharedBuf(hs.big.as_mut_ptr());
            let abp = SharedBuf(hs.ab.as_mut_ptr());
            let bw = hs.big_w;
            let next = std::sync::atomic::AtomicUsize::new(0);
            let body = |_wk: usize, _nw: usize| loop {
                let b = next.fetch_add(1, AO::Relaxed);
                if b >= nb {
                    break;
                }
                // SAFETY: row b of `big` and of `ab` is claimed by this worker.
                let y = unsafe { bigp.slice(b * bw, cd) };
                crate::vmath::silu(y);
                for (base, scale) in [(0usize, q_scale), (kd, 1.0)] {
                    for h in 0..nk {
                        let v = &mut y[base + h * dk..base + (h + 1) * dk];
                        let n = crate::vmath::sumsq_f64(v).sqrt() as f32;
                        let inv = 1.0 / n.max(1e-6) * scale;
                        for x in v.iter_mut() {
                            *x *= inv;
                        }
                    }
                }
                let ab = unsafe { abp.slice(b * 2 * nv, 2 * nv) };
                for h in 0..nv {
                    let beta = sigmoid(ab[h]);
                    let g = a_log[h] * softplus(ab[nv + h] + dt_bias[h]);
                    ab[h] = beta;
                    ab[nv + h] = g.exp();
                }
            };
            if nb >= ROW_PARALLEL && pool.width() > 1 {
                pool.run(body);
                stats.barriers += 1;
            } else {
                body(0, 1);
            }
        }

        drop(tc);
        let td = Tick::new(6);
        // Delta rule: tasks are v-heads; a task runs its head through every row in order.
        let (rp, spp) = {
            rec_ptrs(caches, idx, nb, &mut hs.runs, &mut hs.ptrs.rec);
            snap_ptrs(caches, idx, nb, &mut hs.runs, &mut hs.ptrs.snap);
            (
                RowPtrs(hs.ptrs.rec.as_ptr()),
                SnapPtrs(hs.ptrs.snap.as_ptr()),
            )
        };
        let big: &[f32] = &hs.big;
        let abv: &[f32] = &hs.ab;
        let out = SharedBuf(hs.o.as_mut_ptr());
        let bw = hs.big_w;
        let shared = matches!(caches, Caches::Shared(_));
        // Two tasks per head (each half of the state's rows), claimed dynamically so fast cores take more.
        const SPLIT: usize = 2;
        let next = std::sync::atomic::AtomicUsize::new(0);
        pool.run(|_wk, _nw| loop {
            let task = next.fetch_add(1, AO::Relaxed);
            if task >= nv * SPLIT {
                break;
            }
            let (hv, part) = (task / SPLIT, task % SPLIT);
            let (j0, j1) = (part * dv / SPLIT, (part + 1) * dv / SPLIT);
            for b in 0..nb {
                // SAFETY: `ptrs` outlives the call; rows j0..j1 of head `hv` of row `b`'s state are this task's.
                let (sp, _) = unsafe { rp.at(b) };
                let s = unsafe {
                    std::slice::from_raw_parts_mut(sp.add((hv * dv + j0) * dk), (j1 - j0) * dk)
                };
                let y = &big[b * bw..];
                let kh = hv % nk;
                let q = &y[kh * dk..(kh + 1) * dk];
                let k = &y[kd + kh * dk..kd + (kh + 1) * dk];
                let v = &y[2 * kd + hv * dv..2 * kd + (hv + 1) * dv];
                let beta = abv[b * 2 * nv + hv];
                let decay = abv[b * 2 * nv + nv + hv];
                let o = unsafe { out.slice(b * vd + hv * dv + j0, j1 - j0) };
                crate::delta::delta_head(s, q, k, &v[j0..j1], beta, decay, o, dk);
                // A speculative verification's snapshot: this task's slice of the state after its row.
                if let Some((sp, sstate, _)) = unsafe { spp.at(b) } {
                    if positions[b] == sp {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                s.as_ptr(),
                                sstate.add((hv * dv + j0) * dk),
                                s.len(),
                            )
                        };
                    }
                }
            }
        });
        let _ = shared;
        stats.barriers += 1;
        stats.bytes_kv += (2
            * hc.state_floats()
            * 4
            * if matches!(caches, Caches::Shared(_)) {
                1
            } else {
                nb
            }) as u64;
        stats.macs += (4 * nv * dv * dk * nb) as u64;

        drop(td);
        // The gated RMSNorm runs in the ssm_out rotation's pass (`gated_norm_row`).
        let _ = norm_w;
    }

    /// Gated RMSNorm of one row of the GDN output, per head: rms(o_h)·w ⊙ silu(z_h).
    fn gated_norm_row(&self, o: &mut [f32], z: &[f32], norm_w: &[f32]) {
        let hc = self.hy().cfg;
        let (nv, dv) = (hc.n_v_heads, hc.dv);
        for h in 0..nv {
            let oh = &mut o[h * dv..(h + 1) * dv];
            let zh = &z[h * dv..(h + 1) * dv];
            let ss = crate::vmath::sumsq_f64(oh) / dv as f64;
            let inv = (1.0 / (ss + self.cfg.eps as f64).sqrt()) as f32;
            for i in 0..dv {
                oh[i] *= inv * norm_w[i];
            }
            crate::vmath::mul_silu(oh, zh);
        }
    }
}

/// Every row's (state, conv) pointers for recurrent layer `idx`, taken once per cache (rows of one cache share
/// them): re-borrowing a cache per row would invalidate the pointers taken before. `runs` and `out` are reused
/// scratch — a decode step runs this four times per layer and must not allocate.
fn rec_ptrs(
    caches: &mut Caches,
    idx: usize,
    nb: usize,
    runs: &mut Vec<(usize, usize, usize)>,
    out: &mut [(*mut f32, *mut f32)],
) {
    caches.runs_into(nb, runs);
    for &(r0, len, ci) in runs.iter() {
        let p = caches.cache_mut(ci).recurrent_ptrs(idx);
        for slot in &mut out[r0..r0 + len] {
            *slot = p;
        }
    }
}

/// Per row, the snapshot request of its cache for recurrent layer `idx`: (position, state, conv) pointers.
#[derive(Clone, Copy)]
struct SnapPtrs(*const Option<(usize, *mut f32, *mut f32)>);
// SAFETY: as `RowPtrs`: each snapshot region is written by the task that owns it (a head slice, a channel range).
unsafe impl Send for SnapPtrs {}
unsafe impl Sync for SnapPtrs {}
impl SnapPtrs {
    /// # Safety
    /// `b` must be inside the array, which must outlive the call.
    #[inline(always)]
    unsafe fn at(self, b: usize) -> Option<(usize, *mut f32, *mut f32)> {
        *self.0.add(b)
    }
}

fn snap_ptrs(
    caches: &mut Caches,
    idx: usize,
    nb: usize,
    runs: &mut Vec<(usize, usize, usize)>,
    out: &mut [Option<(usize, *mut f32, *mut f32)>],
) {
    caches.runs_into(nb, runs);
    for &(r0, len, ci) in runs.iter() {
        let p = caches.cache_mut(ci).snapshot_ptrs(idx);
        for slot in &mut out[r0..r0 + len] {
            *slot = p;
        }
    }
}

/// Matmul outputs land at row stride `from` (the matrix's row count); move them to stride `to` in place.
fn restride(buf: &mut [f32], from: usize, to: usize, nb: usize) {
    if from == to {
        return;
    }
    for b in (0..nb).rev() {
        buf.copy_within(b * from..(b + 1) * from, b * to);
    }
}
