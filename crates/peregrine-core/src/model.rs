//! The Llama block, and the forward pass.
//!
//! # Why the matmul is batched even though decode is not
//!
//! `linear_group` takes a *slice* of activations and reads each weight row once for
//! the whole slice. Plain decode passes a slice of one, which is the GEMV case; prefill
//! passes the whole prompt; a speculative verification passes the drafted block.
//! Writing it this way is not premature generality — it is the entire mechanism by
//! which prefill and speculation are cheap. A decode step reads 267 MB of weights to
//! produce one token; running twenty prompt tokens through that path one at a time
//! would read 5.3 GB, where the batched path reads 267 MB once. The arithmetic
//! intensity of the batched call is `b` times higher, which is precisely what the
//! planner in `planner.rs` decides on.
//!
//! # Byte accounting
//!
//! Every weight read goes through `linear_group` or `embed`, and both add the exact
//! on-disk byte length of what they touched to `StepStats`. That total is the
//! number the engine reports as bytes moved, and it is a sum of tensor lengths
//! rather than an estimate from parameter count — the two differ by the block
//! metadata, which at Q4_K is 12.5% of the traffic.

use std::any::Any;

use crate::arch::{Architecture, MtpState};
use crate::blocks::attn::AttnScratch;
use crate::gguf::{Format, Gguf, TensorInfo};
use crate::kv::KvCache;
use crate::lbq::Slb;
use crate::pool::Pool;
use crate::quant::QuantAct;
use crate::tokenizer::Tokenizer;

#[allow(unused_imports)]
pub(crate) use crate::pool::share;

// Qwen3.5-style hybrid models (Bonsai 2).
pub mod hybrid;
// Bonsai 2's multi-token-prediction block (`mtp.rs`).
pub mod mtp;

/// The transformer families this engine implements. Qwen2 is llama with biases on
/// the attention projections and NeoX-style rotary pairing; everything else — norms,
/// SwiGLU, grouped-query attention — is identical, so it shares the forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Llama,
    Qwen2,
    /// Qwen3: no attention biases, a per-head RMSNorm on queries and keys before
    /// rotation, and a head width independent of the model width (Qwen3-4B runs 32
    /// heads of 128 over a 2560-wide residual). The targets DeepSeek's DSpark
    /// drafters were trained against.
    Qwen3,
    /// Qwen3.5-style hybrid (Gated DeltaNet + gated attention), PrismML's ternary Bonsai 2 (`hybrid.rs`).
    Qwen35,
}

impl Arch {
    pub fn from_name(name: &str) -> Option<Arch> {
        Some(match name {
            "llama" => Arch::Llama,
            "qwen2" => Arch::Qwen2,
            "qwen3" => Arch::Qwen3,
            "qwen35" => Arch::Qwen35,
            _ => return None,
        })
    }
    /// The metadata key prefix (`llama.block_count`, `qwen2.block_count`, …).
    pub fn key_prefix(self) -> &'static str {
        match self {
            Arch::Llama => "llama",
            Arch::Qwen2 => "qwen2",
            Arch::Qwen3 => "qwen3",
            Arch::Qwen35 => "qwen35",
        }
    }
}

/// Architecture names this crate implements, for the compatibility probe's message.
pub const ARCHS: &[&str] = &["llama", "qwen2", "qwen3", "qwen35"];

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub arch: Arch,
    /// NeoX rotary pairing (`i` with `i + rope_dim/2`) instead of adjacent pairs.
    pub rope_neox: bool,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub d_model: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub rope_dim: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub ctx_train: usize,
    // the linear-attention geometry of a hybrid model; None for the dense architectures.
    pub hybrid: Option<hybrid::HybridCfg>,
}

impl Config {
    // layers that keep K/V (a hybrid model's full-attention layers only).
    pub fn kv_layers(&self) -> usize {
        self.hybrid.map_or(self.n_layer, |h| h.n_attn)
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_head * self.head_dim
    }
    /// Width of the concatenated query heads — the attention output's width, which
    /// is `d_model` for llama and Qwen2 and not for Qwen3.
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }
    /// Query heads per key/value head. 1 means multi-head, >1 is grouped-query.
    pub fn gqa(&self) -> usize {
        self.n_head / self.n_kv_head
    }
}

/// What one forward call touched. Summed into the per-token telemetry record.
#[derive(Default, Clone, Copy, Debug)]
pub struct StepStats {
    /// Exact bytes of weight tensors read.
    pub bytes_weights: u64,
    /// Exact bytes read from the KV cache — `κ(T)`.
    pub bytes_kv: u64,
    /// Multiply-accumulates, counted rather than estimated. Two FLOPs each.
    pub macs: u64,
    /// Fork-join barriers executed. Proposition 5 of the paper claims this is not
    /// the binding constraint; reporting it is what makes that checkable.
    pub barriers: u64,
    /// Wall nanoseconds inside the weight matmuls (every layer's projections).
    pub ns_matmul: u64,
    /// ... inside the output-head matmul, kept apart: at 49 152 rows it is a quarter
    /// of a token's bytes and behaves differently from the layer matmuls.
    pub ns_head: u64,
    /// ... inside attention: scores, softmax and the value mix.
    pub ns_attn: u64,
    /// Wall nanoseconds of the whole forward call. Everything that is none of the
    /// above — norms, rope, residuals, SwiGLU, activation quantisation, the cache
    /// writes — is `ns_total − ns_matmul − ns_head − ns_attn`, derived rather than
    /// timed so the parts always sum to the whole.
    pub ns_total: u64,
}

impl StepStats {
    /// Nanoseconds that are not matmul, head or attention.
    pub fn ns_other(&self) -> u64 {
        self.ns_total
            .saturating_sub(self.ns_matmul + self.ns_head + self.ns_attn)
    }
    pub fn bytes_total(&self) -> u64 {
        self.bytes_weights + self.bytes_kv
    }
    /// FLOP per byte — the x-axis of the roofline.
    pub fn arithmetic_intensity(&self) -> f64 {
        let b = self.bytes_total();
        if b == 0 {
            0.0
        } else {
            2.0 * self.macs as f64 / b as f64
        }
    }
    pub fn add(&mut self, o: &StepStats) {
        self.bytes_weights += o.bytes_weights;
        self.bytes_kv += o.bytes_kv;
        self.macs += o.macs;
        self.barriers += o.barriers;
        self.ns_matmul += o.ns_matmul;
        self.ns_head += o.ns_head;
        self.ns_attn += o.ns_attn;
        self.ns_total += o.ns_total;
    }
}

/// One projection matrix: a dense GGUF tensor, or an index into the LittleBit
/// sidecar when the converter has replaced it (`attach_lb`). The dispatch lives in
/// `linear_group`, so a forward call never knows which it got — and the telemetry
/// reports whichever bytes actually moved.
pub enum Linear {
    Dense(TensorInfo),
    Lb(usize),
}

pub struct Model {
    /// The file and the repacked weights: what every forward reads matrices from.
    pub(crate) w: Weights,
    pub cfg: Config,
    pub tk: Tokenizer,
    /// Llama 3.x's per-frequency rotary factors (`rope_freqs.weight`, `rope_dim/2`
    /// floats): each angle is divided by its factor. Absent on every other model.
    pub rope_factors: Option<Vec<f32>>,
    pub(crate) lb_replaced: u64,
    pub(crate) lb_count: usize,
    pub(crate) arch: Box<dyn Architecture>,
}

/// The weights a forward pass reads: the memory-mapped file, the repacked tiles, and LittleBit factors.
/// Split from `Model` so layer code can take the weights without the rest of the model (ADR 0006).
pub struct Weights {
    pub(crate) g: Gguf,
    // PQ2_0 matrices repacked into 16-row tiles at load (`TensorInfo::tile_off` points here); empty if not.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))] // read only by the x86 tile kernels
    pub(crate) tiles: TileStore,
    pub(crate) lb: Option<Slb>,
}

impl Weights {
    /// A tensor's bytes in the mapped file.
    #[inline]
    pub fn bytes(&self, t: &TensorInfo) -> &[u8] {
        self.g.bytes(t)
    }
    /// The repacked tile store (see `TensorInfo::tile_off`).
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    #[inline]
    pub fn tiles(&self) -> &[u8] {
        self.tiles.bytes()
    }
    pub fn lb(&self) -> Option<&Slb> {
        self.lb.as_ref()
    }
}

/// Reusable scratch. Allocated once per session so a decode step does no allocation.
pub struct State {
    pub x: Vec<f32>,
    pub(crate) xb: Vec<f32>,
    pub(crate) xb2: Vec<f32>,
    pub(crate) hb: Vec<f32>,
    pub(crate) hb2: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    /// Attention's scratch: per worker, and for split heads.
    pub(crate) attn: AttnScratch,
    /// `(cos, sin)` of every rotary angle for every position, computed once. Rope was
    /// recomputing `powf`, `sin` and `cos` for every pair of every head at every
    /// layer of every token — 20 000 transcendental calls per decode step.
    pub(crate) rope: Vec<f32>,
    pub(crate) acts: Vec<QuantAct>,
    pub logits: Vec<f32>,
    pub(crate) max_batch: usize,
    /// `start_pos..start_pos + n` for a `Model::forward` call, filled in place so the
    /// convenience wrapper does not allocate a position vector per call.
    pub(crate) positions: Vec<usize>,
    /// Layers whose output (the residual stream after the layer) `forward` copies
    /// out, in this order — what a DSpark drafter conditions on. Empty: no copy.
    pub(crate) tap_layers: Vec<usize>,
    /// `[batch row][tap][d_model]`, written by every `forward` while taps are set.
    pub tap: Vec<f32>,
    /// LittleBit intermediates — the per-stage `g ⊙ x` copies (reused for `ℓ ⊙ t`)
    /// and the raw latents. Sized by `Model::lb_scratch` after the model is known;
    /// empty when nothing was converted, so the dense path pays nothing.
    pub(crate) lb_x: Vec<f32>,
    pub(crate) lb_t: Vec<f32>,
    // family scratch (`HyScratch` today), an opaque box taken by split borrow.
    pub(crate) fam: Option<Box<dyn Any + Send + Sync>>,
    // when set, a hybrid forward writes every row's final normed hidden state (`output_norm(x)`, the
    // MTP block's input) to `hidden`, `[row][d_model]`.
    pub want_hidden: bool,
    pub hidden: Vec<f32>,
}

impl State {
    /// `max_logit_rows` is sized separately from `max_batch` on purpose. A logits
    /// row is `n_vocab` floats — 192 KB for this model — so sizing the logits buffer
    /// by the prefill batch would allocate 39 MB to hold output that prefill never
    /// reads. Only speculation needs more than one row.
    pub fn new(
        cfg: &Config,
        max_batch: usize,
        max_logit_rows: usize,
        max_ctx: usize,
        threads: usize,
        rope_factors: Option<&[f32]>,
    ) -> State {
        let b = max_batch;
        let rope = crate::blocks::rope::build_rope_table(
            cfg.rope_dim,
            cfg.rope_base,
            max_ctx,
            rope_factors,
        );
        State {
            x: vec![0.0; b * cfg.d_model],
            // Holds both a normed residual row (d_model wide) and an attention output
            // row (q_dim wide); each phase indexes it with its own stride.
            xb: vec![0.0; b * cfg.d_model.max(cfg.q_dim())],
            xb2: vec![0.0; b * cfg.d_model],
            hb: vec![0.0; b * cfg.n_ff],
            hb2: vec![0.0; b * cfg.n_ff],
            q: vec![0.0; b * cfg.n_head * cfg.head_dim],
            k: vec![0.0; b * cfg.kv_dim()],
            v: vec![0.0; b * cfg.kv_dim()],
            attn: AttnScratch::new(cfg, b, threads),
            rope,
            acts: (0..b)
                .map(|_| {
                    let mut a = QuantAct::with_capacity(cfg.n_ff);
                    a.ternary = cfg.hybrid.is_some();
                    // The int16 PTQ1_0 layout has only the AVX-512 VBMI kernel; elsewhere the AVX2 one reads int8.
                    a.ptq1 = cfg.hybrid.is_some_and(|h| h.ptq1)
                        && crate::table::MatKernels::get().ptq1_layout();
                    a.tb_span = cfg.hybrid.map_or(0, |h| h.tb_span);
                    a
                })
                .collect(),
            // One row per batch element, not one row total. Speculative verification
            // needs every drafted position's logits at once; a single shared row
            // would leave each verification reading the last row's distribution and
            // accepting drafts it should have rejected.
            logits: vec![0.0; max_logit_rows.max(1) * cfg.n_vocab],
            max_batch: b,
            positions: vec![0; b],
            tap_layers: Vec::new(),
            tap: Vec::new(),
            lb_x: Vec::new(),
            lb_t: Vec::new(),
            fam: cfg.hybrid.map(|h| {
                Box::new(crate::families::qwen35::HyScratch::new(cfg, &h, b))
                    as Box<dyn Any + Send + Sync>
            }),
            want_hidden: false,
            hidden: Vec::new(),
        }
    }

    /// Allocate the LittleBit scratch once, like everything else in this struct —
    /// a decode step allocates nothing. Takes `Model::lb_scratch`'s two sizes.
    pub fn set_lb_scratch(&mut self, x: usize, t: usize) {
        self.lb_x = vec![0.0; x];
        self.lb_t = vec![0.0; t];
    }

    /// Copy the residual stream after each of `layers` out of every forward call,
    /// into `tap` as `[row][tap][d_model]` (the layout a DSpark `fc` reads).
    pub fn set_taps(&mut self, layers: &[usize], d_model: usize) {
        self.tap_layers = layers.to_vec();
        self.tap = vec![0.0; self.max_batch * layers.len() * d_model];
    }

    /// The tapped hidden states of batch row `b`: `n_taps × d_model` floats.
    pub fn tap_row(&self, b: usize, d_model: usize) -> &[f32] {
        let w = self.tap_layers.len() * d_model;
        &self.tap[b * w..(b + 1) * w]
    }

    /// Logits for the `i`-th row of the last `want_logits` request.
    pub fn logits_row(&self, i: usize, n_vocab: usize) -> &[f32] {
        &self.logits[i * n_vocab..(i + 1) * n_vocab]
    }
}

/// the repacked weights. An anonymous mapping that asks for transparent huge pages (decode streams all of
/// it every token: one TLB entry per 2 MB instead of per 4 KB); `PEREGRINE_TILE_HUGEPAGES=0` uses a plain heap
/// buffer, for the A/B.
#[derive(Default)]
pub struct TileStore {
    map: Option<memmap2::MmapMut>,
    heap: Vec<u8>,
}

impl TileStore {
    pub fn new(len: usize) -> TileStore {
        let huge = crate::options::get().tile_hugepages;
        if huge && len > 0 {
            if let Ok(m) = memmap2::MmapMut::map_anon(len) {
                #[cfg(target_os = "linux")]
                let _ = m.advise(memmap2::Advice::HugePage);
                return TileStore {
                    map: Some(m),
                    heap: Vec::new(),
                };
            }
        }
        TileStore {
            map: None,
            heap: vec![0u8; len],
        }
    }
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        self.map.as_deref().unwrap_or(&self.heap)
    }
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        match &mut self.map {
            Some(m) => m,
            None => &mut self.heap,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.bytes().is_empty()
    }
}

// where each row of a batch writes and reads its attention cache.
//
// Both shapes run the same forward pass, because every matmul in it is the same work whatever a row belongs to:
// one read of the weights serves the whole batch. `Shared` is one sequence over several positions (prefill, a
// speculative block). `PerRow` is one position of each of several sequences — several users in one weight pass,
// which is what makes concurrent serving on a memory-bound machine cost little more than serving one.
pub enum Caches<'a> {
    /// Every row writes to one cache, at its own position.
    Shared(&'a mut KvCache),
    /// Row `b` writes to cache `b`.
    PerRow(&'a mut [&'a mut KvCache]),
    /// Row `b` writes to cache `map[b]`; rows of one cache are consecutive positions in order (a prompt chunk
    /// beside other users' decode rows: one weight pass for both).
    Mapped(&'a mut [&'a mut KvCache], &'a [usize]),
}

/// Iterator over [`Caches::runs`]: runs of consecutive rows that share one cache, yielded without allocating.
pub(crate) struct Runs<'a> {
    /// The `Mapped` shape's cache index per row; `None` for one shared cache or one cache per row.
    map: Option<&'a [usize]>,
    /// Every row writes one cache (`Caches::Shared`).
    shared: bool,
    nb: usize,
    next: usize,
    done: bool,
}

impl Iterator for Runs<'_> {
    type Item = (usize, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.shared {
            if self.done || self.nb == 0 {
                return None;
            }
            self.done = true;
            return Some((0, self.nb, 0));
        }
        match self.map {
            None => {
                // `PerRow`: one row per run.
                if self.next >= self.nb {
                    None
                } else {
                    let r = (self.next, 1, self.next);
                    self.next += 1;
                    Some(r)
                }
            }
            Some(map) => {
                let start = self.next;
                if start >= map.len() {
                    return None;
                }
                let c = map[start];
                let mut end = start + 1;
                while end < map.len() && map[end] == c {
                    end += 1;
                }
                self.next = end;
                Some((start, end - start, c))
            }
        }
    }
}

impl<'a> Caches<'a> {
    #[inline]
    pub(crate) fn get(&self, b: usize) -> &KvCache {
        match self {
            Caches::Shared(kv) => kv,
            Caches::PerRow(kvs) => kvs[b],
            Caches::Mapped(kvs, map) => kvs[map[b]],
        }
    }
    #[inline]
    pub(crate) fn get_mut(&mut self, b: usize) -> &mut KvCache {
        match self {
            Caches::Shared(kv) => kv,
            Caches::PerRow(kvs) => kvs[b],
            Caches::Mapped(kvs, map) => kvs[map[b]],
        }
    }
    /// Runs of consecutive rows sharing one cache: `(first row, rows, cache)` (one run per row for `PerRow`).
    ///
    /// An iterator, not a `Vec`: the dense and hybrid decode loops call this once per layer, and a decode step
    /// must not allocate. Callers that also need the caches mutably use [`Caches::runs_into`] with a reused
    /// buffer instead (the iterator borrows the caches, so it cannot be held across `cache_mut`).
    pub(crate) fn runs(&self, nb: usize) -> Runs<'a> {
        Runs {
            map: match self {
                Caches::Mapped(_, map) => {
                    let map: &'a [usize] = map;
                    Some(&map[..map.len().min(nb)])
                }
                _ => None,
            },
            shared: matches!(self, Caches::Shared(_)),
            nb,
            next: 0,
            done: false,
        }
    }

    /// `runs`, written into `out` (cleared first) so a caller that also needs mutable cache access can hold
    /// the runs without keeping the caches borrowed. `out` keeps its capacity across calls.
    pub(crate) fn runs_into(&self, nb: usize, out: &mut Vec<(usize, usize, usize)>) {
        out.clear();
        out.extend(self.runs(nb));
    }
    /// The cache of run `c`, mutably.
    pub(crate) fn cache_mut(&mut self, c: usize) -> &mut KvCache {
        match self {
            Caches::Shared(kv) => kv,
            Caches::PerRow(kvs) | Caches::Mapped(kvs, _) => kvs[c],
        }
    }
    /// The cache of run `c` (as numbered by `runs`).
    pub(crate) fn cache(&self, c: usize) -> &KvCache {
        match self {
            Caches::Shared(kv) => kv,
            Caches::PerRow(kvs) | Caches::Mapped(kvs, _) => kvs[c],
        }
    }
    /// Rows this holds caches for, or None when one cache serves every row.
    pub(crate) fn rows(&self) -> Option<usize> {
        match self {
            Caches::Shared(_) => None,
            Caches::PerRow(kvs) => Some(kvs.len()),
            Caches::Mapped(_, map) => Some(map.len()),
        }
    }
}

impl Model {
    pub fn load(g: Gguf) -> Result<Model, String> {
        let arch_name = g.str("general.architecture")?;
        let family = crate::arch::FAMILIES
            .iter()
            .find(|f| f.names().contains(&arch_name))
            .ok_or_else(|| {
                format!(
                    "Peregrine implements {}; this model is {arch_name:?}",
                    ARCHS.join(" and ")
                )
            })?;
        let tk = Tokenizer::from_gguf(&g)?;
        let loaded = family.load(&g, &tk)?;
        Ok(Model {
            w: Weights {
                g,
                tiles: loaded.tiles,
                lb: None,
            },
            cfg: loaded.cfg,
            tk,
            rope_factors: loaded.rope_factors,
            lb_replaced: 0,
            lb_count: 0,
            arch: loaded.arch,
        })
    }

    // a cache shaped for this model: K/V for the layers that attend, plus the recurrent state of a
    // hybrid model's linear-attention layers.
    pub fn new_cache(&self, prec: crate::kv::KvPrec, max_ctx: usize) -> KvCache {
        let c = &self.cfg;
        let kv = KvCache::new(prec, c.kv_layers(), c.n_kv_head, c.head_dim, max_ctx);
        match self.arch.recurrent() {
            Some(r) => kv.with_recurrent(r.n_layers, r.state_floats, r.conv_floats),
            None => kv,
        }
    }

    // whether the model carries recurrent state (a cache rolls back only to its checkpoint).
    pub fn is_hybrid(&self) -> bool {
        self.arch.recurrent().is_some()
    }

    /// Whether this model uses rotated KV cache in Session runtime.
    pub fn session_kv_rotation(&self) -> bool {
        self.arch.session_kv_rotation()
    }

    /// Swap every projection the sidecar covers from its dense tensor to the
    /// LittleBit factors, after checking each one's geometry against the tensor it
    /// replaces. The embedding table is never swapped — a tied head therefore stays
    /// dense even when the sidecar carries `output.weight` for an untied model.
    pub fn attach_lb(&mut self, slb: Slb) -> Result<usize, String> {
        let mut n = 0usize;
        let mut replaced = 0u64;
        {
            let mut swap = |lin: &mut Linear| -> Result<(), String> {
                if let Linear::Dense(ti) = lin {
                    if let Some(&idx) = slb.by_name.get(&ti.name) {
                        let t = &slb.tensors[idx];
                        let (d_in, d_out) = (
                            ti.dims[0] as usize,
                            ti.dims.get(1).map_or(0, |&v| v as usize),
                        );
                        if t.d_in != d_in || t.d_out != d_out {
                            return Err(format!(
                                "slb: {} is {}×{} in the sidecar but {d_out}×{d_in} in the model",
                                ti.name, t.d_out, t.d_in
                            ));
                        }
                        replaced += ti.bytes();
                        n += 1;
                        *lin = Linear::Lb(idx);
                    }
                }
                Ok(())
            };
            self.arch.linears_mut(&mut swap)?;
        }
        if n == 0 {
            return Err("slb: the sidecar names no tensor of this model".into());
        }
        self.lb_replaced = replaced;
        self.lb_count = n;
        self.w.lb = Some(slb);
        Ok(n)
    }

    /// How many projections run as LittleBit, and the sidecar bytes they read.
    pub fn lb_summary(&self) -> Option<(usize, u64)> {
        self.w.lb.as_ref().map(|s| (self.lb_count, s.data_bytes))
    }

    /// Scratch floats one forward call needs for LittleBit intermediates, as
    /// (`x′` floats, latent floats) — the maximum over the fused groups, times the
    /// batch. Zero when nothing was converted.
    pub fn lb_scratch(&self, max_batch: usize) -> (usize, usize) {
        let Some(slb) = self.w.lb.as_ref() else {
            return (0, 0);
        };
        let (mut mx, mut mt) = (0usize, 0usize);
        let mut group = |lins: &[&Linear]| {
            let (mut x, mut t) = (0usize, 0usize);
            for lin in lins {
                if let Linear::Lb(ti) = lin {
                    let tt = &slb.tensors[*ti];
                    x += tt.stages.len() * tt.d_in;
                    t += tt.stages.len() * tt.r_pad;
                }
            }
            mx = mx.max(x);
            mt = mt.max(t);
        };
        self.arch.linear_groups(&mut group);
        (mx * max_batch, mt * max_batch)
    }

    pub fn weight_bytes(&self) -> u64 {
        self.w.g.total_weight_bytes() - self.lb_replaced
            + self.w.lb.as_ref().map_or(0, |s| s.data_bytes)
    }

    /// The file's metadata, for the template detector and the host's telemetry.
    pub fn gguf(&self) -> &Gguf {
        &self.w.g
    }

    /// `general.name`, or an empty string.
    pub fn name(&self) -> &str {
        self.w.g.str("general.name").unwrap_or("")
    }

    /// Formats present in this model and how many tensors use each — the input to
    /// the mixed-precision report.
    pub fn format_mix(&self) -> Vec<(Format, usize, u64)> {
        let mut out: Vec<(Format, usize, u64)> = Vec::new();
        for t in self.w.g.tensors.values() {
            match out.iter_mut().find(|e| e.0 == t.kind) {
                Some(e) => {
                    e.1 += 1;
                    e.2 += t.bytes();
                }
                None => out.push((t.kind, 1, t.bytes())),
            }
        }
        out.sort_by_key(|e| e.0.label());
        out
    }

    /// Bits per weight averaged over the whole model, weighted by tensor size.
    /// This is the `bpw` of the performance model, and it is measured, not quoted
    /// from the file's `general.file_type` — which for the reference model claims
    /// Q4_K_M while the tensors are mostly Q5_0.
    pub fn effective_bpw(&self) -> f64 {
        let bytes = self.weight_bytes();
        let params: u64 = self.w.g.tensors.values().map(|t| t.numel()).sum();
        if params == 0 {
            0.0
        } else {
            8.0 * bytes as f64 / params as f64
        }
    }

    /// Run `tokens` at positions `start_pos..start_pos+tokens.len()`.
    ///
    /// Writes logits for the batch rows named in `want_logits`. Prefill asks for the
    /// last row only: computing all of them would cost an extra 47 M multiply-adds
    /// per prompt token for output nobody reads.
    // Eight arguments, and each is a separate axis: scratch, cache, pool, the tokens,
    // where they start, which rows want logits, and where the byte accounting goes.
    // Bundling them into a config struct would hide that `want_logits` is what makes
    // prefill cheap and speculation possible.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        st8: &mut State,
        kv: &mut KvCache,
        pool: &Pool,
        tokens: &[u32],
        start_pos: usize,
        want_logits: &[usize],
        stats: &mut StepStats,
    ) -> Result<(), String> {
        // Fill the state-owned buffer instead of collecting a `Vec` per call: `forward` is the decode entry point
        // and a per-token allocation breaks invariant 4.
        let n = tokens.len();
        let mut positions = std::mem::take(&mut st8.positions);
        let fits = n <= positions.len();
        if fits {
            for (i, p) in positions[..n].iter_mut().enumerate() {
                *p = start_pos + i;
            }
        }
        let pos: &[usize] = if fits { &positions[..n] } else { &[] };
        let r = self.forward_rows(
            st8,
            &mut Caches::Shared(kv),
            pool,
            tokens,
            pos,
            want_logits,
            stats,
        );
        st8.positions = positions;
        r
    }

    // the general pass. Row `b` carries `tokens[b]` at `positions[b]` of the sequence whose cache
    // `caches` gives for `b`; every row shares one read of the weights.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_rows(
        &self,
        st8: &mut State,
        caches: &mut Caches,
        pool: &Pool,
        tokens: &[u32],
        positions: &[usize],
        want_logits: &[usize],
        stats: &mut StepStats,
    ) -> Result<(), String> {
        let c = self.cfg;
        let nb = tokens.len();
        if nb == 0 {
            return Ok(());
        }
        // the batch width is decided per step when users are served together, so an oversized batch is
        // a request to refuse, not a slice to overrun.
        if nb > st8.max_batch {
            return Err(format!(
                "batch of {nb} rows exceeds the scratch of {}",
                st8.max_batch
            ));
        }
        if positions.len() != nb || caches.rows().is_some_and(|r| r != nb) {
            return Err(format!(
                "batch of {nb} tokens with {} positions and {:?} caches",
                positions.len(),
                caches.rows()
            ));
        }
        for b in 0..nb {
            let kv = caches.get(b);
            if positions[b] >= kv.max_ctx() {
                return Err(format!(
                    "context limit reached: position {} requested, cache holds {}",
                    positions[b],
                    kv.max_ctx()
                ));
            }
        }
        for (i, &t) in tokens.iter().enumerate() {
            if t as usize >= c.n_vocab {
                return Err(format!(
                    "token {t} at batch index {i} is outside the vocabulary"
                ));
            }
        }
        self.arch.forward_rows(
            &self.w,
            &self.cfg,
            st8,
            caches,
            pool,
            tokens,
            positions,
            want_logits,
            stats,
        )
    }

    pub fn has_mtp(&self) -> bool {
        self.arch.mtp().is_some()
    }

    /// Scratch and cache for one sequence's MTP block.
    pub fn mtp_state(&self, max_ctx: usize, threads: usize) -> Result<MtpState, String> {
        let mtp = self.arch.mtp().ok_or("this model has no MTP block")?;
        mtp.new_state(&self.w, &self.cfg, max_ctx, threads)
    }

    /// Run the MTP block over `tokens` at `positions`.
    #[allow(clippy::too_many_arguments)]
    pub fn mtp_forward(
        &self,
        ms: &mut MtpState,
        pool: &Pool,
        tokens: &[u32],
        hidden: &[f32],
        positions: &[usize],
        draft: bool,
        stats: &mut StepStats,
    ) -> Result<Option<u32>, String> {
        let mtp = self.arch.mtp().ok_or("this model has no MTP block")?;
        mtp.forward(
            &self.w, &self.cfg, ms, pool, tokens, hidden, positions, draft, stats,
        )
    }

    /// Drop the MTP block's cache from position `pos` on.
    pub fn mtp_truncate(&self, ms: &mut MtpState, pos: usize) {
        if let Some(mtp) = self.arch.mtp() {
            mtp.truncate(ms, pos);
        } else {
            ms.kv.truncate(pos);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_stats_intensity_is_flops_over_bytes() {
        let s = StepStats {
            bytes_weights: 100,
            bytes_kv: 100,
            macs: 400,
            barriers: 3,
            ..Default::default()
        };
        assert_eq!(s.bytes_total(), 200);
        // 2 FLOP per MAC, so 800 FLOP over 200 bytes.
        assert_eq!(s.arithmetic_intensity(), 4.0);
        // An empty step must not divide by zero.
        assert_eq!(StepStats::default().arithmetic_intensity(), 0.0);
    }

    #[test]
    fn step_stats_accumulate() {
        let mut a = StepStats {
            bytes_weights: 1,
            bytes_kv: 2,
            macs: 3,
            barriers: 4,
            ..Default::default()
        };
        a.add(&StepStats {
            bytes_weights: 10,
            bytes_kv: 20,
            macs: 30,
            barriers: 40,
            ..Default::default()
        });
        assert_eq!(
            (a.bytes_weights, a.bytes_kv, a.macs, a.barriers),
            (11, 22, 33, 44)
        );
    }

    #[test]
    fn gqa_ratio_is_derived_not_assumed() {
        let c = Config {
            n_layer: 32,
            n_head: 15,
            n_kv_head: 5,
            head_dim: 64,
            d_model: 960,
            n_ff: 2560,
            n_vocab: 49152,
            rope_dim: 64,
            eps: 1e-5,
            rope_base: 1e5,
            ctx_train: 8192,
            hybrid: None,
            arch: Arch::Llama,
            rope_neox: false,
        };
        assert_eq!(c.gqa(), 3);
        assert_eq!(c.kv_dim(), 320);
        // Query head 14 must land on kv head 4, not 14.
        assert_eq!(14 / c.gqa(), 4);
    }
}

// ------------------------------------------------------------- the kept model --

/// The loaded model a host may keep for the next session on the same file.
///
/// A `Model` is a mapping plus the tensor directory: dropping it unmaps the weights,
/// and the next `Session::open` on the same file maps them again and minor-faults
/// every page on the first token (2.5 GB for Qwen3 4B). The pages were never gone —
/// the page cache had them — but the mapping was, and so were the TLB and the
/// faulted-in state. The host keeps one `Arc<Model>` here across a switch to the
/// *same* model (`residency.rs` says when: a release whose next session is this
/// model), and lets it go on a switch to another, so the rule that the previous
/// model's memory comes back before the next loads still holds.
pub mod keep {
    use super::Model;
    use crate::gguf::Gguf;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::{Duration, Instant};

    /// How long an unclaimed model is kept before a `sweep` drops it.
    pub const TTL: Duration = Duration::from_secs(600);

    /// Canonical path, size and mtime of the GGUF, plus the sidecar's size and
    /// mtime when one is present — the identity under which a load may be reused.
    type Key = (PathBuf, u64, u64, Option<(u64, u64)>);

    struct Slot {
        key: Key,
        model: Arc<Model>,
        since: Instant,
    }

    fn slot() -> &'static Mutex<Option<Slot>> {
        static S: OnceLock<Mutex<Option<Slot>>> = OnceLock::new();
        S.get_or_init(|| Mutex::new(None))
    }

    fn key(path: &Path, lb: bool) -> Result<Key, String> {
        let m = std::fs::metadata(path).map_err(|e| format!("cannot stat {path:?}: {e}"))?;
        let mtime = m
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // The LittleBit sidecar's identity is part of the key: converting (or
        // removing) a sidecar must invalidate the kept dense (or converted) model.
        // A session that declines LittleBit keys as "no sidecar", so switching the
        // setting reloads rather than reusing the other kind of load.
        let side = lb
            .then(|| crate::lbq::sidecar_path(path))
            .flatten()
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| {
                let mt = m
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                (m.len(), mt)
            });
        Ok((
            std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()),
            m.len(),
            mtime,
            side,
        ))
    }

    /// The model at `path`: the kept one when it is the same file (same canonical
    /// path, size and modification time, same sidecar state), else freshly loaded —
    /// and kept in place of whatever was there. `Err` is `Gguf::open`/`Model::load`'s,
    /// or a corrupt or stale sidecar's: that one refuses loudly rather than running
    /// dense when the user built a sidecar.
    pub fn open(path: &Path) -> Result<Arc<Model>, String> {
        open_with(path, true)
    }

    /// `open`, choosing whether a LittleBit sidecar beside the model is used
    /// (`false` runs dense even when one is present).
    pub fn open_with(path: &Path, lb: bool) -> Result<Arc<Model>, String> {
        open_with_template(path, lb, None)
    }

    /// `open_with`, with the chat template the caller chose (`--chat-template`):
    /// a file that declares no template is then loaded with that one.
    pub fn open_with_template(
        path: &Path,
        lb: bool,
        template: Option<crate::template::Template>,
    ) -> Result<Arc<Model>, String> {
        let k = key(path, lb)?;
        let mut g = slot().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = g.as_mut() {
            if s.key == k {
                s.since = Instant::now();
                return Ok(Arc::clone(&s.model));
            }
        }
        let file = Gguf::open(path)?;
        crate::check_compatible_with(&file, template)?;
        let mut model = Model::load(file)?;
        if lb {
            if let Some(slb) = crate::lbq::Slb::open_for(path)? {
                model.attach_lb(slb)?;
            }
        }
        let model = Arc::new(model);
        *g = Some(Slot {
            key: k,
            model: Arc::clone(&model),
            since: Instant::now(),
        });
        Ok(model)
    }

    /// Whether a load of `path` would find the kept model.
    pub fn has(path: &Path) -> bool {
        has_with(path, true)
    }

    pub fn has_with(path: &Path, lb: bool) -> bool {
        let Ok(k) = key(path, lb) else { return false };
        slot()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|s| s.key == k)
    }

    /// The kept model's path and how many sessions still hold it (besides the slot).
    pub fn status() -> Option<(PathBuf, usize, f64)> {
        slot()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|s| {
                (
                    s.key.0.clone(),
                    Arc::strong_count(&s.model) - 1,
                    s.since.elapsed().as_secs_f64(),
                )
            })
    }

    /// Let the kept model go. Its mapping ends when the last session using it ends.
    pub fn drop_kept() -> bool {
        slot()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .is_some()
    }

    /// Drop the kept model when nothing uses it and it has sat for `TTL`.
    pub fn sweep() -> bool {
        let mut g = slot().lock().unwrap_or_else(|e| e.into_inner());
        if g.as_ref()
            .is_some_and(|s| Arc::strong_count(&s.model) == 1 && s.since.elapsed() >= TTL)
        {
            *g = None;
            return true;
        }
        false
    }
}
