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

use crate::blocks::attn::AttnScratch;
use crate::blocks::par::{Disjoint, OutPtr, SharedBuf};
use crate::gguf::{Format, Gguf, TensorInfo};
use crate::kv::KvCache;
use crate::lbq::Slb;
use crate::pool::Pool;
use crate::quant::QuantAct;
use crate::tokenizer::Tokenizer;
use std::time::Instant;

pub(crate) use crate::blocks::linear::{LinJob, MatJob};
pub(crate) use crate::blocks::norm::head_rmsnorm;
pub(crate) use crate::blocks::ROW_PARALLEL;
pub(crate) use crate::pool::share;
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_imports))]
pub(crate) use crate::quant::row_bytes;

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

struct Layer {
    attn_norm: TensorInfo,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    /// Qwen2's attention biases, decoded once at load (a few KB per layer). `None`
    /// for llama, which has none — and a file that is missing one it should have
    /// fails at load, not by silently running without it.
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    /// Qwen3's per-head query and key RMSNorm weights (`head_dim` floats each).
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    wo: Linear,
    ffn_norm: TensorInfo,
    gate: Linear,
    up: Linear,
    down: Linear,
}

pub struct Model {
    /// The file and the repacked weights: what every forward reads matrices from.
    w: Weights,
    pub cfg: Config,
    pub tk: Tokenizer,
    tok_embd: TensorInfo,
    out_norm: TensorInfo,
    /// The output projection: `output.weight` when the file has one, else the
    /// embedding table (a tied head). Detected by absence — assuming either way
    /// silently transposes the head.
    head: Linear,
    /// The LittleBit sidecar, when one was attached, with the GGUF bytes it
    /// displaced — `weight_bytes` and `effective_bpw` report the mix honestly.
    lb: Option<Slb>,
    lb_replaced: u64,
    lb_count: usize,
    /// Llama 3.x's per-frequency rotary factors (`rope_freqs.weight`, `rope_dim/2`
    /// floats): each angle is divided by its factor. Absent on every other model.
    pub rope_factors: Option<Vec<f32>>,
    layers: Vec<Layer>,
    // the hybrid model's layers and rotation (`hybrid.rs`); None for the dense architectures.
    hy: Option<hybrid::Hybrid>,
    // the MTP (NextN) block of a hybrid file that carries one (`*-MTP-Q8_0.gguf`).
    mtp: Option<mtp::Mtp>,
}

/// The weights a forward pass reads: the memory-mapped file and the matrices repacked from it at load.
/// Split from `Model` so layer code can take the weights without the rest of the model (ADR 0006).
pub(crate) struct Weights {
    pub(crate) g: Gguf,
    // PQ2_0 matrices repacked into 16-row tiles at load (`TensorInfo::tile_off` points here); empty if not.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))] // read only by the x86 tile kernels
    pub(crate) tiles: TileStore,
}

impl Weights {
    /// A tensor's bytes in the mapped file.
    #[inline]
    pub(crate) fn bytes(&self, t: &TensorInfo) -> &[u8] {
        self.g.bytes(t)
    }
    /// The repacked tile store (see `TensorInfo::tile_off`).
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    #[inline]
    pub(crate) fn tiles(&self) -> &[u8] {
        self.tiles.bytes()
    }
}

/// Reusable scratch. Allocated once per session so a decode step does no allocation.
pub struct State {
    pub x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    /// Attention's scratch: per worker, and for split heads.
    attn: AttnScratch,
    /// `(cos, sin)` of every rotary angle for every position, computed once. Rope was
    /// recomputing `powf`, `sin` and `cos` for every pair of every head at every
    /// layer of every token — 20 000 transcendental calls per decode step.
    rope: Vec<f32>,
    acts: Vec<QuantAct>,
    pub logits: Vec<f32>,
    max_batch: usize,
    /// `start_pos..start_pos + n` for a `Model::forward` call, filled in place so the
    /// convenience wrapper does not allocate a position vector per call.
    positions: Vec<usize>,
    /// Layers whose output (the residual stream after the layer) `forward` copies
    /// out, in this order — what a DSpark drafter conditions on. Empty: no copy.
    tap_layers: Vec<usize>,
    /// `[batch row][tap][d_model]`, written by every `forward` while taps are set.
    pub tap: Vec<f32>,
    /// LittleBit intermediates — the per-stage `g ⊙ x` copies (reused for `ℓ ⊙ t`)
    /// and the raw latents. Sized by `Model::lb_scratch` after the model is known;
    /// empty when nothing was converted, so the dense path pays nothing.
    lb_x: Vec<f32>,
    lb_t: Vec<f32>,
    // the hybrid pass's extra buffers.
    hy: Option<hybrid::HyScratch>,
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
            hy: cfg.hybrid.map(|h| hybrid::HyScratch::new(cfg, &h, b)),
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
struct Runs<'a> {
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
    fn runs(&self, nb: usize) -> Runs<'a> {
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
    fn runs_into(&self, nb: usize, out: &mut Vec<(usize, usize, usize)>) {
        out.clear();
        out.extend(self.runs(nb));
    }
    /// The cache of run `c`, mutably.
    fn cache_mut(&mut self, c: usize) -> &mut KvCache {
        match self {
            Caches::Shared(kv) => kv,
            Caches::PerRow(kvs) | Caches::Mapped(kvs, _) => kvs[c],
        }
    }
    /// The cache of run `c` (as numbered by `runs`).
    fn cache(&self, c: usize) -> &KvCache {
        match self {
            Caches::Shared(kv) => kv,
            Caches::PerRow(kvs) | Caches::Mapped(kvs, _) => kvs[c],
        }
    }
    /// Rows this holds caches for, or None when one cache serves every row.
    fn rows(&self) -> Option<usize> {
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
        if arch_name == "qwen35" {
            return hybrid::load(g);
        }
        let arch = Arch::from_name(arch_name).ok_or_else(|| {
            format!(
                "Peregrine implements {}; this model is {arch_name:?}",
                ARCHS.join(" and ")
            )
        })?;
        let k = |suffix: &str| format!("{}.{suffix}", arch.key_prefix());
        let n_layer = g.u32(&k("block_count"))? as usize;
        let d_model = g.u32(&k("embedding_length"))? as usize;
        let n_head = g.u32(&k("attention.head_count"))? as usize;
        let n_kv_head = g
            .u32(&k("attention.head_count_kv"))
            .unwrap_or(n_head as u32) as usize;
        let n_ff = g.u32(&k("feed_forward_length"))? as usize;
        // Rope scaling other than Llama 3's factor table is not implemented; a file
        // asking for it would run with the wrong angles and drift after a few tokens.
        if let Ok(t) = g.str(&k("rope.scaling.type")) {
            if t != "none" && t != "linear" {
                return Err(format!("rope scaling {t:?} is not implemented"));
            }
        }
        if n_head == 0 || n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) {
            return Err(format!(
                "implausible head configuration: {n_head} query heads over {n_kv_head} kv heads"
            ));
        }
        let head_dim = g
            .u32(&k("attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(d_model / n_head);
        // Qwen3 decouples the head width from the model width; the others must match.
        if arch != Arch::Qwen3 && head_dim * n_head != d_model {
            return Err(format!(
                "head dim {head_dim} × {n_head} heads ≠ embedding length {d_model}"
            ));
        }
        if !(head_dim * n_head).is_multiple_of(32) {
            return Err(format!(
                "query width {} is not a multiple of 32",
                head_dim * n_head
            ));
        }
        let rope_dim = g.u32(&k("rope.dimension_count")).unwrap_or(head_dim as u32) as usize;
        if rope_dim > head_dim || !rope_dim.is_multiple_of(2) {
            return Err(format!(
                "implausible rope dimension {rope_dim} for head dim {head_dim}"
            ));
        }
        // The activation quantiser works in blocks of 32, so both matmul widths must
        // be multiples of it. Checked here rather than tripping a debug assert deep
        // inside the first token.
        for (what, n) in [("embedding length", d_model), ("feed-forward length", n_ff)] {
            if !n.is_multiple_of(32) {
                return Err(format!("{what} {n} is not a multiple of 32"));
            }
        }

        let rope_scale = g
            .f32(&k("rope.scaling.factor"))
            .ok()
            .filter(|f| *f > 0.0)
            .map(|f| 1.0 / f);
        let cfg = Config {
            arch,
            rope_neox: arch != Arch::Llama,
            n_layer,
            n_head,
            n_kv_head,
            head_dim,
            d_model,
            n_ff,
            n_vocab: g.u32(&k("vocab_size")).unwrap_or(0) as usize,
            rope_dim,
            eps: g
                .f32(&k("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-5),
            rope_base: g.f32(&k("rope.freq_base")).unwrap_or(10000.0),
            ctx_train: g.u32(&k("context_length")).unwrap_or(2048) as usize,
            hybrid: None,
        };
        // Linear rope scaling folds into the base angle; it is expressed here as a
        // uniform factor table so one code path serves both it and Llama 3's.
        let mut rope_factors: Option<Vec<f32>> = rope_scale.map(|s| vec![1.0 / s; rope_dim / 2]);
        if let Ok(t) = g.tensor("rope_freqs.weight") {
            let n = t.numel() as usize;
            if n != rope_dim / 2 {
                return Err(format!(
                    "rope_freqs.weight has {n} entries, expected {}",
                    rope_dim / 2
                ));
            }
            let mut f = vec![0.0f32; n];
            crate::quant::dequant_row(t.kind, g.bytes(t), &mut f);
            if let Some(base) = &rope_factors {
                for (x, b) in f.iter_mut().zip(base) {
                    *x *= b;
                }
            }
            rope_factors = Some(f);
        }

        let get = |n: &str| -> Result<TensorInfo, String> { g.tensor(n).cloned() };
        // Biases are tiny and read on every token; decode them once rather than
        // dequantising a 128-float row per layer per step.
        let bias = |n: &str| -> Result<Option<Vec<f32>>, String> {
            match arch {
                Arch::Llama | Arch::Qwen3 | Arch::Qwen35 => Ok(None),
                Arch::Qwen2 => {
                    let t = get(n)?;
                    let mut v = vec![0.0f32; t.numel() as usize];
                    crate::quant::dequant_row(t.kind, g.bytes(&t), &mut v);
                    Ok(Some(v))
                }
            }
        };
        // Qwen3's q/k norms: `head_dim` floats, decoded once like the biases.
        let head_norm = |n: &str| -> Result<Option<Vec<f32>>, String> {
            if arch != Arch::Qwen3 {
                return Ok(None);
            }
            let t = get(n)?;
            if t.numel() as usize != head_dim {
                return Err(format!(
                    "{n} has {} entries, expected the head dim {head_dim}",
                    t.numel()
                ));
            }
            let mut v = vec![0.0f32; head_dim];
            crate::quant::dequant_row(t.kind, g.bytes(&t), &mut v);
            Ok(Some(v))
        };
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            layers.push(Layer {
                attn_norm: get(&format!("blk.{i}.attn_norm.weight"))?,
                wq: Linear::Dense(get(&format!("blk.{i}.attn_q.weight"))?),
                wk: Linear::Dense(get(&format!("blk.{i}.attn_k.weight"))?),
                wv: Linear::Dense(get(&format!("blk.{i}.attn_v.weight"))?),
                bq: bias(&format!("blk.{i}.attn_q.bias"))?,
                bk: bias(&format!("blk.{i}.attn_k.bias"))?,
                bv: bias(&format!("blk.{i}.attn_v.bias"))?,
                q_norm: head_norm(&format!("blk.{i}.attn_q_norm.weight"))?,
                k_norm: head_norm(&format!("blk.{i}.attn_k_norm.weight"))?,
                wo: Linear::Dense(get(&format!("blk.{i}.attn_output.weight"))?),
                ffn_norm: get(&format!("blk.{i}.ffn_norm.weight"))?,
                gate: Linear::Dense(get(&format!("blk.{i}.ffn_gate.weight"))?),
                up: Linear::Dense(get(&format!("blk.{i}.ffn_up.weight"))?),
                down: Linear::Dense(get(&format!("blk.{i}.ffn_down.weight"))?),
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

        let tk = Tokenizer::from_gguf(&g)?;
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

        // Every matmul writes `dims[1]` rows into buffers sized from `cfg`, and every
        // kernel reads `dims[0]` activations unchecked: a file whose tensors disagree
        // with its own metadata must be refused here, not discovered as a heap
        // overflow mid-turn (`catch_unwind` cannot contain that).
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
                if let Linear::Dense(t) = lin {
                    shape(t, d_in, d_out)?;
                }
            }
            shape(&l.attn_norm, dm, 1)?;
            shape(&l.ffn_norm, dm, 1)?;
        }
        shape(&tok_embd, dm, cfg.n_vocab)?;
        shape(&out_norm, dm, 1)?;
        if let Linear::Dense(t) = &head {
            shape(t, dm, cfg.n_vocab)?;
        }

        Ok(Model {
            w: Weights {
                g,
                tiles: TileStore::default(),
            },
            cfg,
            tk,
            tok_embd,
            out_norm,
            head,
            lb: None,
            lb_replaced: 0,
            lb_count: 0,
            rope_factors,
            layers,
            hy: None,
            mtp: None,
        })
    }

    // a cache shaped for this model: K/V for the layers that attend, plus the recurrent state of a
    // hybrid model's linear-attention layers.
    pub fn new_cache(&self, prec: crate::kv::KvPrec, max_ctx: usize) -> KvCache {
        let c = &self.cfg;
        let kv = KvCache::new(prec, c.kv_layers(), c.n_kv_head, c.head_dim, max_ctx);
        match c.hybrid {
            Some(h) => kv.with_recurrent(h.n_gdn, h.state_floats(), h.conv_floats()),
            None => kv,
        }
    }

    // whether the model carries recurrent state (a cache rolls back only to its checkpoint).
    pub fn is_hybrid(&self) -> bool {
        self.hy.is_some()
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
                    swap(lin)?;
                }
            }
            // Only a real `output.weight` may be swapped, never the tied embedding:
            // the embedding lookup still needs its rows dense.
            if let Linear::Dense(ti) = &self.head {
                if ti.name == "output.weight" {
                    swap(&mut self.head)?;
                }
            }
        }
        if n == 0 {
            return Err("slb: the sidecar names no tensor of this model".into());
        }
        self.lb_replaced = replaced;
        self.lb_count = n;
        self.lb = Some(slb);
        Ok(n)
    }

    /// How many projections run as LittleBit, and the sidecar bytes they read.
    pub fn lb_summary(&self) -> Option<(usize, u64)> {
        self.lb.as_ref().map(|s| (self.lb_count, s.data_bytes))
    }

    /// Scratch floats one forward call needs for LittleBit intermediates, as
    /// (`x′` floats, latent floats) — the maximum over the fused groups, times the
    /// batch. Zero when nothing was converted.
    pub fn lb_scratch(&self, max_batch: usize) -> (usize, usize) {
        let Some(slb) = &self.lb else { return (0, 0) };
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
        for l in &self.layers {
            group(&[&l.wq, &l.wk, &l.wv]);
            group(&[&l.wo]);
            group(&[&l.gate, &l.up]);
            group(&[&l.down]);
        }
        group(&[&self.head]);
        (mx * max_batch, mt * max_batch)
    }

    pub fn weight_bytes(&self) -> u64 {
        self.w.g.total_weight_bytes() - self.lb_replaced
            + self.lb.as_ref().map_or(0, |s| s.data_bytes)
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

    #[inline]
    fn embed(&self, token: u32, out: &mut [f32], st: &mut StepStats) {
        crate::blocks::embed::embed(&self.tok_embd, &self.w, token, out, st);
    }

    #[inline]
    fn matmul_group(
        &self,
        jobs: &[MatJob],
        acts: &[QuantAct],
        nb: usize,
        pool: &Pool,
        st: &mut StepStats,
    ) {
        crate::blocks::linear::matmul_group(&self.w, jobs, acts, nb, pool, st);
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn linear_group(
        &self,
        jobs: &[LinJob],
        acts: &[QuantAct],
        xs: &[f32],
        xstride: usize,
        nb: usize,
        lbx: SharedBuf,
        lbt: SharedBuf,
        pool: &Pool,
        st: &mut StepStats,
    ) {
        crate::blocks::linear::linear_group(
            &self.w,
            self.lb.as_ref(),
            jobs,
            acts,
            xs,
            xstride,
            nb,
            lbx,
            lbt,
            pool,
            st,
        );
    }

    #[inline]
    #[allow(dead_code, clippy::too_many_arguments)]
    fn lb_group(
        &self,
        jobs: &[(usize, OutPtr<'_>)],
        xs: &[f32],
        xstride: usize,
        nb: usize,
        lbx: SharedBuf<'_>,
        lbt: SharedBuf<'_>,
        pool: &Pool,
        st: &mut StepStats,
    ) {
        crate::blocks::linear::lb_group(
            self.lb.as_ref(),
            jobs,
            xs,
            xstride,
            nb,
            lbx,
            lbt,
            pool,
            st,
        );
    }

    #[inline]
    fn rmsnorm(&self, x: &[f32], w: &TensorInfo, out: &mut [f32], st: &mut StepStats) {
        crate::blocks::norm::rmsnorm(&self.w, w, self.cfg.eps, x, out, st);
    }

    #[inline]
    fn rmsnorm_into(&self, x: &[f32], w: &TensorInfo, out: &mut [f32]) {
        crate::blocks::norm::rmsnorm_into(&self.w, w, self.cfg.eps, x, out);
    }

    #[inline]
    fn norm_rows(
        &self,
        st8: &mut State,
        w: &TensorInfo,
        nb: usize,
        pool: &Pool,
        stats: &mut StepStats,
    ) {
        crate::blocks::norm::norm_rows(
            &self.w,
            w,
            self.cfg.d_model,
            self.cfg.eps,
            &st8.x,
            Disjoint::new(&mut st8.xb),
            Disjoint::new(&mut st8.acts),
            nb,
            pool,
            stats,
        );
    }

    #[inline]
    fn swiglu_rows(&self, st8: &mut State, nb: usize, pool: &Pool, stats: &mut StepStats) {
        crate::blocks::ffn::swiglu_rows(
            self.cfg.n_ff,
            &mut st8.hb,
            &st8.hb2,
            Disjoint::new(&mut st8.acts),
            nb,
            pool,
            stats,
        );
    }

    #[inline]
    fn prepare_rows(
        &self,
        st8: &mut State,
        l: &Layer,
        nb: usize,
        positions: &[usize],
        pool: &Pool,
        stats: &mut StepStats,
    ) {
        crate::blocks::attn::prepare_rows(
            &self.cfg,
            l.bq.as_deref(),
            l.bk.as_deref(),
            l.bv.as_deref(),
            l.q_norm.as_deref(),
            l.k_norm.as_deref(),
            &st8.rope,
            Disjoint::new(&mut st8.q),
            Disjoint::new(&mut st8.k),
            Disjoint::new(&mut st8.v),
            nb,
            positions,
            pool,
            stats,
        );
    }

    #[inline]
    fn rope(&self, rope: &[f32], v: &mut [f32], n_heads: usize, pos: usize) {
        crate::blocks::rope::rope(
            rope,
            v,
            n_heads,
            pos,
            self.cfg.head_dim,
            self.cfg.rope_dim,
            self.cfg.rope_neox,
        );
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
        let t_fwd = Instant::now();
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
        if self.hy.is_some() {
            return self.forward_hybrid(st8, caches, pool, tokens, positions, want_logits, stats);
        }
        for (i, &t) in tokens.iter().enumerate() {
            self.embed(t, &mut st8.x[i * c.d_model..(i + 1) * c.d_model], stats);
        }

        for (li, l) in self.layers.iter().enumerate() {
            // ---- attention ----
            self.norm_rows(st8, &l.attn_norm, nb, pool, stats);
            {
                // Q, K and V share one activation: one fork–join, not three.
                let lbx = Disjoint::new(&mut st8.lb_x);
                let lbt = Disjoint::new(&mut st8.lb_t);
                let acts = &st8.acts[..nb];
                let xs: &[f32] = &st8.xb;
                let jobs = [
                    LinJob {
                        lin: &l.wq,
                        out: Disjoint::new(&mut st8.q),
                    },
                    LinJob {
                        lin: &l.wk,
                        out: Disjoint::new(&mut st8.k),
                    },
                    LinJob {
                        lin: &l.wv,
                        out: Disjoint::new(&mut st8.v),
                    },
                ];
                self.linear_group(&jobs, acts, xs, c.d_model, nb, lbx, lbt, pool, stats);
            }
            self.prepare_rows(st8, l, nb, positions, pool, stats);
            let qd = c.q_dim();
            crate::blocks::attn::write_kv_rows(
                caches,
                li,
                &st8.k,
                &st8.v,
                nb,
                positions,
                c.n_kv_head,
                c.head_dim,
            );

            let t_attn = Instant::now();
            // one grouped call per run of rows that share a cache (a prompt chunk), one row per call otherwise,
            // so each row computes exactly what it would alone.
            for (r0, len, c) in caches.runs(nb) {
                self.attend(
                    st8,
                    caches.cache(c),
                    pool,
                    li,
                    len,
                    positions[r0],
                    r0,
                    stats,
                );
            }
            stats.ns_attn += t_attn.elapsed().as_nanos() as u64;

            for b in 0..nb {
                st8.acts[b].fill(&st8.xb[b * qd..(b + 1) * qd]);
            }
            {
                let lbx = Disjoint::new(&mut st8.lb_x);
                let lbt = Disjoint::new(&mut st8.lb_t);
                let acts = &st8.acts[..nb];
                let jobs = [LinJob {
                    lin: &l.wo,
                    out: Disjoint::new(&mut st8.xb2),
                }];
                self.linear_group(&jobs, acts, &st8.xb, qd, nb, lbx, lbt, pool, stats);
            }
            for i in 0..nb * c.d_model {
                st8.x[i] += st8.xb2[i];
            }

            // ---- feed-forward ----
            self.norm_rows(st8, &l.ffn_norm, nb, pool, stats);
            {
                let lbx = Disjoint::new(&mut st8.lb_x);
                let lbt = Disjoint::new(&mut st8.lb_t);
                let acts = &st8.acts[..nb];
                let jobs = [
                    LinJob {
                        lin: &l.gate,
                        out: Disjoint::new(&mut st8.hb),
                    },
                    LinJob {
                        lin: &l.up,
                        out: Disjoint::new(&mut st8.hb2),
                    },
                ];
                self.linear_group(&jobs, acts, &st8.xb, c.d_model, nb, lbx, lbt, pool, stats);
            }
            // SwiGLU: silu(gate) ⊙ up, quantised for the down projection.
            self.swiglu_rows(st8, nb, pool, stats);
            {
                let lbx = Disjoint::new(&mut st8.lb_x);
                let lbt = Disjoint::new(&mut st8.lb_t);
                let acts = &st8.acts[..nb];
                let jobs = [LinJob {
                    lin: &l.down,
                    out: Disjoint::new(&mut st8.xb2),
                }];
                self.linear_group(&jobs, acts, &st8.hb, c.n_ff, nb, lbx, lbt, pool, stats);
            }
            for i in 0..nb * c.d_model {
                st8.x[i] += st8.xb2[i];
            }
            if let Some(t) = st8.tap_layers.iter().position(|&x| x == li) {
                let (d, nt) = (c.d_model, st8.tap_layers.len());
                for b in 0..nb {
                    let dst = (b * nt + t) * d;
                    st8.tap[dst..dst + d].copy_from_slice(&st8.x[b * d..(b + 1) * d]);
                }
            }
        }

        // ---- output head ----
        if want_logits.is_empty() {
            stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
            return Ok(());
        }
        let nl = want_logits.len();
        if nl > st8.acts.len() || nl * c.n_vocab > st8.logits.len() {
            return Err(format!(
                "logits requested for {nl} rows; scratch holds {} activations and {} logit rows",
                st8.acts.len(),
                st8.logits.len() / c.n_vocab
            ));
        }
        for (i, &b) in want_logits.iter().enumerate() {
            if b >= nb {
                return Err(format!("logits requested for row {b} of a {nb}-row batch"));
            }
            // The normed rows are gathered to the *front* of the scratch (`i`, not
            // `b`), so the head sees them as a contiguous batch — which is what the
            // LittleBit path requires and costs the dense path nothing.
            let (xs, xe) = (b * c.d_model, (b + 1) * c.d_model);
            let (os, oe) = (i * c.d_model, (i + 1) * c.d_model);
            let (xv, xbv) = (&st8.x, &mut st8.xb);
            self.rmsnorm(&xv[xs..xe], &self.out_norm, &mut xbv[os..oe], stats);
            st8.acts[i].fill(&st8.xb[os..oe]);
        }
        // The output head is 50 MB — a quarter of all traffic for this model. One
        // batched call reads it once for every row; a loop would read it `nl` times,
        // which would make speculation cost more bandwidth than it saves.
        let before = stats.ns_matmul;
        {
            let lbx = Disjoint::new(&mut st8.lb_x);
            let lbt = Disjoint::new(&mut st8.lb_t);
            let acts = &st8.acts[..nl];
            let jobs = [LinJob {
                lin: &self.head,
                out: Disjoint::new(&mut st8.logits),
            }];
            self.linear_group(&jobs, acts, &st8.xb, c.d_model, nl, lbx, lbt, pool, stats);
        }
        stats.ns_head += stats.ns_matmul - before;
        stats.ns_matmul = before;
        stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
        Ok(())
    }

    /// Attention for one layer over a batch, written into `st8.xb` as
    /// `[nb][n_head][head_dim]`.
    ///
    /// Grouped by KV head with an online softmax (`attn.rs`): each task decodes a block
    /// of cached positions once and scores every query head that shares the KV head
    /// against it, for every batch row in the task. When the heads alone cannot fill
    /// the pool, a head's positions are split and the partial results combined.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        st8: &mut State,
        kv: &KvCache,
        pool: &Pool,
        li: usize,
        nb: usize,
        start_pos: usize,
        row0: usize,
        stats: &mut StepStats,
    ) {
        crate::blocks::attn::attend(
            &self.cfg,
            &mut st8.attn,
            &st8.q,
            &mut st8.xb,
            kv,
            pool,
            li,
            nb,
            start_pos,
            row0,
            stats,
        );
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
        let k = key(path, lb)?;
        let mut g = slot().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = g.as_mut() {
            if s.key == k {
                s.since = Instant::now();
                return Ok(Arc::clone(&s.model));
            }
        }
        let file = Gguf::open(path)?;
        crate::check_compatible(&file)?;
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
