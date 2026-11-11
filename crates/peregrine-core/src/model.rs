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

use crate::arch::{Architecture, MtpState};
use crate::gguf::{Format, Gguf, TensorInfo};
use crate::kv::KvCache;
use crate::lbq::Slb;
use crate::pool::Pool;
pub use crate::state::{Caches, State, StepStats};
use crate::tokenizer::Tokenizer;

#[allow(unused_imports)]
pub(crate) use crate::pool::share;

// Bonsai 2's multi-token-prediction block (`mtp.rs`).
pub mod mtp;

/// Transformer model configuration describing dimensional shapes and rotary settings.
///
/// Decoupled from specific model families (ADR 0006). Architecture-specific state
/// and kernels are queried via the `Architecture` trait rather than fields here.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Architecture name (e.g. `"llama"`, `"qwen2"`, `"qwen3"`, `"qwen35"`), for messages and telemetry.
    pub arch_name: &'static str,
    /// NeoX rotary pairing (`i` with `i + rope_dim/2`) instead of adjacent pairs.
    pub rope_neox: bool,
    pub n_layer: usize,
    /// Layers that keep K/V cache (all layers for dense models, full-attention layers only for hybrid).
    pub n_kv_layer: usize,
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
    /// YaRN's parameters, when the file asks for YaRN scaling
    /// (`rope.scaling.type = "yarn"`, gpt-oss). `None` for every other model:
    /// the table is then the plain one, with the file's per-frequency factors.
    pub rope_yarn: Option<crate::blocks::rope::Yarn>,
}

impl Config {
    /// Layers that keep K/V (all layers for dense, full-attention layers only for hybrid).
    #[inline]
    pub fn kv_layers(&self) -> usize {
        self.n_kv_layer
    }

    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.n_kv_head * self.head_dim
    }

    /// Width of the concatenated query heads — the attention output's width, which
    /// is `d_model` for llama and Qwen2 and not for Qwen3.
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// Query heads per key/value head. 1 means multi-head, >1 is grouped-query.
    #[inline]
    pub fn gqa(&self) -> usize {
        self.n_head / self.n_kv_head
    }
}

/// One projection matrix: a dense GGUF tensor, or an index into the LittleBit
/// sidecar when the converter has replaced it (`attach_lb`). The dispatch lives in
/// `linear_group`, so a forward call never knows which it got — and the telemetry
/// reports whichever bytes actually moved.
pub enum Linear {
    Dense(TensorInfo),
    #[cfg(feature = "littlebit")]
    Lb(usize),
}

impl Linear {
    #[inline]
    pub fn dense(&self) -> Option<&TensorInfo> {
        match self {
            Linear::Dense(t) => Some(t),
            #[cfg(feature = "littlebit")]
            Linear::Lb(_) => None,
        }
    }
}

pub struct Model {
    /// The file and the repacked weights: what every forward reads matrices from.
    pub(crate) w: Weights,
    pub cfg: Config,
    pub tk: Tokenizer,
    /// Llama 3.x's per-frequency rotary factors (`rope_freqs.weight`, `rope_dim/2`
    /// floats): each angle is divided by its factor. Absent on every other model.
    pub rope_factors: Option<Vec<f32>>,
    #[cfg(feature = "littlebit")]
    pub(crate) lb_replaced: u64,
    #[cfg(feature = "littlebit")]
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
    // The k-quant matrices repacked into eight-row tiles at load (`TensorInfo::kq_off` points
    // here); empty when the repack is off (`PEREGRINE_KREPACK=0`).
    pub(crate) kq: KqStore,
    // The float matrices packed into 16-row panels at load (`TensorInfo::fp_off` points here);
    // empty when the pack is off (`PEREGRINE_FPACK=0`) or the target has no packed kernel.
    pub(crate) fp: FpStore,
    #[cfg(feature = "littlebit")]
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
    /// The repacked k-quant store (see `TensorInfo::kq_off`).
    #[inline]
    pub fn kq(&self) -> &[u8] {
        self.kq.bytes()
    }
    /// The packed float store (see `TensorInfo::fp_off`).
    #[inline]
    pub fn fp(&self) -> &[u8] {
        self.fp.bytes()
    }
    #[cfg(feature = "littlebit")]
    pub fn lb(&self) -> Option<&Slb> {
        self.lb.as_ref()
    }
    #[cfg(not(feature = "littlebit"))]
    pub fn lb(&self) -> Option<&Slb> {
        None
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

/// The repacked k-quant weights: the Q4_K and Q6_K matrices, and the Q5_0 matrices
/// of a model whose shape forces the legacy format, repacked into eight-row tiles at load (`TensorInfo::kq_off` points here). Built the way the [`TileStore`]
/// is — an anonymous mapping that asks for transparent huge pages, since a decode
/// step streams all of it every token — and `PEREGRINE_TILE_HUGEPAGES=0` uses a plain
/// heap buffer, for the A/B.
///
/// The store is empty when the repack is off — `PEREGRINE_KREPACK=0` — and it is not built at
/// all where the kernel table has no repacked kernel to read it, so it never costs resident
/// memory that nothing uses.
#[derive(Default)]
pub struct KqStore {
    map: Option<memmap2::MmapMut>,
    /// A read-only mapping of a pack cache file (S4.9e), or `None`.
    ro: Option<memmap2::Mmap>,
    heap: Vec<u8>,
}

impl KqStore {
    pub fn new(len: usize) -> KqStore {
        let huge = crate::options::get().tile_hugepages;
        if huge && len > 0 {
            if let Ok(m) = memmap2::MmapMut::map_anon(len) {
                #[cfg(target_os = "linux")]
                let _ = m.advise(memmap2::Advice::HugePage);
                return KqStore {
                    map: Some(m),
                    ro: None,
                    heap: Vec::new(),
                };
            }
        }
        KqStore {
            map: None,
            ro: None,
            heap: vec![0u8; len],
        }
    }

    /// A store that reads its bytes from a mapped pack cache file (S4.9e).
    pub fn from_map(map: memmap2::Mmap) -> KqStore {
        KqStore {
            map: None,
            ro: Some(map),
            heap: Vec::new(),
        }
    }

    /// Whether the bytes come from a file mapping (a pack cache hit).
    pub fn is_file_map(&self) -> bool {
        self.ro.is_some()
    }

    #[inline]
    pub fn bytes(&self) -> &[u8] {
        self.ro
            .as_deref()
            .or(self.map.as_deref())
            .unwrap_or(&self.heap)
    }
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        debug_assert!(self.ro.is_none(), "a mapped pack cache is read-only");
        match &mut self.map {
            Some(m) => m,
            None => &mut self.heap,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.bytes().is_empty()
    }
}

/// The packed float weights: the F32, F16 and BF16 matrices in 16-output-row
/// panels (`TensorInfo::fp_off` points here), built like the [`TileStore`] — an
/// anonymous mapping that asks for transparent huge pages, since a decode step
/// streams all of it every token — and `PEREGRINE_TILE_HUGEPAGES=0` uses a plain
/// heap buffer, for the A/B.
///
/// The store is empty when the pack is off — `PEREGRINE_FPACK=0` — and it is not
/// built at all where the kernel table has no packed float kernel, so it never
/// costs resident memory that nothing reads. Its size is one copy of every packed
/// matrix in the file's own element type, plus the zero padding of each matrix's
/// last panel (see `crate::fpack`).
#[derive(Default)]
pub struct FpStore {
    map: Option<memmap2::MmapMut>,
    /// A read-only mapping of a pack cache file (S4.9e), or `None`.
    ro: Option<memmap2::Mmap>,
    heap: Vec<u8>,
}

impl FpStore {
    pub fn new(len: usize) -> FpStore {
        let huge = crate::options::get().tile_hugepages;
        if huge && len > 0 {
            if let Ok(m) = memmap2::MmapMut::map_anon(len) {
                #[cfg(target_os = "linux")]
                let _ = m.advise(memmap2::Advice::HugePage);
                return FpStore {
                    map: Some(m),
                    ro: None,
                    heap: Vec::new(),
                };
            }
        }
        FpStore {
            map: None,
            ro: None,
            heap: vec![0u8; len],
        }
    }

    /// A store that reads its bytes from a mapped pack cache file (S4.9e).
    pub fn from_map(map: memmap2::Mmap) -> FpStore {
        FpStore {
            map: None,
            ro: Some(map),
            heap: Vec::new(),
        }
    }

    /// Whether the bytes come from a file mapping (a pack cache hit).
    pub fn is_file_map(&self) -> bool {
        self.ro.is_some()
    }

    #[inline]
    pub fn bytes(&self) -> &[u8] {
        self.ro
            .as_deref()
            .or(self.map.as_deref())
            .unwrap_or(&self.heap)
    }
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        debug_assert!(self.ro.is_none(), "a mapped pack cache is read-only");
        match &mut self.map {
            Some(m) => m,
            None => &mut self.heap,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.bytes().is_empty()
    }
}

impl Model {
    /// Load a model from a parsed GGUF file, with the k-quant repack as
    /// `PEREGRINE_KREPACK` says — and only where the kernel table has a kernel
    /// that will read the store.
    pub fn load_with(g: Gguf, families: &[&dyn crate::arch::Family]) -> Result<Model, String> {
        // The repack costs the resident memory of every Q4_K, Q6_K and Q5_0 matrix, so it is
        // built only where a kernel multiplies from it. With the switch on but the
        // repacked kernel unavailable (a `PEREGRINE_ISA` cap below AVX2, a build with
        // the k-quant kernels off) the store would be RSS nothing reads.
        let mk = peregrine_kernels::table::MatKernels::get();
        let kq = crate::options::get().kq_repack && mk.kq_tiles();
        // Same rule for the float pack (S4.9f): on where the packed kernel runs.
        let fp = crate::options::get().fpack && mk.fp_panels();
        let cache = crate::packcache::PackCache::from_options(crate::options::get());
        Self::load_with_kq_fp_cache(g, families, kq, fp, cache.as_ref())
    }

    /// Load a model from a parsed GGUF file, with the k-quant repack forced on or
    /// off whatever `PEREGRINE_KREPACK` says, and regardless of whether a kernel
    /// will read the store.
    ///
    /// This is the switch an A/B of the repack needs: the same file, loaded both
    /// ways, and the two compared. `true` builds the store even on a tier that
    /// would not use it, which is what the store's own test does.
    pub fn load_with_kq(
        g: Gguf,
        families: &[&dyn crate::arch::Family],
        kq: bool,
    ) -> Result<Model, String> {
        let fp =
            crate::options::get().fpack && peregrine_kernels::table::MatKernels::get().fp_panels();
        Self::load_with_kq_fp(g, families, kq, fp)
    }

    /// Load a model from a parsed GGUF file with both load-time packs chosen
    /// explicitly (S3.4's k-quant eight-row tiles, S4.9f's float panels).
    ///
    /// This is the switch an A/B needs: the same file, loaded both ways, and the
    /// two compared. It never uses the pack cache; [`Model::load_with`] does.
    pub fn load_with_kq_fp(
        g: Gguf,
        families: &[&dyn crate::arch::Family],
        kq: bool,
        fp: bool,
    ) -> Result<Model, String> {
        Self::load_with_kq_fp_cache(g, families, kq, fp, None)
    }

    /// [`Model::load_with_kq_fp`], with an explicit pack cache (S4.9e).
    ///
    /// `Some(cache)` maps the packed stores from the cache file when its key
    /// matches, and writes the file after packing when it does not. `None` always
    /// packs in memory.
    pub fn load_with_kq_fp_cache(
        g: Gguf,
        families: &[&dyn crate::arch::Family],
        kq: bool,
        fp: bool,
        cache: Option<&crate::packcache::PackCache>,
    ) -> Result<Model, String> {
        let arch_name = g.str("general.architecture")?;
        let family = families
            .iter()
            .find(|f| f.names().contains(&arch_name))
            .ok_or_else(|| {
                let feat = match arch_name {
                    "llama" => Some("arch-llama"),
                    "qwen2" | "qwen3" => Some("arch-qwen"),
                    "qwen25" | "qwen35" => Some("arch-qwen35"),
                    "gemma3" => Some("arch-gemma3"),
                    "gemma-embedding" => Some("arch-gemma-embedding"),
                    "phi3" => Some("arch-phi3"),
                    _ => None,
                };
                if let Some(feat) = feat {
                    format!(
                        "architecture {arch_name:?} is not enabled in this build (recompile with feature '{feat}')"
                    )
                } else {
                    let known: Vec<&str> = families
                        .iter()
                        .flat_map(|f| f.names().iter().copied())
                        .collect();
                    format!(
                        "Peregrine implements {}; this model is {arch_name:?}",
                        known.join(" and ")
                    )
                }
            })?;
        let mut g = g;
        // The k-quant repack (S3.4): the Q4_K and Q6_K matrices, and Q5_0, are
        // rewritten into eight-row tiles before the family reads the file, so every matrix the
        // family holds knows where its repacked bytes are.
        let kq_layout = if kq {
            crate::kq::layout(&g)
        } else {
            Vec::new()
        };
        let fp_layout = if fp {
            crate::fpack::layout(&g)
        } else {
            Vec::new()
        };
        let key = match cache {
            Some(c) => c.key(&g, &kq_layout, &fp_layout, kq, fp).ok(),
            None => None,
        };
        let mut kq_store = None;
        let mut fp_store = None;
        let mut hit = false;
        if let (Some(c), Some(k)) = (cache, key.as_ref()) {
            if let Some(h) = c.open(k) {
                // The packed bytes are the ones the key described, so the tensor
                // directory only has to learn where each matrix sits.
                hit = true;
                for e in &kq_layout {
                    if let Some(t) = g.tensors.get_mut(&e.name) {
                        t.kq_off = Some(e.offset);
                    }
                }
                for e in &fp_layout {
                    if let Some(t) = g.tensors.get_mut(&e.name) {
                        t.fp_off = Some(e.offset);
                    }
                }
                kq_store = h.kq.map(KqStore::from_map);
                fp_store = h.fp.map(FpStore::from_map);
            }
        }
        let kq = match kq_store {
            Some(s) => s,
            None if kq => crate::kq::repack(&mut g),
            None => KqStore::default(),
        };
        // The float pack (S4.9f): every F32, F16 and BF16 matrix into 16-row panels,
        // before the family reads the file, so each matrix knows where its panels are.
        let fp = match fp_store {
            Some(s) => s,
            None if fp => crate::fpack::repack(&mut g),
            None => FpStore::default(),
        };
        // A miss packed the stores above: write them beside the model cache so the
        // next start maps instead of packing. Best effort, never an error.
        if !hit {
            if let (Some(c), Some(k)) = (cache, key.as_ref()) {
                c.write(k, kq.bytes(), fp.bytes());
            }
        }
        let tk = Tokenizer::from_gguf(&g)?;
        let loaded = family.load(&g, &tk)?;
        Ok(Model {
            w: Weights {
                g,
                tiles: loaded.tiles,
                kq,
                fp,
                #[cfg(feature = "littlebit")]
                lb: None,
            },
            cfg: loaded.cfg,
            tk,
            rope_factors: loaded.rope_factors,
            #[cfg(feature = "littlebit")]
            lb_replaced: 0,
            #[cfg(feature = "littlebit")]
            lb_count: 0,
            arch: loaded.arch,
        })
    }

    /// Allocate reusable runtime state for this model. Sized once per engine/session.
    ///
    /// Applies rope factors, activation layout, LittleBit scratch sizes, and family-specific scratch.
    pub fn new_state(
        &self,
        max_batch: usize,
        max_logit_rows: usize,
        max_ctx: usize,
        threads: usize,
    ) -> State {
        let mut st = State::new_impl(
            &self.cfg,
            max_batch,
            max_logit_rows,
            max_ctx,
            threads,
            self.rope_factors.as_deref(),
            self.arch.act_layout(),
            self.arch.new_scratch(&self.cfg, max_batch),
        );
        let (lx, lt) = self.lb_scratch(max_batch);
        st.set_lb_scratch(lx, lt);
        st
    }

    /// Recurrent state geometry for hybrid architectures, if any.
    pub fn recurrent_shape(&self) -> Option<crate::arch::RecurrentShape> {
        self.arch.recurrent()
    }

    /// Whether this model's rows attend causally or to a whole text (design §2.2).
    pub fn causality(&self) -> crate::arch::Causality {
        self.arch.causality()
    }

    /// One bidirectional pass over whole texts (design §2.2).
    ///
    /// The engine schedules whole texts for a bidirectional model: a text is never split
    /// and no KV cache is written. On return `st.hidden` holds one vector per text at
    /// that text's first global row, with the family's head applied.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_texts(
        &self,
        st8: &mut State,
        pool: &Pool,
        texts: &[&[u32]],
        poolings: &[crate::embed::Pooling],
        stats: &mut StepStats,
    ) -> Result<(), String> {
        if texts.len() != poolings.len() {
            return Err(format!(
                "{} texts but {} pooling types",
                texts.len(),
                poolings.len()
            ));
        }
        let total: usize = texts.iter().map(|t| t.len()).sum();
        if total == 0 {
            return Ok(());
        }
        if total > st8.max_batch {
            return Err(format!(
                "batch of {total} rows exceeds the scratch of {}",
                st8.max_batch
            ));
        }
        for (i, t) in texts.iter().enumerate() {
            if t.is_empty() {
                return Err(format!("text {i} has no tokens"));
            }
            for &id in t.iter() {
                if id as usize >= self.cfg.n_vocab {
                    return Err(format!("token {id} in text {i} is outside the vocabulary"));
                }
            }
        }
        self.arch
            .forward_texts(&self.w, &self.cfg, st8, pool, texts, poolings, stats)
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
    #[cfg(feature = "littlebit")]
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
    #[cfg(feature = "littlebit")]
    pub fn lb_summary(&self) -> Option<(usize, u64)> {
        self.w.lb.as_ref().map(|s| (self.lb_count, s.data_bytes))
    }

    #[cfg(not(feature = "littlebit"))]
    pub fn lb_summary(&self) -> Option<(usize, u64)> {
        None
    }

    /// Scratch floats one forward call needs for LittleBit intermediates, as
    /// (`x′` floats, latent floats) — the maximum over the fused groups, times the
    /// batch. Zero when nothing was converted.
    #[cfg(feature = "littlebit")]
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

    #[cfg(not(feature = "littlebit"))]
    pub fn lb_scratch(&self, _max_batch: usize) -> (usize, usize) {
        (0, 0)
    }

    pub fn weight_bytes(&self) -> u64 {
        #[cfg(feature = "littlebit")]
        {
            self.w.g.total_weight_bytes() - self.lb_replaced
                + self.w.lb.as_ref().map_or(0, |s| s.data_bytes)
        }
        #[cfg(not(feature = "littlebit"))]
        {
            self.w.g.total_weight_bytes()
        }
    }

    /// The file's metadata, for the template detector and the host's telemetry.
    pub fn gguf(&self) -> &Gguf {
        &self.w.g
    }

    /// The pooling type the file declares (`<arch>.pooling_type`), read as llama.cpp reads it.
    ///
    /// `Ok(None)` when the key is absent or is `0` (`none`: per-token rows, not one vector) or
    /// `4` (`rank`: a classifier head). `Err` names an integer llama.cpp has no pooling type
    /// for, which it would abort on.
    pub fn pooling(&self) -> Result<Option<crate::embed::Pooling>, String> {
        let key = format!("{}.pooling_type", self.cfg.arch_name);
        let Ok(v) = self.w.g.u32(&key) else {
            return Ok(None);
        };
        match crate::embed::Pooling::from_gguf(v) {
            Some(p) => Ok(Some(p)),
            None if v == 0 => Ok(None),
            None if v == 4 => Ok(None),
            None => Err(format!("{key} = {v} is not a pooling type llama.cpp knows")),
        }
    }

    /// The width of one embedding: the final hidden state's width.
    pub fn embedding_width(&self) -> usize {
        self.cfg.d_model
    }

    /// The repacked k-quant store: the Q4_K and Q6_K matrices, and Q5_0, rewritten
    /// into eight-row tiles at load (S3.4). Empty when the repack is off, and empty
    /// where the kernel table has no kernel to read it.
    pub fn kq(&self) -> &[u8] {
        self.w.kq.bytes()
    }

    /// The packed float store: the F32, F16 and BF16 matrices, in 16-row panels
    /// (S4.9f). Empty when the pack is off, and empty where the kernel table has no
    /// packed kernel to read it.
    pub fn fp(&self) -> &[u8] {
        self.w.fp.bytes()
    }

    /// Whether the packed stores were mapped from a pack cache file (S4.9e)
    /// rather than built in memory at this load.
    pub fn packed_from_cache(&self) -> bool {
        self.w.kq.is_file_map() || self.w.fp.is_file_map()
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
    fn gqa_ratio_is_derived_not_assumed() {
        let c = Config {
            arch_name: "llama",
            rope_neox: false,
            n_layer: 32,
            n_kv_layer: 32,
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
            rope_yarn: None,
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

    fn key(path: &Path, _lb: bool) -> Result<Key, String> {
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
        #[cfg(feature = "littlebit")]
        let side = _lb
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
        #[cfg(not(feature = "littlebit"))]
        let side = None;
        Ok((
            std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()),
            m.len(),
            mtime,
            side,
        ))
    }

    /// `open_with`, choosing whether a LittleBit sidecar beside the model is used
    /// (`false` runs dense even when one is present).
    pub fn open_with(
        path: &Path,
        lb: bool,
        families: &[&dyn crate::arch::Family],
    ) -> Result<Arc<Model>, String> {
        open_with_template(path, lb, None, families)
    }

    /// `open_with`, with the chat template the caller chose (`--chat-template`):
    /// a file that declares no template is then loaded with that one.
    pub fn open_with_template(
        path: &Path,
        lb: bool,
        template: Option<crate::template::Template>,
        families: &[&dyn crate::arch::Family],
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
        crate::check_compatible_with(&file, template, families)?;
        #[allow(unused_mut)]
        let mut model = Model::load_with(file, families)?;
        if lb {
            #[cfg(feature = "littlebit")]
            {
                if let Some(slb) = crate::lbq::Slb::open_for(path)? {
                    model.attach_lb(slb)?;
                }
            }
            #[cfg(not(feature = "littlebit"))]
            {
                let mut sidecar = path.to_path_buf().into_os_string();
                sidecar.push(".slb");
                let sidecar = PathBuf::from(sidecar);
                if sidecar.is_file() {
                    eprintln!(
                        "info: LittleBit sidecar {} ignored (recompile with feature 'littlebit')",
                        sidecar.display()
                    );
                }
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
