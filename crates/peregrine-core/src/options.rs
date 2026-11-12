//! Engine options: every `PEREGRINE_*` switch in one place (roadmap S0.1).
//!
//! The options are read once, from the environment, the first time any of them is needed, or set
//! explicitly with [`init`] before that (a server or a benchmark can then print and record exactly what
//! it ran with, [`EngineOptions::describe`]). Each field keeps the parsing rule its switch always had, so
//! behaviour is unchanged.
//!
//! The options are process-wide for now. Passing them down explicitly comes with the kernel table
//! (roadmap S1.1), which is what makes varying them inside one process possible.

use std::path::PathBuf;
use std::sync::OnceLock;

/// Where LittleBit's sidecar (`.slb`) is looked for: `PEREGRINE_LB`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LbSidecar {
    /// `<model>.slb` beside the model (unset, empty or `1`).
    BesideModel,
    /// Never (`0`).
    Off,
    /// This path (any other value).
    Path(PathBuf),
}

/// DSpark drafter research switches (`PEREGRINE_DSPARK_*`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DsparkOptions {
    /// `PEREGRINE_DSPARK_LOGITS_START`: first block slot whose logits are used (default 0).
    pub logits_start: usize,
    /// `PEREGRINE_DSPARK_QUANT=q8`: keep the drafter at Q8_0 instead of Q4_K.
    pub quant_q8: bool,
    /// `PEREGRINE_DSPARK_CONF_PRENORM=1` sets this to false: the confidence head reads the pre-norm state.
    pub conf_on_normed: bool,
    /// `PEREGRINE_DSPARK_WIDTH`: run a narrower block than the trained one.
    pub width: Option<usize>,
    /// `PEREGRINE_DSPARK_K`: pin an exact draft count (measurement).
    pub k: Option<usize>,
    /// `PEREGRINE_DSPARK_PROF`: print drafter timings.
    pub prof: bool,
    /// `PEREGRINE_DSPARK_DUMP=<dir>`: dump the drafter's inputs for the PyTorch reference.
    pub dump: Option<PathBuf>,
}

/// The switches a tuning profile may set: `(environment variable, profile key)`. All are output-neutral —
/// the golden tests check that tokens do not depend on the thread count, and spinning and prefetch only move
/// time. The prefill chunk only changes how a prompt is *read*: the same prompt, read in pieces of 16,
/// 32, 64 or 128 tokens, gives the same KV cache, the same logits and the same tokens
/// (`crates/peregrine-models/tests/prefill_chunk.rs`). Activation-layout switches such as
/// `PEREGRINE_TQ_SPAN` / `PEREGRINE_TB_SPAN` change the quantised activations, hence the tokens, and are
/// deliberately absent.
pub const TUNABLE: &[(&str, &str)] = &[
    ("PEREGRINE_SPIN_US", "spin_us"),
    ("PEREGRINE_YIELD_US", "yield_us"),
    ("PEREGRINE_DECODE_THREADS", "decode_threads"),
    ("PEREGRINE_PF", "prefetch_bytes"),
    ("PEREGRINE_PREFILL_CHUNK", "prefill_chunk"),
    ("PEREGRINE_PASS_ROWS", "pass_rows"),
];

/// The default prompt chunk on a pool no wider than the machine's physical cores: the prompt tokens one
/// weight pass carries (`PEREGRINE_PREFILL_CHUNK`).
///
/// 16, from the A/B on the reference machine (12 threads, `peregrine bench --prompt-tokens 512
/// --tokens 1`, four rotating rounds): Llama-3.2-1B 165.9 tok/s against 154.3 at 64 (+7.5 %),
/// Bonsai-2B 239.2 against 227.0 (+5.4 %), Qwen2.5-0.5B 257.3 against 270.7 (-5.0 %). The tenth
/// user's first token does not get later (19.8 s against 21.8 s at 64).
pub const PREFILL_CHUNK_DEFAULT: usize = 16;

/// The default prompt chunk when the worker pool is wider than the machine's physical cores (an SMT
/// machine, where `available_parallelism` reports the logical CPUs).
///
/// A 16-row pass is only 16 rows deep: the rotation hands one row to 16 of the workers and leaves the rest
/// spinning for the 1 ms window on the siblings of the cores doing the work, and every parallel region is
/// short enough that its slowest worker is a large share of it. The fixed part of that cost is about 20 ms
/// per pass on both Bonsai sizes — it does not grow with rows or with the model — and a 512-token prompt
/// pays it 32 times instead of 8. 64 rows amortise it: at 24 workers, medians of four rounds give
/// Bonsai-2B 304 prompt tok/s against 212 (+43 %), Bonsai-9B 90.7 against 79.4 (+14 %), Llama-3.2-1B 225
/// against 189 (+19 %), with decode unchanged on all three. The same measurements on a 12-worker pool
/// keep 16: 257 against 248, 69.0 against 64.9, and 180 against 168. See [`prefill_chunk_default_for`].
pub const PREFILL_CHUNK_SMT: usize = 64;

/// The default prompt chunk for a pool of `threads` workers.
///
/// [`PREFILL_CHUNK_DEFAULT`] was tuned on a pool as wide as the physical cores; when the pool is wider,
/// the per-pass cost of rows spread thinly over SMT siblings dominates, and [`PREFILL_CHUNK_SMT`] wins.
/// The chunk only changes how the prompt is read, not what it produces, so the choice is output-neutral
/// (`crates/peregrine-models/tests/prefill_chunk.rs`).
pub fn prefill_chunk_default_for(threads: usize) -> usize {
    match crate::pool::physical_cores() {
        Some(cores) if threads > cores => PREFILL_CHUNK_SMT,
        _ => PREFILL_CHUNK_DEFAULT,
    }
}

/// The prompt tokens one pass carries when no slot is decoding: 64, on both x86 tiers measured.
///
/// The narrow chunk ([`prefill_chunk_default_for`]) exists to bound what a *decoding* user waits
/// for while somebody else's prompt is read. When nobody is decoding there is no such user, and a
/// wider pass is simply faster: it reads the weights once for more rows.
///
/// Measured on the desktop (Ryzen AI 9 HX 370, 12 threads, `peregrine bench --prompt-tokens 512`
/// and `4096`, medians of three interleaved rounds; the narrow chunk is 16 and a long prompt is
/// already read in 64-token pieces, so the 4096 rows are flat):
///
/// | model | tier | 16 | 64 | 128 |
/// |---|---|---|---|---|
/// | Llama 3.2 1B | avx512 | 301.9 | **336.4** | 327.8 |
/// | Qwen 2.5 0.5B | avx512 | 487.6 | 588.9 | **619.9** |
/// | Gemma 3 1B | avx512 | 268.8 | **329.8** | 322.4 |
/// | Llama 3.2 1B | avx2 | 244.9 | **272.1** | 263.3 |
/// | Qwen 2.5 0.5B | avx2 | 371.2 | 456.5 | **459.6** |
/// | Gemma 3 1B | avx2 | 229.6 | **261.4** | 260.5 |
///
/// 64 wins on both 1B models on both tiers and loses 5 % (avx512) and 1 % (avx2) to 128 on Qwen,
/// so the two are within noise of each other overall; 64 is also free in memory, because the
/// scratch already holds 64 prompt rows for speculation (`SCRATCH_CHUNK` in `serve`). The
/// embedding path's pass-size rule ([`pass_rows_default_for`]) returns 16 here, i.e. today's
/// chunk, so it is not the wide one.
pub const WIDE_CHUNK_DEFAULT: usize = 64;

/// [`WIDE_CHUNK_DEFAULT`] on the process's kernel tier.
///
/// Both measured tiers chose the same number; the parameter is here because the choice is made per
/// tier from measurement, and a future tier may disagree.
pub fn wide_chunk_default_for(_tier: crate::cpu::Tier) -> usize {
    WIDE_CHUNK_DEFAULT
}

/// The prompt tokens one pass carries when no slot is decoding, with `--fast-prompt` on:
/// 256 (S5.4b's first look; the mode reads the weights from the resident 8-bit form, so a
/// wider pass amortises that read further than the exact one).
///
/// Measured on the desktop on Llama 3.2 1B Q4_K_M, one round of interleaved runs per cell,
/// 1-minute load recorded (`benchmarks/results/desktop-zen5-2026-11-12-s5.4b/`). Prompt tok/s:
///
/// | threads | tokens | default | 64 | 128 | 256 | 512 |
/// |---|---|---|---|---|---|---|
/// | 12 | 512 | 327.0 | 410.8 | **437.6** | 428.3 | 402.9 |
/// | 12 | 4096 | 226.7 | 273.2 | 283.2 | 285.0 | **288.2** |
/// | 1 | 512 | 46.0 | 53.8 | 67.5 | 67.5 | **67.6** |
/// | 1 | 4096 | 25.7 | 34.1 | 33.7 | 36.6 | **42.5** |
///
/// 256 is within 2.2 % of the best on three of the four rows (128 wins the short 12-thread
/// row, 512 the long ones) and 14 % off on the fourth; 512 loses 8 % on the short 12-thread
/// row. Both are inside this first look's noise on the 12-thread rows. 256 is the balanced
/// choice and its scratch is half the size. S5.4c re-measures this in a quiet window.
pub const WIDE_CHUNK_DEFAULT_FAST: usize = 256;

/// The fast mode's wide chunk on aarch64: **512** (S5.4e). Measured on the DGX Spark at ten
/// threads on Llama 3.2 1B and Qwen 2.5 0.5B, 512 is 7.7–26.6 % ahead of 256 on both prompt
/// lengths (512 and 4 096 tokens) and level at one thread (81.0 against 81.0 tok/s on the 1B,
/// 88.5 against 89.0 on the 0.5B). That clears the 5 %-on-both-models bar the x86 first look
/// set, so ARM widens; x86 keeps 256.
pub const WIDE_CHUNK_DEFAULT_FAST_ARM: usize = 512;

/// [`wide_chunk_default_for`] for the process's mode: the fast prompt mode widens further
/// (`WIDE_CHUNK_DEFAULT_FAST`, or `WIDE_CHUNK_DEFAULT_FAST_ARM` on aarch64), because a prompt
/// row read from the 8-bit form pays one scale multiply per 256 weights and wants more rows
/// under it.
pub fn wide_chunk_default_for_mode(tier: crate::cpu::Tier, fast_prompt: bool) -> usize {
    if fast_prompt {
        if cfg!(target_arch = "aarch64") {
            WIDE_CHUNK_DEFAULT_FAST_ARM
        } else {
            WIDE_CHUNK_DEFAULT_FAST
        }
    } else {
        wide_chunk_default_for(tier)
    }
}

/// The largest prompt chunk one weight pass may carry: the matmul scratch's row limit
/// (`blocks::linear::MAX_BATCH`). The engine lowers it further when the decode rows of the
/// other users need the room.
pub const MAX_PREFILL_CHUNK: usize = crate::blocks::linear::MAX_BATCH;

/// Where the packed-weight cache lives (`PEREGRINE_PACK_CACHE`, S4.9e).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum PackCacheSetting {
    /// `$PEREGRINE_CACHE_DIR`/`packs`, `$XDG_CACHE_HOME/peregrine/packs`, or
    /// `~/.cache/peregrine/packs`.
    #[default]
    Default,
    /// `PEREGRINE_PACK_CACHE=0`: pack in memory on every load, as before S4.9e.
    Off,
    /// `PEREGRINE_PACK_CACHE=<dir>`: the packs of every model go in this directory.
    Path(PathBuf),
}

/// The row budget of a pass that carries work which never decodes (S4.9b): one weight pass
/// carries up to this many rows of pooled or label slots, taken from as many waiting slots as
/// fit.
///
/// The default is derived from the machine and the model ([`pass_rows_default_for`]): a pass's
/// activation rows should fit in the caches a few workers write, and a batch of short texts
/// should be one or two passes. `PEREGRINE_PASS_ROWS` overrides it.
pub const PASS_ROWS_DEFAULT: usize = 256;

/// The widest pass budget `PEREGRINE_PASS_ROWS` may ask for. The matmul driver splits a wider
/// batch into blocks of its own row limit, so this only bounds the engine's scratch.
pub const MAX_PASS_ROWS: usize = 512;

/// The L2 cache of one core, in bytes: from sysfs, or 256 KiB when the machine does not say.
///
/// The pass budget keeps a pass's activation rows near the caches: no extra dependency, one
/// small read at engine construction.
pub fn l2_bytes() -> usize {
    let base = std::path::Path::new("/sys/devices/system/cpu/cpu0/cache");
    if let Ok(entries) = std::fs::read_dir(base) {
        let mut dirs: Vec<_> = entries.flatten().map(|e| e.path()).collect();
        dirs.sort();
        for dir in dirs {
            let level = std::fs::read_to_string(dir.join("level"))
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok());
            if level != Some(2) {
                continue;
            }
            if let Some(n) = std::fs::read_to_string(dir.join("size"))
                .ok()
                .and_then(|s| parse_cache_size(s.trim()))
            {
                return n;
            }
        }
    }
    256 << 10
}

/// `512K` / `1M` (sysfs's cache-size spelling) in bytes.
fn parse_cache_size(s: &str) -> Option<usize> {
    let (digits, mult) = match s.chars().last()? {
        'K' | 'k' => (&s[..s.len() - 1], 1 << 10),
        'M' | 'm' => (&s[..s.len() - 1], 1 << 20),
        _ => (s, 1),
    };
    digits.trim().parse::<usize>().ok().map(|n| n * mult)
}

/// Rows of a pass per byte of weights: the measured ridge of this engine's kernels.
///
/// A pass is memory-bound below this many rows and compute-bound above it. It is
/// `FLOP/s ÷ (2 × bytes/s)`: every step reads the weights once (a pass's bytes are the model's
/// weight bytes whatever its row count) and does `2 × params` FLOPs per row. Measured on bluefin
/// at 4 threads, Q8_0: a 0.6B model does 417 GFLOP/s in 16-row passes and 373 GFLOP/s at 256, so
/// 16 rows is already at the ridge; the ratio 420 GFLOP/s over 2 × 19.5 GB/s is [`ROWS_PER_WEIGHT_BYTE`].
pub const ROWS_PER_WEIGHT_BYTE: f64 = 10.8;

/// The default row budget of a non-decoding pass (S4.9b): what keeps a pass's activations hot
/// and stops it before it turns compute-bound.
///
/// Two bounds, the smaller wins.
///
/// * The activation rows of one pass should stay inside half the L2. The measured files show the
///   pass's buffers are shared by the workers — the design's `threads × L2/2` overcounts them by
///   the thread count — so the bound is `L2/2 / (4 × d_model)` (four f32 buffers of `d_model` per
///   row).
/// * A pass turns compute-bound at `bytes_per_param × `[`ROWS_PER_WEIGHT_BYTE`] rows; packing
///   more only spends compute at lower efficiency.
///
/// The sweep on bluefin (4 threads, 512 KiB L2 per core) pinned it: Q8_0 (1.06 bytes per weight)
/// is best at 16 rows and 7-16% slower at 32 and above; F16 (2 bytes per weight) gains 5-15% at
/// 32; the 300M encoder is flat from 32 and 12-14% ahead of 16. A bidirectional model packs whole
/// texts, so its floor is 32.
///
/// A **small float encoder** — every matrix F32/F16/BF16, and at most
/// [`SMALL_FLOAT_ENCODER_BYTES`] of packed float weights — is the exception: one pass of 128
/// rows (eight whole 16-token texts, one 128-token text) is 14-24% faster than 32 rows on
/// bluefin (measured on all-MiniLM-L6-v2 F16, 44 MiB, and bge-small-en-v1.5 F32, 132 MiB),
/// because the pass streams the whole model's weights and 128 rows amortise that stream while
/// the packed float kernel's activation block is still capped at 128 rows anyway. A quantised
/// (Q8_0) encoder keeps 32: its tile kernels' activation working set grows past L2 above 32,
/// and bge-m3 Q8_0 measured 3% slower at 128 (S4.9m). `PEREGRINE_PASS_ROWS` overrides the
/// result.
pub const SMALL_FLOAT_ENCODER_BYTES: u64 = 256 << 20;

/// The largest Q8_0 `bert` file whose matrices are dequantised into f32 panels (S4.9m):
/// its weights become four times the file's size, so the packed form stays under
/// [`SMALL_FLOAT_ENCODER_BYTES`]. Measured on MiniLM Q8_0 (23 MiB) and bge-small-en-v1.5
/// Q8_0 (34 MiB), which win; bge-m3 Q8_0 (604 MiB) and EmbeddingGemma 300M Q8_0 are
/// past it and keep the int8 tiles. aarch64's `modern-bert` files use
/// [`Q8_FLOAT_MAX_BYTES_ARM`] instead (S4.9o).
pub const Q8_FLOAT_MAX_BYTES: u64 = 64 << 20;

/// The largest Q8_0 `modern-bert` file whose matrices are dequantised into f32 panels on the
/// aarch64 tiers that carry the Q8_0 int8 tiles (`dotprod` and up; S4.9o). It equals
/// [`SMALL_FLOAT_ENCODER_BYTES`], so a converted file is still a "small float encoder" for
/// [`pass_rows_default_for`] and its panels stay under 1 GiB.
///
/// Measured on this ARM host (10 threads, medians, cache off) against the int8 tiles the
/// int8 tiers would otherwise run: gte-modernbert-base Q8_0 (151 MiB) 12.25 → 20.63 texts/s
/// at 128 tokens, 76.3 → 97.7 at 16, 3.13 → 3.93 at 512 and 47.5 → 92.4 on a batch of 16 ×
/// 32, on aarch64-neon-i8mm. The 64 MiB [`Q8_FLOAT_MAX_BYTES`] was measured on x86, where
/// the int8 tile is the faster kernel past that size; on ARM the Q8_0 tile has no `i8mm`
/// body and loses to the float panel kernel at every measured cell. Only the 151 MiB gte
/// file was measured; the Granite Embedding R2 Q8_0 files are smaller than this constant and
/// take the same route.
pub const Q8_FLOAT_MAX_BYTES_ARM: u64 = 256 << 20;

/// [`pass_rows_default_for`] with the model's weight bytes, for the small-float-encoder rule.
pub fn pass_rows_default_for(
    d_model: usize,
    bytes_per_param: f64,
    bidirectional: bool,
    weight_bytes: u64,
    float_only: bool,
) -> usize {
    let l2_rows = (l2_bytes() / 2) / (4 * d_model.max(1));
    let ridge = (bytes_per_param.max(0.1) * ROWS_PER_WEIGHT_BYTE) as usize;
    let rows = l2_rows.min(ridge);
    if bidirectional {
        let floor = if float_only && weight_bytes <= SMALL_FLOAT_ENCODER_BYTES {
            PASS_ROWS_DEFAULT / 2
        } else {
            PASS_ROWS_DEFAULT / 8
        };
        rows.max(floor).clamp(16, 128)
    } else {
        rows.clamp(16, 64)
    }
}

/// Whether the fast prompt mode's 8-bit form has a SIMD body on this tier (AVX-512 or
/// AVX2 on x86; `smmla` or `sdot` on aarch64 since S5.4e). A tier without one runs the
/// mode correctly through the scalar twin, just no faster than the exact kernel.
pub fn fast_prompt_simd() -> bool {
    peregrine_kernels::table::MatKernels::get().q8k_simd()
}

/// The name of that body, for the startup line: `None` where the mode runs through
/// the scalar twin.
pub fn fast_prompt_body() -> Option<&'static str> {
    peregrine_kernels::table::MatKernels::get().q8k_body()
}

/// The weight formats whose prompt rows take the 8-bit form on this tier (S5.4f).
///
/// Empty where the tier has no faster 8-bit kernel: the option then computes
/// exactly what the exact engine computes, and the startup line says so.
pub fn fast_prompt_formats() -> &'static [peregrine_kernels::format::Format] {
    peregrine_kernels::table::MatKernels::get().q8k_formats()
}

/// [`fast_prompt_formats`] as the human-readable list the startup line, `peregrine
/// tier` and the option's help quote (`"Q6_K and Q5_0"`, or `"none"`).
pub fn fast_prompt_format_list() -> String {
    let formats = fast_prompt_formats();
    match formats {
        [] => "none".to_string(),
        [one] => one.label().to_string(),
        [.., last] => {
            let head = formats[..formats.len() - 1]
                .iter()
                .map(|f| f.label())
                .collect::<Vec<_>>()
                .join(", ");
            format!("{head} and {}", last.label())
        }
    }
}

/// Every engine switch. Defaults are the tuned values; see each field's environment variable.
/// The k-quant repack is on by default on every tier, the AVX-512 tier included
/// ([`EngineOptions::kq_repack_note`] says whether the switch overrode it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineOptions {
    // Thread pool.
    /// `PEREGRINE_SPIN_US`: how long a waiting worker spins before yielding or parking (µs).
    pub spin_us: u64,
    /// `PEREGRINE_YIELD_US`: how long it then yields before parking (µs).
    pub yield_us: u64,
    /// `PEREGRINE_ADAPTIVE_SPIN=0`: keep the configured spin window even when the host is
    /// oversubscribed. On by default: the window shortens (and finally drops) when the
    /// thread's run-queue delay says there are more runnable threads than cores.
    pub adaptive_spin: bool,
    /// `PEREGRINE_DECODE_THREADS`: width of the narrower pool for one-row steps (default: physical cores).
    pub decode_threads: Option<usize>,
    /// `PEREGRINE_AFFINITY=1`: pin each pool worker to its own physical core (Linux), so a pool as wide as
    /// the cores never puts two workers on one core's SMT siblings and the scheduler cannot migrate them.
    pub affinity: bool,

    // Kernels.
    /// `PEREGRINE_ISA` (`scalar`, `avx2`, `avx512`; default `native`): the highest kernel tier to use.
    pub isa: Option<crate::cpu::Tier>,
    /// `PEREGRINE_ZMM=0` keeps the 256-bit ternary kernels on an AVX-512 machine.
    pub zmm: bool,
    /// `PEREGRINE_NO_WIDE=1` turns off the wide batched k-quant kernels.
    pub wide: bool,
    /// `PEREGRINE_TILES=0` turns off the repacked row tiles for ternary weights.
    pub tiles: bool,
    /// `PEREGRINE_KREPACK`: repack the k-quant matrices (and Q5_0) into eight-row tiles and
    /// multiply them with the repacked kernels. On by default on every tier, the AVX-512 tier
    /// included (it runs the AVX2 repacked kernels; the A/B accepts a small decode cost for the
    /// prompt win — S3.4 §10). The switch overrides the default in both directions.
    pub kq_repack: bool,
    /// Whether `PEREGRINE_KREPACK` was given, so [`EngineOptions::kq_repack_note`] can say why the
    /// switch has its value.
    pub kq_repack_switch: bool,
    /// `PEREGRINE_FPACK=0`: keep the F32, F16 and BF16 matrices in the file's own rows and
    /// multiply them with the pre-S4.9f float tile (S4.9f's A/B switch).
    pub fpack: bool,
    /// Whether `PEREGRINE_FPACK` was given, for the same reason as `kq_repack_switch`.
    pub fpack_switch: bool,
    /// `PEREGRINE_Q8_FLOAT=0`: a small Q8_0 encoder keeps its int8 tiles instead of being
    /// dequantised once into the float kernel's f32 panels (S4.9m). On by default for a
    /// `bert` file of at most [`Q8_FLOAT_MAX_BYTES`], and on aarch64 for a `modern-bert`
    /// file of at most [`Q8_FLOAT_MAX_BYTES_ARM`] (S4.9o).
    pub q8_float: bool,
    /// `PEREGRINE_DYNAMIC=0` uses the static row partition instead of dynamic work splitting.
    pub dynamic_rows: bool,
    /// `PEREGRINE_PF`: software prefetch distance in bytes (0 = off).
    pub prefetch_bytes: usize,
    /// `PEREGRINE_TQ_SPAN` (16, 32, 64, 128): values sharing one activation scale in the ternary layout.
    pub tq_span: usize,
    /// `PEREGRINE_TB_SPAN` (32, 64, 128): tile-batch span for the hybrid model's tiled matmuls.
    pub tb_span: usize,
    /// `PEREGRINE_EXACT=1`: ternary matmuls in f32 against decoded weights (slow reference path).
    pub exact: bool,
    /// `PEREGRINE_FAST_PROMPT=1` (`--fast-prompt`): a *prompt* row's weights are
    /// read from an 8-bit form built at load, one scale per 256 weights, instead of
    /// the file's own codes — for the weight formats whose form's kernel is faster
    /// than the tier's exact kernel for that format
    /// ([`peregrine_kernels::table::MatKernels::q8k_uses_form`], S5.4f). On AVX-512
    /// that is Q4_K and Q6_K; on AVX2 Q6_K (Q4_K's exact kernel is the stronger one
    /// there, and Q5_0 is level); on aarch64 Q4_K and Q6_K on every cap, and Q5_0
    /// where the multiply is a NEON body (`neon-dotprod`, `neon-i8mm`; S5.4h); on a
    /// tier with no SIMD body, none, so the option changes nothing. Prompt
    /// passes get faster where a format takes it and their logits move (the accuracy
    /// report that says by how much is roadmap S5.4c, not published yet); generated
    /// rows, embeddings, rerank and decisions keep the exact kernels. Off by default
    /// (ADR 0003); `docs/design/s5.4-fast-prompt.md`.
    pub fast_prompt: bool,

    // Memory.
    /// `PEREGRINE_HUGEPAGES=1`: copy the model into huge-page-backed memory at load (Linux).
    pub hugepages: bool,
    /// `PEREGRINE_TILE_HUGEPAGES=0`: keep the repacked tiles in ordinary pages.
    pub tile_hugepages: bool,
    /// `PEREGRINE_PACK_CACHE`: where the load-time packed stores are cached on disk
    /// (`~/.cache/peregrine/packs` by default; `0` turns the cache off, a path
    /// overrides the directory). S4.9e.
    pub pack_cache: PackCacheSetting,
    /// `PEREGRINE_PACK_CACHE_MB`: the cache directory's size cap in MiB (default 4096;
    /// `0` disables eviction). Least-recently-used pack files are removed to stay under it.
    pub pack_cache_mb: usize,
    /// `PEREGRINE_KV_ROTATE=0`: plain int8 KV cache instead of the Hadamard-rotated one.
    pub kv_rotate: bool,
    /// `PEREGRINE_KV` (`f16`, `f32`, `i8`): the precision of the KV cache. The
    /// default is f16; `f32` keeps the keys and values in full precision, which is
    /// what a like-for-like comparison against another engine needs.
    pub kv_prec: crate::kv::KvPrec,

    // Scheduling and speculation.
    /// `PEREGRINE_MIX=0`: no mixed prefill/decode steps. Speculation (MTP, suffix, DSpark) is verified only in
    /// the mixed step, so this also turns speculation off, for the server and for conversations (`Session`).
    pub mix: bool,
    /// `PEREGRINE_PREFILL_CHUNK`: prompt tokens one weight pass carries (default: 16, or 64 when the
    /// pool is wider than the physical cores — [`prefill_chunk_default_for`]).
    ///
    /// The engine prefills a prompt in pieces of this size, so a long prompt is read in several passes.
    /// Smaller pieces keep the activation scratch in cache and read the weights more often; the tokens
    /// do not depend on it (`crates/peregrine-models/tests/prefill_chunk.rs` pins that). A pass with no
    /// decoding slot beside it widens to [`wide_chunk_default_for`]; setting this switch forces one
    /// chunk, the wide one included.
    pub prefill_chunk: Option<usize>,
    /// `PEREGRINE_PASS_ROWS`: rows one weight pass carries for work that never decodes — pooled
    /// (embedding) and label (decision) slots (default: [`pass_rows_default_for`], the machine's
    /// L2 and the model's width).
    ///
    /// It is scheduling only: a text's vector and a question's label logits do not depend on it
    /// (`crates/peregrine-models/tests/embeddings.rs` and `labels.rs` pin that for budgets 1, 16,
    /// 64, 256 and the default).
    pub pass_rows: Option<usize>,
    /// `PEREGRINE_WORK_CACHE_MB`: memory for the caches of slots that never decode — pooled
    /// (embedding) and label (decision) slots (S4.9b; default 512 MiB).
    ///
    /// A non-decoding slot's cache is sized to its own prompt, so this bounds how many texts of
    /// a request are in flight at once, not how many the request may hold: the rest wait in the
    /// engine's queue. `--slots` bounds chat slots only.
    pub work_cache_mb: Option<usize>,
    /// `PEREGRINE_EMBED_CACHE_MB`: memory for the exact cache of finished embedding vectors,
    /// keyed by token ids, pooling and dimensions (S4.9k; default 64 MiB, `0` disables).
    ///
    /// The cache is serving-only: a hit returns the vector a computation would, bit for bit.
    /// 64 MiB holds about 16 000 vectors of 1024 f32 (21 000 of 768), with their keys counted.
    pub embed_cache_mb: usize,
    /// `PEREGRINE_MTP=0`: no MTP speculation.
    pub mtp: bool,
    /// `PEREGRINE_MTP_VOCAB`: vocabulary rows the MTP drafter scores (default 32768).
    pub mtp_vocab: usize,
    /// DSpark drafter switches.
    pub dspark: DsparkOptions,

    // LittleBit.
    /// `PEREGRINE_LB`: where the sidecar is looked for.
    pub lb_sidecar: LbSidecar,
    /// `PEREGRINE_LB_I8=0`: f32 instead of int8 sign dots in LittleBit.
    pub lb_i8: bool,

    // Diagnostics.
    /// `PEREGRINE_PROF=1`: per-phase timings of the hybrid forward.
    pub prof: bool,
    /// `PEREGRINE_TRACE` (any value): speculation trace on stderr.
    pub trace: bool,

    // Calibration (S2.6 PR 3/4).
    /// Run the startup calibration: on for `peregrine serve` when nothing else pins the knobs,
    /// off for library construction, `bench` and `tune` unless `PEREGRINE_CALIBRATE` asks for it.
    /// `PEREGRINE_CALIBRATE=0` (`off`, `false`) turns it off everywhere (S2.6 decision 6).
    pub calibrate: bool,
    /// Source of active tuning parameters (0=default, 1=startup, 2=profile, 3=env).
    pub tuning_source: crate::calib::TuningSource,
    /// Time spent during startup calibration in milliseconds.
    pub calibration_us: u64,
}

/// Default spin window: 1000 µs. On the Ryzen AI 9 HX 370 under WSL2 a Bonsai 2 decode step has serial
/// stretches between its matmuls that outlast 50 µs, and every park then paid a VM wake-up: 2.9 tok/s at
/// 50 µs, 10.9 at 500 µs, 11.4 at 1000 µs (12 threads).
pub const SPIN_DEFAULT_US: u64 = 1000;
pub use peregrine_kernels::options::PF_DEFAULT;

impl Default for EngineOptions {
    fn default() -> Self {
        EngineOptions {
            spin_us: SPIN_DEFAULT_US,
            yield_us: 0,
            adaptive_spin: true,
            decode_threads: None,
            affinity: false,
            isa: None,
            zmm: true,
            wide: true,
            tiles: true,
            // The tier's default, from detection alone (`resolved_under` reads these options).
            kq_repack: peregrine_kernels::options::kq_repack_default(crate::cpu::resolved_under(
                None,
            )),
            kq_repack_switch: false,
            fpack: true,
            fpack_switch: false,
            q8_float: true,
            dynamic_rows: true,
            prefetch_bytes: PF_DEFAULT,
            tq_span: 16,
            tb_span: 32,
            exact: false,
            fast_prompt: false,
            hugepages: false,
            tile_hugepages: true,
            pack_cache: PackCacheSetting::Default,
            pack_cache_mb: 4096,
            kv_rotate: true,
            kv_prec: crate::kv::KvPrec::F16,
            mix: true,
            prefill_chunk: None,
            pass_rows: None,
            work_cache_mb: None,
            embed_cache_mb: 64,
            mtp: true,
            mtp_vocab: 32768,
            dspark: DsparkOptions {
                logits_start: 0,
                quant_q8: false,
                conf_on_normed: true,
                width: None,
                k: None,
                prof: false,
                dump: None,
            },
            lb_sidecar: LbSidecar::BesideModel,
            lb_i8: true,
            prof: false,
            trace: false,
            calibrate: false,
            tuning_source: crate::calib::TuningSource::Default,
            calibration_us: 0,
        }
    }
}

impl EngineOptions {
    /// Calibration duration in milliseconds.
    pub fn calibration_ms(&self) -> f64 {
        self.calibration_us as f64 / 1000.0
    }

    /// The options from the environment, with each switch's historical parsing rule.
    pub fn from_env() -> Self {
        Self::from_env_and_profile(None)
    }

    /// The options from the environment and an optional tuning profile:
    /// environment variables win over the profile, the profile wins over defaults.
    ///
    /// Startup calibration stays off here: a library or benchmark caller gets the settings the
    /// environment and profile name, never a measurement. See
    /// [`EngineOptions::from_env_and_profile_calibrating`] for the server's variant.
    pub fn from_env_and_profile(profile: Option<&crate::profile::Profile>) -> Self {
        Self::from_chained_mode(|k| std::env::var(k).ok(), profile, false)
    }

    /// The options a long-lived server starts with: like [`Self::from_env_and_profile`], but the
    /// startup calibration runs when no profile exists, neither `decode_threads` nor `spin_us` is
    /// pinned, and `PEREGRINE_CALIBRATE` is not `0`.
    ///
    /// `bench` and `tune` keep [`Self::from_env_and_profile`]: a published number must record the
    /// settings it ran with, not a measurement taken before it started.
    pub fn from_env_and_profile_calibrating(profile: Option<&crate::profile::Profile>) -> Self {
        Self::from_chained_mode(|k| std::env::var(k).ok(), profile, true)
    }

    /// Build options from a custom key-lookup (e.g. environment) and a profile:
    /// `get` wins over the profile, the profile wins over calibration, calibration wins over
    /// defaults. The calibration runs only when `PEREGRINE_CALIBRATE=1` and nothing pins the
    /// values it would set.
    pub fn from_chained(
        get: impl Fn(&str) -> Option<String>,
        profile: Option<&crate::profile::Profile>,
    ) -> Self {
        Self::from_chained_mode(get, profile, false)
    }

    /// [`Self::from_chained`], with the server's default: the calibration runs when
    /// `PEREGRINE_CALIBRATE` says nothing and nothing pins the knobs it would set.
    pub fn from_chained_calibrating(
        get: impl Fn(&str) -> Option<String>,
        profile: Option<&crate::profile::Profile>,
    ) -> Self {
        Self::from_chained_mode(get, profile, true)
    }

    /// [`Self::from_chained`], with the calibration's default chosen by the caller:
    /// `calibrate_when_unset` makes it run when `PEREGRINE_CALIBRATE` says nothing.
    fn from_chained_mode(
        get: impl Fn(&str) -> Option<String>,
        profile: Option<&crate::profile::Profile>,
        calibrate_when_unset: bool,
    ) -> Self {
        // Only settings that cannot change a token are read from a profile (ADR 0003, ADR 0005):
        // the ones `TUNABLE` lists. Anything else in the file is ignored, and `get` wins over it.
        let mut opts = Self::from_lookup(|k| {
            if let Some(v) = get(k) {
                return Some(v);
            }
            let prof = profile?;
            let key_name = TUNABLE
                .iter()
                .find(|(env, _)| *env == k)
                .map(|(_, key)| *key)?;
            prof.get(key_name).map(str::to_string)
        });
        if calibrate_when_unset && get("PEREGRINE_CALIBRATE").is_none() {
            opts.calibrate = true;
        }

        let env_spin = get("PEREGRINE_SPIN_US").is_some();
        let env_dec = get("PEREGRINE_DECODE_THREADS").is_some();

        if profile.is_some() {
            opts.tuning_source = if env_spin && env_dec {
                crate::calib::TuningSource::Env
            } else {
                crate::calib::TuningSource::Profile
            };
        } else if env_spin && env_dec {
            opts.tuning_source = crate::calib::TuningSource::Env;
        } else if opts.calibrate {
            let cal = crate::calib::calibrate();
            opts.tuning_source = cal.source;
            opts.calibration_us = (cal.duration_ms * 1000.0).round() as u64;
            if !env_dec {
                opts.decode_threads = Some(cal.decode_threads);
            }
            if !env_spin {
                opts.spin_us = cal.spin_us;
            }
        }

        opts
    }

    /// The options from any key → value lookup (the environment, a file, a test's map).
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let d = EngineOptions::default();
        let s = |k: &str| get(&format!("PEREGRINE_{k}"));
        // "on unless set to `off`" / "off unless set to `on`".
        let unless = |k: &str, off: &str| s(k).is_none_or(|v| v != off);
        let only = |k: &str, on: &str| s(k).is_some_and(|v| v == on);
        let num = |k: &str| s(k).and_then(|v| v.parse::<usize>().ok());
        let isa = crate::cpu::parse_isa(s("ISA").as_deref());
        let krepack = s("KREPACK");
        EngineOptions {
            spin_us: s("SPIN_US")
                .and_then(|v| v.parse().ok())
                .unwrap_or(d.spin_us),
            yield_us: s("YIELD_US")
                .and_then(|v| v.parse().ok())
                .unwrap_or(d.yield_us),
            adaptive_spin: unless("ADAPTIVE_SPIN", "0"),
            decode_threads: num("DECODE_THREADS").filter(|&v| v > 0),
            affinity: only("AFFINITY", "1"),
            isa,
            zmm: unless("ZMM", "0"),
            wide: unless("NO_WIDE", "1"),
            tiles: unless("TILES", "0"),
            kq_repack: peregrine_kernels::options::kq_repack_from(
                krepack.as_deref(),
                crate::cpu::resolved_under(isa),
            ),
            kq_repack_switch: krepack.is_some(),
            fpack: unless("FPACK", "0"),
            fpack_switch: s("FPACK").is_some(),
            q8_float: unless("Q8_FLOAT", "0"),
            dynamic_rows: unless("DYNAMIC", "0"),
            prefetch_bytes: num("PF").unwrap_or(d.prefetch_bytes),
            tq_span: num("TQ_SPAN")
                .filter(|v| [16, 32, 64, 128].contains(v))
                .unwrap_or(d.tq_span),
            tb_span: num("TB_SPAN")
                .filter(|v| [32, 64, 128].contains(v))
                .unwrap_or(d.tb_span),
            exact: only("EXACT", "1"),
            fast_prompt: only("FAST_PROMPT", "1"),
            hugepages: only("HUGEPAGES", "1"),
            tile_hugepages: unless("TILE_HUGEPAGES", "0"),
            pack_cache: match s("PACK_CACHE").as_deref() {
                Some("0") | Some("off") | Some("false") => PackCacheSetting::Off,
                Some(p) if !p.is_empty() && p != "1" => PackCacheSetting::Path(PathBuf::from(p)),
                _ => PackCacheSetting::Default,
            },
            pack_cache_mb: num("PACK_CACHE_MB").unwrap_or(d.pack_cache_mb),
            kv_rotate: unless("KV_ROTATE", "0"),
            kv_prec: s("KV")
                .and_then(|v| crate::kv::KvPrec::parse(&v))
                .unwrap_or(d.kv_prec),
            mix: unless("MIX", "0"),
            prefill_chunk: num("PREFILL_CHUNK").filter(|&v| (1..=MAX_PREFILL_CHUNK).contains(&v)),
            pass_rows: num("PASS_ROWS").filter(|&v| (1..=MAX_PASS_ROWS).contains(&v)),
            work_cache_mb: num("WORK_CACHE_MB").filter(|&v| v > 0),
            embed_cache_mb: num("EMBED_CACHE_MB").unwrap_or(d.embed_cache_mb),
            mtp: unless("MTP", "0"),
            mtp_vocab: num("MTP_VOCAB").unwrap_or(d.mtp_vocab),
            dspark: DsparkOptions {
                logits_start: num("DSPARK_LOGITS_START").unwrap_or(0),
                quant_q8: only("DSPARK_QUANT", "q8"),
                conf_on_normed: unless("DSPARK_CONF_PRENORM", "1"),
                width: num("DSPARK_WIDTH"),
                k: num("DSPARK_K"),
                prof: s("DSPARK_PROF").is_some(),
                dump: s("DSPARK_DUMP").map(PathBuf::from),
            },
            lb_sidecar: match s("LB").as_deref() {
                Some("0") => LbSidecar::Off,
                Some(p) if !p.is_empty() && p != "1" => LbSidecar::Path(PathBuf::from(p)),
                _ => LbSidecar::BesideModel,
            },
            lb_i8: unless("LB_I8", "0"),
            prof: only("PROF", "1"),
            trace: s("TRACE").is_some(),
            calibrate: match s("CALIBRATE").as_deref() {
                Some("0") | Some("off") | Some("false") => false,
                Some(_) => true,
                None => d.calibrate,
            },
            tuning_source: crate::calib::TuningSource::Default,
            calibration_us: 0,
        }
    }

    /// `name=value` for every option, one per line, in a fixed order: what a server prints at startup and
    /// a benchmark records.
    pub fn describe(&self) -> String {
        self.pairs()
            .iter()
            .map(|(k, v)| format!("{k}={v}\n"))
            .collect()
    }

    /// The repack setting and why, for `peregrine tier`: `on (tier default)`, or
    /// `off (PEREGRINE_KREPACK)` when the switch turned it off.
    pub fn kq_repack_note(&self) -> String {
        Self::kq_repack_note_for(
            self.kq_repack,
            self.kq_repack_switch,
            crate::cpu::resolved_under(self.isa),
        )
    }

    /// [`Self::kq_repack_note`] for a tier given directly, without reading the machine.
    pub fn kq_repack_note_for(on: bool, switch: bool, _tier: crate::cpu::Tier) -> String {
        let state = if on { "on" } else { "off" };
        let why = if switch {
            "PEREGRINE_KREPACK"
        } else {
            "tier default"
        };
        format!("{state} ({why})")
    }

    /// The float pack setting and why, for `peregrine tier`: `on`, or
    /// `off (PEREGRINE_FPACK)` when the switch turned it off.
    pub fn fpack_note(&self) -> String {
        let state = if self.fpack { "on" } else { "off" };
        let why = if self.fpack_switch {
            "PEREGRINE_FPACK"
        } else {
            "default"
        };
        format!("{state} ({why})")
    }

    /// The pack cache setting and its directory, for `peregrine tier`: the directory,
    /// `off (PEREGRINE_PACK_CACHE)`, or the default directory with `(default)`.
    pub fn pack_cache_note(&self) -> String {
        match &self.pack_cache {
            PackCacheSetting::Off => "off (PEREGRINE_PACK_CACHE)".into(),
            PackCacheSetting::Path(p) => format!("{} (PEREGRINE_PACK_CACHE)", p.display()),
            PackCacheSetting::Default => format!(
                "{} (default, cap {} MiB)",
                crate::packcache::default_dir().display(),
                self.pack_cache_mb
            ),
        }
    }

    /// The options that differ from the defaults, as `name=value` pairs.
    pub fn non_default(&self) -> Vec<(&'static str, String)> {
        let d = EngineOptions::default().pairs();
        self.pairs()
            .into_iter()
            .zip(d)
            .filter(|(a, b)| a.1 != b.1)
            .map(|(a, _)| a)
            .collect()
    }

    /// Every option as (name, value), in a fixed order.
    pub fn pairs(&self) -> Vec<(&'static str, String)> {
        let opt = |v: Option<usize>| v.map_or("default".to_string(), |v| v.to_string());
        vec![
            ("spin_us", self.spin_us.to_string()),
            ("yield_us", self.yield_us.to_string()),
            ("adaptive_spin", self.adaptive_spin.to_string()),
            ("decode_threads", opt(self.decode_threads)),
            ("affinity", self.affinity.to_string()),
            (
                "isa",
                match self.isa {
                    None => "native".into(),
                    Some(t) => t.as_str().into(),
                },
            ),
            ("zmm", self.zmm.to_string()),
            ("wide", self.wide.to_string()),
            ("tiles", self.tiles.to_string()),
            ("kq_repack", self.kq_repack.to_string()),
            ("fast_prompt", self.fast_prompt.to_string()),
            ("fpack", self.fpack.to_string()),
            ("dynamic_rows", self.dynamic_rows.to_string()),
            ("prefetch_bytes", self.prefetch_bytes.to_string()),
            ("tq_span", self.tq_span.to_string()),
            ("tb_span", self.tb_span.to_string()),
            ("exact", self.exact.to_string()),
            ("hugepages", self.hugepages.to_string()),
            ("tile_hugepages", self.tile_hugepages.to_string()),
            (
                "pack_cache",
                match &self.pack_cache {
                    PackCacheSetting::Default => "default".into(),
                    PackCacheSetting::Off => "off".into(),
                    PackCacheSetting::Path(p) => p.display().to_string(),
                },
            ),
            ("pack_cache_mb", self.pack_cache_mb.to_string()),
            ("kv_rotate", self.kv_rotate.to_string()),
            ("kv_prec", self.kv_prec.name().to_string()),
            ("mix", self.mix.to_string()),
            ("prefill_chunk", opt(self.prefill_chunk)),
            ("pass_rows", opt(self.pass_rows)),
            ("work_cache_mb", opt(self.work_cache_mb)),
            ("embed_cache_mb", self.embed_cache_mb.to_string()),
            ("mtp", self.mtp.to_string()),
            ("mtp_vocab", self.mtp_vocab.to_string()),
            ("dspark.logits_start", self.dspark.logits_start.to_string()),
            ("dspark.quant_q8", self.dspark.quant_q8.to_string()),
            (
                "dspark.conf_on_normed",
                self.dspark.conf_on_normed.to_string(),
            ),
            ("dspark.width", opt(self.dspark.width)),
            ("dspark.k", opt(self.dspark.k)),
            ("dspark.prof", self.dspark.prof.to_string()),
            (
                "dspark.dump",
                self.dspark
                    .dump
                    .as_ref()
                    .map_or("none".into(), |p| p.display().to_string()),
            ),
            (
                "lb_sidecar",
                match &self.lb_sidecar {
                    LbSidecar::BesideModel => "beside-model".into(),
                    LbSidecar::Off => "off".into(),
                    LbSidecar::Path(p) => p.display().to_string(),
                },
            ),
            ("lb_i8", self.lb_i8.to_string()),
            ("prof", self.prof.to_string()),
            ("trace", self.trace.to_string()),
            ("calibrate", self.calibrate.to_string()),
            ("tuning_source", self.tuning_source.as_str().to_string()),
            ("calibration_ms", format!("{:.1}", self.calibration_ms())),
        ]
    }
}

/// Every option's variable name without the `PEREGRINE_` prefix: `PEREGRINE_<NAME>` in the environment,
/// `--opt <NAME>=...` on the server's command line.
pub const NAMES: &[&str] = &[
    "SPIN_US",
    "YIELD_US",
    "ADAPTIVE_SPIN",
    "DECODE_THREADS",
    "AFFINITY",
    "ISA",
    "ZMM",
    "NO_WIDE",
    "TILES",
    "KREPACK",
    "FPACK",
    "Q8_FLOAT",
    "DYNAMIC",
    "PF",
    "TQ_SPAN",
    "TB_SPAN",
    "EXACT",
    "FAST_PROMPT",
    "HUGEPAGES",
    "TILE_HUGEPAGES",
    "PACK_CACHE",
    "PACK_CACHE_MB",
    "KV_ROTATE",
    "KV",
    "MIX",
    "PREFILL_CHUNK",
    "PASS_ROWS",
    "WORK_CACHE_MB",
    "EMBED_CACHE_MB",
    "MTP",
    "MTP_VOCAB",
    "DSPARK_LOGITS_START",
    "DSPARK_QUANT",
    "DSPARK_CONF_PRENORM",
    "DSPARK_WIDTH",
    "DSPARK_K",
    "DSPARK_PROF",
    "DSPARK_DUMP",
    "LB",
    "LB_I8",
    "PROF",
    "TRACE",
    "CALIBRATE",
];

impl EngineOptions {
    /// The switches the kernels read.
    pub fn kernel(&self) -> peregrine_kernels::options::KernelOptions {
        peregrine_kernels::options::KernelOptions {
            isa: self.isa,
            zmm: self.zmm,
            wide: self.wide,
            prefetch_bytes: self.prefetch_bytes,
            tq_span: self.tq_span,
            exact: self.exact,
            kq_repack: self.kq_repack,
            fpack: self.fpack,
            fast_prompt: self.fast_prompt,
        }
    }
}

static OPTIONS: OnceLock<EngineOptions> = OnceLock::new();

/// The process's options: those given to [`init`], else the environment's, read on first use.
pub fn get() -> &'static EngineOptions {
    OPTIONS.get_or_init(|| {
        let o = EngineOptions::from_env();
        // The kernels read the same values; if they already read them, they read the same environment.
        let _ = peregrine_kernels::options::init(o.kernel());
        o
    })
}

/// Set the process's options. Must run before anything reads them; an `Err` means they were already set
/// (by an earlier `init` or a first read) and are unchanged.
pub fn init(opts: EngineOptions) -> Result<(), &'static str> {
    let kernel = opts.kernel();
    OPTIONS
        .set(opts)
        .map_err(|_| "engine options were already set or read")?;
    peregrine_kernels::options::init(kernel).map_err(|_| "kernel options were already read")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn with(pairs: &[(&str, &str)]) -> EngineOptions {
        let m: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (format!("PEREGRINE_{k}"), v.to_string()))
            .collect();
        EngineOptions::from_lookup(|k| m.get(k).cloned())
    }

    #[test]
    fn an_empty_environment_gives_the_defaults() {
        assert_eq!(with(&[]), EngineOptions::default());
        assert!(with(&[]).non_default().is_empty());
    }

    #[test]
    fn the_default_chunk_follows_the_pool() {
        // A pool no wider than the physical cores keeps the tuned 16; a pool that adds SMT siblings
        // reads the prompt in 64-row pieces, which amortise the per-pass cost (S3.5 step 2).
        let Some(cores) = crate::pool::physical_cores() else {
            return; // no physical-core count on this host: the plain default
        };
        assert_eq!(prefill_chunk_default_for(1), PREFILL_CHUNK_DEFAULT);
        assert_eq!(prefill_chunk_default_for(cores), PREFILL_CHUNK_DEFAULT);
        assert_eq!(prefill_chunk_default_for(cores + 1), PREFILL_CHUNK_SMT);
    }

    #[test]
    fn the_wide_chunk_is_wider_than_the_narrow_one() {
        // The wide pass of a step with no decoder (S5.1) may not be narrower than the chunk a
        // pass beside a decoder takes, and it fits the scratch's row limit.
        for tier in [
            crate::cpu::Tier::Scalar,
            crate::cpu::Tier::Avx2,
            crate::cpu::Tier::Avx512,
        ] {
            let wide = wide_chunk_default_for(tier);
            assert_eq!(wide, WIDE_CHUNK_DEFAULT, "{tier}");
            assert!(wide >= PREFILL_CHUNK_DEFAULT, "{tier}");
            assert!(wide <= MAX_PREFILL_CHUNK, "{tier}");
        }
    }

    #[test]
    fn the_fast_modes_wide_chunk_is_the_tiers_choice() {
        // S5.4e: aarch64 widens the fast mode's wide chunk to 512 (7.7–26.6 % over 256 on both
        // models at ten threads); x86 keeps the first look's 256. With the mode off the exact
        // default is unchanged on both, and both fit the scratch.
        let fast_expected = if cfg!(target_arch = "aarch64") {
            512
        } else {
            WIDE_CHUNK_DEFAULT_FAST
        };
        for tier in [crate::cpu::Tier::Avx512, crate::cpu::Tier::NeonI8mm] {
            assert_eq!(wide_chunk_default_for_mode(tier, true), fast_expected);
            assert_eq!(wide_chunk_default_for_mode(tier, false), WIDE_CHUNK_DEFAULT);
        }
        assert!(fast_expected >= PREFILL_CHUNK_DEFAULT);
        assert!(fast_expected <= MAX_PREFILL_CHUNK);
    }

    #[test]
    fn the_fast_prompt_mode_is_off_unless_asked_for() {
        assert!(!with(&[]).fast_prompt);
        assert!(with(&[("FAST_PROMPT", "1")]).fast_prompt);
        assert!(!with(&[("FAST_PROMPT", "0")]).fast_prompt);
        assert!(!with(&[("FAST_PROMPT", "yes")]).fast_prompt);
        // It is recorded with the other options, so `/metrics`, `tier` and `bench --json` see it.
        assert!(with(&[("FAST_PROMPT", "1")])
            .pairs()
            .iter()
            .any(|&(k, ref v)| k == "fast_prompt" && v == "true"));
    }

    #[test]
    fn each_switch_keeps_its_historical_rule() {
        // On unless "0".
        assert!(!with(&[("ZMM", "0")]).zmm);
        assert!(with(&[("ZMM", "no")]).zmm);
        assert!(!with(&[("MTP", "0")]).mtp);
        assert!(!with(&[("TILE_HUGEPAGES", "0")]).tile_hugepages);
        // The k-quant repack is on unless it is turned off.
        assert!(!with(&[("KREPACK", "0")]).kq_repack);
        assert!(with(&[("KREPACK", "yes")]).kq_repack);
        assert!(with(&[("KREPACK", "1")]).kq_repack);
        // The float pack (S4.9f) is the same rule, and its note says which it was.
        assert!(with(&[]).fpack);
        assert!(!with(&[("FPACK", "0")]).fpack);
        assert!(with(&[("FPACK", "1")]).fpack);
        assert!(!with(&[]).fpack_switch);
        assert!(with(&[("FPACK", "0")]).fpack_switch);
        assert_eq!(with(&[]).fpack_note(), "on (default)");
        assert_eq!(
            with(&[("FPACK", "0")]).fpack_note(),
            "off (PEREGRINE_FPACK)"
        );
        // Off only with "1".
        assert!(with(&[("EXACT", "1")]).exact);
        assert!(!with(&[("EXACT", "yes")]).exact);
        assert!(with(&[("HUGEPAGES", "1")]).hugepages);
        // NO_WIDE is inverted.
        assert!(!with(&[("NO_WIDE", "1")]).wide);
        assert!(with(&[("NO_WIDE", "0")]).wide);
        // Spans accept only their listed values.
        assert_eq!(with(&[("TQ_SPAN", "64")]).tq_span, 64);
        assert_eq!(with(&[("TQ_SPAN", "48")]).tq_span, 16);
        assert_eq!(with(&[("TB_SPAN", "16")]).tb_span, 32);
        // The adaptive window is on unless it is turned off.
        assert!(with(&[]).adaptive_spin);
        assert!(with(&[("ADAPTIVE_SPIN", "1")]).adaptive_spin);
        assert!(!with(&[("ADAPTIVE_SPIN", "0")]).adaptive_spin);
        // Presence-only switches.
        assert!(with(&[("TRACE", "")]).trace);
        assert!(with(&[("DSPARK_PROF", "0")]).dspark.prof);
        // Inverted DSpark switch.
        assert!(!with(&[("DSPARK_CONF_PRENORM", "1")]).dspark.conf_on_normed);
        // The tier cap.
        assert_eq!(with(&[("ISA", "avx2")]).isa, Some(crate::cpu::Tier::Avx2));
        assert_eq!(
            with(&[("ISA", "avx-vnni")]).isa,
            Some(crate::cpu::Tier::AvxVnni)
        );
        assert_eq!(with(&[("ISA", "native")]).isa, None);
        // The KV cache precision; an unknown name leaves the default (f16).
        assert_eq!(with(&[("KV", "f32")]).kv_prec, crate::kv::KvPrec::F32);
        assert_eq!(with(&[("KV", "i8")]).kv_prec, crate::kv::KvPrec::I8);
        assert_eq!(with(&[("KV", "f16")]).kv_prec, crate::kv::KvPrec::F16);
        assert_eq!(with(&[("KV", "f64")]).kv_prec, crate::kv::KvPrec::F16);
        // Zero decode threads means "default".
        assert_eq!(with(&[("DECODE_THREADS", "0")]).decode_threads, None);
        assert_eq!(with(&[("DECODE_THREADS", "6")]).decode_threads, Some(6));
        // LittleBit sidecar.
        assert_eq!(with(&[("LB", "0")]).lb_sidecar, LbSidecar::Off);
        assert_eq!(with(&[("LB", "1")]).lb_sidecar, LbSidecar::BesideModel);
        assert_eq!(
            with(&[("LB", "/x.slb")]).lb_sidecar,
            LbSidecar::Path("/x.slb".into())
        );
        // Startup calibration: off unless it is asked for.
        assert!(!with(&[("CALIBRATE", "0")]).calibrate);
        assert!(with(&[("CALIBRATE", "1")]).calibrate);
        assert!(!with(&[]).calibrate);
        // The prefill chunk: from one token to the matmul scratch's row limit. Anything
        // else leaves the choice to the pool (`prefill_chunk_default_for`).
        assert_eq!(with(&[("PREFILL_CHUNK", "32")]).prefill_chunk, Some(32));
        assert_eq!(with(&[("PREFILL_CHUNK", "0")]).prefill_chunk, None);
        assert_eq!(with(&[("PREFILL_CHUNK", "1000")]).prefill_chunk, None);
        assert_eq!(with(&[("PREFILL_CHUNK", "x")]).prefill_chunk, None);
        // The non-decoding pass budget: 1 to `MAX_PASS_ROWS`.
        assert_eq!(with(&[("PASS_ROWS", "64")]).pass_rows, Some(64));
        assert_eq!(with(&[("PASS_ROWS", "0")]).pass_rows, None);
        assert_eq!(with(&[("PASS_ROWS", "513")]).pass_rows, None);
        assert_eq!(with(&[("PASS_ROWS", "x")]).pass_rows, None);
        // The work cache budget.
        assert_eq!(with(&[("WORK_CACHE_MB", "128")]).work_cache_mb, Some(128));
        assert_eq!(with(&[("WORK_CACHE_MB", "0")]).work_cache_mb, None);
        // The embedding result cache: a size in MiB, and 0 disables it.
        assert_eq!(with(&[("EMBED_CACHE_MB", "32")]).embed_cache_mb, 32);
        assert_eq!(with(&[("EMBED_CACHE_MB", "0")]).embed_cache_mb, 0);
    }

    #[test]
    fn the_pass_budget_default_follows_the_machine_and_the_model() {
        // 16 rows is the floor and the ceiling is a format's measured ridge: Q8_0 (1.06 bytes per
        // weight) sits at the ridge at 16, F16 (2) fits more, and an encoder starts at 32.
        assert!(pass_rows_default_for(1024, 1.06, false, 0, false) >= 16);
        assert!(pass_rows_default_for(1024, 2.0, false, 0, true) >= 16);
        assert!(pass_rows_default_for(4096, 4.0, false, 0, true) <= 64);
        assert!(pass_rows_default_for(1024, 1.06, true, 0, false) >= 32);
        // A small float encoder takes whole 128-row blocks; a Q8_0 one or a large float
        // model keeps the 32-row floor.
        assert_eq!(pass_rows_default_for(384, 2.0, true, 22 << 20, true), 128);
        assert_eq!(pass_rows_default_for(384, 4.0, true, 132 << 20, true), 128);
        assert!(pass_rows_default_for(384, 1.06, true, 34 << 20, false) <= 32);
        assert!(pass_rows_default_for(384, 2.0, true, 600 << 20, true) <= 32);
        if l2_bytes() >= 64 << 10 {
            assert!(
                pass_rows_default_for(1024, 4.0, false, 0, true)
                    >= pass_rows_default_for(8192, 4.0, false, 0, true),
                "a narrower model fits more activation rows in the same caches"
            );
        }
        assert!(parse_cache_size("512K") == Some(512 << 10));
        assert!(parse_cache_size("2M") == Some(2 << 20));
        assert!(parse_cache_size("1024") == Some(1024));
        assert_eq!(parse_cache_size("x"), None);
    }

    #[test]
    fn the_repack_setting_follows_the_resolved_tier() {
        let d = crate::cpu::resolved_under(None);
        // Unset: the tier's default, which is the pure resolver's answer for the resolved tier.
        assert_eq!(
            with(&[]).kq_repack,
            peregrine_kernels::options::kq_repack_default(d)
        );
        // A cap is part of the resolution: a lower tier moves the default with it.
        for isa in ["scalar", "avx2", "avx-vnni", "avx512"] {
            let cap = peregrine_kernels::cpu::parse_isa(Some(isa));
            assert_eq!(
                with(&[("ISA", isa)]).kq_repack,
                peregrine_kernels::options::kq_repack_default(crate::cpu::resolved_under(cap)),
                "PEREGRINE_ISA={isa}"
            );
        }
        // The switch overrides the default in both directions on every tier.
        for isa in ["scalar", "avx2", "avx512"] {
            assert!(with(&[("ISA", isa), ("KREPACK", "1")]).kq_repack);
            assert!(!with(&[("ISA", isa), ("KREPACK", "0")]).kq_repack);
        }
    }

    #[test]
    fn the_repack_note_names_the_switch_or_the_tier() {
        use crate::cpu::Tier;
        assert_eq!(
            EngineOptions::kq_repack_note_for(true, false, Tier::Avx512),
            "on (tier default)"
        );
        assert_eq!(
            EngineOptions::kq_repack_note_for(true, false, Tier::Avx2),
            "on (tier default)"
        );
        assert_eq!(
            EngineOptions::kq_repack_note_for(true, true, Tier::Avx512),
            "on (PEREGRINE_KREPACK)"
        );
        assert_eq!(
            EngineOptions::kq_repack_note_for(false, true, Tier::Avx2),
            "off (PEREGRINE_KREPACK)"
        );
    }

    #[test]
    fn names_lists_exactly_the_variables_the_parser_reads() {
        let read = std::cell::RefCell::new(Vec::new());
        let _ = EngineOptions::from_lookup(|k| {
            read.borrow_mut()
                .push(k.trim_start_matches("PEREGRINE_").to_string());
            None
        });
        let mut read = read.into_inner();
        read.sort();
        read.dedup();
        let mut names: Vec<String> = NAMES.iter().map(|s| s.to_string()).collect();
        names.sort();
        assert_eq!(read, names);
    }

    #[test]
    fn describe_lists_every_option_and_non_default_finds_changes() {
        let o = with(&[("SPIN_US", "50"), ("MIX", "0")]);
        assert_eq!(o.describe().lines().count(), o.pairs().len());
        let nd = o.non_default();
        assert_eq!(
            nd,
            vec![("spin_us", "50".to_string()), ("mix", "false".to_string())]
        );
    }

    #[test]
    fn env_over_profile_over_default_precedence() {
        let prof = crate::profile::Profile {
            peregrine_version: "0.1.0".into(),
            cpu: "Test CPU".into(),
            cpu_features: "x86_64-avx2".into(),
            model_size: 1024,
            model_fnv: 0x1234,
            created: 1700000000,
            options: vec![
                ("spin_us".into(), "500".into()),
                ("prefetch_bytes".into(), "2048".into()),
                ("decode_threads".into(), "4".into()),
            ],
        };

        // 1. Profile wins over default (env is empty)
        let o1 = EngineOptions::from_chained(|_| None, Some(&prof));
        assert_eq!(o1.spin_us, 500);
        assert_eq!(o1.prefetch_bytes, 2048);
        assert_eq!(o1.decode_threads, Some(4));
        assert_eq!(o1.tb_span, 32); // unmentioned defaults to 32

        // 2. Env wins over profile
        let env_map: HashMap<String, String> = [
            ("PEREGRINE_SPIN_US".to_string(), "200".to_string()),
            ("PEREGRINE_TB_SPAN".to_string(), "64".to_string()),
        ]
        .into_iter()
        .collect();

        let o2 = EngineOptions::from_chained(|k| env_map.get(k).cloned(), Some(&prof));
        assert_eq!(o2.spin_us, 200); // env wins over profile (200 vs 500)
        assert_eq!(o2.prefetch_bytes, 2048); // profile wins over default (2048 vs default)
        assert_eq!(o2.decode_threads, Some(4)); // profile wins over default
        assert_eq!(o2.tb_span, 64); // env wins over default
    }

    #[test]
    fn a_profile_cannot_set_an_output_changing_option() {
        let prof = crate::profile::Profile {
            peregrine_version: "x".into(),
            cpu: "x".into(),
            cpu_features: "x".into(),
            model_size: 0,
            model_fnv: 0,
            created: 0,
            options: vec![
                ("tb_span".into(), "64".into()),
                ("tq_span".into(), "64".into()),
                ("exact".into(), "true".into()),
                ("mtp".into(), "false".into()),
                ("spin_us".into(), "7".into()),
                ("prefill_chunk".into(), "32".into()),
            ],
        };
        let o = EngineOptions::from_chained(|_| None, Some(&prof));
        let d = EngineOptions::default();
        assert_eq!(
            (o.tb_span, o.tq_span, o.exact, o.mtp),
            (d.tb_span, d.tq_span, d.exact, d.mtp)
        );
        assert_eq!(o.spin_us, 7);
        // The prefill chunk is output-neutral, so a profile may set it.
        assert_eq!(o.prefill_chunk, Some(32));
    }
    #[test]
    fn calibration_disabled_gives_defaults() {
        let o = EngineOptions::from_chained(
            |k| {
                if k == "PEREGRINE_CALIBRATE" {
                    Some("0".into())
                } else {
                    None
                }
            },
            None,
        );
        assert!(!o.calibrate);
        assert_eq!(o.decode_threads, None);
        assert_eq!(o.spin_us, SPIN_DEFAULT_US);
        assert_eq!(o.tuning_source, crate::calib::TuningSource::Default);
    }

    #[test]
    fn env_pins_win_over_calibration() {
        let o = EngineOptions::from_chained(
            |k| match k {
                "PEREGRINE_DECODE_THREADS" => Some("3".into()),
                "PEREGRINE_SPIN_US" => Some("123".into()),
                _ => None,
            },
            None,
        );
        assert_eq!(o.decode_threads, Some(3));
        assert_eq!(o.spin_us, 123);
        assert_eq!(o.tuning_source, crate::calib::TuningSource::Env);
    }

    #[test]
    fn calibration_is_off_unless_it_is_asked_for() {
        let o = EngineOptions::from_chained(|_| None, None);
        assert!(!o.calibrate);
        assert_eq!(o.decode_threads, None);
        assert_eq!(o.spin_us, SPIN_DEFAULT_US);
        assert_eq!(o.tuning_source, crate::calib::TuningSource::Default);
    }

    /// The two documented outcomes of a startup calibration that ran.
    ///
    /// It measured, and the tuning source is `Startup`; or its fork-join probe did not come
    /// back inside the product's 2 s budget on a host too busy to run it, and it fell back to
    /// the safe settings with source `Default` (`calib::calibrate`'s give-up path). Both are
    /// correct behaviour on a loaded machine. With `calibrate` on and nothing pinned, `Default`
    /// has no other cause, so accepting it here does not weaken the `Startup` check: the
    /// fallback's exact settings are asserted instead.
    fn assert_a_calibration_that_ran(o: &EngineOptions) {
        assert!(o.calibrate);
        match o.tuning_source {
            crate::calib::TuningSource::Startup => {
                assert!(o.decode_threads.is_some());
                assert!(o.spin_us >= 50 && o.spin_us <= 2000);
            }
            crate::calib::TuningSource::Default => {
                // The give-up path: the physical cores stand, and so does the default window.
                assert_eq!(
                    o.decode_threads,
                    Some(crate::pool::physical_cores().unwrap_or(1))
                );
                assert_eq!(o.spin_us, SPIN_DEFAULT_US);
            }
            other => {
                panic!("a calibration that ran must end Startup or on the fallback, not {other:?}")
            }
        }
    }

    #[test]
    fn calibration_runs_when_asked_for_and_nothing_is_pinned() {
        let o = EngineOptions::from_chained(
            |k| (k == "PEREGRINE_CALIBRATE").then(|| "1".to_string()),
            None,
        );
        assert_a_calibration_that_ran(&o);
    }

    #[test]
    fn the_server_calibrates_when_nothing_pins_the_knobs() {
        let o = EngineOptions::from_chained_calibrating(|_| None, None);
        assert_a_calibration_that_ran(&o);
    }

    #[test]
    fn a_pin_or_calibrate_zero_keeps_the_server_off_the_probe() {
        let off = EngineOptions::from_chained_calibrating(
            |k| (k == "PEREGRINE_CALIBRATE").then(|| "0".to_string()),
            None,
        );
        assert!(!off.calibrate);
        assert_eq!(off.decode_threads, None);
        assert_eq!(off.spin_us, SPIN_DEFAULT_US);
        assert_eq!(off.tuning_source, crate::calib::TuningSource::Default);

        let pinned = EngineOptions::from_chained_calibrating(
            |k| match k {
                "PEREGRINE_DECODE_THREADS" => Some("3".into()),
                "PEREGRINE_SPIN_US" => Some("123".into()),
                _ => None,
            },
            None,
        );
        assert_eq!((pinned.decode_threads, pinned.spin_us), (Some(3), 123));
        assert_eq!(pinned.tuning_source, crate::calib::TuningSource::Env);
    }

    #[test]
    fn a_profile_wins_over_the_server_calibration_default() {
        let prof = crate::profile::Profile {
            peregrine_version: "x".into(),
            cpu: "x".into(),
            cpu_features: "x".into(),
            model_size: 0,
            model_fnv: 0,
            created: 0,
            options: vec![
                ("decode_threads".into(), "4".into()),
                ("spin_us".into(), "500".into()),
            ],
        };
        let o = EngineOptions::from_chained_calibrating(|_| None, Some(&prof));
        assert_eq!((o.decode_threads, o.spin_us), (Some(4), 500));
        assert_eq!(o.tuning_source, crate::calib::TuningSource::Profile);
    }
}
