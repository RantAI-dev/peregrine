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

/// Every engine switch. Defaults are the tuned values; see each field's environment variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineOptions {
    // Thread pool.
    /// `PEREGRINE_SPIN_US`: how long a waiting worker spins before yielding or parking (µs).
    pub spin_us: u64,
    /// `PEREGRINE_YIELD_US`: how long it then yields before parking (µs).
    pub yield_us: u64,
    /// `PEREGRINE_DECODE_THREADS`: width of the narrower pool for one-row steps (default: physical cores).
    pub decode_threads: Option<usize>,

    // Kernels.
    /// `PEREGRINE_ZMM=0` keeps the 256-bit ternary kernels on an AVX-512 machine.
    pub zmm: bool,
    /// `PEREGRINE_NO_WIDE=1` turns off the wide batched k-quant kernels.
    pub wide: bool,
    /// `PEREGRINE_TILES=0` turns off the repacked row tiles for ternary weights.
    pub tiles: bool,
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

    // Memory.
    /// `PEREGRINE_HUGEPAGES=1`: copy the model into huge-page-backed memory at load (Linux).
    pub hugepages: bool,
    /// `PEREGRINE_TILE_HUGEPAGES=0`: keep the repacked tiles in ordinary pages.
    pub tile_hugepages: bool,
    /// `PEREGRINE_KV_ROTATE=0`: plain int8 KV cache instead of the Hadamard-rotated one.
    pub kv_rotate: bool,

    // Scheduling and speculation.
    /// `PEREGRINE_MIX=0`: no mixed prefill/decode steps.
    pub mix: bool,
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
}

/// Default spin window: 1000 µs. On the Ryzen AI 9 HX 370 under WSL2 a Bonsai 2 decode step has serial
/// stretches between its matmuls that outlast 50 µs, and every park then paid a VM wake-up: 2.9 tok/s at
/// 50 µs, 10.9 at 500 µs, 11.4 at 1000 µs (12 threads).
pub const SPIN_DEFAULT_US: u64 = 1000;
/// Default prefetch distance (bytes), the value the k-quant kernels were tuned with.
pub const PF_DEFAULT: usize = 1024;

impl Default for EngineOptions {
    fn default() -> Self {
        EngineOptions {
            spin_us: SPIN_DEFAULT_US,
            yield_us: 0,
            decode_threads: None,
            zmm: true,
            wide: true,
            tiles: true,
            dynamic_rows: true,
            prefetch_bytes: PF_DEFAULT,
            tq_span: 16,
            tb_span: 32,
            exact: false,
            hugepages: false,
            tile_hugepages: true,
            kv_rotate: true,
            mix: true,
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
        }
    }
}

impl EngineOptions {
    /// The options from the environment, with each switch's historical parsing rule.
    pub fn from_env() -> Self {
        Self::from_lookup(|k| std::env::var(k).ok())
    }

    /// The options from any key → value lookup (the environment, a file, a test's map).
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let d = EngineOptions::default();
        let s = |k: &str| get(&format!("PEREGRINE_{k}"));
        // "on unless set to `off`" / "off unless set to `on`".
        let unless = |k: &str, off: &str| s(k).is_none_or(|v| v != off);
        let only = |k: &str, on: &str| s(k).is_some_and(|v| v == on);
        let num = |k: &str| s(k).and_then(|v| v.parse::<usize>().ok());
        EngineOptions {
            spin_us: s("SPIN_US")
                .and_then(|v| v.parse().ok())
                .unwrap_or(d.spin_us),
            yield_us: s("YIELD_US")
                .and_then(|v| v.parse().ok())
                .unwrap_or(d.yield_us),
            decode_threads: num("DECODE_THREADS").filter(|&v| v > 0),
            zmm: unless("ZMM", "0"),
            wide: unless("NO_WIDE", "1"),
            tiles: unless("TILES", "0"),
            dynamic_rows: unless("DYNAMIC", "0"),
            prefetch_bytes: num("PF").unwrap_or(d.prefetch_bytes),
            tq_span: num("TQ_SPAN")
                .filter(|v| [16, 32, 64, 128].contains(v))
                .unwrap_or(d.tq_span),
            tb_span: num("TB_SPAN")
                .filter(|v| [32, 64, 128].contains(v))
                .unwrap_or(d.tb_span),
            exact: only("EXACT", "1"),
            hugepages: only("HUGEPAGES", "1"),
            tile_hugepages: unless("TILE_HUGEPAGES", "0"),
            kv_rotate: unless("KV_ROTATE", "0"),
            mix: unless("MIX", "0"),
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
            ("decode_threads", opt(self.decode_threads)),
            ("zmm", self.zmm.to_string()),
            ("wide", self.wide.to_string()),
            ("tiles", self.tiles.to_string()),
            ("dynamic_rows", self.dynamic_rows.to_string()),
            ("prefetch_bytes", self.prefetch_bytes.to_string()),
            ("tq_span", self.tq_span.to_string()),
            ("tb_span", self.tb_span.to_string()),
            ("exact", self.exact.to_string()),
            ("hugepages", self.hugepages.to_string()),
            ("tile_hugepages", self.tile_hugepages.to_string()),
            ("kv_rotate", self.kv_rotate.to_string()),
            ("mix", self.mix.to_string()),
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
        ]
    }
}

/// Every option's variable name without the `PEREGRINE_` prefix: `PEREGRINE_<NAME>` in the environment,
/// `--opt <NAME>=...` on the server's command line.
pub const NAMES: &[&str] = &[
    "SPIN_US",
    "YIELD_US",
    "DECODE_THREADS",
    "ZMM",
    "NO_WIDE",
    "TILES",
    "DYNAMIC",
    "PF",
    "TQ_SPAN",
    "TB_SPAN",
    "EXACT",
    "HUGEPAGES",
    "TILE_HUGEPAGES",
    "KV_ROTATE",
    "MIX",
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
];

static OPTIONS: OnceLock<EngineOptions> = OnceLock::new();

/// The process's options: those given to [`init`], else the environment's, read on first use.
pub fn get() -> &'static EngineOptions {
    OPTIONS.get_or_init(EngineOptions::from_env)
}

/// Set the process's options. Must run before anything reads them; an `Err` means they were already set
/// (by an earlier `init` or a first read) and are unchanged.
pub fn init(opts: EngineOptions) -> Result<(), &'static str> {
    OPTIONS
        .set(opts)
        .map_err(|_| "engine options were already set or read")
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
    fn each_switch_keeps_its_historical_rule() {
        // On unless "0".
        assert!(!with(&[("ZMM", "0")]).zmm);
        assert!(with(&[("ZMM", "no")]).zmm);
        assert!(!with(&[("MTP", "0")]).mtp);
        assert!(!with(&[("TILE_HUGEPAGES", "0")]).tile_hugepages);
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
        // Presence-only switches.
        assert!(with(&[("TRACE", "")]).trace);
        assert!(with(&[("DSPARK_PROF", "0")]).dspark.prof);
        // Inverted DSpark switch.
        assert!(!with(&[("DSPARK_CONF_PRENORM", "1")]).dspark.conf_on_normed);
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
}
