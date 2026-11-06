//! The kernel switches: the part of the engine options that kernel code reads.
//!
//! `peregrine-core`'s `EngineOptions` holds these too and sets them here once, at startup (`init`); a
//! program that uses the kernels alone gets them from the environment on first use.

use std::sync::OnceLock;

/// Kernel switches. Defaults are the tuned values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelOptions {
    /// `PEREGRINE_ISA`: the highest kernel tier (None = native).
    pub isa: Option<crate::cpu::Tier>,
    /// `PEREGRINE_ZMM=0` keeps the 256-bit ternary kernels on an AVX-512 machine.
    pub zmm: bool,
    /// `PEREGRINE_NO_WIDE=1` turns off the wide batched k-quant kernels.
    pub wide: bool,
    /// `PEREGRINE_PF`: software prefetch distance in bytes (0 = off).
    pub prefetch_bytes: usize,
    /// `PEREGRINE_TQ_SPAN` (16, 32, 64, 128): values sharing one activation scale in the ternary layout.
    pub tq_span: usize,
    /// `PEREGRINE_EXACT=1`: ternary matmuls in f32 against decoded weights (slow reference path).
    pub exact: bool,
}

/// Default prefetch distance (bytes), the value the k-quant kernels were tuned with.
pub const PF_DEFAULT: usize = 1024;

impl Default for KernelOptions {
    fn default() -> Self {
        KernelOptions {
            isa: None,
            zmm: true,
            wide: true,
            prefetch_bytes: PF_DEFAULT,
            tq_span: 16,
            exact: false,
        }
    }
}

static OPTIONS: OnceLock<KernelOptions> = OnceLock::new();

/// The kernel switches: those given to [`init`], else read from the environment on first use.
pub fn get() -> &'static KernelOptions {
    OPTIONS.get_or_init(|| {
        let s = |k: &str| std::env::var(format!("PEREGRINE_{k}")).ok();
        let d = KernelOptions::default();
        let num = |k: &str| s(k).and_then(|v| v.parse::<usize>().ok());
        KernelOptions {
            isa: crate::cpu::parse_isa(s("ISA").as_deref()),
            zmm: s("ZMM").is_none_or(|v| v != "0"),
            wide: s("NO_WIDE").is_none_or(|v| v != "1"),
            prefetch_bytes: num("PF").unwrap_or(d.prefetch_bytes),
            tq_span: num("TQ_SPAN")
                .filter(|v| [16, 32, 64, 128].contains(v))
                .unwrap_or(d.tq_span),
            exact: s("EXACT").is_some_and(|v| v == "1"),
        }
    })
}

/// Set the kernel switches. Must run before any kernel reads them; `Err` if they were already set.
pub fn init(opts: KernelOptions) -> Result<(), &'static str> {
    OPTIONS
        .set(opts)
        .map_err(|_| "kernel options were already set or read")
}
