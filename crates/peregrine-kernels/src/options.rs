//! The kernel switches: the part of the engine options that kernel code reads.
//!
//! `peregrine-core`'s `EngineOptions` holds these too and sets them here once, at startup (`init`); a
//! program that uses the kernels alone gets them from the environment on first use.

use crate::cpu::Tier;
use std::sync::OnceLock;

/// Kernel switches. Defaults are the tuned values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelOptions {
    /// `PEREGRINE_ISA`: the highest kernel tier (None = native).
    pub isa: Option<Tier>,
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
    /// `PEREGRINE_KREPACK`: repack the k-quant matrices into eight-row tiles and multiply them with
    /// the repacked kernels. On by default on every tier, the AVX-512 tier included: it runs the
    /// AVX2 repacked kernel there, and the A/B measured a large prompt win for a decode cost the
    /// decision accepts (see `kq_repack_default`). The switch is the A/B of the repack.
    pub kq_repack: bool,
    /// `PEREGRINE_FPACK=0`: keep the F32, F16 and BF16 matrices in the file's own rows and
    /// multiply them with the pre-S4.9f float tile. On by default on x86-64, where the packed
    /// kernels run; it is the A/B switch of the float pack.
    pub fpack: bool,
}

/// Default prefetch distance (bytes), the value the k-quant kernels were tuned with.
pub const PF_DEFAULT: usize = 1024;

/// The k-quant repack's default for a resolved kernel tier: on.
///
/// The repacked Q4_K kernel is the AVX2 one, and on the AVX-512 tier the kernel table picks the
/// same AVX2 code. That tier used to keep the repack off because the first A/B measured decode
/// −3.5 % (prefill +24.9 %). The 2026-11-08 re-measurement on the same host (three rounds,
/// interleaved, Llama-3.2-1B Q4_K_M) found prefill +26 % at 512 tokens and decode −1.2 %, inside
/// the desktop's ~1 % null-A/B band and inside the run spread; the user accepted that trade and
/// turned the repack on for the tier. `PEREGRINE_KREPACK=0` remains the control, and the
/// 16-column AVX-512 kernel (S3.4 follow-up) is measured against this default.
pub const fn kq_repack_default(_tier: Tier) -> bool {
    true
}

/// `PEREGRINE_KREPACK` resolved against a tier: `0` is off, any other value is on (the switch
/// overrides the default in both directions), and unset is [`kq_repack_default`].
pub fn kq_repack_from(value: Option<&str>, tier: Tier) -> bool {
    value.map_or(kq_repack_default(tier), |v| v != "0")
}

impl Default for KernelOptions {
    fn default() -> Self {
        KernelOptions {
            isa: None,
            zmm: true,
            wide: true,
            prefetch_bytes: PF_DEFAULT,
            tq_span: 16,
            exact: false,
            // `resolved_under`, not `resolved`: the latter reads these options, and this is the
            // default they are read into.
            kq_repack: kq_repack_default(crate::cpu::resolved_under(None)),
            fpack: true,
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
        let isa = crate::cpu::parse_isa(s("ISA").as_deref());
        KernelOptions {
            isa,
            zmm: s("ZMM").is_none_or(|v| v != "0"),
            wide: s("NO_WIDE").is_none_or(|v| v != "1"),
            prefetch_bytes: num("PF").unwrap_or(d.prefetch_bytes),
            tq_span: num("TQ_SPAN")
                .filter(|v| [16, 32, 64, 128].contains(v))
                .unwrap_or(d.tq_span),
            exact: s("EXACT").is_some_and(|v| v == "1"),
            kq_repack: kq_repack_from(s("KREPACK").as_deref(), crate::cpu::resolved_under(isa)),
            fpack: s("FPACK").is_none_or(|v| v != "0"),
        }
    })
}

/// Set the kernel switches. Must run before any kernel reads them; `Err` if they were already set.
pub fn init(opts: KernelOptions) -> Result<(), &'static str> {
    OPTIONS
        .set(opts)
        .map_err(|_| "kernel options were already set or read")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The repack is on by default on every tier — the AVX-512 tier included — and both switch
    /// values override the default in either direction.
    #[test]
    fn kq_repack_default_is_on_everywhere() {
        assert!(kq_repack_default(Tier::Avx512));
        assert!(kq_repack_default(Tier::AvxVnni));
        assert!(kq_repack_default(Tier::Avx2));
        assert!(kq_repack_default(Tier::Scalar));

        for tier in [Tier::Scalar, Tier::Avx2, Tier::AvxVnni, Tier::Avx512] {
            let default = kq_repack_default(tier);
            assert_eq!(kq_repack_from(None, tier), default);
            assert!(!kq_repack_from(Some("0"), tier));
            assert!(kq_repack_from(Some("1"), tier));
            // The historical rule for the other values is unchanged: anything but "0" is on.
            assert!(kq_repack_from(Some("yes"), tier));
        }
    }
}
