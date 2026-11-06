//! CPU feature checks with a cap (roadmap S0.3).
//!
//! Every kernel choice asks `cpu_has!` instead of `is_x86_feature_detected!`. The answer is the real
//! detection AND the tier cap from `PEREGRINE_ISA` (`scalar`, `avx2`, `avx512`; default `native`, no cap).
//! Forcing a lower tier runs that tier's kernels on any machine that has it, so one host can test every
//! tier below its own, and two hosts can be checked against each other on a shared tier.
//!
//! Hot paths ask cached answers (`simd::have_batched`, `tern::have_zmm`, ...), never `cpu_has!` per row.

use std::sync::OnceLock;

/// Kernel tiers, lowest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// Portable code only.
    Scalar,
    /// AVX2, FMA, F16C (x86-64-v3).
    Avx2,
    /// AVX-512 and the extensions used with it (VNNI, GFNI, VBMI).
    Avx512,
}

impl Tier {
    /// String name of the kernel tier: `scalar`, `avx2`, `avx512`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Tier::Scalar => "scalar",
            Tier::Avx2 => "avx2",
            Tier::Avx512 => "avx512",
        }
    }
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The highest tier kernels may use: `PEREGRINE_ISA`, or no cap.
#[inline]
pub fn cap() -> Tier {
    static CAP: OnceLock<Tier> = OnceLock::new();
    *CAP.get_or_init(|| crate::options::get().isa.unwrap_or(Tier::Avx512))
}

/// `PEREGRINE_ISA`'s value as a tier: `scalar`, `avx2`, `avx512`; anything else (or unset) is no cap.
pub fn parse_isa(v: Option<&str>) -> Option<Tier> {
    match v {
        Some("scalar") => Some(Tier::Scalar),
        Some("avx2") => Some(Tier::Avx2),
        Some("avx512") => Some(Tier::Avx512),
        _ => None,
    }
}

/// The tier a CPU feature belongs to. Unknown features count as the top tier, so a cap never lets one
/// through by accident.
#[inline]
pub fn tier_of(feature: &str) -> Tier {
    match feature {
        "sse2" | "sse3" | "ssse3" | "sse4.1" | "sse4.2" | "popcnt" => Tier::Scalar,
        "avx" | "avx2" | "fma" | "f16c" | "bmi1" | "bmi2" | "lzcnt" | "movbe" => Tier::Avx2,
        _ => Tier::Avx512,
    }
}

/// Whether the cap lets kernels use `feature` (detection is separate).
#[inline]
pub fn allowed(feature: &str) -> bool {
    tier_of(feature) <= cap()
}

/// `is_x86_feature_detected!` under the tier cap. Use this for every kernel choice.
#[macro_export]
macro_rules! cpu_has {
    ($f:tt) => {
        std::arch::is_x86_feature_detected!($f) && $crate::cpu::allowed($f)
    };
}

/// The kernel tier in effect on this machine, as a name: `x86_64-avx2-avx512-vnni-gfni-vbmi` and so on,
/// or the architecture name elsewhere. Golden outputs are stored per tier name.
pub fn tier_name() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let mut t = String::from("x86_64");
        for (name, on) in [
            ("avx2", cpu_has!("avx2") && cpu_has!("fma")),
            ("avx512", cpu_has!("avx512bw") && cpu_has!("avx512vl")),
            ("vnni", cpu_has!("avx512vnni")),
            ("gfni", cpu_has!("gfni")),
            ("vbmi", cpu_has!("avx512vbmi")),
        ] {
            if on {
                t.push('-');
                t.push_str(name);
            }
        }
        t
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        std::env::consts::ARCH.to_string()
    }
}

/// Cached check for AVX-512 VNNI + VL support under tier cap.
#[inline]
pub fn have_vnni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| crate::cpu_has!("avx512vnni") && crate::cpu_has!("avx512vl"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_map_to_tiers() {
        assert_eq!(tier_of("sse4.2"), Tier::Scalar);
        assert_eq!(tier_of("avx2"), Tier::Avx2);
        assert_eq!(tier_of("f16c"), Tier::Avx2);
        assert_eq!(tier_of("avx512vnni"), Tier::Avx512);
        assert_eq!(tier_of("gfni"), Tier::Avx512);
        assert_eq!(tier_of("something-new"), Tier::Avx512);
        assert!(Tier::Scalar < Tier::Avx2 && Tier::Avx2 < Tier::Avx512);
    }
}
