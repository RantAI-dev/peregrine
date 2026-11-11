//! CPU feature checks with a cap (roadmap S0.3, S2.8).
//!
//! Every kernel choice asks `cpu_has!` instead of `is_x86_feature_detected!`. The answer is the real
//! detection AND the tier cap from `PEREGRINE_ISA` (`scalar`, `avx2`, `avx-vnni`, `avx512`, and on
//! aarch64 `neon`, `neon-dotprod`; default `native`, no cap). Forcing a lower tier runs that tier's
//! kernels on any machine that has it, so one host can test every tier below its own, and two hosts
//! can be checked against each other on a shared tier.
//!
//! The ladder is one ordering across both architectures: the x86 rungs come first, the aarch64 rungs
//! after them, and a host only ever detects its own. [`Tier::top`] is therefore the no-cap sentinel,
//! not `Avx512`.
//!
//! Hot paths ask cached answers (`simd::have_batched`, `neon::have_dotprod`, `tern::have_zmm`, ...),
//! never `cpu_has!` per row.

use std::sync::OnceLock;

/// Kernel tiers, lowest first.
///
/// The aarch64 rungs sit above the x86 ones because the ordering is only ever compared within one
/// host's detection: a cap is a ceiling, and a ceiling named for another architecture is simply no
/// ceiling here (the detected tier is always the lower of the two).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// Portable code only.
    Scalar,
    /// AVX2, FMA, F16C (x86-64-v3).
    Avx2,
    /// AVX-VNNI (VEX, 256-bit) without AVX-512.
    AvxVnni,
    /// AVX-512 and the extensions used with it (VNNI, GFNI, VBMI).
    Avx512,
    /// aarch64 Advanced SIMD (`asimd`), without the dot-product extension.
    Neon,
    /// aarch64 `asimd` + `asimddp` (`sdot`): the shipping ARM tier.
    NeonDotprod,
    /// aarch64 `asimd` + `asimddp` + `i8mm` (`smmla`): the repacked tile GEMM.
    NeonI8mm,
}

impl Tier {
    /// The highest rung: the value of "no `PEREGRINE_ISA` cap".
    pub const fn top() -> Tier {
        Tier::NeonI8mm
    }

    /// String name of the kernel tier: `scalar`, `avx2`, `avx512`, `neon`, `neon-dotprod`,
    /// `neon-i8mm`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Tier::Scalar => "scalar",
            Tier::Avx2 => "avx2",
            Tier::AvxVnni => "avx-vnni",
            Tier::Avx512 => "avx512",
            Tier::Neon => "neon",
            Tier::NeonDotprod => "neon-dotprod",
            Tier::NeonI8mm => "neon-i8mm",
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
    *CAP.get_or_init(|| crate::options::get().isa.unwrap_or(Tier::top()))
}

/// The highest tier this CPU reaches on its own, before `PEREGRINE_ISA` caps it.
///
/// Detection only: unlike [`cap`], this reads no options, so the options parser can call it
/// while it is resolving them.
pub fn detected() -> Tier {
    static DETECTED: OnceLock<Tier> = OnceLock::new();
    *DETECTED.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            let avx2 = std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma");
            if std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vl")
            {
                Tier::Avx512
            } else if avx2 && std::arch::is_x86_feature_detected!("avxvnni") {
                Tier::AvxVnni
            } else if avx2 {
                Tier::Avx2
            } else {
                Tier::Scalar
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("i8mm") {
                    Tier::NeonI8mm
                } else if std::arch::is_aarch64_feature_detected!("dotprod") {
                    Tier::NeonDotprod
                } else {
                    Tier::Neon
                }
            } else {
                Tier::Scalar
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Tier::Scalar
        }
    })
}

/// The tier the kernels run at under a `PEREGRINE_ISA` cap (`None` = native): the lower of what the
/// CPU detects and the cap. This is the tier a tier-dependent default is resolved against.
pub fn resolved_under(isa: Option<Tier>) -> Tier {
    detected().min(isa.unwrap_or(Tier::top()))
}

/// The tier the kernels run at: [`detected`] capped by `PEREGRINE_ISA`.
pub fn resolved() -> Tier {
    resolved_under(crate::options::get().isa)
}

/// `PEREGRINE_ISA`'s value as a tier: `scalar`, `avx2`, `avx-vnni`, `avx512`, `neon`,
/// `neon-dotprod`; anything else (or unset) is no cap.
pub fn parse_isa(v: Option<&str>) -> Option<Tier> {
    match v {
        Some("scalar") => Some(Tier::Scalar),
        Some("avx2") => Some(Tier::Avx2),
        Some("avx-vnni") => Some(Tier::AvxVnni),
        Some("avx512") => Some(Tier::Avx512),
        Some("neon") => Some(Tier::Neon),
        Some("neon-dotprod") => Some(Tier::NeonDotprod),
        Some("neon-i8mm") => Some(Tier::NeonI8mm),
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
        "avxvnni" => Tier::AvxVnni,
        "avx512vnni" | "avx512bw" | "avx512vl" | "avx512vbmi" | "avx512f" | "avx512dq"
        | "avx512cd" | "avx512bitalg" | "avx512vpopcntdq" | "gfni" => Tier::Avx512,
        "neon" | "asimd" => Tier::Neon,
        "fp16" | "asimdhp" => Tier::NeonDotprod,
        "dotprod" | "asimddp" => Tier::NeonDotprod,
        "i8mm" => Tier::NeonI8mm,
        _ => Tier::top(),
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
/// or on aarch64 `aarch64-neon-dotprod` / `aarch64-neon` / `aarch64`. Golden outputs are stored per
/// tier name.
pub fn tier_name() -> String {
    tier_name_under(crate::options::get().isa)
}

/// [`tier_name`] under a given `PEREGRINE_ISA` cap (`None` = native), without reading the process
/// options. A tuning profile is checked before the command initialises the options, and that check
/// must not latch them; the cap can only come from the environment at that point.
pub fn tier_name_under(isa: Option<Tier>) -> String {
    let cap = isa.unwrap_or(Tier::top());
    #[cfg(target_arch = "x86_64")]
    {
        macro_rules! has_under {
            ($f:tt) => {
                std::arch::is_x86_feature_detected!($f) && tier_of($f) <= cap
            };
        }
        let has_avx2 = has_under!("avx2") && has_under!("fma");
        let has_avx512 = has_under!("avx512bw") && has_under!("avx512vl");
        if has_avx512 {
            let mut t = String::from("x86_64");
            for (name, on) in [
                ("avx2", has_avx2),
                ("avx512", has_avx512),
                ("vnni", has_under!("avx512vnni")),
                ("gfni", has_under!("gfni")),
                ("vbmi", has_under!("avx512vbmi")),
            ] {
                if on {
                    t.push('-');
                    t.push_str(name);
                }
            }
            t
        } else if has_avx2 && has_under!("avxvnni") {
            String::from("x86_64-avx2-vnni256")
        } else if has_avx2 {
            String::from("x86_64-avx2")
        } else {
            String::from("x86_64")
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        macro_rules! has_under {
            ($f:tt) => {
                std::arch::is_aarch64_feature_detected!($f) && tier_of($f) <= cap
            };
        }
        if !has_under!("neon") {
            return String::from("aarch64");
        }
        if has_under!("i8mm") {
            String::from("aarch64-neon-i8mm")
        } else if has_under!("dotprod") {
            String::from("aarch64-neon-dotprod")
        } else {
            String::from("aarch64-neon")
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = cap;
        std::env::consts::ARCH.to_string()
    }
}

/// Cached check for Advanced SIMD under the tier cap (`PEREGRINE_ISA=scalar` turns it off).
#[inline]
pub fn have_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::arch::is_aarch64_feature_detected!("neon") && allowed("neon"))
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Cached check for the aarch64 half-precision widening/arithmetic extension (`fp16`,
/// `asimdhp`) under the tier cap: the packed float kernel uses `fcvtl` to widen f16
/// weights exactly, and falls back to the scalar chain where it is absent.
#[inline]
pub fn have_fp16() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            have_neon() && std::arch::is_aarch64_feature_detected!("fp16") && allowed("fp16")
        })
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Cached check for the aarch64 dot-product extension (`asimddp`, `sdot`) under the tier cap.
#[inline]
pub fn have_dotprod() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            have_neon() && std::arch::is_aarch64_feature_detected!("dotprod") && allowed("dotprod")
        })
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Cached check for the aarch64 int8 matrix-multiply extension (`i8mm`, `smmla`) under the tier
/// cap. It implies [`have_dotprod`]: the `i8mm` rung sits above `neon-dotprod`.
#[inline]
pub fn have_i8mm() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            have_dotprod() && std::arch::is_aarch64_feature_detected!("i8mm") && allowed("i8mm")
        })
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Cached check for the FMA instruction under tier cap.
///
/// `fma` belongs to the AVX2 tier, so `PEREGRINE_ISA=scalar` turns this off and the
/// portable code (a call to the C library's `fmaf`) runs instead.
#[inline]
pub fn have_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| crate::cpu_has!("fma"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
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

/// Cached check for AVX-VNNI (256-bit VEX encoding) support under tier cap.
#[inline]
pub fn have_avx_vnni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            crate::cpu_has!("avxvnni") && crate::cpu_has!("avx2") && crate::cpu_has!("fma")
        })
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
        assert_eq!(tier_of("avxvnni"), Tier::AvxVnni);
        assert_eq!(tier_of("avx512vnni"), Tier::Avx512);
        assert_eq!(tier_of("gfni"), Tier::Avx512);
        assert_eq!(tier_of("neon"), Tier::Neon);
        assert_eq!(tier_of("asimddp"), Tier::NeonDotprod);
        assert_eq!(tier_of("i8mm"), Tier::NeonI8mm);
        assert_eq!(tier_of("something-new"), Tier::top());
        assert!(
            Tier::Scalar < Tier::Avx2 && Tier::Avx2 < Tier::AvxVnni && Tier::AvxVnni < Tier::Avx512
        );
        assert!(
            Tier::Avx512 < Tier::Neon
                && Tier::Neon < Tier::NeonDotprod
                && Tier::NeonDotprod < Tier::NeonI8mm
        );
        assert_eq!(Tier::top(), Tier::NeonI8mm);
    }

    /// A cap never lets a feature above it through, on either architecture's rungs.
    #[test]
    fn the_cap_gates_both_ladders() {
        assert_eq!(tier_of("avx512f"), Tier::Avx512);
        assert_eq!(tier_of("avx512bw"), Tier::Avx512);
        assert_eq!(tier_of("dotprod"), Tier::NeonDotprod);
    }

    #[test]
    fn the_resolved_tier_is_the_lower_of_detection_and_the_cap() {
        let d = detected();
        assert_eq!(resolved_under(None), d);
        assert_eq!(resolved_under(Some(d)), d);
        assert_eq!(resolved_under(Some(Tier::Scalar)), Tier::Scalar);
        assert_eq!(resolved_under(Some(Tier::top())), d);
        // A cap below the detected tier lowers it; the rungs of the other architecture are either
        // above this host's detection (no cap) or below it (a cap).
        assert_eq!(resolved_under(Some(Tier::Avx2)), d.min(Tier::Avx2));
        #[cfg(target_arch = "x86_64")]
        assert_eq!(resolved_under(Some(Tier::Avx512)), d);
        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(resolved_under(Some(Tier::Avx512)), Tier::Avx512);
            assert_eq!(resolved_under(Some(Tier::Neon)), d.min(Tier::Neon));
        }
    }

    #[test]
    fn parse_isa_values() {
        assert_eq!(parse_isa(Some("scalar")), Some(Tier::Scalar));
        assert_eq!(parse_isa(Some("avx2")), Some(Tier::Avx2));
        assert_eq!(parse_isa(Some("avx-vnni")), Some(Tier::AvxVnni));
        assert_eq!(parse_isa(Some("avx512")), Some(Tier::Avx512));
        assert_eq!(parse_isa(Some("neon")), Some(Tier::Neon));
        assert_eq!(parse_isa(Some("neon-dotprod")), Some(Tier::NeonDotprod));
        assert_eq!(parse_isa(Some("neon-i8mm")), Some(Tier::NeonI8mm));
        assert_eq!(parse_isa(Some("native")), None);
        assert_eq!(parse_isa(None), None);
    }
}
