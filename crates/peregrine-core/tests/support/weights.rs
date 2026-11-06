//! Seeded random weight generators for synthetic test models.

use peregrine_core::gguf::Format;
use peregrine_core::quant::f32_to_f16 as to_f16;

/// Deterministic SplitMix64 pseudo-random generator.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    pub fn next_u8(&mut self) -> u8 {
        (self.next_u64() & 0xff) as u8
    }

    pub fn f32_range(&mut self, min: f32, max: f32) -> f32 {
        let f = (self.next_u32() >> 8) as f32 / (1 << 24) as f32;
        min + (max - min) * f
    }
}

/// Generates valid raw byte representations of random weights for a given format.
pub fn generate_weights(fmt: Format, numel: usize, rng: &mut Rng) -> Vec<u8> {
    assert_eq!(
        numel % fmt.block(),
        0,
        "numel {numel} must be a multiple of {} block {}",
        fmt.label(),
        fmt.block()
    );
    match fmt {
        Format::F32 => {
            let mut out = Vec::with_capacity(numel * 4);
            for _ in 0..numel {
                let v = rng.f32_range(-0.04, 0.04);
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        }
        Format::F16 => {
            let mut out = Vec::with_capacity(numel * 2);
            for _ in 0..numel {
                let v = rng.f32_range(-0.04, 0.04);
                out.extend_from_slice(&to_f16(v).to_le_bytes());
            }
            out
        }
        Format::Q8_0 => {
            let num_blocks = numel / 32;
            let mut out = Vec::with_capacity(num_blocks * 34);
            for _ in 0..num_blocks {
                let d = to_f16(rng.f32_range(0.001, 0.003));
                out.extend_from_slice(&d.to_le_bytes());
                for _ in 0..32 {
                    let q = (rng.next_u32() % 101) as i8 - 50; // -50..=50
                    out.push(q as u8);
                }
            }
            out
        }
        Format::Q4_0 => {
            let num_blocks = numel / 32;
            let mut out = Vec::with_capacity(num_blocks * 18);
            for _ in 0..num_blocks {
                let d = to_f16(rng.f32_range(0.001, 0.003));
                out.extend_from_slice(&d.to_le_bytes());
                for _ in 0..16 {
                    out.push(rng.next_u8());
                }
            }
            out
        }
        Format::Q5_0 => {
            let num_blocks = numel / 32;
            let mut out = Vec::with_capacity(num_blocks * 22);
            for _ in 0..num_blocks {
                let d = to_f16(rng.f32_range(0.001, 0.003));
                out.extend_from_slice(&d.to_le_bytes());
                let qh = rng.next_u32();
                out.extend_from_slice(&qh.to_le_bytes());
                for _ in 0..16 {
                    out.push(rng.next_u8());
                }
            }
            out
        }
        Format::Q4_K => {
            let num_blocks = numel / 256;
            let mut out = Vec::with_capacity(num_blocks * 144);
            for _ in 0..num_blocks {
                let d = to_f16(rng.f32_range(0.001, 0.002));
                let dmin = to_f16(rng.f32_range(0.0005, 0.001));
                out.extend_from_slice(&d.to_le_bytes());
                out.extend_from_slice(&dmin.to_le_bytes());
                for _ in 0..12 {
                    out.push(rng.next_u8() & 0x3f);
                }
                for _ in 0..128 {
                    out.push(rng.next_u8());
                }
            }
            out
        }
        Format::Q5_K => {
            let num_blocks = numel / 256;
            let mut out = Vec::with_capacity(num_blocks * 176);
            for _ in 0..num_blocks {
                let d = to_f16(rng.f32_range(0.001, 0.002));
                let dmin = to_f16(rng.f32_range(0.0005, 0.001));
                out.extend_from_slice(&d.to_le_bytes());
                out.extend_from_slice(&dmin.to_le_bytes());
                for _ in 0..12 {
                    out.push(rng.next_u8() & 0x3f);
                }
                for _ in 0..32 {
                    out.push(rng.next_u8());
                }
                for _ in 0..128 {
                    out.push(rng.next_u8());
                }
            }
            out
        }
        Format::Q2_K => {
            let num_blocks = numel / 256;
            let mut out = Vec::with_capacity(num_blocks * 84);
            for _ in 0..num_blocks {
                for _ in 0..16 {
                    out.push(rng.next_u8());
                }
                for _ in 0..64 {
                    out.push(rng.next_u8());
                }
                let d = to_f16(rng.f32_range(0.001, 0.002));
                let dmin = to_f16(rng.f32_range(0.0005, 0.001));
                out.extend_from_slice(&d.to_le_bytes());
                out.extend_from_slice(&dmin.to_le_bytes());
            }
            out
        }
        Format::Q3_K => {
            let num_blocks = numel / 256;
            let mut out = Vec::with_capacity(num_blocks * 110);
            for _ in 0..num_blocks {
                for _ in 0..32 {
                    out.push(rng.next_u8());
                }
                for _ in 0..64 {
                    out.push(rng.next_u8());
                }
                for _ in 0..12 {
                    out.push(rng.next_u8());
                }
                let d = to_f16(rng.f32_range(0.0008, 0.0015));
                out.extend_from_slice(&d.to_le_bytes());
            }
            out
        }
        Format::Q6_K => {
            let num_blocks = numel / 256;
            let mut out = Vec::with_capacity(num_blocks * 210);
            for _ in 0..num_blocks {
                for _ in 0..128 {
                    out.push(rng.next_u8());
                }
                for _ in 0..64 {
                    out.push(rng.next_u8());
                }
                for _ in 0..16 {
                    let s = (rng.next_u32() % 21) as i8 - 10; // -10..=10
                    out.push(s as u8);
                }
                let d = to_f16(rng.f32_range(0.0008, 0.0015));
                out.extend_from_slice(&d.to_le_bytes());
            }
            out
        }
        Format::IQ4_NL => {
            let num_blocks = numel / 32;
            let mut out = Vec::with_capacity(num_blocks * 18);
            for _ in 0..num_blocks {
                let d = to_f16(rng.f32_range(0.001, 0.003));
                out.extend_from_slice(&d.to_le_bytes());
                for _ in 0..16 {
                    out.push(rng.next_u8());
                }
            }
            out
        }
        Format::IQ4_XS => {
            let num_blocks = numel / 256;
            let mut out = Vec::with_capacity(num_blocks * 136);
            for _ in 0..num_blocks {
                let d = to_f16(rng.f32_range(0.0005, 0.002));
                out.extend_from_slice(&d.to_le_bytes());
                let sh = rng.next_u32() as u16;
                out.extend_from_slice(&sh.to_le_bytes());
                for _ in 0..4 {
                    out.push(rng.next_u8());
                }
                for _ in 0..128 {
                    out.push(rng.next_u8());
                }
            }
            out
        }
        Format::PQ2_0 => {
            let num_blocks = numel / 128;
            let mut out = Vec::with_capacity(num_blocks * 34);
            for _ in 0..num_blocks {
                let d = to_f16(rng.f32_range(0.01, 0.03));
                out.extend_from_slice(&d.to_le_bytes());
                for _ in 0..32 {
                    let mut b = 0u8;
                    for s in 0..4 {
                        let trit = (rng.next_u32() % 3) as u8;
                        b |= trit << (2 * s);
                    }
                    out.push(b);
                }
            }
            out
        }
        Format::PTQ1_0 => {
            let num_blocks = numel / 128;
            let mut out = Vec::with_capacity(num_blocks * 28);
            let enc = |t: &[u8]| -> u8 {
                let mut q: u16 = 0;
                for &x in t {
                    q = q * 3 + x as u16;
                }
                for _ in t.len()..5 {
                    q *= 3;
                }
                (q * 256).div_ceil(243) as u8
            };
            for _ in 0..num_blocks {
                for _ in 0..24 {
                    let t = [
                        (rng.next_u32() % 3) as u8,
                        (rng.next_u32() % 3) as u8,
                        (rng.next_u32() % 3) as u8,
                        (rng.next_u32() % 3) as u8,
                        (rng.next_u32() % 3) as u8,
                    ];
                    out.push(enc(&t));
                }
                for _ in 0..2 {
                    let t = [
                        (rng.next_u32() % 3) as u8,
                        (rng.next_u32() % 3) as u8,
                        (rng.next_u32() % 3) as u8,
                        (rng.next_u32() % 3) as u8,
                    ];
                    out.push(enc(&t));
                }
                let d = to_f16(rng.f32_range(0.01, 0.03));
                out.extend_from_slice(&d.to_le_bytes());
            }
            out
        }
        Format::BF16 => {
            let mut out = Vec::with_capacity(numel * 2);
            for _ in 0..numel {
                let v = rng.f32_range(-0.04, 0.04);
                let bits = (v.to_bits() >> 16) as u16;
                out.extend_from_slice(&bits.to_le_bytes());
            }
            out
        }
    }
}

/// Generates valid RMSNorm weights (centered around 1.0).
pub fn generate_norm(numel: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(numel * 4);
    for _ in 0..numel {
        let v = rng.f32_range(0.95, 1.05);
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
