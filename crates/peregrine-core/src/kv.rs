//! The key–value cache, and the `κ(T)` term of the performance model.
//!
//! This is the one part of the weight traffic that grows with context. At short
//! prompts it is negligible against the 267 MB of weights this model reads per
//! token; by a few thousand positions it is a visible fraction, which is exactly
//! why the cache precision is a lever worth having and why `bytes_read_per_token`
//! is reported rather than assumed.
//!
//! # Why int8 and not int4
//!
//! The evidence on cache quantisation is asymmetric. int8 keys and values are close
//! to lossless across model families; int4 degrades measurably, and the degradation
//! concentrates in long contexts — which is precisely the regime that motivates
//! quantising the cache in the first place. Offering int4 would be offering a
//! setting that is worst exactly where it is advertised as best, so this cache
//! stops at int8 and the decision is recorded rather than left to a future reader.
//!
//! # Layout
//!
//! `[layer][kv_head][position][head_dim]`, so the positions of one head are
//! contiguous. Attention (`attn.rs`) decodes a block of positions of one head at a
//! time and reuses it for every query head sharing that KV head, and that is the
//! only access pattern the cache has.
//!
//! # Rotation (int8)
//!
//! An int8 vector shares one absmax scale, so a single outlier channel — common in
//! keys — stretches the scale and costs every other channel its resolution. Following
//! uzu (`prepare_int8.rs`), int8 vectors are rotated by a fixed sign flip and an
//! orthonormal Walsh–Hadamard transform before quantising, which spreads an outlier
//! across the whole vector. Attention runs in the rotated space; see `rotated`.
//!
//! # The f32 cache (opt-in)
//!
//! The default cache is f16, and that is a *choice*: the keys and values are rounded
//! to half precision before attention sees them. A comparison against another engine
//! is only like-for-like if both engines keep their operands in the same precision, so
//! the f32 cache exists to make that comparison honest — llama.cpp's cache is f16 by
//! default, and its attention reads f16 operands, while Peregrine's queries are f32.
//! With `KvPrec::F32` both engines' attention runs on f32 keys and values, and the
//! difference that remains is the difference of the *engines*, not of their caches.
//!
//! It is opt-in because it costs twice the memory of the f16 cache and buys nothing
//! for a deployment: f16 keys are within a fraction of a percent of f32 ones. It is a
//! measurement tool, and `docs/dev/accuracy.md` says how to use it.

use crate::gguf::{f16, to_f16};

/// Widest head the int8 rotation handles (on the stack, per `put`).
const MAX_ROTATED_DIM: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPrec {
    F16,
    /// Full float precision: the keys and values attention reads are the ones the
    /// model produced, with no rounding. Opt-in; see the module docs.
    F32,
    I8,
}

impl KvPrec {
    pub fn label(self) -> &'static str {
        match self {
            KvPrec::F16 => "f16",
            KvPrec::F32 => "f32",
            KvPrec::I8 => "int8",
        }
    }
    /// The name a command line or an environment variable uses: `f16`, `f32`, `i8`.
    pub fn name(self) -> &'static str {
        match self {
            KvPrec::F16 => "f16",
            KvPrec::F32 => "f32",
            KvPrec::I8 => "i8",
        }
    }
    /// Parse the name a command line or an environment variable carries. `int8` is
    /// accepted for the int8 cache, whose label says `int8`.
    pub fn parse(s: &str) -> Option<KvPrec> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f16" | "fp16" => Some(KvPrec::F16),
            "f32" | "fp32" => Some(KvPrec::F32),
            "i8" | "int8" => Some(KvPrec::I8),
            _ => None,
        }
    }
    /// Bytes per stored element, excluding the per-vector scale that I8 adds.
    const fn elem(self) -> usize {
        match self {
            KvPrec::F16 => 2,
            KvPrec::F32 => 4,
            KvPrec::I8 => 1,
        }
    }
}

/// Half-precision encode with F16C, round-to-nearest-even — the same rounding as
/// `to_f16`, which the cache test pins.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn encode_f16_avx(x: &[f32], out: &mut [u8]) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= x.len() {
        let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(x.as_ptr().add(i)));
        _mm_storeu_si128(out.as_mut_ptr().add(i * 2) as *mut __m128i, h);
        i += 8;
    }
}

/// Full-precision encode with AVX2: the floats are stored as they are, so the only
/// rounding the f32 cache ever applies is the one the model itself did.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn encode_f32_avx(x: &[f32], out: &mut [u8]) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= x.len() {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i * 4) as *mut f32, v);
        i += 8;
    }
}

/// Whether the vector attention path may run. F16C is what widens the stored
/// half-precision keys; FMA and AVX2 are the arithmetic.
#[cfg(target_arch = "x86_64")]
#[inline]
fn vector_path() -> bool {
    crate::cpu_has!("avx2") && crate::cpu_has!("fma") && crate::cpu_has!("f16c")
}

// the first `len` positions of a cache, packed (no gaps between the per-head regions).
pub struct KvSnapshot {
    prec: KvPrec,
    rotated: bool,
    n_layer: usize,
    n_kv_head: usize,
    head_dim: usize,
    len: usize,
    k: Vec<u8>,
    v: Vec<u8>,
    ks: Vec<f32>,
    vs: Vec<f32>,
}

impl KvSnapshot {
    /// Positions held.
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Bytes this snapshot occupies in memory.
    pub fn bytes(&self) -> usize {
        self.k.len() + self.v.len() + (self.ks.len() + self.vs.len()) * 4
    }
}

pub struct KvCache {
    prec: KvPrec,
    n_layer: usize,
    n_kv_head: usize,
    head_dim: usize,
    max_ctx: usize,
    /// Encoded keys and values, one flat buffer each.
    k: Vec<u8>,
    v: Vec<u8>,
    /// One scale per (layer, head, position) vector, for `I8` only. Per-vector
    /// rather than per-tensor: coarse scaling is where the reported int8 accuracy
    /// loss comes from, and a scale per 64 values costs 4 bytes against 64.
    ks: Vec<f32>,
    vs: Vec<f32>,
    /// I8 vectors are stored Hadamard-rotated (see `rotated`).
    rotated: bool,
    /// Positions currently filled.
    pub len: usize,
    // the recurrent state of a hybrid model's linear-attention layers (Bonsai 2's Gated DeltaNet), which
    // a position index cannot roll back: it is the state after exactly `len` positions.
    rec: Option<Recurrent>,
}

/// Per linear-attention layer: the delta-rule state and the causal conv's last inputs, plus at most one copy
/// taken at a known position, so a cancelled or rolled-back turn returns to it exactly.
struct Recurrent {
    n: usize,
    state_floats: usize,
    conv_floats: usize,
    state: Vec<f32>,
    conv: Vec<f32>,
    ckpt: Option<(usize, Vec<f32>, Vec<f32>)>,
    /// A snapshot taken inside one forward call: the recurrent state after position `snap_pos` (a speculative
    /// verification keeps it so a rejected draft rolls back exactly). Requested before the call (`snap_pos`),
    /// written by the delta-rule and conv passes, used by `truncate(snap_pos + 1)`.
    snap_pos: Option<usize>,
    snap_state: Vec<f32>,
    snap_conv: Vec<f32>,
    snap_valid: bool,
}

impl KvCache {
    // ask the next forward call to keep the recurrent state after position `pos` (see `Recurrent`).
    pub fn request_snapshot(&mut self, pos: usize) {
        if let Some(r) = self.rec.as_mut() {
            if r.snap_state.len() != r.state.len() {
                r.snap_state = vec![0.0; r.state.len()];
                r.snap_conv = vec![0.0; r.conv.len()];
            }
            r.snap_pos = Some(pos);
            r.snap_valid = false;
        }
    }

    /// The snapshot request and buffers for recurrent layer `i`: (position, state, conv history).
    pub fn snapshot_ptrs(&mut self, i: usize) -> Option<(usize, *mut f32, *mut f32)> {
        let r = self.rec.as_mut()?;
        let p = r.snap_pos?;
        r.snap_valid = true;
        let (s, c) = (r.state_floats, r.conv_floats);
        Some((
            p,
            r.snap_state[i * s..].as_mut_ptr(),
            r.snap_conv[i * c..].as_mut_ptr(),
        ))
    }

    /// Forget the snapshot (the speculation was accepted, or is over).
    pub fn drop_snapshot(&mut self) {
        if let Some(r) = self.rec.as_mut() {
            r.snap_pos = None;
            r.snap_valid = false;
        }
    }
    // give the cache `n` recurrent layers of `state_floats` + `conv_floats` floats, zeroed.
    pub fn with_recurrent(mut self, n: usize, state_floats: usize, conv_floats: usize) -> KvCache {
        if n > 0 {
            self.rec = Some(Recurrent {
                n,
                state_floats,
                conv_floats,
                state: vec![0.0; n * state_floats],
                conv: vec![0.0; n * conv_floats],
                ckpt: None,
                snap_pos: None,
                snap_state: Vec::new(),
                snap_conv: Vec::new(),
                snap_valid: false,
            });
        }
        self
    }

    pub fn has_recurrent(&self) -> bool {
        self.rec.is_some()
    }

    /// The delta-rule state and conv inputs of recurrent layer `i`.
    pub fn recurrent(&mut self, i: usize) -> (&mut [f32], &mut [f32]) {
        let r = self.rec.as_mut().expect("recurrent state");
        debug_assert!(i < r.n);
        let (s, c) = (r.state_floats, r.conv_floats);
        (
            &mut r.state[i * s..(i + 1) * s],
            &mut r.conv[i * c..(i + 1) * c],
        )
    }

    /// Raw pointers to recurrent layer `i`'s state and conv buffers, for workers that split it by head.
    pub fn recurrent_ptrs(&mut self, i: usize) -> (*mut f32, *mut f32) {
        let (s, c) = self.recurrent(i);
        (s.as_mut_ptr(), c.as_mut_ptr())
    }

    /// Remember the recurrent state at the current length (a turn's starting point). Costs one copy of it.
    pub fn checkpoint(&mut self) {
        let len = self.len;
        if let Some(r) = self.rec.as_mut() {
            match &mut r.ckpt {
                Some((p, s, c)) => {
                    *p = len;
                    s.copy_from_slice(&r.state);
                    c.copy_from_slice(&r.conv);
                }
                None => r.ckpt = Some((len, r.state.clone(), r.conv.clone())),
            }
        }
    }
    pub fn new(
        prec: KvPrec,
        n_layer: usize,
        n_kv_head: usize,
        head_dim: usize,
        max_ctx: usize,
    ) -> KvCache {
        let vectors = n_layer * n_kv_head * max_ctx;
        let elems = vectors * head_dim;
        let scales = if prec == KvPrec::I8 { vectors } else { 0 };
        KvCache {
            prec,
            n_layer,
            n_kv_head,
            head_dim,
            max_ctx,
            k: vec![0u8; elems * prec.elem()],
            v: vec![0u8; elems * prec.elem()],
            ks: vec![0.0; scales],
            vs: vec![0.0; scales],
            // `PEREGRINE_KV_ROTATE=0` is the measurement switch `examples/kvacc` uses
            // to price the rotation against the plain int8 cache.
            rotated: prec == KvPrec::I8
                && head_dim.is_power_of_two()
                && head_dim <= MAX_ROTATED_DIM
                && crate::options::get().kv_rotate,
            len: 0,
            rec: None,
        }
    }

    // positions currently filled.
    pub fn filled(&self) -> usize {
        self.len
    }

    // a copy of the first `len` positions, to be put back later (the session's prefix store).
    pub fn snapshot(&self, len: usize) -> KvSnapshot {
        let len = len.min(self.len);
        let vectors = self.n_layer * self.n_kv_head;
        let elem = self.head_dim * self.prec.elem();
        let mut snap = KvSnapshot {
            prec: self.prec,
            rotated: self.rotated,
            n_layer: self.n_layer,
            n_kv_head: self.n_kv_head,
            head_dim: self.head_dim,
            len,
            k: Vec::with_capacity(vectors * len * elem),
            v: Vec::with_capacity(vectors * len * elem),
            ks: Vec::new(),
            vs: Vec::new(),
        };
        for vec in 0..vectors {
            let at = vec * self.max_ctx * elem;
            snap.k.extend_from_slice(&self.k[at..at + len * elem]);
            snap.v.extend_from_slice(&self.v[at..at + len * elem]);
            if self.prec == KvPrec::I8 {
                let sc = vec * self.max_ctx;
                snap.ks.extend_from_slice(&self.ks[sc..sc + len]);
                snap.vs.extend_from_slice(&self.vs[sc..sc + len]);
            }
        }
        snap
    }

    // put the first `len` positions of `snap` back and make them the filled length. False, and no
    // change, when the snapshot comes from a cache of another shape or holds fewer than `len` positions.
    pub fn restore(&mut self, snap: &KvSnapshot, len: usize) -> bool {
        // a snapshot holds no recurrent state, so it cannot restore a hybrid model's cache.
        if self.rec.is_some() {
            return false;
        }
        if snap.prec != self.prec
            || snap.rotated != self.rotated
            || snap.n_layer != self.n_layer
            || snap.n_kv_head != self.n_kv_head
            || snap.head_dim != self.head_dim
            || len > snap.len
            || len > self.max_ctx
        {
            return false;
        }
        let vectors = self.n_layer * self.n_kv_head;
        let elem = self.head_dim * self.prec.elem();
        for vec in 0..vectors {
            let dst = vec * self.max_ctx * elem;
            let src = vec * snap.len * elem;
            self.k[dst..dst + len * elem].copy_from_slice(&snap.k[src..src + len * elem]);
            self.v[dst..dst + len * elem].copy_from_slice(&snap.v[src..src + len * elem]);
            if self.prec == KvPrec::I8 {
                let (d, sc) = (vec * self.max_ctx, vec * snap.len);
                self.ks[d..d + len].copy_from_slice(&snap.ks[sc..sc + len]);
                self.vs[d..d + len].copy_from_slice(&snap.vs[sc..sc + len]);
            }
        }
        self.len = len;
        true
    }

    pub fn prec(&self) -> KvPrec {
        self.prec
    }
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn clear(&mut self) {
        self.len = 0;
        if let Some(r) = self.rec.as_mut() {
            r.state.fill(0.0);
            r.conv.fill(0.0);
            // A checkpoint describes the conversation that was cleared; restoring it later would be another one.
            r.ckpt = None;
            r.snap_pos = None;
            r.snap_valid = false;
        }
    }

    /// Drop everything from `pos` onward. Used when a speculative block is only
    /// partly accepted: the rejected positions were written before verification and
    /// must not be visible to the next step.
    ///
    /// with recurrent layers the state can only go back to the checkpoint's position; any other cut
    /// empties the cache. Read `filled()` afterwards for what is kept.
    pub fn truncate(&mut self, pos: usize) {
        if pos >= self.len {
            return;
        }
        if let Some(r) = self.rec.as_mut() {
            if let (Some(p), true) = (r.snap_pos, r.snap_valid) {
                if p + 1 == pos {
                    r.state.copy_from_slice(&r.snap_state);
                    r.conv.copy_from_slice(&r.snap_conv);
                    r.snap_pos = None;
                    r.snap_valid = false;
                    self.len = pos;
                    return;
                }
            }
            if let Some((p, s, c)) = &r.ckpt {
                if *p == pos {
                    r.state.copy_from_slice(s);
                    r.conv.copy_from_slice(c);
                    self.len = pos;
                    return;
                }
            }
            self.clear();
            return;
        }
        self.len = self.len.min(pos);
    }

    #[inline]
    fn vec_index(&self, layer: usize, head: usize, pos: usize) -> usize {
        debug_assert!(layer < self.n_layer && head < self.n_kv_head && pos < self.max_ctx);
        (layer * self.n_kv_head + head) * self.max_ctx + pos
    }

    /// Bytes one layer holds for one position, K and V together. This is the unit
    /// the attention loop accumulates, because that loop runs once per layer — using
    /// the whole-model figure there overcounts cache traffic by `n_layer`.
    pub fn bytes_per_position_per_layer(&self) -> u64 {
        let per_vec =
            self.head_dim * self.prec.elem() + if self.prec == KvPrec::I8 { 4 } else { 0 };
        (self.n_kv_head * per_vec * 2) as u64 // ×2 for K and V
    }

    /// Total bytes the cache holds for one position across all layers — `κ` per
    /// position. Includes the I8 scales, because they are read too.
    pub fn bytes_per_position(&self) -> u64 {
        self.n_layer as u64 * self.bytes_per_position_per_layer()
    }

    /// `κ(T)`: bytes read from the cache to produce one token at the current length.
    pub fn bytes_read_at(&self, len: usize) -> u64 {
        self.bytes_per_position() * len as u64
    }

    fn encode_into(&mut self, is_k: bool, idx: usize, x: &[f32]) {
        let hd = self.head_dim;
        let off = idx * hd;
        match self.prec {
            KvPrec::F16 => {
                #[cfg(target_arch = "x86_64")]
                {
                    if hd.is_multiple_of(8) && vector_path() {
                        let buf = if is_k { &mut self.k } else { &mut self.v };
                        // SAFETY: features checked; `off + hd` elements fit (vec_index).
                        unsafe { encode_f16_avx(x, &mut buf[off * 2..(off + hd) * 2]) };
                        return;
                    }
                }
                let buf = if is_k { &mut self.k } else { &mut self.v };
                for (i, &val) in x.iter().enumerate() {
                    let h = to_f16(val).to_le_bytes();
                    buf[(off + i) * 2] = h[0];
                    buf[(off + i) * 2 + 1] = h[1];
                }
            }
            KvPrec::F32 => {
                #[cfg(target_arch = "x86_64")]
                {
                    if hd.is_multiple_of(8) && crate::cpu_has!("avx2") {
                        let buf = if is_k { &mut self.k } else { &mut self.v };
                        // SAFETY: features checked; `off + hd` elements fit (vec_index).
                        unsafe { encode_f32_avx(x, &mut buf[off * 4..(off + hd) * 4]) };
                        return;
                    }
                }
                let buf = if is_k { &mut self.k } else { &mut self.v };
                for (d, &val) in buf[off * 4..(off + hd) * 4]
                    .chunks_exact_mut(4)
                    .zip(x.iter())
                {
                    d.copy_from_slice(&val.to_le_bytes());
                }
            }
            KvPrec::I8 => {
                let mut rot = [0.0f32; MAX_ROTATED_DIM];
                let x = if self.rotated {
                    let r = &mut rot[..hd];
                    r.copy_from_slice(x);
                    crate::vmath::hadamard(r);
                    &*r
                } else {
                    x
                };
                let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                let buf = if is_k { &mut self.k } else { &mut self.v };
                for (i, &val) in x.iter().enumerate() {
                    buf[off + i] = ((val * inv).round().clamp(-127.0, 127.0) as i8) as u8;
                }
                if is_k {
                    self.ks[idx] = d;
                } else {
                    self.vs[idx] = d;
                }
            }
        }
    }

    /// Append one position's key and value for one head. Callers must write every
    /// layer and head for a position before advancing `len`.
    pub fn put(&mut self, layer: usize, head: usize, pos: usize, k: &[f32], v: &[f32]) {
        debug_assert_eq!(k.len(), self.head_dim);
        debug_assert_eq!(v.len(), self.head_dim);
        let idx = self.vec_index(layer, head, pos);
        self.encode_into(true, idx, k);
        self.encode_into(false, idx, v);
    }

    pub fn advance(&mut self, to: usize) {
        self.len = to;
    }

    /// Whether stored vectors are Hadamard-rotated (`vmath::hadamard`). True for the
    /// int8 cache whenever `head_dim` is a power of two: the rotation is what keeps a
    /// per-vector scale accurate when one channel is an outlier. Attention then works
    /// in the rotated space — `q` rotated the same way, the value mix un-rotated once
    /// per head — which is exact because the rotation is orthonormal.
    pub fn rotated(&self) -> bool {
        self.rotated
    }

    /// The same cache with the int8 rotation off. For a family whose keys and queries
    /// share a few huge channels (Qwen2's attention biases): unrotated, the outlier
    /// channel sets the scale and is stored exactly while the error lands in channels
    /// the query barely weighs; rotated, a smaller total error spreads evenly — onto
    /// the one channel the query depends on. Measured on Qwen2.5-0.5B (`examples/
    /// kvacc`): rotation halves the key quantisation error yet raises the next-token
    /// KL against f16 from 5.1e-2 to 6.1e-2, where on Qwen3-4B it cuts it 4×.
    pub fn without_rotation(mut self) -> KvCache {
        self.rotated = false;
        self
    }

    /// Decode positions `p0..p0+n` of one (layer, head) into `out` as `n × head_dim`
    /// floats — keys when `is_k`, values otherwise, in the stored (possibly rotated)
    /// space.
    ///
    /// Attention decodes a block once and reuses it for every query head that shares
    /// this KV head (and every batch row), instead of re-widening the same f16 or int8
    /// bytes once per query head as the old per-head scan did.
    pub fn decode_block(
        &self,
        is_k: bool,
        layer: usize,
        head: usize,
        p0: usize,
        n: usize,
        out: &mut [f32],
    ) {
        let hd = self.head_dim;
        debug_assert!(out.len() >= n * hd && p0 + n <= self.max_ctx);
        if n == 0 {
            return;
        }
        let idx0 = self.vec_index(layer, head, p0);
        #[cfg(target_arch = "x86_64")]
        {
            if hd.is_multiple_of(8) {
                // One path per precision, chosen once per block: the f16 path must
                // not pay for the f32 one, and the f32 cache needs no F16C (it is a
                // plain copy of the stored floats).
                match self.prec {
                    KvPrec::F32 => {
                        if crate::cpu_has!("avx2") {
                            // SAFETY: features checked; `vec_index` and `p0 + n <= max_ctx`
                            // keep every read inside this head's run of positions.
                            unsafe { return self.decode_block_f32_avx2(is_k, idx0, n, out) };
                        }
                    }
                    KvPrec::F16 => {
                        if vector_path() {
                            // SAFETY: features checked; `vec_index` and `p0 + n <= max_ctx`
                            // keep every read inside this head's run of positions.
                            unsafe { return self.decode_block_f16_avx2(is_k, idx0, n, out) };
                        }
                    }
                    KvPrec::I8 => {
                        if vector_path() {
                            // SAFETY: features checked; `vec_index` and `p0 + n <= max_ctx`
                            // keep every read inside this head's run of positions.
                            unsafe { return self.decode_block_i8_avx2(is_k, idx0, n, out) };
                        }
                    }
                }
            }
        }
        self.decode_block_scalar(is_k, idx0, n, out);
    }

    /// The scalar specification of `decode_block`; the vector path is tested against it.
    fn decode_block_scalar(&self, is_k: bool, idx0: usize, n: usize, out: &mut [f32]) {
        let hd = self.head_dim;
        let buf = if is_k { &self.k } else { &self.v };
        // The precision is decided once, not per position: the f16 path must not pay
        // for the f32 one.
        match self.prec {
            KvPrec::F16 => {
                for p in 0..n {
                    let row = &mut out[p * hd..(p + 1) * hd];
                    let b = &buf[(idx0 + p) * hd * 2..(idx0 + p + 1) * hd * 2];
                    for (i, o) in row.iter_mut().enumerate() {
                        *o = f16(u16::from_le_bytes([b[i * 2], b[i * 2 + 1]]));
                    }
                }
            }
            KvPrec::F32 => {
                for p in 0..n {
                    let row = &mut out[p * hd..(p + 1) * hd];
                    let b = &buf[(idx0 + p) * hd * 4..(idx0 + p + 1) * hd * 4];
                    for (i, o) in row.iter_mut().enumerate() {
                        *o = f32::from_le_bytes([
                            b[i * 4],
                            b[i * 4 + 1],
                            b[i * 4 + 2],
                            b[i * 4 + 3],
                        ]);
                    }
                }
            }
            KvPrec::I8 => {
                for p in 0..n {
                    let row = &mut out[p * hd..(p + 1) * hd];
                    let b = &buf[(idx0 + p) * hd..(idx0 + p + 1) * hd];
                    let s = if is_k {
                        self.ks[idx0 + p]
                    } else {
                        self.vs[idx0 + p]
                    };
                    for (o, &c) in row.iter_mut().zip(b) {
                        *o = (c as i8) as f32 * s;
                    }
                }
            }
        }
    }

    /// The f32 cache's vector decode: the stored floats, copied out.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn decode_block_f32_avx2(&self, is_k: bool, idx0: usize, n: usize, out: &mut [f32]) {
        use std::arch::x86_64::*;
        let hd = self.head_dim;
        let buf = if is_k { &self.k } else { &self.v };
        let op = out.as_mut_ptr();
        // Contiguous positions: one flat run of `n · hd` floats.
        let src = buf.as_ptr().add(idx0 * hd * 4) as *const f32;
        let total = n * hd;
        let mut i = 0;
        while i < total {
            _mm256_storeu_ps(op.add(i), _mm256_loadu_ps(src.add(i)));
            i += 8;
        }
    }

    /// The f16 cache's vector decode: the stored halves, widened.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn decode_block_f16_avx2(&self, is_k: bool, idx0: usize, n: usize, out: &mut [f32]) {
        use std::arch::x86_64::*;
        let hd = self.head_dim;
        let buf = if is_k { &self.k } else { &self.v };
        let op = out.as_mut_ptr();
        // Contiguous positions: one flat run of `n · hd` halves.
        let src = buf.as_ptr().add(idx0 * hd * 2);
        let total = n * hd;
        let mut i = 0;
        while i < total {
            _mm256_storeu_ps(
                op.add(i),
                _mm256_cvtph_ps(_mm_loadu_si128(src.add(i * 2) as *const __m128i)),
            );
            i += 8;
        }
    }

    /// The int8 cache's vector decode: the stored bytes, scaled.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn decode_block_i8_avx2(&self, is_k: bool, idx0: usize, n: usize, out: &mut [f32]) {
        use std::arch::x86_64::*;
        let hd = self.head_dim;
        let buf = if is_k { &self.k } else { &self.v };
        let op = out.as_mut_ptr();
        let scales = if is_k { &self.ks } else { &self.vs };
        for p in 0..n {
            let src = buf.as_ptr().add((idx0 + p) * hd);
            let s = _mm256_set1_ps(scales[idx0 + p]);
            let mut i = 0;
            while i < hd {
                let x = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                    src.add(i) as *const __m128i
                )));
                _mm256_storeu_ps(op.add(p * hd + i), _mm256_mul_ps(x, s));
                i += 8;
            }
        }
    }

    /// Resident bytes of the cache itself, for the memory report.
    pub fn resident_bytes(&self) -> u64 {
        (self.k.len() + self.v.len() + (self.ks.len() + self.vs.len()) * 4) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // a hybrid cache returns to its checkpoint exactly and empties on any other cut.
    #[test]
    fn recurrent_state_rolls_back_only_to_its_checkpoint() {
        let mut kv = KvCache::new(KvPrec::F16, 1, 1, 8, 64).with_recurrent(2, 4, 3);
        assert!(kv.has_recurrent());
        kv.recurrent(1).0.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        kv.advance(5);
        kv.checkpoint();
        kv.recurrent(1).0.copy_from_slice(&[9.0; 4]);
        kv.recurrent(0).1.copy_from_slice(&[7.0; 3]);
        kv.advance(9);
        kv.truncate(5);
        assert_eq!(kv.filled(), 5);
        assert_eq!(kv.recurrent(1).0, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(kv.recurrent(0).1, &[0.0; 3]);
        kv.truncate(5); // a no-op at the current length
        assert_eq!(kv.filled(), 5);
        kv.advance(7);
        kv.truncate(6); // not the checkpoint: everything goes
        assert_eq!(kv.filled(), 0);
        assert_eq!(kv.recurrent(1).0, &[0.0; 4]);
        // and a snapshot cannot restore a recurrent cache
        let snap = KvCache::new(KvPrec::F16, 1, 1, 8, 64).snapshot(0);
        assert!(!kv.restore(&snap, 0));
    }

    fn ramp(n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) / n as f32 - 0.5) * scale)
            .collect()
    }

    // a snapshot restores exactly the positions it was taken at, into the same or a fresh cache.
    #[test]
    fn a_snapshot_restores_its_positions_into_a_fresh_cache() {
        for prec in [KvPrec::F16, KvPrec::F32, KvPrec::I8] {
            let hd = 32;
            let mut a = KvCache::new(prec, 3, 2, hd, 16);
            for pos in 0..6 {
                for layer in 0..3 {
                    for head in 0..2 {
                        let k = ramp(hd, 1.0 + (layer * 2 + head + pos) as f32);
                        let v = ramp(hd, 0.5 + pos as f32);
                        a.put(layer, head, pos, &k, &v);
                    }
                }
            }
            a.advance(6);
            let snap = a.snapshot(4);
            assert_eq!(snap.len(), 4);
            let mut b = KvCache::new(prec, 3, 2, hd, 16);
            assert!(b.restore(&snap, 3), "{prec:?}");
            assert_eq!(b.len, 3);
            for layer in 0..3 {
                for head in 0..2 {
                    for pos in 0..3 {
                        assert_eq!(
                            decoded(&a, true, layer, head, pos),
                            decoded(&b, true, layer, head, pos),
                            "{prec:?} l{layer} h{head} p{pos}"
                        );
                    }
                }
            }
            // A snapshot of another shape, or asked for more than it holds, is refused and changes nothing.
            assert!(!b.restore(&snap, 5));
            assert_eq!(b.len, 3);
            let mut other = KvCache::new(prec, 3, 4, hd, 16);
            assert!(!other.restore(&snap, 2));
        }
    }

    fn decoded(c: &KvCache, is_k: bool, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; c.head_dim];
        c.decode_block(is_k, layer, head, pos, 1, &mut out);
        if c.rotated() {
            crate::vmath::hadamard_inv(&mut out);
        }
        out
    }

    #[test]
    fn f16_cache_round_trips_keys_and_values() {
        let hd = 64;
        let mut c = KvCache::new(KvPrec::F16, 2, 2, hd, 8);
        let k = ramp(hd, 3.0);
        let v = ramp(hd, 1.0);
        c.put(1, 1, 3, &k, &v);
        assert!(!c.rotated(), "only the int8 cache is rotated");
        for (got, want) in decoded(&c, true, 1, 1, 3).iter().zip(&k) {
            assert!((got - want).abs() <= 1e-3 * want.abs().max(1.0));
        }
        for (got, want) in decoded(&c, false, 1, 1, 3).iter().zip(&v) {
            assert!((got - want).abs() <= 1e-3 * want.abs().max(1.0));
        }
    }

    /// The f32 cache is exact: the key attention reads is the key the model
    /// produced, bit for bit. That is the whole point of it — a like-for-like
    /// comparison against an engine whose attention reads f16 operands must not
    /// differ because of *our* rounding.
    #[test]
    fn f32_cache_round_trips_keys_and_values_exactly() {
        let hd = 64;
        let mut c = KvCache::new(KvPrec::F32, 2, 2, hd, 8);
        assert!(!c.rotated(), "only the int8 cache is rotated");
        // Values a half-precision cache would round: a subnormal, a tiny and a huge
        // one, and something f16 cannot represent at all.
        let k: Vec<f32> = (0..hd)
            .map(|i| match i % 4 {
                0 => 1e-40,           // f16 rounds this to zero
                1 => 1.0 / 3.0,       // not a half
                2 => 65520.0,         // f16 overflows this to infinity
                _ => -12.345_678_f32, // more digits than a half holds
            })
            .collect();
        let v: Vec<f32> = (0..hd).map(|i| (i as f32) * 1e-3 - 0.5).collect();
        c.put(1, 1, 3, &k, &v);
        for (got, want) in decoded(&c, true, 1, 1, 3).iter().zip(&k) {
            assert_eq!(got.to_bits(), want.to_bits(), "key {got} != {want}");
        }
        for (got, want) in decoded(&c, false, 1, 1, 3).iter().zip(&v) {
            assert_eq!(got.to_bits(), want.to_bits(), "value {got} != {want}");
        }
        // The f16 cache of the same vector is not exact — that is the difference the
        // option exists to remove.
        let mut h = KvCache::new(KvPrec::F16, 2, 2, hd, 8);
        h.put(1, 1, 3, &k, &v);
        let inexact = decoded(&h, true, 1, 1, 3)
            .iter()
            .zip(&k)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert!(
            inexact > 0,
            "the f16 cache was exact; the test proves nothing"
        );
    }

    /// The f32 cache costs twice the bytes of the f16 one and no scale at all.
    #[test]
    fn the_f32_cache_holds_twice_the_bytes_of_the_f16_one() {
        let f16 = KvCache::new(KvPrec::F16, 32, 5, 64, 2048);
        let f32 = KvCache::new(KvPrec::F32, 32, 5, 64, 2048);
        assert_eq!(f16.bytes_per_position(), 32 * 5 * 64 * 2 * 2);
        assert_eq!(f32.bytes_per_position(), 32 * 5 * 64 * 4 * 2);
        assert_eq!(f32.bytes_per_position(), 2 * f16.bytes_per_position());
        assert_eq!(
            f32.bytes_per_position_per_layer() * 32,
            f32.bytes_per_position()
        );
        // The int8 scale is the only per-vector extra; f32 has none.
        assert_eq!(f32.resident_bytes(), f32.bytes_per_position() * 2048);
    }

    /// The names a command line carries: `f16`, `f32`, `i8`.
    #[test]
    fn a_cache_precision_is_named_and_parsed() {
        assert_eq!(KvPrec::parse("f16"), Some(KvPrec::F16));
        assert_eq!(KvPrec::parse("f32"), Some(KvPrec::F32));
        assert_eq!(KvPrec::parse("F32"), Some(KvPrec::F32));
        assert_eq!(KvPrec::parse("i8"), Some(KvPrec::I8));
        assert_eq!(KvPrec::parse("int8"), Some(KvPrec::I8));
        assert_eq!(KvPrec::parse("f64"), None);
        assert_eq!(KvPrec::parse(""), None);
        for p in [KvPrec::F16, KvPrec::F32, KvPrec::I8] {
            assert_eq!(KvPrec::parse(p.name()), Some(p));
        }
    }

    #[test]
    fn int8_cache_is_accurate_to_its_quantisation_step() {
        let hd = 64;
        let mut c = KvCache::new(KvPrec::I8, 1, 1, hd, 4);
        assert!(c.rotated());
        let k = ramp(hd, 3.0);
        let v = ramp(hd, 1.0);
        c.put(0, 0, 0, &k, &v);
        // Rotation is orthonormal, so the L2 error of the round trip is the L2 error
        // of quantising the rotated vector: at most half a step per element, where
        // the step is the rotated vector's absmax / 127.
        for (x, is_k) in [(&k, true), (&v, false)] {
            let mut r = x.clone();
            crate::vmath::hadamard(&mut r);
            let step = r.iter().fold(0.0f32, |m, e| m.max(e.abs())) / 127.0;
            let err: f32 = decoded(&c, is_k, 0, 0, 0)
                .iter()
                .zip(x.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            let bound = (hd as f32).sqrt() * step * 0.5 + 1e-5;
            assert!(err <= bound, "is_k={is_k}: L2 error {err} > bound {bound}");
        }
    }

    /// The point of rotating: a vector with one outlier channel — the shape real
    /// keys have — loses far less to a shared int8 scale once rotated.
    #[test]
    fn rotation_cuts_int8_error_on_an_outlier_channel() {
        let hd = 128;
        let mut x: Vec<f32> = (0..hd)
            .map(|i| ((i * 37 % 101) as f32 / 101.0 - 0.5) * 0.2)
            .collect();
        x[5] = 6.0;
        let err = |c: &KvCache| -> f32 {
            decoded(c, true, 0, 0, 0)
                .iter()
                .zip(&x)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt()
        };
        let mut rotated = KvCache::new(KvPrec::I8, 1, 1, hd, 2);
        rotated.put(0, 0, 0, &x, &x);
        let mut plain = KvCache::new(KvPrec::I8, 1, 1, hd, 2).without_rotation();
        plain.put(0, 0, 0, &x, &x);
        let (er, ep) = (err(&rotated), err(&plain));
        assert!(
            er < 0.5 * ep,
            "rotated L2 error {er} should be well under unrotated {ep}"
        );
    }

    /// The vector decode must be the scalar decode, for both precisions, every head
    /// width it accepts, and a block that starts mid-run.
    #[test]
    fn vector_decode_matches_the_scalar_decode() {
        for prec in [KvPrec::F16, KvPrec::F32, KvPrec::I8] {
            for hd in [64usize, 24, 128] {
                let n = 37;
                let mut c = KvCache::new(prec, 2, 2, hd, 64);
                for p in 0..n {
                    let k: Vec<f32> = ramp(hd, 2.0 + p as f32 * 0.01)
                        .iter()
                        .map(|v| v * (1.0 + p as f32 * 0.05))
                        .collect();
                    let v: Vec<f32> = ramp(hd, 1.5).iter().map(|x| x - p as f32 * 0.003).collect();
                    c.put(1, 1, p, &k, &v);
                }
                c.advance(n);
                for is_k in [true, false] {
                    let (p0, m) = (3, 29);
                    let mut fast = vec![0.0f32; m * hd];
                    c.decode_block(is_k, 1, 1, p0, m, &mut fast);
                    let mut slow = vec![0.0f32; m * hd];
                    c.decode_block_scalar(is_k, c.vec_index(1, 1, p0), m, &mut slow);
                    assert_eq!(fast, slow, "{prec:?} hd={hd} is_k={is_k}");
                }
            }
        }
    }

    /// The whole reason the cache has a precision setting: int8 must actually halve
    /// the bytes read. A layout that stored scales per element would not.
    #[test]
    fn int8_moves_about_half_the_bytes_of_f16() {
        let f = KvCache::new(KvPrec::F16, 32, 5, 64, 2048);
        let i = KvCache::new(KvPrec::I8, 32, 5, 64, 2048);
        let (bf, bi) = (f.bytes_per_position(), i.bytes_per_position());
        assert_eq!(bf, 32 * 5 * 64 * 2 * 2);
        // And the per-layer unit must be exactly the whole divided by the layers.
        assert_eq!(f.bytes_per_position_per_layer() * 32, bf);
        // int8 adds 4 bytes of scale per 64-element vector, so the saving is just
        // under half — stating it exactly keeps the telemetry honest.
        assert_eq!(bi, 32 * 5 * (64 + 4) * 2);
        let ratio = bi as f64 / bf as f64;
        assert!((0.5..0.55).contains(&ratio), "ratio {ratio}");
    }

    /// The F16C encoder must round exactly as the scalar `to_f16` does, including
    /// subnormals, overflow to infinity and the round-half-to-even ties.
    #[test]
    fn vector_f16_encode_matches_scalar_rounding() {
        let mut xs: Vec<f32> = vec![
            0.0, -0.0, 1.0, -2.5, 65504.0, 65520.0, 1e9, -1e9, 6.0e-8, 3.0e-8, 1.0e-5, -1.0e-7,
        ];
        // Ties at several exponents: 1 + 2^-11 sits exactly between two halves.
        for e in -14i32..15 {
            let base = 2f32.powi(e);
            xs.push(base * (1.0 + 2f32.powi(-11)));
            xs.push(base * (1.0 + 3.0 * 2f32.powi(-11)));
            xs.push(-base * 1.3333);
        }
        while !xs.len().is_multiple_of(8) {
            xs.push(0.123);
        }
        let hd = xs.len();
        let mut c = KvCache::new(KvPrec::F16, 1, 1, hd, 2);
        c.put(0, 0, 0, &xs, &xs);
        for (i, &x) in xs.iter().enumerate() {
            let got = u16::from_le_bytes([c.k[i * 2], c.k[i * 2 + 1]]);
            assert_eq!(got, to_f16(x), "x = {x:e} (index {i})");
        }
    }

    #[test]
    fn truncate_discards_rejected_speculative_positions() {
        let mut c = KvCache::new(KvPrec::F16, 1, 1, 8, 16);
        c.advance(10);
        c.truncate(4);
        assert_eq!(c.len, 4);
        // Truncating above the current length must not extend it.
        c.truncate(9);
        assert_eq!(c.len, 4);
    }

    #[test]
    fn kappa_grows_linearly_with_context() {
        let c = KvCache::new(KvPrec::F16, 32, 5, 64, 4096);
        assert_eq!(c.bytes_read_at(0), 0);
        assert_eq!(c.bytes_read_at(1000), c.bytes_per_position() * 1000);
        assert_eq!(c.bytes_read_at(2000), 2 * c.bytes_read_at(1000));
    }
}
