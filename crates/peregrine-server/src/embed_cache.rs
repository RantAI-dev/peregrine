//! An exact, bounded cache of finished embedding vectors (S4.9k).
//!
//! The key is everything that changes a vector: the token ids, the pooling and
//! the requested dimensions. The value is the **finished** vector (truncated and
//! L2-normalised), so a hit returns the same bits as a computation. A batch's
//! duplicate texts are planned onto one job (`plan`), so a text that appears
//! twice is computed once.
//!
//! The bound is bytes, not entries: keys and vectors both count, plus a small
//! per-entry overhead. Eviction is insertion order (the oldest entry goes
//! first). The default is 64 MiB (`PEREGRINE_EMBED_CACHE_MB`); `0` disables the
//! cache. 64 MiB holds about 16,000 1024-wide f32 vectors or 21,000 768-wide
//! ones, and the same again in keys is included in the accounting.
//!
//! Concurrent requests share one `Mutex<Inner>`: a lookup and an insertion are
//! two short critical sections and no lock is held while the engine computes.
//! Two requests that miss the same text may both compute it; both results are
//! bit-identical by the engine's batch invariance, so the second insert is
//! harmless.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use peregrine_core::embed::Pooling;

/// One cache key. The fields are exactly the inputs a vector is a function of.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Key {
    pub tokens: Vec<u32>,
    pub pooling: u8,
    /// Requested Matryoshka width (`None` is the model's full width).
    pub dimensions: Option<usize>,
}

impl Key {
    pub fn new(tokens: Vec<u32>, pooling: Pooling, dimensions: Option<usize>) -> Key {
        let pooling = match pooling {
            Pooling::Last => 0,
            Pooling::Mean => 1,
            Pooling::Cls => 2,
        };
        Key {
            tokens,
            pooling,
            dimensions,
        }
    }

    fn bytes(&self) -> usize {
        self.tokens.len() * 4 + 32
    }
}

struct Inner {
    map: HashMap<Key, Vec<f32>>,
    order: VecDeque<Key>,
    bytes: usize,
}

/// What one batch needs from the cache and the engine.
pub(crate) struct Plan {
    /// Per input, the cached vector when the key was a hit.
    pub cached: Vec<Option<Vec<f32>>>,
    /// The distinct keys to compute, in first-appearance order.
    pub jobs: Vec<Key>,
    /// Per input, the job that produces it (`None` for a hit).
    pub job_of: Vec<Option<usize>>,
}

pub(crate) struct EmbedCache {
    cap_bytes: usize,
    inner: Mutex<Inner>,
}

impl EmbedCache {
    pub fn new(cap_mb: usize) -> EmbedCache {
        EmbedCache {
            cap_bytes: cap_mb.saturating_mul(1 << 20),
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
                bytes: 0,
            }),
        }
    }

    pub fn enabled(&self) -> bool {
        self.cap_bytes > 0
    }

    pub fn len(&self) -> usize {
        self.lock().map.len()
    }

    pub fn bytes(&self) -> usize {
        self.lock().bytes
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn get(&self, key: &Key) -> Option<Vec<f32>> {
        if !self.enabled() {
            return None;
        }
        self.lock().map.get(key).cloned()
    }

    pub fn put(&self, key: Key, vector: Vec<f32>) {
        if !self.enabled() {
            return;
        }
        let cost = key.bytes() + vector.len() * 4 + 64;
        if cost > self.cap_bytes {
            return;
        }
        let mut inner = self.lock();
        if let Some(old) = inner.map.remove(&key) {
            inner.bytes = inner.bytes.saturating_sub(key.bytes() + old.len() * 4 + 64);
        }
        while inner.bytes + cost > self.cap_bytes {
            let Some(evict) = inner.order.pop_front() else {
                break;
            };
            if let Some(v) = inner.map.remove(&evict) {
                inner.bytes = inner.bytes.saturating_sub(evict.bytes() + v.len() * 4 + 64);
            }
        }
        inner.bytes += cost;
        inner.map.insert(key.clone(), vector);
        inner.order.push_back(key);
    }

    /// Split one request's keys into cache hits and the distinct keys to compute.
    pub fn plan(&self, keys: Vec<Key>) -> Plan {
        let mut cached = Vec::with_capacity(keys.len());
        let mut jobs: Vec<Key> = Vec::new();
        let mut job_index: HashMap<Key, usize> = HashMap::new();
        let mut job_of = Vec::with_capacity(keys.len());
        for key in &keys {
            if let Some(v) = self.get(key) {
                cached.push(Some(v));
                job_of.push(None);
                continue;
            }
            cached.push(None);
            let idx = match job_index.get(key) {
                Some(&i) => i,
                None => {
                    jobs.push(key.clone());
                    let i = jobs.len() - 1;
                    job_index.insert(key.clone(), i);
                    i
                }
            };
            job_of.push(Some(idx));
        }
        Plan {
            cached,
            jobs,
            job_of,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(s: &str) -> Key {
        Key::new(s.bytes().map(u32::from).collect(), Pooling::Mean, None)
    }

    #[test]
    fn a_hit_is_the_same_bits_and_a_duplicate_shares_one_job() {
        let c = EmbedCache::new(1);
        c.put(key("a"), vec![1.0, 2.0]);
        let p = c.plan(vec![key("a"), key("b"), key("b"), key("a")]);
        assert_eq!(p.cached[0], Some(vec![1.0, 2.0]));
        assert_eq!(p.cached[3], Some(vec![1.0, 2.0]));
        assert_eq!(p.jobs.len(), 1);
        assert_eq!(p.job_of, vec![None, Some(0), Some(0), None]);
        let p = c.plan(vec![key("b"), key("b")]);
        assert_eq!(p.jobs.len(), 1);
        assert_eq!(p.job_of, vec![Some(0), Some(0)]);
    }

    #[test]
    fn the_bound_is_bytes_and_the_oldest_entry_goes_first() {
        let c = EmbedCache::new(0);
        c.put(key("a"), vec![0.0; 8]);
        assert_eq!(c.len(), 0, "0 MB disables the cache");
        // A 150-byte cap: one 8-value vector costs 36 (key) + 32 (values) + 64,
        // so one entry fits and the second evicts the first.
        let c = EmbedCache {
            cap_bytes: 150,
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
                bytes: 0,
            }),
        };
        c.put(key("a"), vec![0.0; 8]);
        assert_eq!(c.len(), 1);
        c.put(key("b"), vec![0.0; 8]);
        assert_eq!(c.len(), 1);
        assert!(c.get(&key("a")).is_none());
        assert!(c.get(&key("b")).is_some());
    }

    #[test]
    fn a_vector_bigger_than_the_cache_is_not_stored() {
        let c = EmbedCache {
            cap_bytes: 100,
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
                bytes: 0,
            }),
        };
        c.put(key("a"), vec![0.0; 16]);
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn concurrent_readers_and_writers_see_complete_vectors() {
        let c = std::sync::Arc::new(EmbedCache::new(1));
        let mut hs = Vec::new();
        for t in 0..4u32 {
            let c = c.clone();
            hs.push(std::thread::spawn(move || {
                for i in 0..200u32 {
                    let k = Key::new(vec![t, i], Pooling::Mean, None);
                    c.put(k.clone(), vec![t as f32; 4]);
                    if let Some(v) = c.get(&k) {
                        assert_eq!(v, vec![t as f32; 4]);
                    }
                }
            }));
        }
        for h in hs {
            h.join().unwrap();
        }
        assert!(c.bytes() <= 1 << 20);
    }
}
