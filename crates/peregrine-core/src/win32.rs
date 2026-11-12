//! The Win32 calls `std` does not expose, in one module (S7.2).
//!
//! ADR 0002 says no new dependencies, so what the standard library lacks is written
//! against the system API by hand. Every `unsafe` block carries its SAFETY comment,
//! and every declaration here states the ABI (`extern "system"`, which is `stdcall`
//! on 32-bit Windows and `C` on x86-64 and aarch64).
//!
//! Today there is one function: the physical core count, for the thread pool's default
//! width. `std` knows only the logical processor count (`available_parallelism`).

/// `LOGICAL_PROCESSOR_RELATIONSHIP::RelationProcessorCore`: one entry per physical core.
const RELATION_PROCESSOR_CORE: u32 = 0;

/// `ERROR_INSUFFICIENT_BUFFER`: the topology query needs a larger buffer.
const ERROR_INSUFFICIENT_BUFFER: i32 = 122;

/// `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX`'s fixed part: the relationship and the
/// record's size in bytes. The payload follows, and its shape depends on the relationship;
/// this module only walks the records, so it never reads the payload.
const RECORD_HEADER: usize = 8;

#[link(name = "kernel32")]
extern "system" {
    /// `BOOL GetLogicalProcessorInformationEx(LOGICAL_PROCESSOR_RELATIONSHIP, PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, PDWORD)`.
    ///
    /// Called with a null buffer and a zero length it sets the length the buffer needs and
    /// fails with `ERROR_INSUFFICIENT_BUFFER`; that is the documented size query.
    fn GetLogicalProcessorInformationEx(
        relationship: u32,
        buffer: *mut u8,
        returned_length: *mut u32,
    ) -> i32;
}

/// The number of physical cores, from the processor topology, or `None` when the call fails.
///
/// `GetLogicalProcessorInformationEx(RelationProcessorCore, …)` returns one record per core,
/// SMT siblings included as one, which is the width a decode step wants: the siblings share
/// the core's caches and only slow a synchronous pool down.
pub fn physical_core_count() -> Option<usize> {
    let mut len: u32 = 0;
    // SAFETY: a null buffer with a zero length is the documented size query; the function
    // writes only `len`, which is a live, initialized `u32` for the duration of the call.
    let ok = unsafe {
        GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut len)
    };
    // A successful zero-length query cannot happen (there is always one core), and a failure
    // that reports no length has nothing to retry with.
    if len == 0 || (ok != 0 && len as usize <= RECORD_HEADER) {
        return None;
    }
    // The topology can grow between the query and the call (a hot-plugged CPU, a changed
    // processor group); two retries cover the realistic case and the loop is bounded.
    for _ in 0..3 {
        let mut buf = vec![0u8; len as usize];
        // SAFETY: `buf` is `len` bytes long and the call writes at most `len` bytes into it,
        // setting `len` to what it wrote. `buf` is live and distinct for the call.
        let ok = unsafe {
            GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf.as_mut_ptr(), &mut len)
        };
        if ok != 0 {
            return Some(count_records(&buf[..len as usize]));
        }
        let err = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        if err != ERROR_INSUFFICIENT_BUFFER || len as usize <= RECORD_HEADER {
            // Anything else (access denied, unsupported) is not worth a second try.
            return None;
        }
        // `len` now names the size that fits; loop and ask again.
    }
    None
}

/// Count the `RelationProcessorCore` records in a topology buffer.
fn count_records(buf: &[u8]) -> usize {
    let mut cores = 0usize;
    let mut at = 0usize;
    while at + RECORD_HEADER <= buf.len() {
        // The header is two little-endian `DWORD`s on every Windows target.
        let rel = u32::from_le_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]]);
        let size =
            u32::from_le_bytes([buf[at + 4], buf[at + 5], buf[at + 6], buf[at + 7]]) as usize;
        // A malformed record would loop forever; treat it as the end of the buffer.
        if size < RECORD_HEADER || at + size > buf.len() {
            break;
        }
        if rel == RELATION_PROCESSOR_CORE {
            cores += 1;
        }
        at += size;
    }
    cores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_topology_reports_between_one_and_the_logical_count() {
        let logical = std::thread::available_parallelism()
            .expect("the processor count")
            .get();
        let cores = physical_core_count().expect("GetLogicalProcessorInformationEx works");
        assert!(
            (1..=logical).contains(&cores),
            "{cores} physical cores against {logical} logical processors"
        );
    }

    #[test]
    fn a_record_header_is_walked_and_a_truncated_one_stops_the_walk() {
        let mut buf = Vec::new();
        // Two core records of 16 bytes, then one group record that is not a core.
        for rel in [RELATION_PROCESSOR_CORE, RELATION_PROCESSOR_CORE, 2u32] {
            buf.extend_from_slice(&rel.to_le_bytes());
            buf.extend_from_slice(&16u32.to_le_bytes());
            buf.extend_from_slice(&[0u8; 8]);
        }
        assert_eq!(count_records(&buf), 2);
        // A record whose size runs past the buffer ends the walk rather than being counted.
        buf.extend_from_slice(&RELATION_PROCESSOR_CORE.to_le_bytes());
        buf.extend_from_slice(&64u32.to_le_bytes());
        assert_eq!(count_records(&buf), 2);
    }
}
