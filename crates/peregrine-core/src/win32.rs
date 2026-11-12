//! The Win32 calls `std` does not expose, in one module (S7.2).
//!
//! ADR 0002 says no new dependencies, so what the standard library lacks is written
//! against the system API by hand. Every `unsafe` block carries its SAFETY comment,
//! and every declaration here states the ABI (`extern "system"`, which is `stdcall`
//! on 32-bit Windows and `C` on x86-64 and aarch64).
//!
//! Two things live here: the physical core count, for the thread pool's default width
//! (`std` knows only the logical processor count, `available_parallelism`), and the process
//! calls the lifetime commands need (S7.3: Windows has no `kill -0` or `SIGTERM`, so
//! `peregrine status` and `stop` ask `OpenProcess`, `GetExitCodeProcess` and
//! `TerminateProcess`).

/// `LOGICAL_PROCESSOR_RELATIONSHIP::RelationProcessorCore`: one entry per physical core.
const RELATION_PROCESSOR_CORE: u32 = 0;

/// `PROCESS_QUERY_LIMITED_INFORMATION`: enough to read the exit code of a process of ours.
const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
/// `PROCESS_TERMINATE`: the right `TerminateProcess` needs.
const PROCESS_TERMINATE: u32 = 0x0001;
/// `STILL_ACTIVE`: the exit-code value `GetExitCodeProcess` reports for a running process.
const STILL_ACTIVE: u32 = 259;

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
    /// `HANDLE OpenProcess(DWORD dwDesiredAccess, BOOL bInheritHandle, DWORD dwProcessId)`.
    ///
    /// A null handle means the process does not exist (or is not ours to open); the last
    /// error says which.
    fn OpenProcess(desired_access: u32, inherit_handle: i32, process_id: u32) -> isize;
    /// `BOOL GetExitCodeProcess(HANDLE hProcess, LPDWORD lpExitCode)`.
    fn GetExitCodeProcess(process: isize, exit_code: *mut u32) -> i32;
    /// `BOOL TerminateProcess(HANDLE hProcess, UINT uExitCode)`.
    fn TerminateProcess(process: isize, exit_code: u32) -> i32;
    /// `BOOL CloseHandle(HANDLE hObject)`.
    fn CloseHandle(object: isize) -> i32;
}

/// Whether `pid` names a running process.
///
/// A process that has exited is not running even while its handle is still open somewhere, so
/// the exit code (`STILL_ACTIVE` while it runs) is the answer, not whether the handle opens.
pub fn process_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    // SAFETY: the call opens a kernel handle for the pid and writes nothing into this process;
    // a null handle means failure, which is all this function needs to know.
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if handle == 0 {
        return false;
    }
    let mut code: u32 = 0;
    // SAFETY: `handle` is a live process handle from `OpenProcess` and `code` is a live,
    // initialized `u32` that the call writes only to.
    let ok = unsafe { GetExitCodeProcess(handle, &mut code) };
    // SAFETY: `handle` came from `OpenProcess` above and is closed exactly once here.
    let _ = unsafe { CloseHandle(handle) };
    ok != 0 && code == STILL_ACTIVE
}

/// End `pid` (`TerminateProcess`). The last resort of `peregrine stop` on Windows, used only
/// after the stop file's drain timeout; the caller has already asked it to drain.
pub fn terminate_process(pid: u32) -> std::io::Result<()> {
    // SAFETY: the call opens a kernel handle for the pid and writes nothing into this process;
    // a null handle means failure, reported with the last OS error.
    let handle = unsafe { OpenProcess(PROCESS_TERMINATE, 0, pid) };
    if handle == 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `handle` is a live process handle from `OpenProcess` with `PROCESS_TERMINATE`;
    // the exit code is arbitrary and only the success flag is read.
    let ok = unsafe { TerminateProcess(handle, 1) };
    let error = if ok == 0 {
        Some(std::io::Error::last_os_error())
    } else {
        None
    };
    // SAFETY: `handle` came from `OpenProcess` above and is closed exactly once here.
    let _ = unsafe { CloseHandle(handle) };
    match error {
        Some(error) => Err(error),
        None => Ok(()),
    }
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
    fn a_running_process_is_alive_and_an_impossible_one_is_not() {
        assert!(
            process_alive(std::process::id()),
            "the test process must be alive"
        );
        assert!(!process_alive(0), "there is no pid 0 process");
        assert!(
            !process_alive(u32::MAX),
            "u32::MAX is not a process id Windows accepts"
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
