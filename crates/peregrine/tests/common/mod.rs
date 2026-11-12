//! Starting `peregrine serve` for a test, on a port that is really the test's own.
//!
//! A test cannot be handed a port by the server (`serve` takes the port it is told), so it
//! asks the system for a free one, releases it, and passes it on. Between the release and
//! the server's `bind` the port can be taken again: by another test in this binary that
//! asked at the same moment, or as the local port of some other test's HTTP connection.
//! The server then cannot listen and exits, and a test that only polled `/health` on that
//! port either waited out its deadline or, worse, talked to the other test's server until
//! that test ended and killed it (both seen in CI).
//!
//! [`start`] closes the gap from the test's side: it waits for *this* child to write its
//! own `listening on` line, and starts again on another port when the child exits because
//! it could not listen. Anything else the child dies of fails the test with its log.

#![allow(dead_code)]

use std::net::TcpListener;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Stands for the port in the arguments given to [`start`].
pub const PORT: &str = "{port}";

/// How long a server may take to load its model and listen.
const START_TIMEOUT: Duration = Duration::from_secs(120);

/// How many ports are tried before the test gives up.
const ATTEMPTS: usize = 8;

/// A port nothing listened on a moment ago.
pub fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind a free port");
    let p = l.local_addr().expect("the bound address").port();
    drop(l);
    p
}

/// Start `peregrine <args>` with [`PORT`] replaced by a free port, and return the child
/// once it is listening, with the port it listens on. Its output goes to a log of its own
/// under `dir` (`serve-<n>.log`), so two servers of one test never share a file.
///
/// The arguments must serve on `127.0.0.1` (`--host 127.0.0.1 --port {port}`).
pub fn start(dir: &Path, envs: &[(&str, &str)], args: &[&str]) -> (Child, u16) {
    start_on(dir, envs, args, &mut free_port)
}

/// [`start`] with the ports to try given by the caller, so the retry can be tested.
pub fn start_on(
    dir: &Path,
    envs: &[(&str, &str)],
    args: &[&str],
    next_port: &mut dyn FnMut() -> u16,
) -> (Child, u16) {
    static STARTED: AtomicUsize = AtomicUsize::new(0);
    assert!(
        args.contains(&PORT),
        "the arguments must name the port as common::PORT"
    );
    let mut last = String::new();
    for _ in 0..ATTEMPTS {
        let port = next_port();
        let port_text = port.to_string();
        let n = STARTED.fetch_add(1, Ordering::SeqCst);
        let log = &dir.join(format!("serve-{}-{n}.log", std::process::id()));
        let log_file = std::fs::File::create(log).expect("create the server's log");
        let mut command = Command::new(env!("CARGO_BIN_EXE_peregrine"));
        command
            .args(
                args.iter()
                    .map(|a| if *a == PORT { port_text.as_str() } else { *a }),
            )
            .stdout(Stdio::from(
                log_file.try_clone().expect("clone the log handle"),
            ))
            .stderr(Stdio::from(log_file));
        // The readiness line is logged at `info`; a developer's own RUST_LOG must not hide it.
        command.env("RUST_LOG", "info");
        for (key, value) in envs {
            command.env(key, value);
        }
        let mut child = command.spawn().expect("spawn peregrine serve");
        let ready = format!("listening on http://127.0.0.1:{port}");
        let deadline = Instant::now() + START_TIMEOUT;
        loop {
            let text = std::fs::read_to_string(log).unwrap_or_default();
            if text.contains(&ready) {
                return (child, port);
            }
            if let Ok(Some(status)) = child.try_wait() {
                // `serve` says this and exits when the port was taken after it was chosen.
                if text.contains("cannot listen on") {
                    last = text;
                    break;
                }
                panic!("peregrine serve exited ({status}) before it listened:\n{text}");
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "peregrine serve was not listening after {} s:\n{text}",
                    START_TIMEOUT.as_secs()
                );
            }
            thread::sleep(Duration::from_millis(20));
        }
    }
    panic!("no port could be listened on in {ATTEMPTS} attempts; the last server's log:\n{last}");
}
