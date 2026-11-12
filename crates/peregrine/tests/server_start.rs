//! The server tests' own start helper (`common::start`): a test's server is the one it
//! started, also when the port it was given is taken before the server can listen.

#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

mod common;

use peregrine_core::gguf::Format;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::Child;
use std::time::Duration;
use support::models::{write_dense, Dense};

struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "peregrine-server-start-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&d);
    d
}

fn health(port: u16) -> String {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .unwrap();
    let mut out = String::new();
    stream.read_to_string(&mut out).unwrap();
    out
}

fn args(model: &str) -> Vec<&str> {
    vec![
        "serve",
        "-m",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        common::PORT,
        "--ctx",
        "64",
        "--threads",
        "1",
        "--no-profile",
    ]
}

/// The port the test chose is taken by the time the server binds: the server exits, and
/// the helper starts another on the next port instead of waiting for a server that is not
/// there (or talking to whoever holds the port).
#[test]
fn a_taken_port_is_retried_on_another() {
    let dir = tmpdir("taken");
    let model = write_dense(&dir, "llama", Format::F16, 5, Dense::default());
    // Something else listens on the first port for the whole test.
    let squatter = TcpListener::bind("127.0.0.1:0").unwrap();
    let taken = squatter.local_addr().unwrap().port();
    let mut handed_out = Vec::new();
    let mut next = || {
        let port = if handed_out.is_empty() {
            taken
        } else {
            common::free_port()
        };
        handed_out.push(port);
        port
    };
    let (child, port) = common::start_on(&dir, &[], &args(model.to_str().unwrap()), &mut next);
    let _guard = Guard(child);
    assert_eq!(
        handed_out.len(),
        2,
        "one failed attempt, then one that listened"
    );
    assert_eq!(handed_out[0], taken);
    assert_eq!(port, handed_out[1]);
    assert_ne!(port, taken);
    assert!(health(port).starts_with("HTTP/1.1 200"), "{}", health(port));
    drop(squatter);
}

/// The ordinary case: the first port works and the server that answers is the child.
#[test]
fn a_started_server_is_listening_when_start_returns() {
    let dir = tmpdir("plain");
    let model = write_dense(&dir, "llama", Format::F16, 6, Dense::default());
    let (child, port) = common::start(&dir, &[], &args(model.to_str().unwrap()));
    let _guard = Guard(child);
    // No polling: `start` returned, so the socket is bound and the request is answered.
    assert!(health(port).starts_with("HTTP/1.1 200"));
}

/// A server that dies for any other reason fails the test with its own words.
#[test]
#[should_panic(expected = "before it listened")]
fn a_server_that_cannot_start_fails_the_test_with_its_log() {
    let dir = tmpdir("broken");
    let missing = dir.join("no-such-model.gguf");
    let _ = common::start(&dir, &[], &args(missing.to_str().unwrap()));
}
