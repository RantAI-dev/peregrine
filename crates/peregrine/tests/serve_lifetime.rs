//! The server's lifetime is unbounded: it exits when a shutdown signal arrives, not when a drain
//! timeout elapses. v0.1.1 shipped the opposite (a 30 s `timeout` around the whole serving future),
//! so this runs the real binary past a short, test-only drain timeout.
//!
//! `PEREGRINE_DRAIN_TIMEOUT_MS` is the override (`docs/dev/testing.md`); users never set it.

#![cfg(unix)]

#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

use peregrine_core::gguf::Format;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use support::models::{write_dense, Dense};

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "peregrine-serve-lifetime-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Kill the child even when an assertion fails first, so a failed test leaves no server behind.
struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn http_request(port: u16, request: &str) -> Option<String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .ok()?;
    stream.write_all(request.as_bytes()).ok()?;
    let mut out = String::new();
    stream.read_to_string(&mut out).ok()?;
    Some(out)
}

fn wait_for_health(port: u16, timeout: Duration) -> Result<String, String> {
    let deadline = Instant::now() + timeout;
    let mut last = String::new();
    while Instant::now() < deadline {
        if let Some(response) = http_request(
            port,
            "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
        ) {
            if response.starts_with("HTTP/1.1 200") || response.starts_with("HTTP/1.0 200") {
                return Ok(response);
            }
            last = response;
        }
        thread::sleep(Duration::from_millis(50));
    }
    Err(format!(
        "no healthy response before the timeout; last: {last}"
    ))
}

fn chat_completion(port: u16) -> String {
    let body = r#"{"model":"peregrine","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0}"#;
    http_request(
        port,
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        ),
    )
    .unwrap_or_else(|| panic!("no response from the chat endpoint"))
}

fn read_log(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

#[test]
fn the_server_outlives_the_drain_timeout_and_stops_on_sigterm() {
    let dir = tmpdir("sigterm");
    let model = write_dense(&dir, "llama", Format::F16, 11, Dense::default());
    let home = dir.join("home");
    std::fs::create_dir_all(&home).unwrap();
    let listen = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listen.local_addr().unwrap().port();
    drop(listen);
    let log = dir.join("serve.log");
    let log_file = std::fs::File::create(&log).unwrap();

    // 1 s instead of the 30 s users get: a regression that ties the server's lifetime to the drain
    // timeout then shows up in seconds rather than half a minute.
    let mut serve = Command::new(env!("CARGO_BIN_EXE_peregrine"));
    let child = serve
        .args([
            "serve",
            "-m",
            model.to_str().unwrap(),
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--no-profile",
            "--threads",
            "1",
            "-c",
            "128",
        ])
        .env("PEREGRINE_DRAIN_TIMEOUT_MS", "1000")
        .env("HOME", &home)
        .env_remove("XDG_STATE_HOME")
        .stdout(Stdio::from(log_file.try_clone().unwrap()))
        .stderr(Stdio::from(log_file))
        .spawn()
        .expect("spawn peregrine serve");
    let mut guard = Guard(child);

    wait_for_health(port, Duration::from_secs(60)).unwrap_or_else(|e| {
        panic!(
            "serve did not answer /health: {e}\nlog:\n{}",
            read_log(&log)
        )
    });

    // Well past the drain timeout, with no signal sent: the v0.1.1 server exited here.
    thread::sleep(Duration::from_millis(2500));
    let health = http_request(
        port,
        "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
    )
    .unwrap_or_else(|| {
        panic!(
            "serve exited on its own before any signal\nlog:\n{}",
            read_log(&log)
        )
    });
    assert!(
        health.starts_with("HTTP/1.1 200"),
        "health after the drain timeout: {health:?}\nlog:\n{}",
        read_log(&log)
    );
    let chat = chat_completion(port);
    assert!(
        chat.starts_with("HTTP/1.1 200"),
        "chat after the drain timeout: {chat:?}\nlog:\n{}",
        read_log(&log)
    );
    assert!(chat.contains("\"choices\""), "chat body: {chat:?}");
    assert!(
        guard.0.try_wait().unwrap().is_none(),
        "serve exited before the shutdown signal\nlog:\n{}",
        read_log(&log)
    );

    // A clean shutdown: SIGTERM drains and exits with status 0.
    let pid = guard.0.id();
    let sent = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()
        .expect("run kill");
    assert!(sent.success(), "kill -TERM {pid} failed");
    let deadline = Instant::now() + Duration::from_secs(20);
    let status = loop {
        if let Some(status) = guard.0.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            panic!(
                "serve did not exit within 20 s of SIGTERM\nlog:\n{}",
                read_log(&log)
            );
        }
        thread::sleep(Duration::from_millis(50));
    };
    assert!(
        status.success(),
        "serve exited with {status} after SIGTERM\nlog:\n{}",
        read_log(&log)
    );
    assert!(
        !read_log(&log).contains("shutdown took longer"),
        "an idle drain hit the timeout\nlog:\n{}",
        read_log(&log)
    );
}
