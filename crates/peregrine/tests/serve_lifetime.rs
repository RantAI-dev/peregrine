//! The server's lifetime is unbounded: it exits when a shutdown signal arrives, not when a drain
//! timeout elapses. v0.1.1 shipped the opposite (a 30 s `timeout` around the whole serving future),
//! so this runs the real binary past a short, test-only drain timeout.
//!
//! `PEREGRINE_DRAIN_TIMEOUT_MS` is the override (`docs/dev/testing.md`); users never set it.
//!
//! The lifetime check runs everywhere. The clean-shutdown half sends `SIGTERM` on Unix and
//! creates the stop file on Windows (S7.3: there is no `SIGTERM`, and `peregrine stop` asks the
//! server to drain through a per-user file); both are the same drain. Ctrl-C is the other clean
//! shutdown on Windows, but the test harness cannot deliver a console control event to another
//! process group without its own Win32 code.

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

/// Start a server on a free port and wait until it answers `/health`. Its log is `dir/serve.log`.
fn start(dir: &Path, model: &Path) -> (Guard, u16, PathBuf) {
    start_with_drain(dir, model, 1000)
}

/// [`start`] with the drain timeout in milliseconds. The default is 1 s instead of the 30 s users
/// get: a regression that ties the server's lifetime to the drain timeout then shows up in seconds
/// rather than half a minute. The Windows stop-file test needs long enough for an in-flight
/// request, so it asks for more.
fn start_with_drain(dir: &Path, model: &Path, drain_ms: u64) -> (Guard, u16, PathBuf) {
    let home = dir.join("home");
    std::fs::create_dir_all(&home).unwrap();
    let state = dir.join("state");
    std::fs::create_dir_all(&state).unwrap();
    let listen = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listen.local_addr().unwrap().port();
    drop(listen);
    let log = dir.join("serve.log");
    let log_file = std::fs::File::create(&log).unwrap();

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
        .env("PEREGRINE_DRAIN_TIMEOUT_MS", drain_ms.to_string())
        // The Windows stop file lives in the state directory; give every test its own.
        .env("PEREGRINE_STATE_DIR", &state)
        .env("HOME", &home)
        .env_remove("XDG_STATE_HOME")
        .stdout(Stdio::from(log_file.try_clone().unwrap()))
        .stderr(Stdio::from(log_file))
        .spawn()
        .expect("spawn peregrine serve");
    let guard = Guard(child);
    wait_for_health(port, Duration::from_secs(60)).unwrap_or_else(|e| {
        panic!(
            "serve did not answer /health: {e}\nlog:\n{}",
            read_log(&log)
        )
    });
    (guard, port, log)
}

#[test]
fn the_server_outlives_the_drain_timeout() {
    let dir = tmpdir("lifetime");
    let model = write_dense(&dir, "llama", Format::F16, 11, Dense::default());
    let (mut guard, port, log) = start(&dir, &model);

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
    // The guard kills the process when the test ends. The clean shutdown is the next test.
}

/// A clean shutdown: `SIGTERM` drains and exits with status 0.
///
/// Unix only: Windows has no `SIGTERM`, and the stop file that replaces it is S7.3. There,
/// Ctrl-C is the clean shutdown and a test cannot send it without the console APIs (see the
/// module comment); the guard's `kill` only checks that the process was still serving above.
#[cfg(unix)]
#[test]
fn a_shutdown_signal_drains_and_exits_cleanly() {
    let dir = tmpdir("sigterm");
    let model = write_dense(&dir, "llama", Format::F16, 11, Dense::default());
    let (mut guard, _port, log) = start(&dir, &model);

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

/// A clean shutdown on Windows: the stop file drains and exits with status 0, an in-flight
/// request finishes, and a request made after the drain starts is refused.
///
/// Same behaviour as the `SIGTERM` test above, through the mechanism Windows has (S7.3). The
/// server watches `serve-<port>.stop` in its state directory every 250 ms.
#[cfg(windows)]
#[test]
fn a_stop_file_drains_and_exits_cleanly() {
    let dir = tmpdir("stopfile");
    let model = write_dense(&dir, "llama", Format::F16, 11, Dense::default());
    // Long enough that the in-flight request below is not cut off by the drain timeout.
    let (mut guard, port, log) = start_with_drain(&dir, &model, 20_000);
    let stop = dir.join("state").join(format!("serve-{port}.stop"));

    // A request that is still generating when the stop file appears.
    let body = r#"{"model":"peregrine","messages":[{"role":"user","content":"hi"}],"max_tokens":128,"temperature":0}"#;
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let in_flight = thread::spawn(move || http_request(port, &request));
    thread::sleep(Duration::from_millis(100));
    std::fs::write(&stop, b"stop\n").expect("write the stop file");

    // The server removes the file as it starts the drain, then closes the listener.
    let deadline = Instant::now() + Duration::from_secs(10);
    while stop.exists() {
        assert!(
            Instant::now() < deadline,
            "the server never noticed the stop file\nlog:\n{}",
            read_log(&log)
        );
        thread::sleep(Duration::from_millis(20));
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    while TcpStream::connect(("127.0.0.1", port)).is_ok() {
        assert!(
            Instant::now() < deadline,
            "a new connection was still accepted while draining\nlog:\n{}",
            read_log(&log)
        );
        thread::sleep(Duration::from_millis(20));
    }

    let finished = in_flight
        .join()
        .expect("the in-flight request thread")
        .unwrap_or_else(|| {
            panic!(
                "the in-flight request got no response\nlog:\n{}",
                read_log(&log)
            )
        });
    assert!(
        finished.starts_with("HTTP/1.1 200"),
        "the in-flight request was cut off: {finished:?}\nlog:\n{}",
        read_log(&log)
    );

    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = guard.0.try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "serve did not exit after the stop file\nlog:\n{}",
            read_log(&log)
        );
        thread::sleep(Duration::from_millis(50));
    };
    assert!(
        status.success(),
        "serve exited with {status} after the stop file\nlog:\n{}",
        read_log(&log)
    );
    assert!(
        !read_log(&log).contains("shutdown took longer"),
        "the drain hit the timeout\nlog:\n{}",
        read_log(&log)
    );
    assert!(
        !stop.exists(),
        "the server must remove the stop file it acted on"
    );
}

/// `serve --background`, `status` and `stop` on Windows (S7.3): the detached start records the
/// same state a Unix background server does, `status` reads it and gets `/health`, and `stop`
/// drains through the stop file; `status` then reports no managed server.
#[cfg(windows)]
#[test]
fn background_serve_status_and_stop_round_trip() {
    struct Managed {
        exe: PathBuf,
        state: PathBuf,
    }
    impl Drop for Managed {
        fn drop(&mut self) {
            let _ = Command::new(&self.exe)
                .arg("stop")
                .env("PEREGRINE_STATE_DIR", &self.state)
                .output();
        }
    }

    let dir = tmpdir("background");
    let model = write_dense(&dir, "llama", Format::F16, 11, Dense::default());
    let state = dir.join("state");
    std::fs::create_dir_all(&state).unwrap();
    let listen = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listen.local_addr().unwrap().port();
    drop(listen);
    let exe = PathBuf::from(env!("CARGO_BIN_EXE_peregrine"));
    let run = |args: &[&str]| {
        Command::new(&exe)
            .args(args)
            .env("PEREGRINE_STATE_DIR", &state)
            .output()
            .expect("run peregrine")
    };
    let port_text = port.to_string();

    let start = run(&[
        "serve",
        "-m",
        model.to_str().unwrap(),
        "--host",
        "127.0.0.1",
        "--port",
        &port_text,
        "--threads",
        "1",
        "-c",
        "128",
        "--no-profile",
        "--background",
    ]);
    assert!(
        start.status.success(),
        "serve --background failed: {}",
        String::from_utf8_lossy(&start.stderr)
    );
    let started = String::from_utf8_lossy(&start.stdout);
    assert!(
        started.contains("background server ready"),
        "the parent did not report readiness: {started}"
    );
    let _managed = Managed {
        exe: exe.clone(),
        state: state.clone(),
    };

    let status = run(&["status"]);
    assert!(
        status.status.success(),
        "status failed: {}",
        String::from_utf8_lossy(&status.stderr)
    );
    let text = String::from_utf8_lossy(&status.stdout);
    assert!(
        text.contains(&format!("127.0.0.1:{port}")),
        "status does not name the address: {text}"
    );
    assert!(
        text.contains("health     ok"),
        "status did not get /health: {text}"
    );

    let stop = run(&["stop"]);
    assert!(
        stop.status.success(),
        "stop failed: {}",
        String::from_utf8_lossy(&stop.stderr)
    );
    let stopped = String::from_utf8_lossy(&stop.stdout);
    assert!(
        stopped.contains("stopped pid"),
        "stop did not stop the server: {stopped}"
    );

    let after = run(&["status"]);
    assert!(
        !after.status.success(),
        "status still reports a server after stop"
    );
    assert!(
        String::from_utf8_lossy(&after.stderr).contains("no managed server"),
        "the stopped state was not removed: {}",
        String::from_utf8_lossy(&after.stderr)
    );
}
