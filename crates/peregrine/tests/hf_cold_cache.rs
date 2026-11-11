//! Cold-cache `hf:` download end to end, without the network.
//!
//! A local HTTP server serves a synthetic GGUF with `Range` support, and the built `peregrine`
//! binary downloads it through `PEREGRINE_HF_BASE`. This is the path that shipped broken in
//! v0.1.1: the first download latched the engine options, so the command that did the download
//! failed right after it.

#[path = "../../peregrine-models/tests/support/mod.rs"]
mod support;

use peregrine_core::gguf::Format;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use support::models::{write_dense_pooled, Dense};

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "peregrine-hf-cold-cache-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Serve `bytes` on 127.0.0.1 with `HEAD` and byte ranges, as a hub's CDN does.
fn serve(bytes: Arc<Vec<u8>>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
            let _ = answer(&mut stream, &bytes);
        }
    });
    port
}

fn answer(stream: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    let mut request = Vec::new();
    let mut buf = [0u8; 4096];
    while !request.windows(4).any(|w| w == b"\r\n\r\n") {
        let n = stream.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        request.extend_from_slice(&buf[..n]);
        if request.len() > 64 * 1024 {
            break;
        }
    }
    let text = String::from_utf8_lossy(&request);
    let mut lines = text.split("\r\n");
    let method = lines
        .next()
        .unwrap_or("")
        .split_whitespace()
        .next()
        .unwrap_or("");
    let range = lines.find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case("range").then(|| value.trim())
    });

    let len = bytes.len() as u64;
    let (status, start, end, body): (&str, u64, u64, &[u8]) = if method == "HEAD" {
        ("200 OK", 0, len.saturating_sub(1), &[])
    } else {
        match range.and_then(|r| r.strip_prefix("bytes=")) {
            Some(spec) => {
                let (a, b) = spec.split_once('-').unwrap_or((spec, ""));
                let start: u64 = if a.is_empty() {
                    0
                } else {
                    a.parse().unwrap_or(0)
                };
                if start >= len {
                    let head = "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    stream.write_all(head.as_bytes())?;
                    return Ok(());
                }
                let end = if b.is_empty() {
                    len - 1
                } else {
                    b.parse::<u64>().unwrap_or(len - 1).min(len - 1)
                };
                (
                    "206 Partial Content",
                    start,
                    end,
                    &bytes[start as usize..=end as usize],
                )
            }
            None => ("200 OK", 0, len.saturating_sub(1), bytes),
        }
    };

    let mut head = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n",
        body.len()
    );
    if status.starts_with("206") {
        head.push_str(&format!("Content-Range: bytes {start}-{end}/{len}\r\n"));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    if !body.is_empty() {
        stream.write_all(body)?;
    }
    stream.flush()
}

/// Kill the child even when an assertion fails first, so a failed test leaves no server behind.
struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn http_get(port: u16, path: &str) -> Option<String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(2))).ok()?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
    )
    .ok()?;
    let mut out = String::new();
    stream.read_to_string(&mut out).ok()?;
    Some(out)
}

fn wait_for_health(port: u16, timeout: Duration) -> Result<String, String> {
    let deadline = Instant::now() + timeout;
    let mut last = String::new();
    while Instant::now() < deadline {
        if let Some(response) = http_get(port, "/health") {
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

fn read_log(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

#[test]
fn a_cold_hf_download_runs_and_serves() {
    let dir = tmpdir("both");
    // Pooling type 3 (last) so the same cold file also serves `peregrine embed`.
    let model = write_dense_pooled(&dir, "llama", Format::F16, 11, Dense::default(), 3);
    let bytes = Arc::new(std::fs::read(&model).unwrap());
    let port = serve(bytes);
    let base = format!("http://127.0.0.1:{port}");
    let cache = dir.join("cache");
    let home = dir.join("home");
    std::fs::create_dir_all(&home).unwrap();
    let hf = "hf:test/model/model.gguf";
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_peregrine"));

    let env = |cmd: &mut Command| {
        cmd.env("PEREGRINE_HF_BASE", &base)
            .env("PEREGRINE_CACHE_DIR", &cache)
            .env("HOME", &home)
            .env_remove("XDG_CACHE_HOME")
            .env_remove("XDG_CONFIG_HOME")
            .env_remove("XDG_STATE_HOME");
    };

    // 1. `run` on a cold cache: this failed right after the download in v0.1.1.
    let mut run = Command::new(&bin);
    env(&mut run);
    let out = run
        .args([
            "run",
            "-m",
            hf,
            "--max-tokens",
            "8",
            "--no-profile",
            "--quiet",
            "hi",
        ])
        .output()
        .expect("run peregrine");
    assert!(
        out.status.success(),
        "cold-cache run failed ({}):\nstdout: {}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let cached = cache
        .join("models")
        .join("test")
        .join("model")
        .join("model.gguf");
    assert!(
        cached.is_file(),
        "the model was not cached at {}",
        cached.display()
    );
    assert!(!cached.with_extension("gguf.part").exists());

    // 2. `embed` on a cold cache: the command must set its options after the download, and the
    //    download must not have read them (the v0.1.2 failure mode).
    std::fs::remove_dir_all(cache.join("models")).expect("clear the cache for the cold embed");
    let mut embed = Command::new(&bin);
    env(&mut embed);
    let out = embed
        .args(["embed", "-m", hf, "--no-profile", "--dimensions", "8", "hi"])
        .output()
        .expect("run peregrine embed");
    assert!(
        out.status.success(),
        "cold-cache embed failed ({}):\nstdout: {}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let floats: Vec<&str> = stdout.split_whitespace().collect();
    assert_eq!(floats.len(), 8, "one truncated vector");
    assert!(cached.is_file(), "embed did not download the model");

    // 3. `serve` on a cold cache: the same first download must not panic before the model loads.
    std::fs::remove_dir_all(cache.join("models")).expect("clear the cache for the cold serve");
    let listen = TcpListener::bind("127.0.0.1:0").unwrap();
    let serve_port = listen.local_addr().unwrap().port();
    drop(listen);
    let log = dir.join("serve.log");
    let log_file = std::fs::File::create(&log).unwrap();
    let mut serve = Command::new(&bin);
    env(&mut serve);
    let child = serve
        .args([
            "serve",
            "-m",
            hf,
            "--host",
            "127.0.0.1",
            "--port",
            &serve_port.to_string(),
            "--no-profile",
        ])
        .stdout(Stdio::from(log_file.try_clone().unwrap()))
        .stderr(Stdio::from(log_file))
        .spawn()
        .expect("spawn peregrine serve");
    let mut guard = Guard(child);
    let health = wait_for_health(serve_port, Duration::from_secs(60)).unwrap_or_else(|e| {
        panic!(
            "cold-cache serve did not answer /health: {e}\nlog:\n{}",
            read_log(&log)
        )
    });
    assert!(
        health.starts_with("HTTP/1.1 200"),
        "health response: {health}"
    );
    assert!(
        cached.is_file(),
        "serve did not download the model to {}",
        cached.display()
    );

    // Drop kills and reaps the server, so the test leaves nothing running.
    let status = guard.0.try_wait().unwrap();
    assert!(
        status.is_none(),
        "serve exited before the check: {status:?}"
    );
}
