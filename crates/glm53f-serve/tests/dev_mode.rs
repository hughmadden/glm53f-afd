//! The daemon end to end in development mode (feature `cuda`): four `glm53f-rank` daemons on
//! loopback ports serving the EXL3 shares of layers 3 and 4, `glm53f-serve --dev-layers 0-4
//! --experts remote` in front of them, and one streamed chat completion through the HTTP API:
//! HTTP -> queue -> scheduler -> forward (layers 0-4, the MoE layers on the ranks) -> sampler ->
//! SSE. The text is meaningless by design (5 of 45 layers); the test checks the loop, not the
//! words.
//!
//! Needs `GLM53F_CHECKPOINT_DIR` (the coordinator's weights, with the tokenizer and the official
//! chat template), `GLM53F_RANK_BIN` (a `glm53f-rank` binary built with `--features cuda`) and
//! `GLM53F_RANK_DIRS` (the four rank directories cut with `glm53f-rank slice --layers 3-4`,
//! comma-separated in rank order), and about 13 GiB of free GPU memory; anything missing makes
//! it print why and pass. Every process it starts is stopped when it ends.
//!
//! ```sh
//! GLM53F_CHECKPOINT_DIR=... GLM53F_RANK_BIN=... GLM53F_RANK_DIRS=a,b,c,d \
//!   cargo test --release -p glm53f-serve --features cuda --test dev_mode -- --nocapture
//! ```
#![cfg(feature = "cuda")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use glm53f_api::json::{self, Json};

/// Child processes, stopped when dropped.
struct Procs {
    children: Vec<Child>,
    _out: Vec<BufReader<ChildStdout>>,
}

impl Drop for Procs {
    fn drop(&mut self) {
        for c in &mut self.children {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

fn env_path(k: &str) -> Option<PathBuf> {
    std::env::var_os(k).map(PathBuf::from)
}

/// Start the four rank daemons on loopback; their addresses in rank order.
fn spawn_ranks(bin: &PathBuf, dirs: &[PathBuf]) -> (Procs, Vec<String>) {
    let mut p = Procs {
        children: Vec::new(),
        _out: Vec::new(),
    };
    for (r, dir) in dirs.iter().enumerate() {
        p.children.push(
            Command::new(bin)
                .args(["serve", "--rank", &r.to_string(), "--dir"])
                .arg(dir)
                .args(["--listen", "127.0.0.1:0", "--allow-partial"])
                .env("GLM53F_WIRE_ALLOW_LAN", "1")
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .expect("spawn glm53f-rank"),
        );
    }
    let mut addrs = Vec::new();
    for r in 0..dirs.len() {
        let mut out = BufReader::new(p.children[r].stdout.take().unwrap());
        let mut line = String::new();
        loop {
            line.clear();
            if out.read_line(&mut line).unwrap_or(0) == 0 {
                panic!("rank {r} exited before listening (see its log above)");
            }
            if let Some(rest) = line.strip_prefix("listening on ") {
                addrs.push(rest.split(' ').next().unwrap().to_string());
                break;
            }
        }
        p._out.push(out);
    }
    (p, addrs)
}

/// A chunked HTTP/1.1 body, decoded.
fn dechunk(mut b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let eol = b
            .windows(2)
            .position(|w| w == b"\r\n")
            .expect("chunk size line");
        let n = usize::from_str_radix(std::str::from_utf8(&b[..eol]).unwrap().trim(), 16).unwrap();
        b = &b[eol + 2..];
        if n == 0 {
            return out;
        }
        out.extend_from_slice(&b[..n]);
        b = &b[n + 2..];
    }
}

fn delta_text(chunk: &Json) -> Option<String> {
    let d = chunk.get("choices")?.as_array()?.first()?.get("delta")?;
    let mut s = String::new();
    for k in ["reasoning", "reasoning_content", "content"] {
        if let Some(t) = d.get(k).and_then(|v| v.as_str()) {
            s.push_str(t);
        }
    }
    (!s.is_empty()).then_some(s)
}

#[test]
fn a_streamed_chat_completion_through_four_ranks_in_development_mode() {
    let (Some(ckpt), Some(bin)) = (
        env_path("GLM53F_CHECKPOINT_DIR"),
        env_path("GLM53F_RANK_BIN").filter(|p| p.is_file()),
    ) else {
        eprintln!("skip: GLM53F_CHECKPOINT_DIR or GLM53F_RANK_BIN is not set");
        return;
    };
    let dirs: Vec<PathBuf> = std::env::var("GLM53F_RANK_DIRS")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();
    if dirs.len() != 4 || !dirs.iter().all(|d| d.join("manifest.txt").is_file()) {
        eprintln!("skip: GLM53F_RANK_DIRS must name the four rank directories");
        return;
    }
    if glm53f_forward::device::device_count() == 0 {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (free, _) = glm53f_forward::device::mem_info().unwrap();
    if free < 13 << 30 {
        eprintln!(
            "skip: {:.1} GiB free on the GPU, the test needs 13",
            free as f64 / (1u64 << 30) as f64
        );
        return;
    }

    let (ranks, addrs) = spawn_ranks(&bin, &dirs);
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let api = format!("127.0.0.1:{port}");
    let mut serve = Procs {
        children: vec![Command::new(env!("CARGO_BIN_EXE_glm53f-serve"))
            .arg("--checkpoint")
            .arg(&ckpt)
            .args([
                "--dev-layers",
                "0-4",
                "--experts",
                "remote",
                "--ranks",
                &addrs.join(","),
            ])
            .args(["--listen", &api, "--slots", "2", "--max-context", "4096"])
            .args(["--kv-gib", "0.25", "--prefill-rows", "64"])
            .env("GLM53F_HOST_CACHE_GB", "0")
            .env("GLM53F_WIRE_ALLOW_LAN", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn glm53f-serve")],
        _out: Vec::new(),
    };
    // Its log, echoed and kept.
    let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let err = serve.children[0].stderr.take().unwrap();
    let l2 = log.clone();
    std::thread::spawn(move || {
        for line in BufReader::new(err).lines().map_while(Result::ok) {
            eprintln!("serve: {line}");
            l2.lock().unwrap().push(line);
        }
    });
    let t0 = Instant::now();
    while TcpStream::connect(&api).is_err() {
        if let Some(st) = serve.children[0].try_wait().unwrap() {
            panic!("glm53f-serve exited ({st}) before serving (see its log above)");
        }
        assert!(
            t0.elapsed() < Duration::from_secs(300),
            "glm53f-serve did not start serving in 300 s"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
    let ready = t0.elapsed().as_secs_f64();

    // One streamed chat completion.
    let body = r#"{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Name three colours."}],"stream":true,"max_tokens":12,"stream_options":{"include_usage":true}}"#;
    let mut s = TcpStream::connect(&api).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(300))).unwrap();
    let t1 = Instant::now();
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {api}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut resp = Vec::new();
    s.read_to_end(&mut resp).expect("read the response");
    let took = t1.elapsed().as_secs_f64();
    let split = resp
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("headers");
    let head = String::from_utf8_lossy(&resp[..split]).to_string();
    assert!(head.starts_with("HTTP/1.1 200"), "response: {head}");
    let raw = &resp[split + 4..];
    let text = if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        dechunk(raw)
    } else {
        raw.to_vec()
    };
    let text = String::from_utf8(text).expect("UTF-8 SSE");
    let events: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .collect();
    assert_eq!(
        events.last(),
        Some(&"[DONE]"),
        "the stream ends with [DONE]:\n{text}"
    );
    let chunks: Vec<Json> = events[..events.len() - 1]
        .iter()
        .map(|e| json::parse(e).expect("a JSON chunk"))
        .collect();
    let deltas: Vec<String> = chunks.iter().filter_map(delta_text).collect();
    let finish = chunks.iter().find_map(|c| {
        c.get("choices")?
            .as_array()?
            .first()?
            .get("finish_reason")?
            .as_str()
            .map(String::from)
    });
    let usage = chunks
        .iter()
        .find_map(|c| c.get("usage")?.get("completion_tokens")?.as_f64());
    eprintln!(
        "ready in {ready:.1} s; streamed completion in {took:.2} s: {} chunks, {} with text, finish_reason {finish:?}, completion_tokens {usage:?}; text (meaningless by design): {:?}",
        chunks.len(),
        deltas.len(),
        deltas.concat()
    );
    assert!(!deltas.is_empty(), "no text streamed:\n{text}");
    assert!(
        matches!(finish.as_deref(), Some("length") | Some("stop")),
        "finish_reason {finish:?}"
    );
    assert!(usage.is_some_and(|n| n >= 1.0), "usage {usage:?}");
    let log = log.lock().unwrap().join("\n");
    assert!(
        log.contains("DEVELOPMENT MODE"),
        "the development mode was not announced"
    );
    drop(serve);
    drop(ranks);
}
