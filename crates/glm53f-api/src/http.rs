//! A minimal HTTP/1.1 server (std-only): one thread per connection, keep-alive
//! (request/response reuse until the client closes), and a plain request/response
//! pair handed to a routing closure. Good enough for the OpenAI API; not
//! a general-purpose server.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

/// A parsed HTTP request.
#[derive(Debug)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

/// An HTTP response body: either fully-buffered bytes, or a streaming writer
/// that emits the body directly to the socket.
pub enum ResponseBody {
    Bytes(Vec<u8>),
    /// Streaming: the closure writes the body events directly to the socket
    /// (the server frames them with chunked encoding). Called once.
    Stream(Box<dyn FnOnce(&mut TcpStream) -> std::io::Result<()> + Send>),
}

pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub body: ResponseBody,
}

pub fn json_response(status: u16, body: &str) -> Response {
    Response { status, content_type: "application/json", body: ResponseBody::Bytes(body.as_bytes().to_vec()) }
}

pub fn sse_response(body: &str) -> Response {
    Response { status: 200, content_type: "text/event-stream", body: ResponseBody::Bytes(body.as_bytes().to_vec()) }
}

/// A streaming SSE response: the closure writes each event directly to the
/// socket (chunked-encoded by the server, flushed per write).
pub fn sse_stream_response(body: Box<dyn FnOnce(&mut TcpStream) -> std::io::Result<()> + Send>) -> Response {
    Response { status: 200, content_type: "text/event-stream", body: ResponseBody::Stream(body) }
}

/// Write one chunked-encoding frame (`<hex-len>\r\n<data>\r\n`), flushed.
pub fn write_chunk(stream: &mut TcpStream, data: &[u8]) -> std::io::Result<()> {
    write!(stream, "{:x}\r\n", data.len())?;
    stream.write_all(data)?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}

fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
}

fn read_request(stream: &mut BufReader<TcpStream>) -> std::io::Result<Option<Request>> {
    let mut line = String::new();
    if stream.read_line(&mut line)? == 0 {
        return Ok(None); // peer closed
    }
    let line = line.trim_end_matches(['\r', '\n']);
    let mut parts = line.split(' ');
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let mut headers = Vec::new();
    loop {
        let mut h = String::new();
        stream.read_line(&mut h)?;
        let h = h.trim_end_matches(['\r', '\n']);
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    let len = headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse::<usize>().ok()).unwrap_or(0);
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body)?;
    Ok(Some(Request { method, path, headers, body }))
}

fn write_response(stream: &mut TcpStream, resp: Response, keep_alive: bool) -> std::io::Result<()> {
    let conn = if keep_alive { "keep-alive" } else { "close" };
    match resp.body {
        ResponseBody::Bytes(body) => {
            // A 429 (the engine's queue is full, perf reset V3) asks the client to retry in a second.
            let retry = if resp.status == 429 { "Retry-After: 1\r\n" } else { "" };
            // A 401 (no valid API key) names the scheme that would do (RFC 9110, 11.6.1).
            let challenge = if resp.status == 401 { "WWW-Authenticate: Bearer\r\n" } else { "" };
            let head = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\n{retry}{challenge}Connection: {}\r\n\r\n",
                resp.status, status_reason(resp.status), resp.content_type, body.len(), conn,
            );
            stream.write_all(head.as_bytes())?;
            stream.write_all(&body)?;
            stream.flush()
        }
        ResponseBody::Stream(write_body) => {
            // No Content-Length: stream the body with chunked encoding, sending
            // the head first so the first event reaches the client immediately.
            let head = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nTransfer-Encoding: chunked\r\nConnection: {}\r\n\r\n",
                resp.status, status_reason(resp.status), resp.content_type, conn,
            );
            stream.write_all(head.as_bytes())?;
            stream.flush()?;
            write_body(stream)?;
            stream.write_all(b"0\r\n\r\n")?;
            stream.flush()
        }
    }
}

/// Serve `handler` on `addr`. Each connection is handled on its own thread with
/// keep-alive until the client closes or sends `Connection: close`.
pub fn serve<F>(addr: &str, handler: F) -> std::io::Result<()>
where
    F: Fn(Request) -> Response + Send + Sync + 'static,
{
    serve_listener(TcpListener::bind(addr)?, handler)
}

/// Serve an already-bound listener (lets tests bind an ephemeral port).
pub fn serve_listener<F>(listener: TcpListener, handler: F) -> std::io::Result<()>
where
    F: Fn(Request) -> Response + Send + Sync + 'static,
{
    let handler = std::sync::Arc::new(handler);
    for conn in listener.incoming() {
        let stream = match conn {
            Ok(s) => s,
            Err(_) => continue,
        };
        let handler = std::sync::Arc::clone(&handler);
        std::thread::spawn(move || {
            let _ = handle_conn(stream, move |r| handler(r));
        });
    }
    Ok(())
}

fn handle_conn<F>(stream: TcpStream, handler: F) -> std::io::Result<()>
where
    F: Fn(Request) -> Response,
{
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    loop {
        let Some(req) = read_request(&mut reader)? else { break };
        let keep_alive = !req.header("connection").map(|c| c.eq_ignore_ascii_case("close")).unwrap_or(false);
        let resp = handler(req);
        write_response(&mut writer, resp, keep_alive)?;
        if !keep_alive {
            break;
        }
    }
    Ok(())
}
