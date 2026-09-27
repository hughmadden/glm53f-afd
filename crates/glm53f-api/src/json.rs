//! A small, self-contained JSON codec for the HTTP API.
//!
//! Hand-rolled so `glm53f-api` stays std-only. The parser accepts the full JSON
//! grammar we need (objects, arrays, strings with `\uXXXX` escapes, int and
//! float numbers, `true`/`false`/`null`) and is lenient about object key order
//! (a chat payload is a client input, not an identity manifest). Serialization
//! preserves object insertion order and renders whole numbers without a
//! trailing `.0`.

use std::fmt::Write as _;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// A number (int or float); whole numbers serialize without `.0`.
    Num(f64),
    Str(String),
    Array(Vec<Json>),
    /// Key/value pairs in insertion order.
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(v) => Some(v),
            _ => None,
        }
    }
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object()?.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

pub fn parse(text: &str) -> Result<Json, String> {
    let mut p = P { s: text.as_bytes(), i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("json: trailing bytes at {}", p.i));
    }
    Ok(v)
}

/// Parse raw request bytes, rejecting invalid UTF-8 (a 400) instead of the
/// lossy U+FFFD substitution a `from_utf8_lossy` would silently apply (D6).
pub fn parse_bytes(bytes: &[u8]) -> Result<Json, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "json: invalid UTF-8".to_string())?;
    parse(text)
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> P<'a> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }
    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.i += 1;
        Some(b)
    }
    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }
    fn value(&mut self) -> Result<Json, String> {
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            Some(b'n') => self.lit("null", Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(format!("json: unexpected byte {:?} at {}", c as char, self.i)),
            None => Err("json: unexpected end".into()),
        }
    }
    fn lit(&mut self, word: &str, v: Json) -> Result<Json, String> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(format!("json: bad literal at {}", self.i))
        }
    }
    fn object(&mut self) -> Result<Json, String> {
        self.bump(); // {
        let mut out = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.bump();
            return Ok(Json::Object(out));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return Err(format!("json: expected string key at {}", self.i));
            }
            let key = self.string()?;
            self.ws();
            if self.bump() != Some(b':') {
                return Err(format!("json: expected ':' at {}", self.i));
            }
            self.ws();
            let v = self.value()?;
            out.push((key, v));
            self.ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b'}') => break,
                _ => return Err(format!("json: expected ',' or '}}' at {}", self.i)),
            }
        }
        Ok(Json::Object(out))
    }
    fn array(&mut self) -> Result<Json, String> {
        self.bump(); // [
        let mut out = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.bump();
            return Ok(Json::Array(out));
        }
        loop {
            self.ws();
            let v = self.value()?;
            out.push(v);
            self.ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b']') => break,
                _ => return Err(format!("json: expected ',' or ']' at {}", self.i)),
            }
        }
        Ok(Json::Array(out))
    }
    fn string(&mut self) -> Result<String, String> {
        self.bump(); // opening quote
        let mut out = String::new();
        // Raw (unescaped) bytes accumulate here and are validated as UTF-8 when
        // flushed (at a closing quote or an escape). This fixes D6: pushing each
        // byte as a char decoded UTF-8 as Latin-1 (mojibake for non-ASCII).
        let mut raw: Vec<u8> = Vec::new();
        // Flush pending raw bytes as one UTF-8 string; any invalid UTF-8 is a 400.
        fn flush_raw(out: &mut String, raw: &mut Vec<u8>) -> Result<(), String> {
            if raw.is_empty() {
                return Ok(());
            }
            let s = String::from_utf8(std::mem::take(raw)).map_err(|_| "json: invalid UTF-8 in string".to_string())?;
            out.push_str(&s);
            Ok(())
        }
        loop {
            match self.bump() {
                None => return Err("json: unterminated string".into()),
                Some(b'"') => {
                    flush_raw(&mut out, &mut raw)?;
                    break;
                }
                Some(b'\\') => {
                    flush_raw(&mut out, &mut raw)?;
                    let e = self.bump().ok_or("json: bad escape")?;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.unicode_escape()?),
                        other => return Err(format!("json: bad escape \\{}", other as char)),
                    }
                }
                Some(b) => raw.push(b),
            }
        }
        Ok(out)
    }
    /// Parse one `\uXXXX` escape as a code point, combining a UTF-16 surrogate
    /// pair (`\uD83D\uDE00`) into one scalar (D6). A lone or reversed surrogate
    /// is an error (a 400).
    fn unicode_escape(&mut self) -> Result<char, String> {
        let unit = self.hex4().map_err(|_| "json: bad \\u codepoint")?;
        let cp = if (0xD800..=0xDBFF).contains(&unit) {
            // High surrogate: the next two bytes must be the low half `\uXXXX`.
            if self.bump() != Some(b'\\') || self.bump() != Some(b'u') {
                return Err("json: lone high surrogate".into());
            }
            let low = self.hex4().map_err(|_| "json: lone high surrogate")?;
            if !(0xDC00..=0xDFFF).contains(&low) {
                return Err("json: lone high surrogate".into());
            }
            0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)
        } else if (0xDC00..=0xDFFF).contains(&unit) {
            return Err("json: lone low surrogate".into());
        } else {
            unit
        };
        char::from_u32(cp).ok_or_else(|| "json: bad \\u codepoint".to_string())
    }
    /// Parse four hex digits as a u32 (the value of one `\uXXXX` unit).
    fn hex4(&mut self) -> Result<u32, ()> {
        let mut hex = 0u32;
        for _ in 0..4 {
            let h = self.bump().ok_or(())?;
            let d = (h as char).to_digit(16).ok_or(())?;
            hex = hex * 16 + d;
        }
        Ok(hex)
    }
    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.bump();
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.i += 1;
        }
        if self.peek() == Some(b'.') {
            self.i += 1;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).map_err(|_| "json: bad number")?;
        text.parse::<f64>().map(Json::Num).map_err(|_| format!("json: bad number {text}"))
    }
}

pub fn serialize(v: &Json) -> String {
    let mut out = String::new();
    write_json(&mut out, v);
    out
}

fn write_json(out: &mut String, v: &Json) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => {
            if n.fract() == 0.0 && n.abs() < 9.0e15 {
                let _ = write!(out, "{}", *n as i64);
            } else {
                let _ = write!(out, "{n}");
            }
        }
        Json::Str(s) => write_string(out, s),
        Json::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(out, item);
            }
            out.push(']');
        }
        Json::Object(pairs) => {
            out.push('{');
            for (i, (k, val)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(out, k);
                out.push(':');
                write_json(out, val);
            }
            out.push('}');
        }
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let doc = r#"{"a": 1, "b": [true, null, "x\n"], "c": -2.5, "d": {"e": "f"}}"#;
        let v = parse(doc).unwrap();
        assert_eq!(v.get("a"), Some(&Json::Num(1.0)));
        assert_eq!(serialize(&parse(&serialize(&v)).unwrap()), serialize(&v));
    }

    #[test]
    fn whole_number_renders_without_dot() {
        assert_eq!(serialize(&Json::Num(3.0)), "3");
        assert_eq!(serialize(&Json::Num(0.5)), "0.5");
        assert_eq!(serialize(&Json::Num(-7.0)), "-7");
    }

    /// D6: raw non-ASCII bytes round-trip exactly (no Latin-1 mojibake).
    #[test]
    fn raw_utf8_round_trips_byte_exactly() {
        // CJK, accented latin, a 4-byte emoji, curly quotes, all unescaped.
        let s = "東京 café \u{1f600} \u{2019}\u{201c}\u{201d}";
        let doc = format!("{{\"s\":\"{s}\"}}");
        let v = parse(&doc).unwrap();
        assert_eq!(v.get("s").and_then(|x| x.as_str()), Some(s));
        // And the serialized form is byte-identical to the input.
        assert_eq!(serialize(&v), doc);
    }

    /// D6: an escaped BMP code point and an escaped surrogate pair decode to the
    /// same strings as their raw forms.
    #[test]
    fn escaped_bmp_and_surrogate_pair_decode() {
        // \u6771\u4eac == 東京; \uD83D\uDE00 == 😀 (a surrogate pair).
        let v = parse(r#"{"a":"\u6771\u4eac","b":"\uD83D\uDE00"}"#).unwrap();
        assert_eq!(v.get("a").and_then(|x| x.as_str()), Some("\u{6771}\u{4eac}"));
        assert_eq!(v.get("b").and_then(|x| x.as_str()), Some("\u{1f600}"));
        // Both decode to the same value as their raw literal.
        let raw = parse("{\"a\":\"東京\",\"b\":\"😀\"}").unwrap();
        assert_eq!(v.get("a"), raw.get("a"));
        assert_eq!(v.get("b"), raw.get("b"));
    }

    /// D6: a lone high surrogate, a lone low surrogate and a reversed pair are
    /// errors (a 400), never silently decoded.
    #[test]
    fn lone_and_reversed_surrogates_are_errors() {
        assert!(parse(r#""\uD800""#).is_err(), "lone high surrogate must fail");
        assert!(parse(r#""\uDC00""#).is_err(), "lone low surrogate must fail");
        assert!(parse(r#""\uDE00\uD83D""#).is_err(), "reversed pair must fail");
        assert!(parse(r#""\uD83D\u0041""#).is_err(), "high followed by non-low must fail");
    }

    /// D6: invalid UTF-8 bytes are an error (a 400), not a Latin-1 substitution.
    #[test]
    fn invalid_utf8_bytes_are_an_error() {
        // 0xFF is never valid UTF-8, even as a lead byte of a multi-byte char.
        let bad = [b'{', b'"', b's', b'"', b':', b'"', 0xFF, b'"', b'}'];
        assert!(parse_bytes(&bad).is_err(), "invalid UTF-8 must be rejected");
        // A truncated multi-byte sequence (a lone 0xE3 lead byte) is also invalid.
        let bad2 = [b'"', 0xE3, 0x81, b'"'];
        assert!(parse_bytes(&bad2).is_err(), "truncated UTF-8 must be rejected");
        // A valid multi-byte sequence at the raw level still parses.
        let ok = [b'"', 0xE6, 0x9D, 0xB1, b'"']; // 東
        let v = parse_bytes(&ok).unwrap();
        assert_eq!(v.as_str(), Some("\u{6771}"));
    }
}
