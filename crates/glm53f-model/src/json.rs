//! A small, self-contained JSON codec.
//!
//! Copied from mimo26f-afd v1.2.0 `crates/mimo26-api/src/json.rs` (MIT); see
//! this crate's `PROVENANCE.md`. Changes from the source:
//!
//! - integer literals parse to [`Json::Int`], so byte offsets and dimensions
//!   are exact and a float where an integer is required is an error instead of
//!   a silent truncation;
//! - [`check_unique_keys`] rejects duplicate object keys (checkpoint headers
//!   and configs are read exactly or not at all);
//! - `as_u64`, `as_i64` and `Json::Int` serialization.
//!
//! The parser accepts objects, arrays, strings with `\uXXXX` escapes (surrogate
//! pairs combined, lone surrogates rejected), numbers, `true`, `false` and
//! `null`. Objects keep insertion order. Invalid UTF-8 is an error.

use std::fmt::Write as _;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// An integer literal (no fraction, no exponent) that fits in an `i64`.
    Int(i64),
    /// Any other number; whole numbers serialize without `.0`.
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
    /// The first value stored under `key` (see [`check_unique_keys`]).
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
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
    /// Any number, integer or not.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            Json::Int(n) => Some(*n as f64),
            _ => None,
        }
    }
    /// An integer literal.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Int(n) => Some(*n),
            _ => None,
        }
    }
    /// A non-negative integer literal.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Int(n) if *n >= 0 => Some(*n as u64),
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
    let mut p = P {
        s: text.as_bytes(),
        i: 0,
    };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("json: trailing bytes at {}", p.i));
    }
    Ok(v)
}

/// Parse raw bytes, rejecting invalid UTF-8 instead of substituting U+FFFD.
pub fn parse_bytes(bytes: &[u8]) -> Result<Json, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "json: invalid UTF-8".to_string())?;
    parse(text)
}

/// Fail if any object in `v` (at any depth) holds the same key twice.
pub fn check_unique_keys(v: &Json) -> Result<(), String> {
    match v {
        Json::Object(pairs) => {
            let mut keys: Vec<&str> = pairs.iter().map(|(k, _)| k.as_str()).collect();
            keys.sort_unstable();
            if let Some(w) = keys.windows(2).find(|w| w[0] == w[1]) {
                return Err(format!("json: duplicate key {:?}", w[0]));
            }
            pairs.iter().try_for_each(|(_, x)| check_unique_keys(x))
        }
        Json::Array(items) => items.iter().try_for_each(check_unique_keys),
        _ => Ok(()),
    }
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
            Some(c) => Err(format!(
                "json: unexpected byte {:?} at {}",
                c as char, self.i
            )),
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
        // flushed (at a closing quote or an escape).
        let mut raw: Vec<u8> = Vec::new();
        fn flush_raw(out: &mut String, raw: &mut Vec<u8>) -> Result<(), String> {
            if raw.is_empty() {
                return Ok(());
            }
            let s = String::from_utf8(std::mem::take(raw))
                .map_err(|_| "json: invalid UTF-8 in string".to_string())?;
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
    /// pair (`\uD83D\uDE00`) into one scalar. A lone or reversed surrogate is an
    /// error.
    fn unicode_escape(&mut self) -> Result<char, String> {
        let unit = self.hex4().map_err(|_| "json: bad \\u codepoint")?;
        let cp = if (0xD800..=0xDBFF).contains(&unit) {
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
        let mut integral = true;
        if self.peek() == Some(b'-') {
            self.bump();
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.i += 1;
        }
        if self.peek() == Some(b'.') {
            integral = false;
            self.i += 1;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            integral = false;
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).map_err(|_| "json: bad number")?;
        if integral {
            if let Ok(n) = text.parse::<i64>() {
                return Ok(Json::Int(n));
            }
        }
        text.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| format!("json: bad number {text}"))
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
        Json::Int(n) => {
            let _ = write!(out, "{n}");
        }
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
        assert_eq!(v.get("a"), Some(&Json::Int(1)));
        assert_eq!(v.get("c"), Some(&Json::Num(-2.5)));
        assert_eq!(serialize(&parse(&serialize(&v)).unwrap()), serialize(&v));
    }

    #[test]
    fn whole_number_renders_without_dot() {
        assert_eq!(serialize(&Json::Num(3.0)), "3");
        assert_eq!(serialize(&Json::Num(0.5)), "0.5");
        assert_eq!(serialize(&Json::Num(-7.0)), "-7");
        assert_eq!(serialize(&Json::Int(-7)), "-7");
    }

    /// Integers are exact far beyond 2^53 and never confused with floats.
    #[test]
    fn integers_are_exact_and_distinct_from_floats() {
        let v =
            parse(r#"[9007199254740993, 4096, 4096.0, 1e3, -1, 18446744073709551615]"#).unwrap();
        let a = v.as_array().unwrap();
        assert_eq!(a[0].as_u64(), Some(9_007_199_254_740_993));
        assert_eq!(a[1].as_u64(), Some(4096));
        assert_eq!(a[2].as_u64(), None, "4096.0 is not an integer literal");
        assert_eq!(a[2].as_f64(), Some(4096.0));
        assert_eq!(a[3].as_u64(), None);
        assert_eq!(a[4].as_u64(), None, "negative");
        assert_eq!(a[4].as_i64(), Some(-1));
        // Beyond i64: kept as a float, so it is not mistaken for an exact integer.
        assert_eq!(a[5].as_u64(), None);
        assert!(a[5].as_f64().is_some());
    }

    #[test]
    fn duplicate_keys_are_found_at_any_depth() {
        assert!(check_unique_keys(&parse(r#"{"a":1,"b":{"c":1,"d":2}}"#).unwrap()).is_ok());
        assert!(check_unique_keys(&parse(r#"{"a":1,"a":2}"#).unwrap()).is_err());
        assert!(check_unique_keys(&parse(r#"{"a":[{"x":1,"x":1}]}"#).unwrap()).is_err());
    }

    /// Raw non-ASCII bytes round-trip exactly (no Latin-1 mojibake).
    #[test]
    fn raw_utf8_round_trips_byte_exactly() {
        let s = "東京 café \u{1f600} \u{2019}\u{201c}\u{201d}";
        let doc = format!("{{\"s\":\"{s}\"}}");
        let v = parse(&doc).unwrap();
        assert_eq!(v.get("s").and_then(|x| x.as_str()), Some(s));
        assert_eq!(serialize(&v), doc);
    }

    #[test]
    fn escaped_bmp_and_surrogate_pair_decode() {
        let v = parse(r#"{"a":"\u6771\u4eac","b":"\uD83D\uDE00"}"#).unwrap();
        assert_eq!(
            v.get("a").and_then(|x| x.as_str()),
            Some("\u{6771}\u{4eac}")
        );
        assert_eq!(v.get("b").and_then(|x| x.as_str()), Some("\u{1f600}"));
        // Both decode to the same value as their raw literal.
        let raw = parse("{\"a\":\"東京\",\"b\":\"😀\"}").unwrap();
        assert_eq!(v.get("a"), raw.get("a"));
        assert_eq!(v.get("b"), raw.get("b"));
    }

    #[test]
    fn lone_and_reversed_surrogates_are_errors() {
        assert!(parse(r#""\uD800""#).is_err());
        assert!(parse(r#""\uDC00""#).is_err());
        assert!(parse(r#""\uDE00\uD83D""#).is_err());
        assert!(parse(r#""\uD83D\u0041""#).is_err());
    }

    #[test]
    fn invalid_utf8_bytes_are_an_error() {
        let bad = [b'{', b'"', b's', b'"', b':', b'"', 0xFF, b'"', b'}'];
        assert!(parse_bytes(&bad).is_err());
        let bad2 = [b'"', 0xE3, 0x81, b'"'];
        assert!(parse_bytes(&bad2).is_err());
        let ok = [b'"', 0xE6, 0x9D, 0xB1, b'"'];
        assert_eq!(parse_bytes(&ok).unwrap().as_str(), Some("\u{6771}"));
    }

    #[test]
    fn trailing_bytes_are_an_error() {
        assert!(parse("{} x").is_err());
        assert!(parse("[1,]").is_err());
    }
}
