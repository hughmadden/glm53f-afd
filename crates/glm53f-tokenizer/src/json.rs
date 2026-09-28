//! A small JSON reader and a Python-compatible JSON writer (std-only).
//!
//! The chat template sees values the way Python's `json.loads` builds them, so the reader keeps
//! what that keeps:
//! - object key order; a duplicate key keeps its first position and takes its last value, as a
//!   dict built by `json.loads` does;
//! - integers as exact decimal text (Python integers are unbounded); `-0` reads as `0`;
//! - numbers with a fraction or an exponent as `f64` (Python floats);
//! - `NaN`, `Infinity` and `-Infinity`, which `json.loads` accepts.
//!
//! Like `json.loads` it rejects raw control characters inside strings. Lone surrogates are
//! rejected too, because a Rust string cannot hold them.
//!
//! [`to_python_json`] is `json.dumps(value, ensure_ascii=False)` with the default separators
//! (`", "` and `": "`): the template's `tojson` filter. [`py_float_repr`] is Python's `repr` of a
//! float, which `json.dumps` uses.

use std::collections::HashMap;
use std::fmt::Write as _;

/// A JSON value, as Python's `json.loads` would hold it.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// An integer, as canonical decimal text.
    Int(String),
    /// A number written with a fraction or an exponent.
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    /// Key/value pairs in document order, duplicate keys merged.
    Object(Vec<(String, Value)>),
}

impl Value {
    /// The value of `key` in an object.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_object()?.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn as_object(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Object(p) => Some(p),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }
    /// A non-negative integer that fits in `u64`.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Int(t) => t.parse().ok(),
            _ => None,
        }
    }
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }
    /// Python truthiness: `None`, `False`, zero and empty containers are false.
    pub fn truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Int(t) => t != "0",
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::Array(a) => !a.is_empty(),
            Value::Object(o) => !o.is_empty(),
        }
    }
}

/// Deepest nesting accepted (arrays and objects).
const MAX_DEPTH: usize = 512;

/// Objects with more keys than this index their keys while parsing (duplicate detection stays
/// linear for the 154,820-key vocabulary).
const INDEX_AFTER: usize = 16;

/// Parse one JSON document (surrounding whitespace allowed).
pub fn parse(text: &str) -> Result<Value, String> {
    let mut p = Parser { s: text.as_bytes(), i: 0, depth: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(p.err("trailing data"));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    depth: usize,
}

impl Parser<'_> {
    fn err(&self, what: &str) -> String {
        format!("json: {what} at byte {}", self.i)
    }
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
    fn literal(&mut self, word: &str, v: Value) -> Result<Value, String> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(self.err("bad literal"))
        }
    }
    fn value(&mut self) -> Result<Value, String> {
        match self.peek() {
            Some(b'{') => self.nested(Self::object),
            Some(b'[') => self.nested(Self::array),
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b't') => self.literal("true", Value::Bool(true)),
            Some(b'f') => self.literal("false", Value::Bool(false)),
            Some(b'n') => self.literal("null", Value::Null),
            Some(b'N') => self.literal("NaN", Value::Float(f64::NAN)),
            Some(b'I') => self.literal("Infinity", Value::Float(f64::INFINITY)),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.err("unexpected byte")),
            None => Err(self.err("unexpected end")),
        }
    }
    fn nested(&mut self, f: fn(&mut Self) -> Result<Value, String>) -> Result<Value, String> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.err("nesting too deep"));
        }
        let v = f(self);
        self.depth -= 1;
        v
    }
    fn object(&mut self) -> Result<Value, String> {
        self.i += 1; // {
        let mut pairs: Vec<(String, Value)> = Vec::new();
        let mut index: Option<HashMap<String, usize>> = None;
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Value::Object(pairs));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("expected a string key"));
            }
            let key = self.string()?;
            self.ws();
            if self.bump() != Some(b':') {
                return Err(self.err("expected ':'"));
            }
            self.ws();
            let v = self.value()?;
            let seen = match &index {
                Some(ix) => ix.get(&key).copied(),
                None => pairs.iter().position(|(k, _)| *k == key),
            };
            match seen {
                Some(at) => pairs[at].1 = v,
                None => {
                    if let Some(ix) = index.as_mut() {
                        ix.insert(key.clone(), pairs.len());
                    }
                    pairs.push((key, v));
                    if index.is_none() && pairs.len() > INDEX_AFTER {
                        index = Some(pairs.iter().enumerate().map(|(i, (k, _))| (k.clone(), i)).collect());
                    }
                }
            }
            self.ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b'}') => break,
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
        Ok(Value::Object(pairs))
    }
    fn array(&mut self) -> Result<Value, String> {
        self.i += 1; // [
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.ws();
            items.push(self.value()?);
            self.ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b']') => break,
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
        Ok(Value::Array(items))
    }
    fn text(&self, a: usize, b: usize) -> Result<&str, String> {
        std::str::from_utf8(&self.s[a..b]).map_err(|_| self.err("invalid UTF-8"))
    }
    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // opening quote
        let mut out = String::new();
        let mut run = self.i;
        loop {
            let Some(b) = self.peek() else {
                return Err(self.err("unterminated string"));
            };
            match b {
                b'"' => {
                    out.push_str(self.text(run, self.i)?);
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    out.push_str(self.text(run, self.i)?);
                    self.i += 1;
                    match self.bump() {
                        Some(b'"') => out.push('"'),
                        Some(b'\\') => out.push('\\'),
                        Some(b'/') => out.push('/'),
                        Some(b'b') => out.push(char::from(0x08u8)),
                        Some(b'f') => out.push(char::from(0x0cu8)),
                        Some(b'n') => out.push('\n'),
                        Some(b'r') => out.push('\r'),
                        Some(b't') => out.push('\t'),
                        Some(b'u') => out.push(self.escaped_char()?),
                        _ => return Err(self.err("bad escape")),
                    }
                    run = self.i;
                }
                0x00..=0x1f => return Err(self.err("control character in string")),
                _ => self.i += 1,
            }
        }
    }
    /// The code point of one escape (after its `u`), joining a surrogate pair.
    fn escaped_char(&mut self) -> Result<char, String> {
        let hi = self.hex4()?;
        let cp = if (0xD800..0xDC00).contains(&hi) {
            if self.bump() != Some(b'\\') || self.bump() != Some(b'u') {
                return Err(self.err("lone surrogate"));
            }
            let lo = self.hex4()?;
            if !(0xDC00..0xE000).contains(&lo) {
                return Err(self.err("lone surrogate"));
            }
            0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
        } else if (0xDC00..0xE000).contains(&hi) {
            return Err(self.err("lone surrogate"));
        } else {
            hi
        };
        char::from_u32(cp).ok_or_else(|| self.err("bad code point"))
    }
    fn hex4(&mut self) -> Result<u32, String> {
        let mut v = 0u32;
        for _ in 0..4 {
            let d = self.bump().and_then(|b| (b as char).to_digit(16)).ok_or_else(|| self.err("bad hex escape"))?;
            v = v * 16 + d;
        }
        Ok(v)
    }
    fn digits(&mut self) -> usize {
        let start = self.i;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        self.i - start
    }
    fn number(&mut self) -> Result<Value, String> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
            if self.s[self.i..].starts_with(b"Infinity") {
                self.i += 8;
                return Ok(Value::Float(f64::NEG_INFINITY));
            }
        }
        match self.peek() {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(self.err("bad number")),
        }
        let mut float = false;
        if self.peek() == Some(b'.') {
            self.i += 1;
            float = true;
            if self.digits() == 0 {
                return Err(self.err("bad number"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            float = true;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if self.digits() == 0 {
                return Err(self.err("bad number"));
            }
        }
        let text = self.text(start, self.i)?;
        if float {
            text.parse::<f64>().map(Value::Float).map_err(|_| self.err("bad number"))
        } else if text == "-0" {
            Ok(Value::Int("0".to_string()))
        } else {
            Ok(Value::Int(text.to_string()))
        }
    }
}

/// `json.dumps(value, ensure_ascii=False)` with the default separators.
pub fn to_python_json(v: &Value) -> String {
    let mut out = String::new();
    write_python_json(&mut out, v);
    out
}

/// Append `json.dumps(value, ensure_ascii=False)` to `out`.
pub fn write_python_json(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Int(t) => out.push_str(t),
        Value::Float(f) => out.push_str(&py_float_repr(*f)),
        Value::Str(s) => write_python_string(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_python_json(out, item);
            }
            out.push(']');
        }
        Value::Object(pairs) => {
            out.push('{');
            for (i, (k, item)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_python_string(out, k);
                out.push_str(": ");
                write_python_json(out, item);
            }
            out.push('}');
        }
    }
}

/// A JSON string as `json.dumps(..., ensure_ascii=False)` writes it: quote, backslash and the
/// control characters are escaped (the short forms for backspace, form feed, newline, carriage
/// return and tab; four hex digits for the rest); everything else is written as is.
pub fn write_python_string(out: &mut String, s: &str) {
    const BACKSLASH: char = '\\';
    out.push('"');
    for c in s.chars() {
        let cp = c as u32;
        match c {
            '"' | '\\' => {
                out.push(BACKSLASH);
                out.push(c);
            }
            '\n' => {
                out.push(BACKSLASH);
                out.push('n');
            }
            '\r' => {
                out.push(BACKSLASH);
                out.push('r');
            }
            '\t' => {
                out.push(BACKSLASH);
                out.push('t');
            }
            _ if cp == 0x08 => {
                out.push(BACKSLASH);
                out.push('b');
            }
            _ if cp == 0x0c => {
                out.push(BACKSLASH);
                out.push('f');
            }
            _ if cp < 0x20 => {
                out.push(BACKSLASH);
                out.push('u');
                let _ = write!(out, "{cp:04x}");
            }
            _ => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)` (what `json.dumps` writes for a float): the shortest digits that
/// round-trip; exponent form (`1e-05`, `1.5e+16`) when the decimal exponent is below -4 or at
/// least 16, else positional with at least one fractional digit (`100.0`). Non-finite values are
/// written as `json.dumps` writes them: `NaN`, `Infinity`, `-Infinity`.
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    // Rust's `{:e}` gives the shortest round-trip digits, as Python's repr does.
    let e = format!("{:e}", x.abs());
    let (mantissa, exp) = e.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("exponent");
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    let n = digits.len() as i32;
    // Python's `decpt`: the value is 0.<digits> * 10^decpt.
    let decpt = exp + 1;
    let mut out = String::new();
    if x < 0.0 {
        out.push('-');
    }
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        let _ = write!(out, "e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..-decpt {
            out.push('0');
        }
        out.push_str(&digits);
    } else if decpt >= n {
        out.push_str(&digits);
        for _ in 0..decpt - n {
            out.push('0');
        }
        out.push_str(".0");
    } else {
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A backslash built at run time, so this file never spells an escape sequence the
    /// tests are about (see tests/hygiene.rs).
    fn bs() -> String {
        char::from(0x5cu8).to_string()
    }

    #[test]
    fn keeps_order_integers_and_floats() {
        let v = parse(r#"{"b": 1, "a": 1.0, "c": -0, "d": 1e2, "e": 12345678901234567890123}"#).unwrap();
        let keys: Vec<&str> = v.as_object().unwrap().iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["b", "a", "c", "d", "e"]);
        assert_eq!(to_python_json(&v), r#"{"b": 1, "a": 1.0, "c": 0, "d": 100.0, "e": 12345678901234567890123}"#);
    }

    #[test]
    fn duplicate_keys_keep_first_position_and_last_value() {
        let v = parse(r#"{"a": 1, "b": 2, "a": 3}"#).unwrap();
        assert_eq!(to_python_json(&v), r#"{"a": 3, "b": 2}"#);
        // Past the indexing threshold too.
        let mut doc = String::from("{");
        for i in 0..40 {
            doc.push_str(&format!("\"k{i}\": {i}, "));
        }
        doc.push_str("\"k3\": 99}");
        let v = parse(&doc).unwrap();
        assert_eq!(v.as_object().unwrap().len(), 40);
        assert_eq!(v.get("k3"), Some(&Value::Int("99".into())));
        assert_eq!(v.as_object().unwrap()[3].0, "k3");
    }

    /// Escapes in, Python's escapes out. The JSON text is assembled at run time from a
    /// backslash and plain letters: if an editing tool had decoded an escape in this file, the
    /// input would no longer contain one and these assertions would fail.
    #[test]
    fn escapes_decode_and_encode_like_python() {
        let b = bs();
        let doc = format!("\"{b}u00e9{b}ud83d{b}ude00{b}n{b}t{b}b{b}f{b}u0001{b}{b}{b}\"{b}/\"");
        assert!(doc.is_ascii());
        let v = parse(&doc).unwrap();
        let expect: String = [0xe9u32, 0x1f600, 0x0a, 0x09, 0x08, 0x0c, 0x01, 0x5c, 0x22, 0x2f]
            .iter()
            .map(|&c| char::from_u32(c).unwrap())
            .collect();
        assert_eq!(v, Value::Str(expect.clone()));
        let out = to_python_json(&v);
        let want: String = [
            "\"".to_string(),
            char::from_u32(0xe9).unwrap().to_string(),
            char::from_u32(0x1f600).unwrap().to_string(),
            format!("{b}n{b}t{b}b{b}f{b}u0001{b}{b}{b}\"/\""),
        ]
        .concat();
        assert_eq!(out, want);
        // Lone surrogates and raw control characters are refused, as by json.loads (strict).
        assert!(parse(&format!("\"{b}ud800\"")).is_err());
        assert!(parse(&format!("\"{b}udc00x\"")).is_err());
        assert!(parse(&format!("\"a{}b\"", char::from(0x01u8))).is_err());
    }

    #[test]
    fn python_float_repr() {
        let cases: &[(f64, &str)] = &[
            (1.0, "1.0"),
            (0.5, "0.5"),
            (100.0, "100.0"),
            (1e-7, "1e-07"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (123456789.123, "123456789.123"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
            (2.5e-5, "2.5e-05"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
            (std::f64::consts::PI, "3.141592653589793"),
            (-3.25e-5, "-3.25e-05"),
            (1234567890123456789.0, "1.2345678901234568e+18"),
            (0.1 + 0.2, "0.30000000000000004"),
            (f64::INFINITY, "Infinity"),
            (f64::NEG_INFINITY, "-Infinity"),
        ];
        for (x, want) in cases {
            assert_eq!(py_float_repr(*x), *want, "{x:e}");
        }
        assert_eq!(py_float_repr(f64::NAN), "NaN");
    }

    #[test]
    fn python_literals_and_strict_numbers() {
        assert_eq!(to_python_json(&parse("[NaN, Infinity, -Infinity, 1E400, -1e-400]").unwrap()),
            "[NaN, Infinity, -Infinity, Infinity, -0.0]");
        for bad in ["01", "1.", ".5", "+1", "-", "1e", "[1,]", "{\"a\" 1}", "tru", "\"x"] {
            assert!(parse(bad).is_err(), "{bad} should not parse");
        }
    }

    #[test]
    fn truthiness_is_pythons() {
        for (doc, t) in [("null", false), ("false", false), ("0", false), ("0.0", false), ("\"\"", false),
            ("[]", false), ("{}", false), ("1", true), ("\"0\"", true), ("[0]", true), ("-0.5", true)] {
            assert_eq!(parse(doc).unwrap().truthy(), t, "{doc}");
        }
    }

    #[test]
    fn nesting_is_bounded() {
        let deep = "[".repeat(MAX_DEPTH + 1) + &"]".repeat(MAX_DEPTH + 1);
        assert!(parse(&deep).is_err());
        let ok = "[".repeat(MAX_DEPTH) + &"]".repeat(MAX_DEPTH);
        assert!(parse(&ok).is_ok());
    }
}
