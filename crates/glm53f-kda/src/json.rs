//! A small JSON reader (RFC 8259) for fixture manifests. Objects keep their key order.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl Value {
    /// The member `key` of an object.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Number(x) => Some(*x),
            _ => None,
        }
    }

    /// A non-negative integer that is exactly representable.
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Value::Number(x) if *x >= 0.0 && x.fract() == 0.0 && *x < 9.007_199_254_740_992e15 => {
                Some(*x as usize)
            }
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Object(m) => Some(m),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Error {
    pub offset: usize,
    pub message: &'static str,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JSON: {} at byte {}", self.message, self.offset)
    }
}

impl std::error::Error for Error {}

/// Parse one JSON document.
pub fn parse(text: &str) -> Result<Value, Error> {
    let mut p = Parser {
        s: text.as_bytes(),
        i: 0,
        depth: 0,
    };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(p.err("trailing characters"));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    depth: usize,
}

impl Parser<'_> {
    fn err(&self, message: &'static str) -> Error {
        Error {
            offset: self.i,
            message,
        }
    }

    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &[u8]) -> bool {
        if self.s[self.i..].starts_with(lit) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<Value, Error> {
        if self.i >= self.s.len() {
            return Err(self.err("unexpected end"));
        }
        let c = self.s[self.i];
        match c {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Value::String(self.string()?)),
            b'-' | b'0'..=b'9' => self.number(),
            _ if self.eat(b"true") => Ok(Value::Bool(true)),
            _ if self.eat(b"false") => Ok(Value::Bool(false)),
            _ if self.eat(b"null") => Ok(Value::Null),
            _ => Err(self.err("unexpected character")),
        }
    }

    fn enter(&mut self) -> Result<(), Error> {
        self.depth += 1;
        if self.depth > 128 {
            return Err(self.err("nesting too deep"));
        }
        Ok(())
    }

    fn object(&mut self) -> Result<Value, Error> {
        self.enter()?;
        self.i += 1;
        let mut members = Vec::new();
        self.ws();
        if self.eat(b"}") {
            self.depth -= 1;
            return Ok(Value::Object(members));
        }
        loop {
            self.ws();
            if self.i >= self.s.len() || self.s[self.i] != b'"' {
                return Err(self.err("expected a key"));
            }
            let key = self.string()?;
            self.ws();
            if !self.eat(b":") {
                return Err(self.err("expected ':'"));
            }
            self.ws();
            let v = self.value()?;
            members.push((key, v));
            self.ws();
            if self.eat(b",") {
                continue;
            }
            if self.eat(b"}") {
                break;
            }
            return Err(self.err("expected ',' or '}'"));
        }
        self.depth -= 1;
        Ok(Value::Object(members))
    }

    fn array(&mut self) -> Result<Value, Error> {
        self.enter()?;
        self.i += 1;
        let mut items = Vec::new();
        self.ws();
        if self.eat(b"]") {
            self.depth -= 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.ws();
            items.push(self.value()?);
            self.ws();
            if self.eat(b",") {
                continue;
            }
            if self.eat(b"]") {
                break;
            }
            return Err(self.err("expected ',' or ']'"));
        }
        self.depth -= 1;
        Ok(Value::Array(items))
    }

    fn hex4(&mut self) -> Result<u32, Error> {
        if self.i + 4 > self.s.len() {
            return Err(self.err("short \\u escape"));
        }
        let h = std::str::from_utf8(&self.s[self.i..self.i + 4])
            .map_err(|_| self.err("bad \\u escape"))?;
        let v = u32::from_str_radix(h, 16).map_err(|_| self.err("bad \\u escape"))?;
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, Error> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.i < self.s.len() && self.s[self.i] != b'"' && self.s[self.i] != b'\\' {
                if self.s[self.i] < 0x20 {
                    return Err(self.err("control character in string"));
                }
                self.i += 1;
            }
            out.push_str(
                std::str::from_utf8(&self.s[start..self.i])
                    .map_err(|_| self.err("invalid UTF-8"))?,
            );
            if self.i >= self.s.len() {
                return Err(self.err("unterminated string"));
            }
            if self.s[self.i] == b'"' {
                self.i += 1;
                return Ok(out);
            }
            self.i += 1;
            if self.i >= self.s.len() {
                return Err(self.err("unterminated escape"));
            }
            let c = self.s[self.i];
            self.i += 1;
            match c {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                b'b' => out.push('\u{8}'),
                b'f' => out.push('\u{c}'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'u' => {
                    let hi = self.hex4()?;
                    let cp = if (0xd800..0xdc00).contains(&hi) {
                        if !self.eat(b"\\u") {
                            return Err(self.err("unpaired surrogate"));
                        }
                        let lo = self.hex4()?;
                        if !(0xdc00..0xe000).contains(&lo) {
                            return Err(self.err("unpaired surrogate"));
                        }
                        0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00)
                    } else {
                        hi
                    };
                    out.push(char::from_u32(cp).ok_or_else(|| self.err("invalid code point"))?);
                }
                _ => return Err(self.err("bad escape")),
            }
        }
    }

    fn number(&mut self) -> Result<Value, Error> {
        let start = self.i;
        if self.s[self.i] == b'-' {
            self.i += 1;
        }
        let digits = |p: &mut Self| {
            let d0 = p.i;
            while p.i < p.s.len() && p.s[p.i].is_ascii_digit() {
                p.i += 1;
            }
            p.i - d0
        };
        let int_start = self.i;
        if digits(self) == 0 {
            return Err(self.err("expected digits"));
        }
        if self.s[int_start] == b'0' && self.i - int_start > 1 {
            return Err(self.err("leading zero"));
        }
        if self.i < self.s.len() && self.s[self.i] == b'.' {
            self.i += 1;
            if digits(self) == 0 {
                return Err(self.err("expected fraction digits"));
            }
        }
        if self.i < self.s.len() && matches!(self.s[self.i], b'e' | b'E') {
            self.i += 1;
            if self.i < self.s.len() && matches!(self.s[self.i], b'+' | b'-') {
                self.i += 1;
            }
            if digits(self) == 0 {
                return Err(self.err("expected exponent digits"));
            }
        }
        let text =
            std::str::from_utf8(&self.s[start..self.i]).map_err(|_| self.err("bad number"))?;
        text.parse::<f64>()
            .map(Value::Number)
            .map_err(|_| self.err("bad number"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_manifest() {
        let v = parse(
            r#" { "kda.0.q": {"file": "q.bin", "dtype": "bf16", "shape": [2, 64, 128],
                  "sha256": "ab"}, "note": "caf\u00e9 \ud83d\ude00", "x": [true, false, null, -1.5e3] } "#,
        )
        .unwrap();
        let q = v.get("kda.0.q").unwrap();
        assert_eq!(q.get("file").unwrap().as_str(), Some("q.bin"));
        let shape: Vec<usize> = q
            .get("shape")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_usize().unwrap())
            .collect();
        assert_eq!(shape, [2, 64, 128]);
        assert_eq!(v.get("note").unwrap().as_str(), Some("café 😀"));
        assert_eq!(
            v.get("x").unwrap().as_array().unwrap()[3],
            Value::Number(-1500.0)
        );
        assert_eq!(v.as_object().unwrap()[0].0, "kda.0.q");
    }

    #[test]
    fn rejects_bad_json() {
        for bad in [
            "",
            "{",
            "[1,]",
            "{\"a\" 1}",
            "01",
            "1.",
            "\"\\x\"",
            "[1] 2",
            "\"\\ud800\"",
            "nul",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} should not parse");
        }
    }
}
