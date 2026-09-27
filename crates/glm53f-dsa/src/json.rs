//! A minimal JSON reader, enough for safetensors headers and fixture manifests.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    /// Object members in document order.
    Obj(Vec<(String, Json)>),
}

#[derive(Debug)]
pub struct JsonError {
    pub offset: usize,
    pub message: String,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "json error at byte {}: {}", self.offset, self.message)
    }
}

impl std::error::Error for JsonError {}

impl Json {
    pub fn parse(text: &str) -> Result<Json, JsonError> {
        let mut p = Parser { s: text.as_bytes(), i: 0 };
        p.ws();
        let v = p.value()?;
        p.ws();
        if p.i != p.s.len() {
            return Err(p.err("trailing characters"));
        }
        Ok(v)
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
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

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as u64),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(m) => Some(m),
            _ => None,
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn err(&self, m: &str) -> JsonError {
        JsonError { offset: self.i, message: m.to_string() }
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

    fn value(&mut self) -> Result<Json, JsonError> {
        match self.s.get(self.i) {
            None => Err(self.err("unexpected end")),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') if self.eat(b"true") => Ok(Json::Bool(true)),
            Some(b'f') if self.eat(b"false") => Ok(Json::Bool(false)),
            Some(b'n') if self.eat(b"null") => Ok(Json::Null),
            Some(c) if *c == b'-' || c.is_ascii_digit() => self.number(),
            Some(_) => Err(self.err("unexpected character")),
        }
    }

    fn object(&mut self) -> Result<Json, JsonError> {
        self.i += 1;
        let mut m = Vec::new();
        self.ws();
        if self.eat(b"}") {
            return Ok(Json::Obj(m));
        }
        loop {
            self.ws();
            if self.s.get(self.i) != Some(&b'"') {
                return Err(self.err("expected key"));
            }
            let k = self.string()?;
            self.ws();
            if !self.eat(b":") {
                return Err(self.err("expected ':'"));
            }
            self.ws();
            let v = self.value()?;
            m.push((k, v));
            self.ws();
            if self.eat(b",") {
                continue;
            }
            if self.eat(b"}") {
                return Ok(Json::Obj(m));
            }
            return Err(self.err("expected ',' or '}'"));
        }
    }

    fn array(&mut self) -> Result<Json, JsonError> {
        self.i += 1;
        let mut a = Vec::new();
        self.ws();
        if self.eat(b"]") {
            return Ok(Json::Arr(a));
        }
        loop {
            self.ws();
            a.push(self.value()?);
            self.ws();
            if self.eat(b",") {
                continue;
            }
            if self.eat(b"]") {
                return Ok(Json::Arr(a));
            }
            return Err(self.err("expected ',' or ']'"));
        }
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        if self.i + 4 > self.s.len() {
            return Err(self.err("short \\u escape"));
        }
        let t = std::str::from_utf8(&self.s[self.i..self.i + 4]).map_err(|_| self.err("bad \\u escape"))?;
        let v = u32::from_str_radix(t, 16).map_err(|_| self.err("bad \\u escape"))?;
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, JsonError> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.i < self.s.len() && self.s[self.i] != b'"' && self.s[self.i] != b'\\' {
                self.i += 1;
            }
            out.push_str(std::str::from_utf8(&self.s[start..self.i]).map_err(|_| self.err("invalid utf-8"))?);
            match self.s.get(self.i) {
                None => return Err(self.err("unterminated string")),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(_) => {
                    self.i += 1;
                    let c = *self.s.get(self.i).ok_or_else(|| self.err("bad escape"))?;
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
                            let mut cp = self.hex4()?;
                            if (0xD800..0xDC00).contains(&cp) && self.eat(b"\\u") {
                                let lo = self.hex4()?;
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo.wrapping_sub(0xDC00) & 0x3FF);
                            }
                            out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err(self.err("bad escape")),
                    }
                }
            }
        }
    }

    fn number(&mut self) -> Result<Json, JsonError> {
        let start = self.i;
        while self.i < self.s.len() && matches!(self.s[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
            self.i += 1;
        }
        let t = std::str::from_utf8(&self.s[start..self.i]).unwrap();
        t.parse::<f64>().map(Json::Num).map_err(|_| self.err("bad number"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_manifest_shape() {
        let j = Json::parse(
            r#"{"a": {"file": "a.bin", "dtype": "float32", "shape": [2, 3], "sha256": "ab"},
                "bé": [true, false, null, -1.5e3, "x\"y"]}"#,
        )
        .unwrap();
        let a = j.get("a").unwrap();
        assert_eq!(a.get("file").unwrap().as_str(), Some("a.bin"));
        let shape: Vec<u64> = a.get("shape").unwrap().as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        assert_eq!(shape, vec![2, 3]);
        let b = j.get("bé").unwrap().as_array().unwrap();
        assert_eq!(b[3].as_f64(), Some(-1500.0));
        assert_eq!(b[4].as_str(), Some("x\"y"));
        assert!(Json::parse("{\"a\":1,}").is_err());
    }
}
