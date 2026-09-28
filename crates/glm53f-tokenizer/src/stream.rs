//! Streaming detokenization that never splits a character.
//!
//! Byte-level BPE tokens do not respect UTF-8: a CJK character or an emoji often spans two or
//! three tokens. [`Utf8Stream`] passes bytes through as text as soon as they are decided and holds
//! back only a trailing sequence that is still a valid prefix of a character. An invalid
//! sequence becomes one U+FFFD per maximal invalid subpart, exactly as `String::from_utf8_lossy`
//! (and the reference `decode`) writes it, so the concatenated deltas always equal the decode of
//! the whole sequence. A prefix still held at [`Utf8Stream::finish`] becomes one U+FFFD, as it
//! would at the end of a whole decode.
//!
//! [`StreamDecoder`] feeds it one token at a time from a [`Tokenizer`].

use crate::tokenizer::Tokenizer;

/// Bytes in, text out, without splitting a character.
#[derive(Debug, Default, Clone)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn new() -> Self {
        Utf8Stream::default()
    }

    /// Add bytes; returns the text they complete (possibly empty).
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        let mut start = 0;
        loop {
            match std::str::from_utf8(&self.pending[start..]) {
                Ok(s) => {
                    out.push_str(s);
                    start = self.pending.len();
                    break;
                }
                Err(e) => {
                    let valid = start + e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[start..valid]).expect("valid prefix"));
                    match e.error_len() {
                        // Decided: this subpart can never become valid.
                        Some(n) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            start = valid + n;
                        }
                        // A valid prefix of a character that the next bytes may complete.
                        None => {
                            start = valid;
                            break;
                        }
                    }
                }
            }
        }
        self.pending.drain(..start);
        out
    }

    /// End of stream: the text of any held prefix (U+FFFD), and reset.
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }

    /// Bytes held back, waiting for the rest of a character.
    pub fn held(&self) -> usize {
        self.pending.len()
    }
}

/// Token ids in, text deltas out.
#[derive(Debug)]
pub struct StreamDecoder<'t> {
    tok: &'t Tokenizer,
    skip_special: bool,
    utf8: Utf8Stream,
}

impl<'t> StreamDecoder<'t> {
    /// `skip_special` as in [`Tokenizer::decode`].
    pub fn new(tok: &'t Tokenizer, skip_special: bool) -> Self {
        StreamDecoder { tok, skip_special, utf8: Utf8Stream::new() }
    }

    /// The text `id` completes (possibly empty).
    pub fn push(&mut self, id: u32) -> String {
        match self.tok.piece_bytes(id, self.skip_special) {
            Some(b) => self.utf8.push(b),
            None => String::new(),
        }
    }

    /// End of stream (see [`Utf8Stream::finish`]).
    pub fn finish(&mut self) -> String {
        self.utf8.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator (xorshift), so the test needs no crate.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Any bytes, split anywhere: the deltas concatenate to `from_utf8_lossy` of the whole.
    #[test]
    fn deltas_equal_lossy_decode_of_the_whole() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let alphabet: Vec<u8> = {
            let mut a: Vec<u8> = b"ab <>".to_vec();
            // Lead and continuation bytes of 2-, 3- and 4-byte sequences, and bytes never valid.
            a.extend_from_slice(&[0xC3, 0xA9, 0xE4, 0xB8, 0xAD, 0xF0, 0x9F, 0x98, 0x80, 0x80, 0xBF, 0xC0, 0xC1, 0xF5, 0xFF, 0xED, 0xA0]);
            a
        };
        for _ in 0..5000 {
            let n = rng.below(24) as usize;
            let bytes: Vec<u8> = (0..n).map(|_| alphabet[rng.below(alphabet.len() as u64) as usize]).collect();
            let mut s = Utf8Stream::new();
            let mut out = String::new();
            let mut i = 0;
            while i < bytes.len() {
                let k = 1 + rng.below(4) as usize;
                let j = (i + k).min(bytes.len());
                out.push_str(&s.push(&bytes[i..j]));
                assert!(s.held() <= 3);
                i = j;
            }
            out.push_str(&s.finish());
            assert_eq!(out, String::from_utf8_lossy(&bytes), "{bytes:02x?}");
        }
    }

    #[test]
    fn holds_a_split_character_until_complete() {
        let mut s = Utf8Stream::new();
        let euro = [0xE2u8, 0x82, 0xAC];
        assert_eq!(s.push(&euro[..1]), "");
        assert_eq!(s.push(&euro[1..2]), "");
        assert_eq!(s.push(&euro[2..]), char::from_u32(0x20AC).unwrap().to_string());
        assert_eq!(s.push(&[0xE2, 0x82]), "");
        assert_eq!(s.push(b"x"), format!("{}x", char::REPLACEMENT_CHARACTER));
        assert_eq!(s.push(&[0xF0, 0x9F]), "");
        assert_eq!(s.finish(), char::REPLACEMENT_CHARACTER.to_string());
    }
}
