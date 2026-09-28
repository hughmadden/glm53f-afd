//! GLM-5.3-Flash's tokenizer.json, reproduced: a byte-level BPE with 36 added tokens.
//!
//! Encoding follows the reference pipeline:
//! 1. **Added tokens** split the text first: at each position the longest added token that
//!    starts there wins, leftmost first (`tokenizers`' leftmost-longest match). Special and
//!    non-special added tokens are matched alike; none of GLM's strips spaces or needs a word
//!    boundary. There is no normalizer.
//! 2. **Pre-tokenizer:** each stretch between added tokens is split by the regex
//!    ([`crate::pretok`]), then byte-level mapped.
//! 3. **BPE:** with `ignore_merges` (set in GLM's file) a piece that is itself a vocabulary
//!    token is that token; otherwise its bytes are merged in the reference's order
//!    ([`crate::bpe`]).
//!
//! The ByteLevel post-processor adds nothing, so `add_special_tokens` does not change the ids.
//!
//! Decoding maps each model token back through the byte-level table and writes added tokens as
//! their text (special ones skipped on request). The bytes are then read as UTF-8, and an invalid
//! sequence becomes U+FFFD, as `String::from_utf8_lossy` and the reference do.
//! [`crate::stream::StreamDecoder`] does the same token by token.
//!
//! Loading refuses a tokenizer.json with anything this module does not implement: another
//! model type, a normalizer, a different pre-tokenizer, dropout, an unknown token, subword
//! affixes, byte fallback, or added tokens that strip spaces. It never tokenizes a different
//! file approximately.

use std::collections::HashMap;

use crate::bpe::{self, Merges};
use crate::json::{self, Value};
use crate::pretok;

/// `<|endoftext|>`, id 154,820: end of text, and the padding token.
pub const ENDOFTEXT: &str = concat!("<", "|endoftext|", ">");
/// `<|user|>`, id 154,827: the next user turn.
pub const USER: &str = concat!("<", "|user|", ">");
/// `<|observation|>`, id 154,829: tool results follow (the model stops here after its calls).
pub const OBSERVATION: &str = concat!("<", "|observation|", ">");

/// GLM-5.3-Flash's stop tokens (generation_config.json `eos_token_id`), in id order.
pub const STOP_TOKENS: [&str; 3] = [ENDOFTEXT, USER, OBSERVATION];
/// Their ids in GLM-5.3-Flash's tokenizer.json. [`Tokenizer::stop_ids`] resolves them from the
/// file itself; the tests check the two agree.
pub const STOP_IDS: [u32; 3] = [154_820, 154_827, 154_829];

/// One added token of tokenizer.json.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddedToken {
    pub id: u32,
    pub content: String,
    /// Special tokens are dropped by `decode(.., skip_special = true)`; the others (the think and
    /// tool-call tags among them) always decode as text.
    pub special: bool,
}

/// GPT-2's byte-to-character map: printable ASCII and most of Latin-1 map to themselves, the
/// other bytes to U+0100 onwards in byte order.
pub fn byte_chars() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let keep = (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        let cp = if keep {
            b
        } else {
            n += 1;
            0xFF + n
        };
        map[b as usize] = char::from_u32(cp).expect("valid scalar");
    }
    map
}

/// A loaded tokenizer. Immutable and `Sync`: share one across requests.
pub struct Tokenizer {
    /// Model vocabulary: byte-level token text -> id.
    vocab: HashMap<String, u32>,
    merges: Merges,
    /// The vocabulary id of each single byte.
    byte_ids: [u32; 256],
    byte_chars: [char; 256],
    /// Per id: the token text (byte-level form for model tokens, content for added tokens).
    tokens: Vec<Option<Box<str>>>,
    /// Per id: the bytes it decodes to.
    bytes: Vec<Option<Box<[u8]>>>,
    special: Vec<bool>,
    added: Vec<AddedToken>,
    /// Added-token indices by first byte, longest first.
    added_by_byte: Vec<Vec<usize>>,
    ignore_merges: bool,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab", &self.vocab.len())
            .field("merges", &self.merges.len())
            .field("added", &self.added.len())
            .field("ignore_merges", &self.ignore_merges)
            .finish()
    }
}

fn field<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.get(key).filter(|x| !x.is_null())
}

fn type_of(v: Option<&Value>) -> Option<&str> {
    v.and_then(|x| x.get("type")).and_then(|t| t.as_str())
}

impl Tokenizer {
    /// Load tokenizer.json from `path`.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Tokenizer, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Tokenizer::from_json(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Load a tokenizer from the text of a tokenizer.json.
    pub fn from_json(text: &str) -> Result<Tokenizer, String> {
        let doc = json::parse(text)?;
        let unsupported = |what: &str| Err(format!("unsupported tokenizer.json: {what}"));

        // The pipeline must be exactly the one implemented here.
        if field(&doc, "normalizer").is_some() {
            return unsupported("a normalizer");
        }
        let pre = field(&doc, "pre_tokenizer");
        let steps = match (type_of(pre), pre.and_then(|p| p.get("pretokenizers")).and_then(|s| s.as_array())) {
            (Some("Sequence"), Some(steps)) if steps.len() == 2 => steps,
            _ => return unsupported("pre_tokenizer is not Sequence[Split, ByteLevel]"),
        };
        let split = &steps[0];
        let pattern = split.get("pattern").and_then(|p| p.get("Regex")).and_then(|r| r.as_str());
        if type_of(Some(split)) != Some("Split")
            || pattern != Some(pretok::PATTERN)
            || split.get("behavior").and_then(|b| b.as_str()) != Some("Isolated")
            || split.get("invert").and_then(|i| i.as_bool()) != Some(false)
        {
            return unsupported("the Split pre-tokenizer differs from GLM-5.3-Flash's");
        }
        let byte_level = &steps[1];
        if type_of(Some(byte_level)) != Some("ByteLevel")
            || byte_level.get("add_prefix_space").and_then(|b| b.as_bool()) != Some(false)
            || byte_level.get("use_regex").and_then(|b| b.as_bool()) != Some(false)
        {
            return unsupported("the ByteLevel pre-tokenizer must not add a prefix space or split");
        }
        if !matches!(type_of(field(&doc, "decoder")), Some("ByteLevel")) {
            return unsupported("decoder is not ByteLevel");
        }
        if !matches!(type_of(field(&doc, "post_processor")), None | Some("ByteLevel")) {
            return unsupported("post_processor adds tokens");
        }
        let model = field(&doc, "model").ok_or("tokenizer.json has no model")?;
        if type_of(Some(model)) != Some("BPE") {
            return unsupported("model is not BPE");
        }
        if field(model, "dropout").is_some_and(|d| d.truthy()) {
            return unsupported("BPE dropout");
        }
        if field(model, "unk_token").is_some() {
            return unsupported("an unknown token");
        }
        for affix in ["continuing_subword_prefix", "end_of_word_suffix"] {
            if field(model, affix).and_then(|a| a.as_str()).is_some_and(|a| !a.is_empty()) {
                return unsupported(affix);
            }
        }
        if field(model, "byte_fallback").and_then(|b| b.as_bool()) == Some(true) {
            return unsupported("byte fallback");
        }
        let ignore_merges = field(model, "ignore_merges").and_then(|b| b.as_bool()).unwrap_or(false);

        // Vocabulary.
        let vocab_obj = field(model, "vocab").and_then(|v| v.as_object()).ok_or("model.vocab is not an object")?;
        let mut vocab = HashMap::with_capacity(vocab_obj.len());
        for (tok, id) in vocab_obj {
            let id = id.as_u64().filter(|&i| i <= u32::MAX as u64).ok_or_else(|| format!("vocab id of {tok:?}"))?;
            vocab.insert(tok.clone(), id as u32);
        }
        let byte_chars = byte_chars();
        let mut byte_ids = [0u32; 256];
        for (b, c) in byte_chars.iter().enumerate() {
            byte_ids[b] = *vocab.get(c.to_string().as_str()).ok_or_else(|| format!("byte {b:#04x} is not in the vocabulary"))?;
        }

        // Merges: [left, right] pairs (or "left right" strings); the later of two equal pairs wins,
        // as in the reference's map.
        let merges_arr = field(model, "merges").and_then(|m| m.as_array()).ok_or("model.merges is not an array")?;
        let mut merges = Merges::with_capacity(merges_arr.len());
        for (rank, m) in merges_arr.iter().enumerate() {
            let (a, b) = match m {
                Value::Array(p) if p.len() == 2 => (
                    p[0].as_str().ok_or("merge part is not a string")?,
                    p[1].as_str().ok_or("merge part is not a string")?,
                ),
                Value::Str(s) => s.split_once(' ').ok_or_else(|| format!("merge {s:?} is not a pair"))?,
                _ => return Err(format!("merge {rank} is not a pair")),
            };
            let id = |t: &str| vocab.get(t).copied().ok_or_else(|| format!("merge token {t:?} is not in the vocabulary"));
            let merged = format!("{a}{b}");
            merges.insert((id(a)?, id(b)?), (rank as u32, id(&merged)?));
        }

        // Added tokens.
        let mut added = Vec::new();
        for t in doc.get("added_tokens").and_then(|a| a.as_array()).unwrap_or(&[]) {
            let id = t.get("id").and_then(|i| i.as_u64()).filter(|&i| i <= u32::MAX as u64).ok_or("added token id")?;
            let content = t.get("content").and_then(|c| c.as_str()).ok_or("added token content")?;
            if content.is_empty() {
                return unsupported("an empty added token");
            }
            for flag in ["lstrip", "rstrip", "single_word"] {
                if t.get(flag).and_then(|f| f.as_bool()) == Some(true) {
                    return unsupported(&format!("added token {content:?} sets {flag}"));
                }
            }
            let special = t.get("special").and_then(|s| s.as_bool()).unwrap_or(false);
            added.push(AddedToken { id: id as u32, content: content.to_string(), special });
        }
        let mut added_by_byte: Vec<Vec<usize>> = vec![Vec::new(); 256];
        for (i, t) in added.iter().enumerate() {
            added_by_byte[t.content.as_bytes()[0] as usize].push(i);
        }
        for list in &mut added_by_byte {
            list.sort_by(|&x, &y| added[y].content.len().cmp(&added[x].content.len()));
        }

        // Per-id tables. Added tokens take precedence over a vocabulary entry with the same id.
        let bound = vocab.values().chain(added.iter().map(|t| &t.id)).map(|&i| i as usize + 1).max().unwrap_or(0);
        let mut tokens: Vec<Option<Box<str>>> = vec![None; bound];
        let mut bytes: Vec<Option<Box<[u8]>>> = vec![None; bound];
        let mut special = vec![false; bound];
        let mut from_char: HashMap<char, u8> = HashMap::with_capacity(256);
        for (b, c) in byte_chars.iter().enumerate() {
            from_char.insert(*c, b as u8);
        }
        for (tok, &id) in &vocab {
            // A token with a character outside the byte map decodes as its own UTF-8 bytes, whole,
            // as the reference's ByteLevel decoder does.
            let mapped: Option<Vec<u8>> = tok.chars().map(|c| from_char.get(&c).copied()).collect();
            bytes[id as usize] = Some(mapped.unwrap_or_else(|| tok.as_bytes().to_vec()).into_boxed_slice());
            tokens[id as usize] = Some(tok.clone().into_boxed_str());
        }
        for t in &added {
            bytes[t.id as usize] = Some(t.content.as_bytes().to_vec().into_boxed_slice());
            tokens[t.id as usize] = Some(t.content.clone().into_boxed_str());
            special[t.id as usize] = t.special;
        }

        Ok(Tokenizer { vocab, merges, byte_ids, byte_chars, tokens, bytes, special, added, added_by_byte, ignore_merges })
    }

    /// Token ids of `text` (`add_special_tokens` makes no difference for this tokenizer).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() / 3 + 1);
        self.encode_into(text, &mut ids);
        ids
    }

    /// Append the token ids of `text` to `ids`.
    pub fn encode_into(&self, text: &str, ids: &mut Vec<u32>) {
        let b = text.as_bytes();
        let (mut seg, mut i) = (0, 0);
        while i < b.len() {
            match self.added_at(b, i) {
                Some(t) => {
                    self.encode_segment(&text[seg..i], ids);
                    ids.push(self.added[t].id);
                    i += self.added[t].content.len();
                    seg = i;
                }
                // Added tokens start with a character's first byte, so stepping by bytes never
                // matches inside a character.
                None => i += 1,
            }
        }
        self.encode_segment(&text[seg..], ids);
    }

    /// Encode `text` in which markers `open ... close` (as the API writes images) stand for token
    /// spans: `expand` receives each marker's inner text and returns its ids. A marker splits the
    /// text as an added token does, so the ids equal those of the text with each marker replaced
    /// by the tokens `expand` returns, when markers sit between added tokens (as the chat template
    /// renders images: `<|begin_of_image|>` marker `<|end_of_image|>`). An unclosed marker is
    /// an error.
    pub fn encode_with_markers(
        &self,
        text: &str,
        open: char,
        close: char,
        expand: &mut dyn FnMut(&str) -> Result<Vec<u32>, String>,
    ) -> Result<Vec<u32>, String> {
        let mut ids = Vec::with_capacity(text.len() / 3 + 1);
        let mut rest = text;
        while let Some(p) = rest.find(open) {
            self.encode_into(&rest[..p], &mut ids);
            let inner = &rest[p + open.len_utf8()..];
            let q = inner.find(close).ok_or("unclosed marker")?;
            ids.extend(expand(&inner[..q])?);
            rest = &inner[q + close.len_utf8()..];
        }
        self.encode_into(rest, &mut ids);
        Ok(ids)
    }

    /// The longest added token that starts at byte `i`.
    fn added_at(&self, b: &[u8], i: usize) -> Option<usize> {
        self.added_by_byte[b[i] as usize].iter().copied().find(|&t| b[i..].starts_with(self.added[t].content.as_bytes()))
    }

    fn encode_segment(&self, seg: &str, ids: &mut Vec<u32>) {
        if seg.is_empty() {
            return;
        }
        let mut word = String::new();
        let mut syms = Vec::new();
        for (a, e) in pretok::split(seg) {
            let piece = &seg.as_bytes()[a..e];
            if self.ignore_merges {
                word.clear();
                word.extend(piece.iter().map(|&x| self.byte_chars[x as usize]));
                if let Some(&id) = self.vocab.get(word.as_str()) {
                    ids.push(id);
                    continue;
                }
            }
            syms.clear();
            syms.extend(piece.iter().map(|&x| self.byte_ids[x as usize]));
            bpe::merge(&syms, &self.merges, ids);
        }
    }

    /// The text of `ids`. `skip_special` drops special added tokens (the think and tool-call tags
    /// are not special and always appear). Unknown ids are skipped, as by the reference.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids, skip_special)).into_owned()
    }

    /// The bytes of `ids`, before UTF-8 decoding.
    pub fn decode_bytes(&self, ids: &[u32], skip_special: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            if let Some(b) = self.piece_bytes(id, skip_special) {
                out.extend_from_slice(b);
            }
        }
        out
    }

    /// The bytes one token decodes to; `None` for an unknown id or a skipped special token.
    pub fn piece_bytes(&self, id: u32, skip_special: bool) -> Option<&[u8]> {
        let i = id as usize;
        if skip_special && self.special.get(i).copied().unwrap_or(false) {
            return None;
        }
        self.bytes.get(i)?.as_deref()
    }

    /// Whether `id` is a special added token.
    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(false)
    }

    /// The token text of `id`: the byte-level form of a model token, or an added token's content.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize)?.as_deref()
    }

    /// The id of an added token's content, else of a model token's byte-level form.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.added.iter().find(|t| t.content == token).map(|t| t.id).or_else(|| self.vocab.get(token).copied())
    }

    /// The added tokens, in file order.
    pub fn added_tokens(&self) -> &[AddedToken] {
        &self.added
    }

    /// Number of model (BPE) tokens.
    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }

    /// Number of merges.
    pub fn merges_len(&self) -> usize {
        self.merges.len()
    }

    /// One past the largest token id (the logits the head must cover at least).
    pub fn id_bound(&self) -> usize {
        self.tokens.len()
    }

    /// The ids of [`STOP_TOKENS`] in this file.
    pub fn stop_ids(&self) -> Result<[u32; 3], String> {
        let mut out = [0u32; 3];
        for (slot, tok) in out.iter_mut().zip(STOP_TOKENS) {
            *slot = self.token_to_id(tok).ok_or_else(|| format!("stop token {tok} is not in this tokenizer"))?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tokenizer.json in GLM's shape over a toy vocabulary: the 256 byte characters, "ab",
    /// "abc", the byte-level form of " a", and two added tokens (one special).
    fn toy() -> Tokenizer {
        let chars = byte_chars();
        let mut vocab: Vec<(String, u32)> = chars.iter().enumerate().map(|(i, c)| (c.to_string(), i as u32)).collect();
        let g = chars[b' ' as usize];
        vocab.push(("ab".into(), 256));
        vocab.push(("abc".into(), 257));
        vocab.push((format!("{g}a"), 258));
        let vocab_json: Vec<String> = vocab.iter().map(|(t, i)| format!("{}: {i}", json::to_python_json(&Value::Str(t.clone())))).collect();
        let doc = format!(
            r#"{{"added_tokens": [
                {{"id": 300, "content": "{eot}", "special": true, "lstrip": false, "rstrip": false, "single_word": false}},
                {{"id": 301, "content": "{think}", "special": false}}],
              "normalizer": null,
              "pre_tokenizer": {{"type": "Sequence", "pretokenizers": [
                {{"type": "Split", "pattern": {{"Regex": {pattern}}}, "behavior": "Isolated", "invert": false}},
                {{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}}]}},
              "post_processor": {{"type": "ByteLevel"}},
              "decoder": {{"type": "ByteLevel"}},
              "model": {{"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": true,
                "vocab": {{{vocab}}}, "merges": [["a", "b"], ["ab", "c"], ["{g}", "a"]]}}}}"#,
            eot = ENDOFTEXT,
            think = concat!("<", "think", ">"),
            pattern = json::to_python_json(&Value::Str(pretok::PATTERN.to_string())),
            vocab = vocab_json.join(", "),
        );
        Tokenizer::from_json(&doc).unwrap()
    }

    #[test]
    fn toy_encode_decode() {
        let t = toy();
        let think = concat!("<", "think", ">");
        assert_eq!(t.encode("abc a"), [257, 258]);
        let text = format!("ab{think}ab{}", ENDOFTEXT);
        assert_eq!(t.encode(&text), [256, 301, 256, 300]);
        assert_eq!(t.decode(&t.encode(&text), false), text);
        assert_eq!(t.decode(&t.encode(&text), true), format!("ab{think}ab"));
        assert!(t.is_special(300) && !t.is_special(301));
        // Unknown ids are skipped; a lone continuation byte decodes to U+FFFD.
        assert_eq!(t.decode(&[9999, 0x80 - 0x21], false), "_");
        assert_eq!(t.decode(&[255], false), char::REPLACEMENT_CHARACTER.to_string());
    }

    #[test]
    fn refuses_what_it_does_not_implement() {
        let base = r#"{"normalizer": {"type": "NFC"}, "model": {"type": "BPE"}}"#;
        assert!(Tokenizer::from_json(base).unwrap_err().contains("normalizer"));
        let no_pre = r#"{"normalizer": null, "pre_tokenizer": null, "model": {"type": "BPE"}}"#;
        assert!(Tokenizer::from_json(no_pre).unwrap_err().contains("pre_tokenizer"));
    }

    #[test]
    fn byte_map_is_gpt2s() {
        let m = byte_chars();
        assert_eq!(m[b'A' as usize], 'A');
        assert_eq!(m[b' ' as usize] as u32, 0x120);
        assert_eq!(m[0] as u32, 0x100);
        assert_eq!(m[0xAD] as u32, 0x143);
        let mut sorted: Vec<char> = m.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 256);
    }
}
