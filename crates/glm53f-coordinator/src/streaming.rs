//! Streaming decode helpers for the `Engine` path: decode a pending token
//! buffer and decide what to emit now, holding back text that could be a stop
//! prefix. A pure function of the decoder, the stop list and the pending ids, so
//! the CPU tests cover it.

/// Decode `pending` (`decode`: token ids to bytes) and decide what to stream now.
/// Returns `(text_to_emit, stop)` where `stop` means a full stop sequence was hit
/// (the text before it is returned and the generation ends). When the pending text
/// ends with a *proper prefix* of a stop sequence (or is itself a proper prefix of
/// one), nothing is emitted (`""`) so the stop is held back and never emitted early.
pub fn flush_pending(decode: &dyn Fn(&[u32]) -> Vec<u8>, stop: &[String], pending: &mut Vec<u32>) -> (String, bool) {
    if pending.is_empty() {
        return (String::new(), false);
    }
    let bytes = decode(pending);
    // D7: hold back an incomplete UTF-8 tail (a character split across byte-level
    // BPE tokens) — emitting it now would turn the partial character into U+FFFD.
    if let Err(e) = std::str::from_utf8(&bytes) {
        if e.error_len().is_none() {
            return (String::new(), false);
        }
    }
    let text = String::from_utf8_lossy(&bytes).into_owned();
    // Full stop: the earliest stop sequence.
    let mut stop_at: Option<usize> = None;
    for s in stop {
        if !s.is_empty() {
            if let Some(pos) = text.find(s) {
                if stop_at.map_or(true, |p| pos < p) {
                    stop_at = Some(pos);
                }
            }
        }
    }
    if let Some(pos) = stop_at {
        let before = text[..pos].to_string();
        pending.clear();
        return (before, true);
    }
    // Partial stop: hold back if the pending text is a prefix of a stop sequence
    // (shorter) or ends with a proper prefix of one (char-level).
    let partial = stop.iter().any(|s| {
        if s.is_empty() {
            return false;
        }
        let chars: Vec<char> = s.chars().collect();
        let is_prefix_of_s = text.len() < s.len() && s.starts_with(&text);
        let ends_with_prefix = (0..chars.len().saturating_sub(1)).any(|i| {
            let prefix: String = chars[..=i].iter().collect();
            text.ends_with(&prefix)
        });
        is_prefix_of_s || ends_with_prefix
    });
    if partial {
        return (String::new(), false);
    }
    pending.clear();
    (text, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A decoder of id 0→"a", 1→"b", 2→"c", 3→"d".
    fn abc_tok() -> impl Fn(&[u32]) -> Vec<u8> {
        |ids: &[u32]| ids.iter().map(|&i| b"abcd"[i as usize]).collect()
    }

    #[test]
    fn no_stop_emits_everything() {
        let tok = abc_tok();
        let mut pending = vec![0, 1, 2]; // "abc"
        let (emit, stop) = flush_pending(&tok, &[], &mut pending);
        assert_eq!(emit, "abc");
        assert!(!stop);
        assert!(pending.is_empty());
    }

    #[test]
    fn full_stop_truncates_and_flags() {
        let tok = abc_tok();
        let mut pending = vec![0, 1, 2, 1, 2]; // "abcbc"
        let (emit, stop) = flush_pending(&tok, &["bc".to_string()], &mut pending);
        assert_eq!(emit, "a"); // text before the first "bc"
        assert!(stop);
        assert!(pending.is_empty());
    }

    #[test]
    fn partial_stop_is_held_back() {
        let tok = abc_tok();
        let mut pending = vec![0, 1]; // "ab", a proper prefix of stop "abc"
        let (emit, stop) = flush_pending(&tok, &["abc".to_string()], &mut pending);
        assert_eq!(emit, ""); // held back
        assert!(!stop);
        assert_eq!(pending, vec![0, 1]); // unchanged
    }

    /// A byte-level decoder: id `i` decodes to the one byte `bytes[i]`, so a
    /// multi-byte character arrives as several tokens.
    fn byte_tok(bytes: &[u8]) -> impl Fn(&[u32]) -> Vec<u8> {
        let table = bytes.to_vec();
        move |ids: &[u32]| ids.iter().map(|&i| table[i as usize]).collect()
    }

    /// D7: a CJK character (3 bytes) split across tokens must be held back until
    /// it completes, never emitted as U+FFFD.
    #[test]
    fn incomplete_utf8_tail_is_held_back() {
        // 東 = E6 9D B1.
        let tok = byte_tok(&[0xE6, 0x9D, 0xB1]);
        let mut pending = vec![0]; // first byte only
        let (emit, stop) = flush_pending(&tok, &[], &mut pending);
        assert_eq!(emit, "");
        assert!(!stop);
        assert_eq!(pending, vec![0]);
        pending.extend_from_slice(&[1, 2]);
        let (emit, _) = flush_pending(&tok, &[], &mut pending);
        assert_eq!(emit, "東");
        assert!(!emit.contains('\u{FFFD}'));
    }

    /// D7: a 4-byte emoji split across two tokens, then a stop sequence right
    /// after it, still works and never emits U+FFFD.
    #[test]
    fn emoji_split_across_tokens_then_stop() {
        // 😀 = F0 9F 98 80.
        let tok = byte_tok(&[0xF0, 0x9F, 0x98, 0x80, 0x61]); // emoji + "a"
        let mut pending = vec![0, 1, 2, 3]; // emoji complete, then no "a"
        let (emit, _) = flush_pending(&tok, &["a".to_string()], &mut pending);
        assert_eq!(emit, "😀");
        assert!(!emit.contains('\u{FFFD}'));
    }
}
