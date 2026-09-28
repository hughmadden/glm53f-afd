//! The pre-tokenizer: GLM-5.3-Flash's `Split` regex (behavior `Isolated`), hand-rolled.
//!
//! The pattern in tokenizer.json is
//!
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! ```
//!
//! (the cl100k form: digits in runs of one to three). The reference engine is a backtracking
//! matcher: at each position the first alternative that matches wins, and a greedy quantifier
//! gives back characters only as far as the rest of its alternative needs. [`split`] computes
//! the same pieces directly:
//!
//! 1. an apostrophe and a contraction letter or pair, case-insensitive; the regex's case folding
//!    also lets U+017F (long s) stand for `s` ([`contraction_folds`]);
//! 2. an optional character that is not CR, LF, a letter or a number, then a run of letters;
//! 3. one to three numbers;
//! 4. an optional space (U+0020), a run of characters that are not whitespace, letters or
//!    numbers, then any CR and LF characters;
//! 5. a whitespace run up to and including its last CR or LF;
//! 6. a whitespace run that ends the text, or all of a run but its last character when a
//!    non-space follows (that last character then prefixes the next piece);
//! 7. any other whitespace run (a single space before a non-space).
//!
//! The character classes are the reference's own ([`crate::unicode`]). The split applies to the
//! text between added tokens, never across one.

use crate::unicode::{is_letter, is_number, is_space};

/// The pre-tokenizer regex this module implements, as tokenizer.json spells it.
pub const PATTERN: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The characters each contraction letter matches under the regex's case-insensitive mode, as
/// probed from the reference (checked against unicode_classes.json by tests/goldens.rs).
pub fn contraction_folds() -> [(char, &'static [u32]); 8] {
    [
        ('s', &[0x53, 0x73, 0x17F]),
        ('t', &[0x54, 0x74]),
        ('m', &[0x4D, 0x6D]),
        ('d', &[0x44, 0x64]),
        ('r', &[0x52, 0x72]),
        ('e', &[0x45, 0x65]),
        ('v', &[0x56, 0x76]),
        ('l', &[0x4C, 0x6C]),
    ]
}

/// Whether `c` matches the contraction letter `letter` (lower-case ASCII).
fn folds_to(c: char, letter: char) -> bool {
    c.to_ascii_lowercase() == letter || (letter == 's' && c as u32 == 0x17F)
}

/// The character at byte `i` and its length.
#[inline]
fn at(s: &str, i: usize) -> Option<(char, usize)> {
    s[i..].chars().next().map(|c| (c, c.len_utf8()))
}

#[inline]
fn is_other(c: char) -> bool {
    !is_space(c) && !is_letter(c) && !is_number(c)
}

fn letters_from(s: &str, mut i: usize) -> usize {
    while let Some((c, l)) = at(s, i) {
        if !is_letter(c) {
            break;
        }
        i += l;
    }
    i
}

/// Alternative 1 after the apostrophe at `i - 1`: the end of the contraction, if one matches.
fn contraction(s: &str, i: usize) -> Option<usize> {
    let (c1, l1) = at(s, i)?;
    for letter in ['s', 't'] {
        if folds_to(c1, letter) {
            return Some(i + l1);
        }
    }
    let second = |first: char, then: char| -> Option<usize> {
        if !folds_to(c1, first) {
            return None;
        }
        let (c2, l2) = at(s, i + l1)?;
        folds_to(c2, then).then_some(i + l1 + l2)
    };
    if let Some(e) = second('r', 'e') {
        return Some(e);
    }
    if let Some(e) = second('v', 'e') {
        return Some(e);
    }
    if folds_to(c1, 'm') {
        return Some(i + l1);
    }
    if let Some(e) = second('l', 'l') {
        return Some(e);
    }
    if folds_to(c1, 'd') {
        return Some(i + l1);
    }
    None
}

/// The end (byte offset) of the piece that starts at byte `p` (`p < s.len()`).
fn piece_end(s: &str, p: usize) -> usize {
    let (c0, l0) = at(s, p).expect("p is inside the text");

    // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if c0 == '\'' {
        if let Some(e) = contraction(s, p + l0) {
            return e;
        }
    }

    // 2. [^\r\n\p{L}\p{N}]?\p{L}+
    if is_letter(c0) {
        return letters_from(s, p + l0);
    }
    if c0 != '\r' && c0 != '\n' && !is_number(c0) {
        if let Some((c1, l1)) = at(s, p + l0) {
            if is_letter(c1) {
                return letters_from(s, p + l0 + l1);
            }
        }
    }

    // 3. \p{N}{1,3}
    if is_number(c0) {
        let mut e = p + l0;
        for _ in 0..2 {
            match at(s, e) {
                Some((c, l)) if is_number(c) => e += l,
                _ => break,
            }
        }
        return e;
    }

    // 4. ' ?[^\s\p{L}\p{N}]+[\r\n]*'
    {
        let start = if c0 == ' ' { p + 1 } else { p };
        let mut e = start;
        while let Some((c, l)) = at(s, e) {
            if !is_other(c) {
                break;
            }
            e += l;
        }
        if e > start {
            while let Some((c, l)) = at(s, e) {
                if c != '\r' && c != '\n' {
                    break;
                }
                e += l;
            }
            return e;
        }
    }

    // 5-7: c0 is whitespace. Scan its run once.
    let mut end = p;
    let mut last_start = p;
    let mut after_last_newline = None;
    while let Some((c, l)) = at(s, end) {
        if !is_space(c) {
            break;
        }
        if c == '\r' || c == '\n' {
            after_last_newline = Some(end + l);
        }
        last_start = end;
        end += l;
    }
    debug_assert!(end > p, "every character is a letter, a number, whitespace or other");
    // 5. \s*[\r\n]+
    if let Some(e) = after_last_newline {
        return e;
    }
    // 6. \s+(?!\S)
    if end == s.len() {
        return end;
    }
    if last_start > p {
        return last_start;
    }
    // 7. \s+
    end
}

/// The pieces of `text`, as byte ranges covering it in order.
pub fn split(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut p = 0;
    while p < text.len() {
        let e = piece_end(text, p);
        out.push((p, e));
        p = e;
    }
    out
}

/// The pieces of `text`, as substrings.
pub fn split_str(text: &str) -> Vec<&str> {
    split(text).into_iter().map(|(a, b)| &text[a..b]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(cps: &[u32]) -> String {
        cps.iter().map(|&c| char::from_u32(c).unwrap()).collect()
    }

    #[test]
    fn ascii_shapes() {
        assert_eq!(split_str("Hello  world 12345"), ["Hello", " ", " world", " ", "123", "45"]);
        assert_eq!(split_str("I'm you're CAN'T"), ["I", "'m", " you", "'re", " CAN", "'T"]);
        assert_eq!(split_str("'salut 'tree"), ["'s", "alut", " '", "tree"]);
        assert_eq!(split_str("a\n\n b"), ["a", "\n\n", " b"]);
        assert_eq!(split_str("x \n \n  y"), ["x", " \n \n", " ", " y"]);
        assert_eq!(split_str("trailing   "), ["trailing", "   "]);
        assert_eq!(split_str("(really) ;:,."), ["(really", ")", " ;:,."]);
        assert_eq!(split_str("end.\n\nNext"), ["end", ".\n\n", "Next"]);
        assert_eq!(split_str(" \t \n \t\n  \r\n \t x"), [" \t \n \t\n  \r\n", " \t", " x"]);
        assert_eq!(split_str(""), Vec::<&str>::new());
    }

    #[test]
    fn non_ascii_shapes() {
        // Long s after an apostrophe is the 's contraction.
        let t = format!("it'{}x", s(&[0x17F]));
        assert_eq!(split_str(&t), ["it", &t[2..5], "x"]);
        // A no-break space prefixes a word; U+001C is not whitespace to the regex.
        let t = format!("a{}b{}c", s(&[0xA0]), s(&[0x1C]));
        assert_eq!(split_str(&t).len(), 3);
        // A combining mark ends a letter run and prefixes the next word.
        let t = format!("e{}x", s(&[0x301]));
        assert_eq!(split_str(&t), ["e", &t[1..]]);
    }

    #[test]
    fn pieces_cover_the_text() {
        let t = format!("{} {} ab12 \r\n\r\n x{}", s(&[0x4F60, 0x597D]), s(&[0x1F600]), s(&[0x3000]));
        let pieces = split(&t);
        let mut p = 0;
        for &(a, b) in &pieces {
            assert_eq!(a, p);
            assert!(b > a);
            p = b;
        }
        assert_eq!(p, t.len());
    }

    #[test]
    fn long_runs_are_linear() {
        let t = " ".repeat(200_000) + "x" + &"=".repeat(200_000) + &"\n".repeat(1000);
        let pieces = split_str(&t);
        assert_eq!(pieces.len(), 3);
        assert_eq!(pieces[0].len(), 199_999);
        assert_eq!(pieces[1], " x");
    }
}
