//! The character classes of the pre-tokenizer regex: `\p{L}`, `\p{N}` and `\s`, as the reference
//! tokenizer's regex engine defines them.
//!
//! The tables are not derived from a Unicode database. They were probed from the reference
//! itself, one scalar value at a time: oracle/tokenizer_goldens.py records
//! oracle/goldens/tokenizer/unicode_classes.json, and examples/gen_unicode_tables.rs turns it into
//! `unicode_tables.rs`. With `tokenizers` 0.23.2 they are Unicode 16.0's General_Category L and
//! N, and White_Space (one version ahead of Python 3.12's `unicodedata`, which the probe
//! records for comparison). tests/goldens.rs checks the compiled tables against that file.

include!("unicode_tables.rs");

/// `\p{L}`: a letter.
pub fn is_letter(c: char) -> bool {
    let cp = c as u32;
    if cp < 0x80 {
        return c.is_ascii_alphabetic();
    }
    in_table(&LETTER, cp)
}

/// `\p{N}`: a number (decimal digit, letter number or other number).
pub fn is_number(c: char) -> bool {
    let cp = c as u32;
    if cp < 0x80 {
        return c.is_ascii_digit();
    }
    in_table(&NUMBER, cp)
}

/// `\s`: whitespace, the regex's definition (tab to carriage return, space, next line, and the
/// space, line and paragraph separators). Not Python's `str.isspace`, which adds U+001C..U+001F.
pub fn is_space(c: char) -> bool {
    let cp = c as u32;
    if cp < 0x80 {
        return matches!(cp, 0x09..=0x0D | 0x20);
    }
    in_table(&SPACE, cp)
}

fn in_table(table: &[(u32, u32)], cp: u32) -> bool {
    table
        .binary_search_by(|&(lo, hi)| {
            if hi < cp {
                std::cmp::Ordering::Less
            } else if lo > cp {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// The classes of every scalar value, as table rows (for the golden comparison).
pub fn class_tables() -> [(&'static str, &'static [(u32, u32)]); 3] {
    [("L", &LETTER), ("N", &NUMBER), ("S", &SPACE)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_sorted_and_disjoint() {
        for (name, t) in class_tables() {
            for w in t.windows(2) {
                assert!(w[0].0 <= w[0].1 && w[0].1 + 1 < w[1].0, "{name}: {:?} {:?}", w[0], w[1]);
            }
        }
        // The three classes never overlap.
        for cp in 0..=0x10FFFFu32 {
            if let Some(c) = char::from_u32(cp) {
                let n = [is_letter(c), is_number(c), is_space(c)].iter().filter(|&&x| x).count();
                assert!(n <= 1, "U+{cp:04X} is in {n} classes");
            }
        }
    }

    #[test]
    fn ascii_fast_path_agrees_with_the_tables() {
        for cp in 0..0x80u32 {
            let c = char::from_u32(cp).unwrap();
            assert_eq!(is_letter(c), in_table(&LETTER, cp), "L U+{cp:04X}");
            assert_eq!(is_number(c), in_table(&NUMBER, cp), "N U+{cp:04X}");
            assert_eq!(is_space(c), in_table(&SPACE, cp), "S U+{cp:04X}");
        }
    }

    #[test]
    fn spot_checks() {
        let ch = |cp: u32| char::from_u32(cp).unwrap();
        // Letters: CJK, Hangul, Arabic, a modifier letter, the long s.
        for cp in [0x4E00, 0xAC00, 0x0645, 0x02B0, 0x017F, 0x3005] {
            assert!(is_letter(ch(cp)), "U+{cp:04X}");
        }
        // Not letters: combining marks, the zero-width space, the byte order mark.
        for cp in [0x0301, 0x094D, 0x200B, 0xFEFF] {
            assert!(!is_letter(ch(cp)) && !is_number(ch(cp)) && !is_space(ch(cp)), "U+{cp:04X}");
        }
        // Numbers: superscript two, one half, a roman numeral, Arabic-Indic one.
        for cp in [0x00B2, 0x00BD, 0x216B, 0x0661] {
            assert!(is_number(ch(cp)), "U+{cp:04X}");
        }
        // Whitespace: no-break space, next line, ideographic space; U+001C is not.
        for cp in [0x00A0, 0x0085, 0x3000, 0x2028] {
            assert!(is_space(ch(cp)), "U+{cp:04X}");
        }
        assert!(!is_space(ch(0x1C)));
    }
}
