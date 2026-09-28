//! Source hygiene for this crate and its two oracle scripts.
//!
//! - **ASCII only.** An editing tool can decode an escape sequence into the character it stands
//!   for. A test that used to feed escaped input then silently feeds the literal character and
//!   still passes. Every file here keeps to ASCII (non-ASCII test text is built from code
//!   points), so a decoded escape fails this test.
//! - **No literal markup.** No file spells a complete think, tool-call, argument or tool-response
//!   tag, so tools that scan text for GLM's markup never misread these sources. The tags are
//!   assembled from pieces where they are needed.
//!
//! The golden JSON files are data written by the oracle scripts: ASCII, with angle brackets
//! escaped (as the MiMo parser goldens are), so they are held to the same two rules.

use std::path::{Path, PathBuf};

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn tags() -> Vec<String> {
    ["think", "tool_call", "arg_key", "arg_value", "tool_response"]
        .iter()
        .flat_map(|t| [format!("<{t}>"), format!("</{t}>")])
        .collect()
}

#[test]
fn sources_are_ascii_and_spell_no_markup() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let oracle = root.join("../../oracle");
    let mut sources = vec![
        root.join("Cargo.toml"),
        root.join("PROVENANCE.md"),
        oracle.join("tokenizer_goldens.py"),
        oracle.join("template_goldens.py"),
    ];
    for dir in ["src", "tests", "examples"] {
        collect(&root.join(dir), &mut sources);
    }
    let mut fails = Vec::new();
    for f in &sources {
        let text = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        if let Some(n) = text.lines().position(|l| !l.is_ascii()) {
            fails.push(format!("{}:{}: non-ASCII character", f.display(), n + 1));
        }
        for t in tags() {
            if let Some(n) = text.lines().position(|l| l.contains(t.as_str())) {
                fails.push(format!("{}:{}: literal markup tag", f.display(), n + 1));
            }
        }
    }
    let mut data = Vec::new();
    for sub in ["tokenizer", "template"] {
        collect(&oracle.join("goldens").join(sub), &mut data);
    }
    for f in &data {
        let bytes = std::fs::read(f).unwrap();
        if !bytes.is_ascii() {
            fails.push(format!("{}: non-ASCII golden file", f.display()));
        }
        let text = String::from_utf8_lossy(&bytes);
        if tags().iter().any(|t| text.contains(t.as_str())) {
            fails.push(format!("{}: literal markup tag in a golden file", f.display()));
        }
    }
    assert!(sources.len() >= 12 && data.len() >= 40, "{} sources, {} golden files", sources.len(), data.len());
    assert!(fails.is_empty(), "{fails:#?}");
}
