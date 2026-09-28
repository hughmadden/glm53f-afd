//! Source checks: the parity copy of the source kernels is still verbatim, and the Rust
//! bindings declare every function of the C header.

use std::path::Path;

fn read(rel: &str) -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)).unwrap()
}

/// The lines of `kernels/parity/tensorfold_kda.cu` between its BEGIN/END VERBATIM markers are
/// lines 1–11 and 14–218 of the source's `kda.cu` at the recorded commit (see PROVENANCE.md).
#[test]
fn parity_source_is_verbatim() {
    let text = read("kernels/parity/tensorfold_kda.cu");
    let lines: Vec<&str> = text.lines().collect();
    let begin = lines
        .iter()
        .position(|l| *l == "// BEGIN VERBATIM")
        .expect("BEGIN marker");
    let end = lines
        .iter()
        .position(|l| *l == "// END VERBATIM")
        .expect("END marker");
    let mut region = String::new();
    for l in &lines[begin + 1..end] {
        region.push_str(l);
        region.push('\n');
    }
    assert_eq!(
        end - begin - 1,
        11 + 205,
        "line count of the verbatim region"
    );
    assert_eq!(
        glm53f_kda::sha256::hex(region.as_bytes()),
        "0444df5018cf3b20b4ac9d612c834d71a80f866b6cc5dd6d3a1e0b260352e544"
    );
}

/// Every function the header declares is bound in `src/ffi.rs`, and nothing else.
#[test]
fn bindings_cover_the_header() {
    // The identifiers that follow each occurrence of `marker`.
    fn after(text: &str, marker: &str) -> Vec<String> {
        let mut v: Vec<String> = text
            .match_indices(marker)
            .map(|(i, _)| {
                let rest = &text[i + marker.len()..];
                rest.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .next()
                    .unwrap()
                    .to_string()
            })
            .collect();
        v.sort();
        v.dedup();
        v
    }
    let header = read("kernels/glm53f_kda.h");
    let mut declared = after(&header, "int glm53f_kda_");
    declared.extend(after(&header, "int64_t glm53f_kda_"));
    declared.sort();
    let bound: Vec<String> = after(&read("src/ffi.rs"), "pub fn glm53f_kda_")
        .into_iter()
        .filter(|n| !n.starts_with("parity_"))
        .collect();
    assert_eq!(declared.len(), 10, "{declared:?}");
    assert_eq!(declared, bound);
}
