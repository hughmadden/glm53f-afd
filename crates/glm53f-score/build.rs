//! Records what the scorer was built from, for the engine line it writes into every output:
//!
//! - `GLM53F_SCORE_REVISION`: the repository's commit (`git rev-parse HEAD`), with `-dirty` when
//!   the engine's sources (`crates/`, the workspace manifest and lock file) differ from it,
//!   untracked files included; `unknown` outside a git checkout;
//! - `GLM53F_SCORE_CUDA_ARCH`: the target the kernel crates compile for (`GLM53F_CUDA_ARCH`,
//!   default `sm_120`, as their build scripts read it).
//!
//! It reruns when a source under `crates/` changes or the checkout's `HEAD` moves.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .arg("--no-optional-locks")
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.join("../..");
    println!("cargo:rerun-if-changed=build.rs");
    for p in ["crates", "Cargo.toml", "Cargo.lock"] {
        println!("cargo:rerun-if-changed={}", root.join(p).display());
    }
    println!("cargo:rerun-if-env-changed=GLM53F_CUDA_ARCH");

    let revision = match git(&root, &["rev-parse", "HEAD"]) {
        Some(head) => {
            // The files that move HEAD: the checkout's HEAD, its branch ref and the packed refs.
            for what in ["HEAD", "packed-refs"] {
                if let Some(p) = git(&root, &["rev-parse", "--git-path", what]) {
                    println!("cargo:rerun-if-changed={}", root.join(p).display());
                }
            }
            if let Some(r) = git(&root, &["symbolic-ref", "-q", "HEAD"]) {
                if let Some(p) = git(&root, &["rev-parse", "--git-path", &r]) {
                    println!("cargo:rerun-if-changed={}", root.join(p).display());
                }
            }
            let dirty = git(
                &root,
                &[
                    "status",
                    "--porcelain",
                    "--",
                    "crates",
                    "Cargo.toml",
                    "Cargo.lock",
                ],
            )
            .is_none_or(|s| !s.is_empty());
            if dirty {
                format!("{head}-dirty")
            } else {
                head
            }
        }
        None => "unknown".to_string(),
    };
    println!("cargo:rustc-env=GLM53F_SCORE_REVISION={revision}");
    let arch = env::var("GLM53F_CUDA_ARCH").unwrap_or_else(|_| "sm_120".into());
    println!("cargo:rustc-env=GLM53F_SCORE_CUDA_ARCH={arch}");
}
