//! Builds the libibverbs shim (`native/rdma.c`) under the `rdma` feature and
//! links the system `libibverbs.so.1` (through an OUT_DIR symlink, since hosts
//! carry the runtime library without the `-dev` package).

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=native/rdma.c");
    println!("cargo:rerun-if-env-changed=CC");
    if env::var_os("CARGO_FEATURE_RDMA").is_none() {
        return;
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cc = env::var("CC").unwrap_or_else(|_| "cc".into());
    let obj = out.join("rdma.o");
    let status = Command::new(&cc)
        .args(["-O2", "-fPIC", "-std=gnu11", "-Wall", "-c"])
        .arg(format!("-I{}", manifest.join("native/include").display()))
        .arg(manifest.join("native/rdma.c"))
        .arg("-o")
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| panic!("failed to run {cc}: {e}"));
    assert!(status.success(), "compile of native/rdma.c failed");
    let lib = out.join("libglm53f_rdma.a");
    let status = Command::new("ar").arg("rcs").arg(&lib).arg(&obj).status().expect("failed to run ar");
    assert!(status.success(), "ar failed");
    let candidates = [
        "/usr/lib/x86_64-linux-gnu/libibverbs.so.1",
        "/usr/lib/aarch64-linux-gnu/libibverbs.so.1",
        "/usr/lib64/libibverbs.so.1",
        "/usr/lib/libibverbs.so.1",
    ];
    let so = candidates
        .iter()
        .find(|p| std::path::Path::new(p).exists())
        .expect("libibverbs.so.1 not found (install rdma-core)");
    let link = out.join("libibverbs.so");
    let _ = std::fs::remove_file(&link);
    std::os::unix::fs::symlink(so, &link).expect("symlink libibverbs.so");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=glm53f_rdma");
    println!("cargo:rustc-link-lib=dylib=ibverbs");
}
