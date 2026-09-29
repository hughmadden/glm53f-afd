//! Feature-gated CUDA build (same pattern as the MiMo-V2.6-Flash engine's
//! attention crate; see PROVENANCE.md).
//!
//! `cargo test` with default features compiles nothing here and needs no CUDA
//! toolkit. With `--features cuda`, the kernel translation units are compiled by
//! nvcc into a static archive and linked with cudart.
//!
//! Environment:
//!   GLM53F_NVCC       nvcc path (default /usr/local/cuda/bin/nvcc)
//!   GLM53F_CUDA_ARCH  target: sm_120 (default; the RTX 5090 coordinator) or sm_89 (an
//!                     RTX 4090, the development GPU)
//!   GLM53F_CUDA_LIB   directory holding libcudart (default: next to nvcc, ../lib64)

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for key in ["GLM53F_NVCC", "GLM53F_CUDA_ARCH", "GLM53F_CUDA_LIB"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest.join("kernels");
    let include = kernels_dir.join("include");
    let units = ["dsa_index.cu", "dsa_mla.cu"];
    for u in &units {
        println!("cargo:rerun-if-changed={}", kernels_dir.join(u).display());
    }
    for h in ["glm53f_dsa.h", "glm53f_dsa_common.cuh"] {
        println!("cargo:rerun-if-changed={}", include.join(h).display());
    }

    let nvcc = PathBuf::from(env::var("GLM53F_NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".into()));
    let arch = env::var("GLM53F_CUDA_ARCH").unwrap_or_else(|_| "sm_120".into());
    let cuda_lib = env::var("GLM53F_CUDA_LIB").map(PathBuf::from).unwrap_or_else(|_| {
        nvcc.parent().and_then(|b| b.parent()).map(|r| r.join("lib64")).unwrap_or_else(|| PathBuf::from("/usr/local/cuda/lib64"))
    });

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut objs = Vec::new();
    for u in &units {
        let src = kernels_dir.join(u);
        let obj = out.join(format!("{}.o", src.file_stem().unwrap().to_string_lossy()));
        let status = Command::new(&nvcc)
            .args(["-O3", "-std=c++17", "-lineinfo", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"])
            .arg(format!("-arch={arch}"))
            .arg("-Xcompiler=-fPIC")
            .arg(format!("-I{}", include.display()))
            .arg("-c")
            .arg(&src)
            .arg("-o")
            .arg(&obj)
            .status()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));
        assert!(status.success(), "nvcc failed on {}", src.display());
        objs.push(obj);
    }
    let lib = out.join("libglm53f_dsa_kernels.a");
    let _ = std::fs::remove_file(&lib);
    let status = Command::new("ar").arg("rcs").arg(&lib).args(&objs).status().expect("failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=glm53f_dsa_kernels");
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
