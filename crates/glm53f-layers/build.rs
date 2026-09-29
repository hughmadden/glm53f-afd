//! Builds the layer kernels when the `cuda` feature is on.
//!
//! Default builds (and `cargo test`) skip this entirely, so they need no CUDA toolkit.
//! With `--features cuda` this compiles `kernels/*.cu` into a static archive and links it
//! together with `cudart`.
//!
//! Environment:
//! - `GLM53F_NVCC`: the nvcc to run (default `/usr/local/cuda/bin/nvcc`);
//! - `GLM53F_CUDA_ARCH`: the target (default `sm_120`, the RTX 5090 coordinator; set `sm_89` to
//!   build for an RTX 4090, the development GPU);
//! - `GLM53F_CUDA_LIB`: the directory holding `libcudart` (default `/usr/local/cuda/lib64`).
//!
//! `--fmad=false` is load-bearing: every fused multiply-add in the kernels is written
//! explicitly (`fmaf`), so nvcc contracts nothing and the CPU reference in `src/` can model
//! the kernels' arithmetic operation by operation. `--ftz=false`, `--prec-div=true` and
//! `--prec-sqrt=true` pin IEEE division, square root and subnormals for the same reason.

use std::env;
use std::path::PathBuf;
use std::process::Command;

const SOURCES: [&str; 4] = ["hc.cu", "router.cu", "fp8_gemm.cu", "elementwise.cu"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/glm53f_layers.h");
    println!("cargo:rerun-if-changed=kernels/common.cuh");
    for key in ["GLM53F_NVCC", "GLM53F_CUDA_ARCH", "GLM53F_CUDA_LIB"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels = manifest_dir.join("kernels");
    let nvcc = env::var("GLM53F_NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".into());
    let arch = env::var("GLM53F_CUDA_ARCH").unwrap_or_else(|_| "sm_120".into());
    let cuda_lib = env::var("GLM53F_CUDA_LIB").unwrap_or_else(|_| "/usr/local/cuda/lib64".into());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    let mut objs = Vec::new();
    for src in SOURCES {
        println!("cargo:rerun-if-changed=kernels/{src}");
        let obj = out.join(src.replace(".cu", ".o"));
        let status = Command::new(&nvcc)
            .args(["-O3", "-std=c++17", "-lineinfo"])
            .args([
                "--fmad=false",
                "--ftz=false",
                "--prec-div=true",
                "--prec-sqrt=true",
            ])
            .arg(format!("-arch={arch}"))
            .arg("-Xcompiler=-fPIC")
            .arg(format!("-I{}", kernels.display()))
            .arg("-c")
            .arg(kernels.join(src))
            .arg("-o")
            .arg(&obj)
            .status()
            .unwrap_or_else(|e| panic!("failed to run {nvcc}: {e}"));
        assert!(status.success(), "nvcc failed on kernels/{src}");
        objs.push(obj);
    }

    let lib = out.join("libglm53f_layers_kernels.a");
    let status = Command::new("ar")
        .arg("rcs")
        .arg(&lib)
        .args(&objs)
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=glm53f_layers_kernels");
    println!("cargo:rustc-link-search=native={cuda_lib}");
    println!("cargo:rustc-link-lib=dylib=cudart");
    // The archive is C++ compiled by nvcc's host compiler.
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
