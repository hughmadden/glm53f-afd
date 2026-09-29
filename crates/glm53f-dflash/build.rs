//! Builds the drafter's CUDA kernels when the `cuda` feature is on, and links cudart and cuBLAS.
//!
//! Default builds (and `cargo test`) skip this entirely, so they need no CUDA toolkit.
//! With `--features cuda` this compiles `kernels/dflash.cu` into a static archive.
//!
//! Environment (the same variables as the other kernel crates):
//! - `GLM53F_NVCC`: the nvcc to run (default `/usr/local/cuda/bin/nvcc`);
//! - `GLM53F_CUDA_ARCH`: the target (default `sm_120`, the RTX 5090 coordinator; set `sm_89` to
//!   build for an RTX 4090, the development GPU);
//! - `GLM53F_CUDA_LIB`: the directory holding `libcudart` and `libcublas` (default
//!   `/usr/local/cuda/lib64`).
//!
//! `--fmad=false --prec-div=true --prec-sqrt=true`: the kernels' f32 arithmetic is the IEEE
//! operation sequence the source spells out, so the norms, RoPE, convolution and selector match
//! the CPU reference's order of operations bit for bit (see `kernels/glm53f_dflash.h`).

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/glm53f_dflash.h");
    println!("cargo:rerun-if-changed=kernels/dflash.cu");
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

    let obj = out.join("dflash.o");
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
        .arg(kernels.join("dflash.cu"))
        .arg("-o")
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| panic!("failed to run {nvcc}: {e}"));
    assert!(status.success(), "nvcc failed on kernels/dflash.cu");

    let lib = out.join("libglm53f_dflash_kernels.a");
    let _ = std::fs::remove_file(&lib);
    let status = Command::new("ar")
        .arg("rcs")
        .arg(&lib)
        .arg(&obj)
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=glm53f_dflash_kernels");
    println!("cargo:rustc-link-search=native={cuda_lib}");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-env=GLM53F_DFLASH_CUDA_ARCH={arch}");
}
