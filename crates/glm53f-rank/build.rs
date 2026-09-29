//! Builds the EXL3 expert kernels when the `cuda` feature is on (after
//! mimo26f-afd's `crates/mimo26-spark/build.rs`: compile, archive, link cudart
//! and the C++ runtime).
//!
//! Default builds and `cargo test` skip this, so they need no CUDA toolkit.
//!
//! Environment:
//! - `GLM53F_NVCC`: the nvcc to run (default `/usr/local/cuda/bin/nvcc`);
//! - `GLM53F_CUDA_ARCH`: the target, default `sm_121` (a DGX Spark, built on
//!   the Spark with CUDA 13). Set `sm_89` for an RTX 4090, the development
//!   GPU. The architecture is baked into the binary, and the daemon refuses to
//!   serve on a device of another architecture;
//! - `GLM53F_CUDA_LIB`: the directory holding `libcudart` (default
//!   `/usr/local/cuda/lib64`).
//!
//! `--fmad=false` keeps every multiply and add of the rotations and epilogues
//! separately rounded, as the CPU reference computes them (the tensor-core
//! products are unaffected). `--ftz=false` and the precise division keep IEEE
//! subnormals and division.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/exl3_rank.cu");
    for key in ["GLM53F_NVCC", "GLM53F_CUDA_ARCH", "GLM53F_CUDA_LIB"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let nvcc = env::var("GLM53F_NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".into());
    let arch = env::var("GLM53F_CUDA_ARCH").unwrap_or_else(|_| "sm_121".into());
    let cuda_lib = env::var("GLM53F_CUDA_LIB").unwrap_or_else(|_| "/usr/local/cuda/lib64".into());
    // "sm_121a" -> 121: the compute capability the daemon checks the device against.
    let baked: String = arch.trim_start_matches("sm_").chars().take_while(|c| c.is_ascii_digit()).collect();
    assert!(!baked.is_empty(), "GLM53F_CUDA_ARCH must look like sm_121 or sm_89, got {arch}");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj = out.join("exl3_rank.o");
    let status = Command::new(&nvcc)
        .args(["-O3", "-std=c++17", "-lineinfo", "--fmad=false", "--ftz=false", "--prec-div=true"])
        .arg(format!("-arch={arch}"))
        .arg("-Xcompiler=-fPIC")
        .arg(format!("-DG53R_BAKED_ARCH={baked}"))
        .arg("-c")
        .arg(manifest_dir.join("kernels/exl3_rank.cu"))
        .arg("-o")
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| panic!("failed to run {nvcc}: {e}"));
    assert!(
        status.success(),
        "nvcc failed on kernels/exl3_rank.cu for GLM53F_CUDA_ARCH={arch}: the default, sm_121 (a DGX Spark), needs CUDA 13; set GLM53F_CUDA_ARCH=sm_89 to build for an RTX 4090"
    );

    let lib = out.join("libglm53f_rank_kernels.a");
    let _ = std::fs::remove_file(&lib);
    let status = Command::new("ar").arg("rcs").arg(&lib).arg(&obj).status().expect("failed to run ar");
    assert!(status.success(), "ar failed");

    // The kernel archive, the CUDA runtime, and the C++ runtime (the nvcc object
    // uses std::vector and Rust links with -nodefaultlibs).
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=glm53f_rank_kernels");
    println!("cargo:rustc-link-search=native={cuda_lib}");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-env=GLM53F_RANK_CUDA_ARCH={arch}");
}
