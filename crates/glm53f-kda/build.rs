//! Builds the KDA CUDA kernels when the `cuda` feature is on.
//!
//! Default builds (and `cargo test`) skip this entirely, so they need no CUDA toolkit.
//! With `--features cuda` this compiles `kernels/kda.cu` into a static archive and links
//! it together with `cudart`.
//!
//! Environment:
//! - `GLM53F_NVCC`: the nvcc to run (default `/usr/local/cuda/bin/nvcc`);
//! - `GLM53F_CUDA_ARCH`: the target (default `sm_120`, the RTX 5090 coordinator; set `sm_89` to
//!   build for an RTX 4090, the development GPU);
//! - `GLM53F_CUDA_LIB`: the directory holding `libcudart` (default `/usr/local/cuda/lib64`).
//!
//! `--fmad=false` is load-bearing: the chain and the replays share one state-update routine,
//! and keeping a prefix of a window gives the bits of serial steps only because no
//! multiply-add is contracted differently in the two kernels. `--ftz=false`,
//! `--prec-div=true` and `--prec-sqrt=true` pin IEEE division, square root and subnormals,
//! which the CPU reference relies on to match the state update bit for bit.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/glm53f_kda.h");
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
    // kda.cu: the chain, replays and conv shift. kda_prefill.cu: the chunked prefill.
    // parity/tensorfold_kda.cu: the source kernels, verbatim, for the parity tests only; it is a
    // separate object, so a binary that does not call it never links it.
    let mut objs = Vec::new();
    for (src, name) in [
        ("kda.cu", "kda.o"),
        ("kda_prefill.cu", "kda_prefill.o"),
        ("parity/tensorfold_kda.cu", "tensorfold_kda.o"),
    ] {
        println!("cargo:rerun-if-changed=kernels/{src}");
        let obj = out.join(name);
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

    let lib = out.join("libglm53f_kda_kernels.a");
    let _ = std::fs::remove_file(&lib);
    let status = Command::new("ar")
        .arg("rcs")
        .arg(&lib)
        .args(&objs)
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=glm53f_kda_kernels");
    println!("cargo:rustc-link-search=native={cuda_lib}");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-env=GLM53F_KDA_CUDA_ARCH={arch}");
}
