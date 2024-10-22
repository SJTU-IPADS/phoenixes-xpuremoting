extern crate glob;
use glob::glob;
use std::{env, path::PathBuf};

pub fn read_env() -> Vec<PathBuf> {
    if let Ok(path) = env::var("CUDA_LIBRARY_PATH") {
        let split_char = ":";
        path.split(split_char).map(|s| PathBuf::from(s)).collect()
    } else {
        vec![]
    }
}

pub fn find_cuda() -> Vec<PathBuf> {
    let mut candidates = read_env();
    candidates.push(PathBuf::from("/opt/cuda"));
    candidates.push(PathBuf::from("/usr/local/cuda"));
    for e in glob("/usr/local/cuda-*").unwrap() {
        if let Ok(path) = e {
            candidates.push(path)
        }
    }

    let mut valid_paths = vec![];
    for base in &candidates {
        let lib = PathBuf::from(base).join("lib64");
        if lib.is_dir() {
            valid_paths.push(lib.clone());
            valid_paths.push(lib.join("stubs"));
        }
        let base = base.join("targets/x86_64-linux");
        let header = base.join("include/cuda.h");
        if header.is_file() {
            valid_paths.push(base.join("lib"));
            valid_paths.push(base.join("lib/stubs"));
            continue;
        }
    }
    eprintln!("Found CUDA paths: {:?}", valid_paths);
    valid_paths
}

fn main() {
    // cc::Build::new()
    //     .cpp(true)
    //     .file("../pos/pos.cpp")
    //     .compile("pos");
    // println!("cargo:rerun-if-changed=../pos/pos.h");
    // println!("cargo:rerun-if-changed=../pos/pos.cpp");

    for path in find_cuda() {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=nvidia-ml");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");

    println!("cargo:rustc-link-search=native=../../lib/");
    println!("cargo:rustc-link-lib=pos");

    // TODO: use bindgen (or cuda_hook) to automatically generate the FFI
}
