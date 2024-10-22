use std::{env, path::PathBuf, process::Command, str};

fn find_std_lib() -> String {
    let start_dir = "/usr/lib/";
    let lib_name = "libelf.so";

    let output = Command::new("find")
        .arg(start_dir)
        .arg("-name")
        .arg(lib_name)
        .output()
        .expect("Failed to execute find command");

    if !output.status.success() {
        panic!("Error executing find: {:?}", output.stderr);
    }

    let paths = str::from_utf8(&output.stdout).unwrap();
    if let Some(path) = paths.lines().next() {
        return PathBuf::from(path).parent().unwrap().to_str().unwrap().to_string();
    } else {
        panic!("{} not found", lib_name);
    }
}

fn main() {
    println!("cargo:rustc-link-lib=static=elfctrl");
    println!("cargo:rustc-link-lib=dylib=elf");
    println!("cargo:rustc-link-lib=dylib=z");
    println!("cargo:rustc-link-lib=dylib=dl");

    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let library_dir = root.join("src/elf/");
    println!("cargo:rustc-link-search=native={}", library_dir.display());
    println!("cargo:rustc-link-search=native={}", find_std_lib());

    println!("cargo:rustc-link-search=native=../../lib/");
    println!("cargo:rustc-link-lib=pos");
    
    println!("cargo:rerun-if-changed=build.rs");
}
