# XPURemoting

This is a virtualization framework for CUDA API.  The developer can easily add customization to the execution CUDA API, e.g., execute it as an RPC at a remote machine. 

## Minimal demo

Consits of 4 parts:

- `server`
- `client`
- `network`
- `codegen`

## Getting started

- How to add customization to a CUDA API:  [implement_new_apis.md](docs/implement_new_apis.md) 



## Requirements

- Environment setup: please run a docker container mounting the `xpuremoting` directory and enter it. (On `meepo3` or `meepo4`, you can use image `xpu_remoting:latest`). An example command is:

```shell
export container_name=xxx
docker run -dit  --shm-size 8G  --name $container_name  --gpus all  --privileged  --network host  -v path/to/xpuremoting:/workspace  xpu_remoting:latest
```

- Note that we need Rust **nightly** toolchain to build the project, which is not installed in the docker image.

- Version checklist
  - Rust: `cargo 1.78.0-nightly`
  - CMake: `3.22.1`
  - Clang: `6.0.0-1ubuntu2`

## Build

First we should build the ELF parsing static library from source:

```shell
cd /path/to/xpuremoting/client/src/elf
make -j
```

Then we can build the project using cargo:

```shell
cd /path/to/xpuremoting && cargo build
```

## Config

You can use `config.toml` file to config communication type, buffer size, RDMA server listener socket and so on.

Due to `cargo` will use cwd as running root folder, use absolute path for config file. The default path will be `/workspace/xpuremoting/config.toml`. If you want a specific path, you can use environment variable `NETWORK_CONFIG` customize it. For example: `NETWORK_CONFG="/path/to/my/config cargo run`.

## Test

### Unit test

```shell
cargo test
```

### Integration test

Launch two terminals, one for server and the other for client.

- server side:

```shell
cargo run [--features shm,rdma] server
```

You should use `features` to decide what communication methods will be compiled. For example: if you do not want rdma communication method to be compiled, run with `cargo run --features shm server`.

- client side:

```shell
cd tests/cuda_api
mkdir -p build && cd build
cmake .. && make
cd ..
./startclient.sh ./build/remoting_test
```

P.S. Can use `RUST_LOG` environment to control the log level (default=debug).



## Appendix

### Build the docker image

Please refer to the [link](https://x8csr71rzs.feishu.cn/docx/DdXFdGSYOo8cktxgj8hcYh12nHf), and use the Dockerfile in the root directory.