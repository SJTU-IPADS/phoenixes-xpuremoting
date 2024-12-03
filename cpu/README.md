# Retrofitted GPU Remoting Frame

## Introduction

We retrofit `cricket` remoting tech by replacing tirpc library with universal proxy library (most code is in `./proxy`).

Now proxy can support different communication methods like shared memory, rdma, tcp/ip. And any new method can be accommodated easily by implementing a few interfaces defined in `./proxy/device_buffer.h`.

## Usage

As for enabling different communication methods, we can use env `VERSION` to specify one in building phase. In root path, run `VERSION=xxx make cpu -j`.

The value of `VERSION` now can be:

```plain
NO_OPTIMIZATION: cricket native
WITH_SHARED_MEMORY
WITH_RDMA
WITH_TCPIP
```
