<div align="center">
<img src="https://github.com/tracel-ai/cubek/tree/main/assets/image.webp" width="150px"/>

<br />

[![Discord](https://img.shields.io/discord/1038839012602941528.svg?color=7289da&&logo=discord)](https://discord.gg/KSBSPhAUCc)
[![Current Crates.io Version](https://img.shields.io/crates/v/cubek.svg)](https://crates.io/crates/cubek)
[![Minimum Supported Rust Version](https://img.shields.io/crates/msrv/cubek)](https://crates.io/crates/cubek)
[![Test Status](https://github.com/tracel-ai/cubek/actions/workflows/ci.yml/badge.svg)](https://github.com/tracel-ai/cubek/actions/workflows/test.yml)
![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)
<br />

---

**CubeK: high-performance multi-platform kernels in CubeCL**
<br/>

</div>

# Algorithms

| Algorithms                                                                             | Variants                                                                 |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [Random](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-random)             | `bernoulli` `normal` `uniform`                                           |
| [Quantization](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-quantization) | `symmetric` `per-block` `per-tensor` `q2` `q4` `q8` `fp4`                |
| [Reduction](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-reduce)          | `mean` `sum` `prod` `max` `min` `arg[max\|min]` `per-cube` `per-plane`   |
| [Matmul](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-matmul)             | `mma` `unit` `tma` `multi-stage` `specialization` `ordered` `multi-rows` |
| [Convolution](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-convolution)   | `mma` `unit` `tma` `multi-stage` `im2col`                                |
| [Attention](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-attention)       | `mma` `unit` `multi-rows`                                                |
