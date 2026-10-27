<div align="center">
<img src="https://raw.githubusercontent.com/tracel-ai/cubek/main/assets/image.webp" width="150px"/>

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

| Algorithms                                                                                         | Variants                                                                 |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [Random](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-random)                         | `bernoulli` `normal` `uniform`                                           |
| [Quantization](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-quant)                    | `symmetric` `per-block` `per-tensor` `q2` `q4` `q8` `fp4`                |
| [Reduction](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-reduce)                      | `mean` `sum` `prod` `max` `min` `arg[max\|min]` `per-cube` `per-plane`   |
| [Matmul](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-matmul)                         | `mma` `unit` `tma` `multi-stage` `specialization` `ordered` `multi-rows` |
| [Convolution](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-convolution)               | `mma` `unit` `tma` `multi-stage` `im2col`                                |
| [Attention](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-attention)                   | `mma` `unit` `multi-rows`                                                |
| [Attention backward](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-attention-backward) | `mma` `unit` `plane` `cube`                                              |

# Runtime configuration

CubeK loads runtime settings from `cubek.toml` or `CubeK.toml` in the current directory or a
parent directory. Missing settings use defaults.

To disable automatic F32-to-TF32 conversion for matrix multiplication:

```toml
[matmul]
allow_tf32 = false
```

`allow_tf32` defaults to `true`. Disabling it preserves F32 operands during automatic selection
and may reduce performance.

Configuration is loaded on first use. Restart the application after editing the file;
no recompilation is needed.

# Contributing

If you want to contribute new kernels, please read the [`GUIDE.md`](./GUIDE.md).

# Running tests

The full testing guide — suites, `CUBE_TEST_MODE`, failure-message format, and
filter syntax — lives in [`cubek-test-utils`](./crates/cubek-test-utils/README.md).
