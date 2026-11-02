<div align="center">
  <img src="assets/image.webp" alt="The CubeK illustration" height="200"/>
  <br /><br />

[![Rust 1.92 or later](https://img.shields.io/badge/Rust-1.92%2B-6b8e23?style=flat-square&logo=rust&logoColor=white)](https://crates.io/crates/cubek)
![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-6b8e23?style=flat-square)
[![Current Crates.io Version](https://img.shields.io/crates/v/cubek.svg?style=flat-square&color=6b8e23)](https://crates.io/crates/cubek)
[![Test Status](https://img.shields.io/github/actions/workflow/status/tracel-ai/cubek/ci.yml?branch=main&label=CI&style=flat-square)](https://github.com/tracel-ai/cubek/actions/workflows/ci.yml)
<br />
[![Discord](https://img.shields.io/discord/1038839012602941528.svg?label=Chat&style=flat-square&color=6b8e23&logo=discord&logoColor=white)](https://discord.gg/KSBSPhAUCc)
[![Documentation](https://img.shields.io/docsrs/cubek?style=flat-square)](https://docs.rs/cubek)

  <p><b>CubeK: high-performance multi-platform kernels in CubeCL</b></p>
</div>

---

CubeK provides reusable compute kernels written in Rust with
[CubeCL](https://github.com/tracel-ai/cubecl). It powers core operations in
[Burn](https://github.com/tracel-ai/burn) and can also be used directly in CubeCL applications, with
specialized implementations for matrix multiplication, convolution, attention, reductions, and more.

## Algorithms

| Algorithms                                                                                         | Variants                                                                 |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [Random](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-random)                         | `bernoulli` `normal` `uniform`                                           |
| [Quantization](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-quant)                    | `symmetric` `per-block` `per-tensor` `q2` `q4` `q8` `fp4`                |
| [Reduction](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-reduce)                      | `mean` `sum` `prod` `max` `min` `arg[max\|min]` `per-cube` `per-plane`   |
| [Matmul](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-matmul)                         | `mma` `unit` `tma` `multi-stage` `specialization` `ordered` `multi-rows` |
| [Convolution](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-convolution)               | `mma` `unit` `tma` `multi-stage` `im2col`                                |
| [Attention](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-attention)                   | `mma` `unit` `multi-rows`                                                |
| [Attention backward](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-attention-backward) | `mma` `unit` `plane` `cube`                                              |

## Runtime configuration

CubeK loads runtime settings from `cubek.toml` or `CubeK.toml` in the current directory or a parent
directory. Missing settings use defaults.

To disable automatic F32-to-TF32 conversion for matrix multiplication:

```toml
[matmul]
allow_tf32 = false
```

`allow_tf32` defaults to `true`. Disabling it preserves F32 operands during automatic selection and
may reduce performance.

Configuration is loaded on first use. Restart the application after editing the file; no
recompilation is needed.

## Contributing

If you want to contribute new kernels, please read the [`GUIDE.md`](./GUIDE.md).

## Running tests

The full testing guide — suites, `CUBE_TEST_MODE`, failure-message format, and filter syntax — lives
in [`cubek-test-utils`](./crates/cubek-test-utils/README.md).
