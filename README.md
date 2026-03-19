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

| Algorithms                                                                           | Variants                                                                 |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| [Random](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-random)           | `bernoulli` `normal` `uniform`                                           |
| [Quantization](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-quant)      | `symmetric` `per-block` `per-tensor` `q2` `q4` `q8` `fp4`                |
| [Reduction](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-reduce)        | `mean` `sum` `prod` `max` `min` `arg[max\|min]` `per-cube` `per-plane`   |
| [Matmul](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-matmul)           | `mma` `unit` `tma` `multi-stage` `specialization` `ordered` `multi-rows` |
| [Convolution](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-convolution) | `mma` `unit` `tma` `multi-stage` `im2col`                                |
| [Attention](https://github.com/tracel-ai/cubek/tree/main/crates/cubek-attention)     | `mma` `unit` `multi-rows`                                                |

# Contributing

If you want to contribute new kernels, please read the [`GUIDE.md`](./GUIDE.md).

# Running tests

> Note: This applies to most kernels, but `reduce` works slightly differently for now, see [its README](./crates/cubek-reduce/README.md).

## Command

Three test suites are available:

- **Smoke test suite**: a tractable subset of representative tests that run on the CI.
- **Extended test suite**: usually auto-generated combinatorial tests covering many configurations. Good to run when developing kernels. Normally kept tractable.
- **Full test suite**: all generable test combinations; may be too large to compile or run practically.

Run tests with

```bash
# Replace <runtime> with cpu, cuda, rocm, wgpu, vulkan or metal

# Smoke test suite
cargo test-<runtime>

# Extended test suite
cargo test-<runtime>-extended

# Full test suite
cargo test-<runtime>-full
```

## Cube test mode

You can control test behavior by setting the `CUBE_TEST_MODE` environment variable.  
For more details, see [Test Mode](./crates/cubek-test-utils/src/test_mode/base.rs).

### Modes

- **`CUBE_TEST_MODE=correct`** _(default)_  
  Tests pass if results are numerically correct **or** if the kernel was launched with an invalid configuration.
  - Useful when tests are auto-generated from multiple parameter combinations, where some invalid configurations are expected.
  - Failing tests display only the first index with a discrepancy.

- **`CUBE_TEST_MODE=strict`**  
  Tests pass **only** if they compile, run, and produce numerically accurate results.
  - Ideal for debugging to avoid false positives that can occur in `correct` mode.

- **`CUBE_TEST_MODE=printfail`**  
  Similar to `correct` mode: tests pass if results are correct or if the kernel is invalid.
  - Failing tests show **all tensor discrepancies**.
  - Supports filtering, e.g.: `CUBE_TEST_MODE=printfail:0,.,10-20` shows elements from the 0th first dimension, all of the second, and elements 10–20 in the third.

- **`CUBE_TEST_MODE=printall`**  
  All tests fail, displaying **all tensor discrepancies**.
  - Filtering works the same as in `printfail`.

- **`CUBE_TEST_MODE=failifrun`**  
  Only tests that **compile and run** will fail; others succeed.
  - Useful for tracking critical tests in large suites.
