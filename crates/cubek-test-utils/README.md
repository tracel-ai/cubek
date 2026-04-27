# cubek-test-utils

Shared building blocks for kernel tests in CubeK: test-tensor builders, host-side
reference comparisons, and a `CUBE_TEST_MODE` policy layer that decides what
constitutes a passing test under a given environment variable.

This README documents the testing workflow used across most CubeK kernels.

---

## Test suites

Four test suites are available:

- **Light test suite** — a tractable subset of representative tests that runs in CI.
- **Basic test suite** — adds tests considered basic but that may hang on CI (slow on CPU).
- **Extended test suite** — usually auto-generated combinatorial tests covering many
  configurations. Good to run when developing kernels. Normally kept tractable.
- **Full test suite** — all generable test combinations; may be too large to compile or
  run practically.

Run them with:

```bash
# Replace <runtime> with cpu, cuda, rocm, wgpu, vulkan or metal

# Basic test suite (light on cpu)
cargo test-<runtime>

# Extended test suite
cargo test-<runtime>-extended

# Full test suite
cargo test-<runtime>-full
```

---

## Cube test mode

Set the `CUBE_TEST_MODE` environment variable to control how tests respond to
numerical errors and compilation errors.

| Mode                  | Numerical error | Compilation error         | Notes                                                                |
| --------------------- | --------------- | ------------------------- | -------------------------------------------------------------------- |
| `correct` _(default)_ | fail            | accept                    | Useful when test grids include invalid configurations on purpose.    |
| `strict`              | fail            | fail                      | Recommended for debugging — surfaces every problem.                  |
| `printall[:filter]`   | fail (printed)  | fail (printed)            | Every test is rejected so you can read the full per-element dump.    |
| `printfail[:filter]`  | fail (printed)  | accept                    | Per-element dump only for tests that compile and produce mismatches. |
| `failifrun`           | accept          | accept (other tests fail) | Inverts `correct` to surface tests that _do_ run.                    |

### Filter syntax

The filter is optional and tells the printers which indices to highlight. A
comma-separated list of dimensions, where each entry is one of:

- `.` — wildcard (any index along that dimension)
- `N` — a single index
- `M-K` — an inclusive range

Example for a 4-D tensor: `CUBE_TEST_MODE=printfail:.,.,10-20,30` selects all
elements where dim 2 is in `10..=20` and dim 3 is exactly `30`.

> The filter rank must match the tensor rank, otherwise the test returns an
> `Error` instead of `Fail`/`Pass`.

---

## Failure messages

`assert_equals_approx` collects up to **8 individual mismatches** plus aggregate
stats and reports them in the test panic message:

```
Test failed: Got incorrect results: 17/4096 elements mismatched
  (max |Δ|=0.014648, mean |Δ|=0.004112, worst at [3, 12]) — shape=[16, 256]
First mismatches:
  [0, 5]: got 1.234, expected 1.220, |Δ|=0.014 > ε=0.001
  [0, 17]: got 0.998, expected 1.001, |Δ|=0.003 > ε=0.001
  ...
  ... and 9 more
```

In `printall` / `printfail` modes, the per-element output is written to stdout
and the panic message keeps only the aggregate header (no duplicated examples).

---

## Pointers

- `TestMode` and `current_test_mode` — `src/test_mode/base.rs`
- Validation result / decision pipeline — `src/test_mode/result.rs`
- `assert_equals_approx`, `assert_equals_approx_in_slice` — `src/correctness/base.rs`
- Tensor builders (`TestInput`, `DataKind`, `StrideSpec`, …) — `src/test_tensor/`
