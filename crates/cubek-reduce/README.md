# CubeK Reduce

Implements a wide variety of reduction algorithms across multiple instruction sets and hardware targets for efficient tensor reduction.

## Running Tests

### Important Environment Variables

Test behavior is controlled by the shared `CUBE_TEST_MODE` env var (see `cubek-test-utils`).

- `CUBE_TEST_MODE=Correct` (default): numerical errors fail the test; compilation / hardware-incompatibility errors are accepted.
- `CUBE_TEST_MODE=Strict`: both numerical and compilation errors fail the test. Useful to surface tests that are silently skipped on your hardware.
- `CUBE_TEST_MODE=PrintAll[:<filter>]` / `PrintFail[:<filter>]`: print tensor elements; see `cubek-test-utils` docs.

### Important Feature Flags

- `heavy`: enables the `Cube` reduction-routine strategy tests, plus the full breadth of the `reduce_dim` matrix: both dtypes, both output-vectorization settings, both forced plane dims, every shape configuration and the whole `k` sweep. All of it is slow on CPU and can stall CI, so the default (light) suite keeps one point per collapsed axis and one shape per layout class. CI never compiles the `heavy` tier while testing, so `cargo clippy -p cubek-reduce --tests --features heavy` guards it separately.
- `extended`: implies `heavy` (room for future growth).
- `full`: implies `extended`, and enables the benchmark catalogue.

#### Examples

```bash
# Default (fast) test run on CUDA
cargo test --features cubecl/cuda

# Run the full suite, including Cube-strategy tests
cargo test --features cubecl/cuda,extended

# Fail on any silently-skipped tests
CUBE_TEST_MODE=Strict cargo test --features cubecl/cuda,extended
```
