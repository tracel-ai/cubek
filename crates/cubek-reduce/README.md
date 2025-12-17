# CubeK Reduce

Implements a wide variety of reduction algorithms across multiple instruction sets and hardware targets for efficient tensor reduction.

## Running Tests

### Important Environment Variables

Two environment variables control test execution behavior:

- `CUBEK_TEST_MODE`  
  Controls handling of tests that cannot run on the current hardware (e.g., due to missing support for certain algorithms).
  - `skip` (default): Skipped tests are silently ignored and reported as passed by the Rust test runner.
  - `verbose`: Skipped tests are reported with an explanation why they were skipped, but still marked as passed.
  - `panic`: Skipped tests cause a failure, printing the reason. The test run will show failures.  
    Useful for discovering which tests are being skipped on your hardware.

- `CUBEK_TEST_FULL`  
  Controls whether time-consuming tests are executed.
  - `0` (default): Long-running tests are skipped with an explanatory message.
  - `1`: All tests are run, including the longer ones.

### Important Feature Flags

The test suite can be run on different CubeCL runtimes by enabling the corresponding feature flag.

#### Examples

```bash
# Run all tests (including long ones) on the CUDA runtime, skipping unsupported tests silently
CUBEK_TEST_FULL=1 cargo test --features cubecl/cuda

# Run all tests on CUDA, failing on any unsupported tests (to see what is skipped)
CUBEK_TEST_MODE=panic CUBEK_TEST_FULL=1 cargo test --features cubecl/cuda

# Run tests on the WGSL (web GPU) runtime with verbose skipping
CUBEK_TEST_MODE=verbose cargo test --features cubecl/wgsl
```
