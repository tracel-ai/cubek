//! Benchmark registry for cubek.

pub mod attention;
pub mod contiguous;
pub mod conv2d;
pub mod fft;
pub mod gemm;
pub mod gemv;
pub mod interpolate;
pub mod memcpy_async;
pub mod quantized_matmul;
pub mod reduce;
pub mod registry;
pub mod unary;

pub use registry::{BenchmarkCategory, ItemDescriptor, RunSamples, all};

// Re-exports so downstream consumers can reach the HostData (de)ser and
// comparison primitives without taking a direct dep on cubek-test-utils.
pub use cubek_test_utils::{
    HostData, ValidationResult, compare_host_data_files, read_host_data, write_host_data,
};
