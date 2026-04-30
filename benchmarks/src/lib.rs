//! Benchmark registry for cubek.

pub mod attention;
pub mod contiguous;
pub mod conv2d;
pub mod fft;
pub mod gemm;
pub mod gemv;
pub mod memcpy_async;
pub mod quantized_matmul;
pub mod reduce;
pub mod registry;
pub mod unary;

pub use registry::{BenchmarkCategory, ItemDescriptor, RunSamples, all};
