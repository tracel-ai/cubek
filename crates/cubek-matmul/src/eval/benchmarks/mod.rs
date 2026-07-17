//! Benchmark catalogues for `cubek-matmul`.

pub mod gemm;
pub mod gemm_cpu;
pub mod gemm_cpu_tiled;
pub mod gemv;
pub mod quantized_matmul;
pub mod split_k;
pub mod tile_quant_stage;
