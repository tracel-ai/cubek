//! The batch-matmul routine family: cooperative, tiled [`BatchMatmulRoutine`] algorithms
//! that all launch through the shared [`launch`] hub (`launch_ref` / `launch_ref_tma`).

pub mod double_buffering;
pub mod double_unit;
pub mod gemv_innerproduct;
pub mod interleaved;
pub mod ordered_double_buffering;
pub mod simple;
pub mod simple_unit;
pub mod specialized;

pub mod launch;
