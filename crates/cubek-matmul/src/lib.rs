/// Components for matrix multiplication
pub mod components;
pub mod definition;
pub mod launch;
/// Contains matmul kernels
pub mod routines;

#[cfg(feature = "cpu-reference")]
pub mod cpu_reference;
