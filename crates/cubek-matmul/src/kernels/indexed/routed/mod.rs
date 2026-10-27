//! The routed product: each lhs row against the slabs of a stacked rhs its routes name, launched as
//! the dense product of its batches of one row, the stack read through the routes. It is measured
//! as any matmul is, under the matmul tuning key ([`RoutedMatmulTensors::key`]).

mod tensors;

pub use tensors::*;
