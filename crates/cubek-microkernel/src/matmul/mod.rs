//! The matmul microkernels: the one product, `lhs · rhs`, whichever of its inputs are packed, a
//! folder per scope ([`unit`](mod@unit), [`plane`], [`cube`], [`hypercube`]), each named by
//! algorithm. Beside them, what the product is, which all of them and its kernels read: the
//! [`MatmulWorkItem`], its axes and operands, the [`MatmulStep`] every load of a product fills
//! whole, the [`MatmulMicrokernel`] trait, the [`MatmulPartitioning`] every product places its work
//! with, and the [`MatmulStage`] a walk through `K` stages, its stages held in the
//! [`StageRing`](crate::StageRing).

mod base;
pub mod cube;
pub mod hypercube;
pub mod plane;
pub mod unit;

pub use base::*;
