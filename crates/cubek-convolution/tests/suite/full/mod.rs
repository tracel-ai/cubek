//! Full tier: macro-driven cartesian sweep.
//!
//! Generates the cross-product of (algorithm × precision × tile size ×
//! partition × stage × swizzle × partition buffering × problem size) for the
//! plane-accelerated convolution routines, on both the CMMA and MMA tile
//! matmuls.

#[macro_use]
mod accelerated;
#[macro_use]
mod common;
#[macro_use]
mod launch;

mod conv2d_accelerated_suite {
    crate::testgen_convolution_accelerated!();
}
