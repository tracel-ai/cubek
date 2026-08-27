//! The extents and strides an operand is addressed by, on the host.

use cubecl::prelude::*;

/// One operand's physical extents and strides, in scalars, one entry per physical dim.
///
/// The two are one value because they are never separately true. Held apart they are two
/// `Vec<usize>` a caller can state at two ranks, and two same-typed arguments a call site can
/// swap in silence; here a dim is an `(extent, stride)` pair and neither is expressible. A bound
/// operand takes its geometry off its binding ([`From`]), an unbound one states it, and the
/// derivation reaches both the same way.
///
/// The kernel-side twin is [`RuntimeGeometry`](crate::RuntimeGeometry), which is what the
/// derivation's settled geometry is handed to a tile as.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Geometry {
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Geometry {
    /// One `(extent, stride)` per physical dim, coarsest first, both in scalars.
    pub fn of_dims(dims: &[(usize, usize)]) -> Self {
        Self {
            shape: dims.iter().map(|&(extent, _)| extent).collect(),
            strides: dims.iter().map(|&(_, stride)| stride).collect(),
        }
    }

    /// The extents, coarsest first.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The strides that step them, in scalars.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// How many physical dims the operand has.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// The dims, coarsest first.
    pub fn dims(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.shape.iter().copied().zip(self.strides.iter().copied())
    }
}

impl<R: Runtime> From<&TensorBinding<R>> for Geometry {
    fn from(binding: &TensorBinding<R>) -> Self {
        Self {
            shape: binding.shape.to_vec(),
            strides: binding.strides.to_vec(),
        }
    }
}
