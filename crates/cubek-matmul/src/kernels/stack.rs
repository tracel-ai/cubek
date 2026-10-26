//! A stacked rhs: slabs laid end to end along one axis of a product, as the routed and grouped
//! products read it.

use cubecl::client::Client;
use cubecl::tile::Axis;
use cubecl::zspace::metadata::Metadata;
use cubek_microkernel::matmul::MatmulLayout;
use cubek_std::matmul_axes::{K, N};
use serde::{Deserialize, Serialize};

use crate::kernels::tensor::PlainTensor;

/// The axis of a product along which the slabs of its stacked rhs lie end
/// to end.
///
/// A model stores its experts along the output, a `[k, experts · n]` stack
/// whose slab `e` owns columns `e · n..(e + 1) · n`. A product that reads the
/// same buffer transposed, as a backward's `d lhs = grad @ slabᵀ` does, is
/// handed the `[experts · n, k]` view and finds the slabs along its
/// contraction. Both are one stack read where it lies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SlabAxis {
    /// Slab `e` owns the view's columns `e · n..(e + 1) · n`.
    Output,
    /// Slab `e` owns the view's rows `e · k..(e + 1) · k`.
    Contraction,
}

impl SlabAxis {
    /// The product's axis the slabs lie along.
    pub fn axis(self) -> Axis {
        match self {
            Self::Output => N,
            Self::Contraction => K,
        }
    }

    /// The axis the same slabs lie along in the view's transpose: a stack's
    /// output becomes its transpose's contraction, and back.
    pub fn transposed(self) -> Self {
        match self {
            Self::Output => Self::Contraction,
            Self::Contraction => Self::Output,
        }
    }

    /// One slab's `[k, n]` out of a `[k, n]` view holding `slabs` of them
    /// along this axis.
    ///
    /// # Panics
    ///
    /// Unless the view is whole slabs.
    pub fn slab_of(self, [k, n]: [usize; 2], slabs: usize) -> [usize; 2] {
        let whole = |extent: usize| {
            assert!(
                extent.is_multiple_of(slabs),
                "a stack of {slabs} slabs has whole slabs along its {self:?} axis, got {extent}"
            );
            extent / slabs
        };
        match self {
            Self::Output => [k, whole(n)],
            Self::Contraction => [whole(k), n],
        }
    }
}

/// The stack as `[groups, k, n]`, one `[k, n]` slab a batch, over the buffer it lies in: a
/// strided batch rather than a gathered dim. A `cmma` or plane-register fragment reads no
/// gathered operand, a gathered dim compiles one slab's extent into its coefficients where a
/// batch's stride is read at runtime, and a col stack stays a col rhs a slab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlabView {
    pub shape: [usize; 3],
    pub strides: [usize; 3],
}

impl SlabView {
    /// `stack`, a `[k, n]` view holding `groups` slabs laid along `along`, one slab a batch. A slab
    /// steps within itself as the stack's view does, whichever way its buffer lies, and the next
    /// slab starts one slab's extent further along the axis the slabs lie along.
    ///
    /// # Panics
    ///
    /// As [`SlabAxis::slab_of`], on a stack that is not whole slabs.
    pub fn new(stack: &PlainTensor, groups: usize, along: SlabAxis) -> Self {
        let meta = stack.meta();
        let (shape, strides) = (meta.shape(), meta.strides());
        let [k, n] = along.slab_of([shape[0], shape[1]], groups);
        let [k_stride, n_stride] = [strides[0], strides[1]];
        let slab_stride = match along {
            SlabAxis::Output => n * n_stride,
            SlabAxis::Contraction => k * k_stride,
        };
        Self {
            shape: [groups, k, n],
            strides: [slab_stride, k_stride, n_stride],
        }
    }

    /// `stack` restated as its slabs, `[slabs, k, n]` one slab a batch, over the buffer it lies in.
    pub fn slabs_of(&self, stack: &PlainTensor) -> PlainTensor {
        stack.viewed(Metadata::new(self.shape, self.strides))
    }

    /// `stack` as a grouped or routed product reads it: where it lies where it is contiguous
    /// along one of its axes ([`MatmulLayout::lines`]), a row-major copy on `client` where it is
    /// not, since every routine reads a slab along one of its axes.
    pub fn readable(client: &Client, stack: PlainTensor) -> PlainTensor {
        let meta = stack.meta();
        let lined = match (&meta.shape()[..], &meta.strides()[..]) {
            ([k, n], [k_stride, n_stride]) => {
                MatmulLayout::lines([K, N], [*k, *n], [*k_stride, *n_stride]).is_some()
            }
            _ => true,
        };
        match lined {
            true => stack,
            false => stack.into_contiguous(client),
        }
    }
}
