//! A stacked rhs: slabs laid end to end along one dimension of a product, as the routed and grouped
//! products read it.

use crate::kernels::MATMUL_SCALED_BESIDE_FLOAT;
use crate::kernels::tensor::{PlainTensor, StoredTensor};
use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::tile::Dimension;
use cubecl::tile::launch::Decline;
use cubecl::zspace::metadata::Metadata;
use cubek_microkernel::matmul::MatmulLayout;
use cubek_std::matmul_dimensions::{K, N};
use serde::{Deserialize, Serialize};

/// A stacked rhs that is no rank-two view of whole slabs, at least one of them.
pub const MATMUL_STACK_NOT_WHOLE_SLABS: Decline = Decline::Unserved {
    why: "a stack is a rank-two view of whole slabs, at least one of them",
};

/// Ends of a grouped product that are no one `u32` per group, at least one.
pub const GROUP_ENDS_UNREAD: Decline = Decline::Unserved {
    why: "the groups' ends are one u32 row per group, at least one group",
};

/// Whether `ends` are one `u32` per group, at least one.
///
/// # Errors
///
/// [`GROUP_ENDS_UNREAD`].
pub(crate) fn admit_ends(ends: &PlainTensor) -> Result<(), Decline> {
    let shape = ends.shape();
    if ends.elem != ElemType::UInt(UIntKind::U32) || shape.rank() != 1 || shape[0] == 0 {
        return Err(GROUP_ENDS_UNREAD);
    }
    Ok(())
}

/// The dimension of a product along which the slabs of its stacked rhs lie end
/// to end.
///
/// A model stores its experts along the output, a `[k, experts · n]` stack
/// whose slab `e` owns columns `e · n..(e + 1) · n`. A product that reads the
/// same buffer transposed, as a backward's `d lhs = grad @ slabᵀ` does, is
/// handed the `[experts · n, k]` view and finds the slabs along its
/// contraction. Both are one stack read where it lies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SlabDimension {
    /// Slab `e` owns the view's columns `e · n..(e + 1) · n`.
    Output,
    /// Slab `e` owns the view's rows `e · k..(e + 1) · k`.
    Contraction,
}

impl SlabDimension {
    /// The product's dimension the slabs lie along.
    pub fn dimension(self) -> Dimension {
        match self {
            Self::Output => N,
            Self::Contraction => K,
        }
    }

    /// The dimension the same slabs lie along in the view's transpose: a stack's
    /// output becomes its transpose's contraction, and back.
    pub fn transposed(self) -> Self {
        match self {
            Self::Output => Self::Contraction,
            Self::Contraction => Self::Output,
        }
    }

    /// One slab's `[k, n]` out of a `[k, n]` view holding `slabs` of them
    /// along this dimension.
    ///
    /// # Assumptions
    ///
    /// The view is whole slabs, at least one.
    ///
    /// # Panics
    ///
    /// On a view that is not whole slabs, or on no slab.
    pub fn slab_of(self, [k, n]: [usize; 2], slabs: usize) -> [usize; 2] {
        let whole = |extent: usize| {
            assert!(
                extent.is_multiple_of(slabs),
                "a stack of {slabs} slabs should be whole slabs along its {self:?} dimension, got {extent}"
            );
            extent / slabs
        };
        match self {
            Self::Output => [k, whole(n)],
            Self::Contraction => [whole(k), n],
        }
    }

    /// One slab's `[k, n]` out of `stack`, float or packed, holding `slabs` of them along this
    /// dimension.
    ///
    /// # Assumptions
    ///
    /// As [`slab_of`](Self::slab_of).
    ///
    /// # Panics
    ///
    /// As [`slab_of`](Self::slab_of).
    pub fn slab_shape(self, stack: &StoredTensor, slabs: usize) -> [usize; 2] {
        self.slab_of(
            [K, N].map(|dimension| stack.extent([K, N], dimension)),
            slabs,
        )
    }

    /// `stack`, holding `slabs` along this dimension, as the product reads it: a float one as its
    /// slabs, `[slabs, k, n]` over the buffer it lies in, a packed one its words and scales read
    /// slab by slab.
    ///
    /// # Errors
    ///
    /// What reading a packed stack slab by slab declines.
    pub fn read_slabs(self, stack: &StoredTensor, slabs: usize) -> Result<StoredTensor, Decline> {
        match stack {
            StoredTensor::Packed(packed) => Ok(StoredTensor::Packed(
                packed.clone().stacked(slabs, self.dimension())?,
            )),
            StoredTensor::Scaled(_) => Err(MATMUL_SCALED_BESIDE_FLOAT),
            StoredTensor::Plain(stack) => {
                let view = SlabView::new(stack, slabs, self);
                Ok(StoredTensor::Plain(view.slabs_of(stack)))
            }
        }
    }
}

impl SlabDimension {
    /// Whether `stack`, float or packed, is whole slabs along this dimension, `slabs` of them.
    ///
    /// # Errors
    ///
    /// [`MATMUL_STACK_NOT_WHOLE_SLABS`].
    pub(crate) fn admit(self, stack: &StoredTensor, slabs: usize) -> Result<(), Decline> {
        if stack.values().shape().rank() != 2 || slabs == 0 {
            return Err(MATMUL_STACK_NOT_WHOLE_SLABS);
        }
        if !stack.extent([K, N], self.dimension()).is_multiple_of(slabs) {
            return Err(MATMUL_STACK_NOT_WHOLE_SLABS);
        }
        Ok(())
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
    /// slab starts one slab's extent further along the dimension the slabs lie along.
    ///
    /// # Assumptions
    ///
    /// As [`SlabDimension::slab_of`].
    ///
    /// # Panics
    ///
    /// As [`SlabDimension::slab_of`].
    pub fn new(stack: &PlainTensor, groups: usize, along: SlabDimension) -> Self {
        let meta = stack.meta();
        let (shape, strides) = (meta.shape(), meta.strides());
        let [k, n] = along.slab_of([shape[0], shape[1]], groups);
        let [k_stride, n_stride] = [strides[0], strides[1]];
        let slab_stride = match along {
            SlabDimension::Output => n * n_stride,
            SlabDimension::Contraction => k * k_stride,
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

    /// `stack` as a grouped or routed product reads it: where it lies where it is contiguous along
    /// one of its dimensions ([`MatmulLayout::lines`]), a row-major copy on `client` where it is
    /// not, since every routine reads a slab along one of its dimensions.
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
