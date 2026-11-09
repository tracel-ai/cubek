use crate::matmul::cube::{MatmulCubeSpace, MatmulCubeWrite};
use crate::matmul::{
    MatmulAxesKind, MatmulBuffers, MatmulDestination, MatmulOperand, MatmulStage, MatmulWorkItem,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::{Delivery, Refusal, VectorRequirement, Vectorization};
use cubek_std::matmul_axes::{M, N};
use cubek_std::{Hardware, TMA_ALIGNMENT_BYTES};

/// Who moves a cube microkernel's box of the output into global memory once its walk ends. The
/// routine's structure, never a setting: each routine stores one way and refuses a problem it
/// cannot store so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeStagedStore {
    /// Each plane's fragments store their own cells, through a scratch where they cannot land
    /// them on their own.
    Fragments,
    /// The planes drain the box into shared memory, laid dense, and the TMA engine stores it in
    /// one bulk copy, which the descriptor clips at the output's edge: every row of the box leaves
    /// whole, where a fragment's cells leave a few bytes a row each.
    Tma,
}

/// Why the TMA engine cannot store a cube microkernel's box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeStagedTmaStoreRefusal {
    /// The device offers no bulk tensor copy ([`Hardware::bulk_tensor_copy`]).
    NoBulkTensorCopy,
    /// The output is a sink a caller binds, which has no address for a descriptor to encode.
    Sink,
    /// The cubes do not each write their box once: they take turns at it, or merge their parts
    /// in it, and those meetings read the output back.
    Shared(MatmulCubeWrite),
    /// The planes split `K` and merge their partials in shared memory, which the cube then copies
    /// out itself.
    SplitAlongK,
    /// A descriptor addresses one matrix, and the problem holds several.
    Batched,
    /// The output's rows are not whole 16-byte lines, which a descriptor's strides must be.
    Misaligned,
    /// The output's view starts at an address a descriptor cannot encode.
    UnalignedStart,
    /// The box's edge is past what one descriptor's box holds.
    BoxPastDescriptor,
    /// The product is a packed one's, its box stated in the blocks the packed input's scales cut
    /// and the positions inside one, where a descriptor's box is the output's rows and columns.
    BlockSplit,
}

impl CubeStagedTmaStoreRefusal {
    /// Why `hw` cannot store the boxes of `item`'s product, over buffers as `buffers` states them,
    /// in the box `cube_space`, its cubes reaching the output as `write` and its planes split
    /// `planes_along_k` ways along `K`, by the TMA engine; `None` where it can.
    pub fn new(
        hw: &Hardware,
        item: &MatmulWorkItem,
        buffers: &MatmulBuffers,
        cube_space: &MatmulCubeSpace,
        write: MatmulCubeWrite,
        planes_along_k: NonZeroUsize,
    ) -> Option<Self> {
        let out = MatmulOperand::Out;
        let bits = item.elements.get(out).size_bits().max(1);
        if !hw.bulk_tensor_copy() {
            Some(Self::NoBulkTensorCopy)
        } else if item.destination == MatmulDestination::Sink {
            Some(Self::Sink)
        } else if write.shares_box() {
            Some(Self::Shared(write))
        } else if planes_along_k.get() > 1 {
            Some(Self::SplitAlongK)
        } else if cube_space.batched {
            Some(Self::Batched)
        } else if cube_space.axes_kind == MatmulAxesKind::BlockSplit {
            Some(Self::BlockSplit)
        } else if buffers.alignments.get(out).is_stored_piece()
            || !buffers
                .alignments
                .get(out)
                .admits((TMA_ALIGNMENT_BYTES * 8 / bits).max(1))
        {
            Some(Self::Misaligned)
        } else if !buffers
            .start_bytes
            .get(out)
            .is_multiple_of(TMA_ALIGNMENT_BYTES as u64)
        {
            Some(Self::UnalignedStart)
        } else if Delivery::Tma
            .moves(&cube_space.space.subspace(&[M, N]))
            .is_err()
        {
            Some(Self::BoxPastDescriptor)
        } else {
            None
        }
    }
}

impl CubeStagedStore {
    /// Bytes of shared memory the box it stores through takes, `elem_bytes` a cell, aligned as a
    /// bulk copy's box is: none where the fragments store.
    pub fn box_bytes(self, stage: &MatmulStage, elem_bytes: usize) -> usize {
        match self {
            Self::Fragments => 0,
            Self::Tma => {
                stage.extent(M) * stage.extent(N) * elem_bytes + Delivery::TMA_STAGE_ALIGNMENT
            }
        }
    }

    /// What it asks of the output's width: a descriptor moves whole elements, so a box the TMA
    /// engine stores binds the output one element wide.
    pub fn vectorize(self, vectorization: &mut Vectorization<MatmulOperand>) {
        if self == Self::Tma {
            vectorization.require(VectorRequirement::Elementwise {
                operand: MatmulOperand::Out,
            });
        }
    }
}

impl From<CubeStagedTmaStoreRefusal> for Refusal {
    fn from(refusal: CubeStagedTmaStoreRefusal) -> Self {
        let why = match refusal {
            CubeStagedTmaStoreRefusal::NoBulkTensorCopy => {
                "this device offers no bulk tensor copy for the TMA engine to store the box with"
            }
            CubeStagedTmaStoreRefusal::Sink => {
                "the output is a sink, which has no address for a descriptor to encode"
            }
            CubeStagedTmaStoreRefusal::Shared(_) => {
                "the cubes share their boxes, and their meetings read the output back"
            }
            CubeStagedTmaStoreRefusal::SplitAlongK => {
                "the planes split K and merge in shared memory, which the cube copies out itself"
            }
            CubeStagedTmaStoreRefusal::Batched => {
                "a descriptor addresses one matrix, and the problem holds several"
            }
            CubeStagedTmaStoreRefusal::Misaligned => {
                "the output's rows are not whole 16-byte lines"
            }
            CubeStagedTmaStoreRefusal::UnalignedStart => {
                "the output's view starts at an address a descriptor cannot encode"
            }
            CubeStagedTmaStoreRefusal::BoxPastDescriptor => {
                "the box's edge is past what one descriptor's box holds"
            }
            CubeStagedTmaStoreRefusal::BlockSplit => {
                "the box is stated in a packed input's scale blocks, and a descriptor's box in rows and columns"
            }
        };
        Refusal::Declined { why }
    }
}
