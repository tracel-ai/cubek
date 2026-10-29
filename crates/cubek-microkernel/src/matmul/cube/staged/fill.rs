use crate::matmul::cube::MatmulCubeSpace;
use crate::matmul::plane::{PlaneFragmentKStatement, PlaneFragmentMicrokernel};
use crate::matmul::{
    MatmulBlocks, MatmulBuffers, MatmulLayout, MatmulOperand, MatmulOperands, MatmulStage,
    MatmulWorkItem,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::{Delivery, VectorRequirement, Vectorization};
use cubecl::tile::stage::Prefetch;
use cubecl::tile::{Axis, Scratch, StageStorage};
use cubek_std::matmul_axes::{K, KB, KI, M, N};
use cubek_std::microkernel::StageRing;
use cubek_std::{Hardware, TMA_ALIGNMENT_BYTES};

/// Who fills the cube microkernel's stages: the cube's units, the copy engine their lines are
/// handed to, or the TMA engine, and how many of the cube's planes do nothing but fill them. A
/// request the microkernel resolves against the device and the problem
/// ([`resolve`](Self::resolve)), so every setting runs on every problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedFill {
    pub delivery: CubeStagedDelivery,
    /// Planes of the cube set aside to fill its stages, beside the ones that contract, the two
    /// meeting only on each slot's barriers.
    pub filling_planes: usize,
}

/// Who moves a cube microkernel's stage out of global memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeStagedDelivery {
    /// The cube's units copy each stage, every unit its share of its lines.
    Copy,
    /// The cube's units hand each line of their share to the copy engine (`cp.async`) instead of
    /// loading and storing it: the lines land on the slot's barrier after the fill returns, never
    /// passing through a register.
    AsyncCopy,
    /// One descriptor a stage an operand, the copy running on the TMA engine while the units
    /// contract: the kernel's operands are tensor maps, read one element wide.
    Tma,
}

impl CubeStagedDelivery {
    /// The delivery the launch states on the operands it binds, which is how the tile DSL's stage
    /// fill learns whether the units load a line or hand it to the engine. A descriptor's stages
    /// are filled from tensor maps and never from the bound operands, which state the loads.
    pub fn bound(self) -> Delivery {
        match self {
            Self::Copy | Self::Tma => Delivery::SyncPerUnit,
            Self::AsyncCopy => Delivery::AsyncPerUnit,
        }
    }
}

/// Why a stage asked of the TMA engine is copied by the cube's units instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeStagedTmaRefusal {
    /// The device offers no bulk tensor copy, or no barrier for one to land on
    /// ([`Hardware::bulk_tensor_copy`]).
    NoBulkTensorCopy,
    /// The ring holds its next stage in the units' registers, which a bulk copy never passes
    /// through.
    PrefetchInRegisters,
    /// A descriptor addresses one matrix, and the problem holds several.
    Batched,
    /// The rhs is contiguous along neither of its axes, and a descriptor's innermost stride is
    /// one element.
    StridedWeight,
    /// An operand is stored in tiles a descriptor cannot land a stage of row by row: a stage not
    /// whole tiles along its rows, or whose depth does not divide a tile's, stage rows that are not
    /// whole 16-byte lines, or tiles the buffers do not state ([`MatmulBuffers::stored_tiles`]).
    StoredInTiles(MatmulOperand),
    /// An operand's rows are not whole 16-byte lines contiguous along their own axis, which a
    /// descriptor's strides must be: a transposed lhs, among others.
    Misaligned(MatmulOperand),
    /// An operand's view starts partway into its buffer at an address a descriptor cannot
    /// encode, which it would refuse at launch rather than here.
    UnalignedStart(MatmulOperand),
    /// One box cannot land the operand's stage as its instruction reads it (the tile DSL's
    /// `StageStorage::tma_swizzle`): its rows longer than the 128 bytes a descriptor swizzles,
    /// among others.
    Unlanded(MatmulOperand),
    /// A stage's edge is past what one descriptor's box holds.
    BoxPastDescriptor,
    /// A block-split product's stage holds a packed operand decoded, which the fill writes value
    /// by value: the engine lands the words only as they are stored.
    Decoded,
    /// A block-split product's lhs is packed: its descriptors land a float lhs beside a packed
    /// rhs's words.
    PackedLhs,
}

/// Why a stage asked of the copy engine is copied by the cube's units instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeStagedAsyncCopyRefusal {
    /// The device offers no copy engine, or no barrier for its copies to land on
    /// ([`Hardware::unit_async_copy`]).
    NoCopyEngine,
    /// The ring holds its next stage in the units' registers, which the engine never passes
    /// through.
    PrefetchInRegisters,
    /// An operand's lines are not 4, 8 or 16 bytes, the only copies the engine makes. Known once
    /// the widths are ([`CubeStagedFill::at_widths`]).
    LineSize(MatmulOperand),
}

impl CubeStagedAsyncCopyRefusal {
    /// Why `hw` cannot hand `ring`'s stages to the copy engine, the widths aside; `None` where it
    /// can.
    pub fn new(hw: &Hardware, ring: &StageRing) -> Option<Self> {
        if !hw.unit_async_copy() {
            Some(Self::NoCopyEngine)
        } else if ring.prefetch == Prefetch::InRegisters {
            Some(Self::PrefetchInRegisters)
        } else {
            None
        }
    }

    /// Why lines `widths` wide, of `item`'s elements, of the `copied` operands are not ones the
    /// engine copies whole.
    fn of_widths(
        copied: &[MatmulOperand],
        widths: &MatmulOperands<usize>,
        item: &MatmulWorkItem,
    ) -> Option<Self> {
        copied
            .iter()
            .copied()
            .find(|&operand| {
                // Bits rather than bytes: an `e2m1` value is half of one.
                let bytes = widths.get(operand) * item.elements.get(operand).size_bits() / 8;
                !matches!(bytes, 4 | 8 | 16)
            })
            .map(Self::LineSize)
    }
}

/// Why planes asked to only fill a stage are not set aside, and every plane fills and contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeStagedFillingRefusal {
    /// The units copy the stages, or hand them to the copy engine, distributing a stage's lines
    /// over every unit of the cube: planes set aside would leave their share unwritten.
    CopiedByUnits,
    /// One stage in flight: its fill is the last event before its read, with nothing to run
    /// ahead of.
    OneStage,
    /// The fragments bounce through a scratch, whose cube barriers the filling planes never reach.
    Bounces,
    /// The planes split `K` and merge in shared memory behind a cube barrier, which the filling
    /// planes never reach.
    SplitAlongK,
}

impl CubeStagedTmaRefusal {
    /// Why `hw` cannot deliver the `stage` of `item`'s product a cube walks through the space
    /// `cube_space`, over buffers as `buffers` states them, held as `storage` (the instruction's
    /// [`landed_stage_storage`](crate::matmul::plane::PlaneFragmentInstruction::landed_stage_storage)),
    /// by the TMA engine, to `microkernel`'s fragments through `ring`; `None` where it can.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hw: &Hardware,
        item: &MatmulWorkItem,
        buffers: &MatmulBuffers,
        cube_space: &MatmulCubeSpace,
        stage: &MatmulStage,
        storage: &StageStorage,
        ring: &StageRing,
        microkernel: &PlaneFragmentMicrokernel,
    ) -> Option<Self> {
        // Bits rather than bytes: an `e2m1` value is half of one.
        let bits = |operand: MatmulOperand| item.elements.get(operand).size_bits().max(1);
        // A stage of whole stored tiles along its rows, bound rows first, `K` along them, at most
        // one tile deep and dividing it, each stage row whole lines: what a descriptor over the
        // tiles lands row by row, a box's inner row a stage row padded to the swizzle's span. A
        // stage shallower than a tile lands part of each tile's run, which never straddles two.
        let lands_tiles = |operand: MatmulOperand| {
            let space = stage.operand_space(operand, microkernel.orders);
            buffers.stored_tiles.get(operand).is_some_and(|tile| {
                space.axis_at(1) == K
                    && space.extent_at(0).is_multiple_of(tile.rows)
                    && tile.columns.is_multiple_of(space.extent_at(1))
                    && (space.extent_at(1) * bits(operand) / 8).is_multiple_of(TMA_ALIGNMENT_BYTES)
            })
        };
        let rows_aligned = |operand: MatmulOperand| {
            lands_tiles(operand)
                || buffers
                    .alignments
                    .get(operand)
                    .admits((TMA_ALIGNMENT_BYTES * 8 / bits(operand)).max(1))
        };
        let staged = MatmulOperand::INPUTS;
        // Each operand's stage lands one cell the descriptor moves a line, as
        // `CubeStagedFill::vectorize` binds it.
        let landed = |operand: MatmulOperand| {
            let space = stage.operand_space(operand, microkernel.orders);
            let element = item.elements.get(operand);
            // A value under a byte moves in the word it is packed in, which is what a packed
            // operand's descriptor is built over.
            let cell = match element.size_bits() {
                bits if bits < 8 => u32::BITS as usize / bits,
                _ => 1,
            };
            storage
                .tma_swizzle(&space, cell, element.size_bits())
                .is_ok()
        };
        if !hw.bulk_tensor_copy() {
            Some(Self::NoBulkTensorCopy)
        } else if ring.prefetch == Prefetch::InRegisters {
            Some(Self::PrefetchInRegisters)
        } else if cube_space.batched {
            Some(Self::Batched)
        } else if let Some(blocks) = &item.blocks {
            Self::of_blocks(item, blocks, buffers, stage, storage, microkernel)
        } else if item.layouts.rhs == MatmulLayout::Strided {
            Some(Self::StridedWeight)
        } else if let Some(operand) = staged.into_iter().find(|&operand| {
            let stored = buffers.alignments.get(operand).is_stored_piece()
                || buffers.stored_tiles.get(operand).is_some();
            stored && !lands_tiles(operand)
        }) {
            Some(Self::StoredInTiles(operand))
        } else if let Some(operand) = staged.into_iter().find(|&operand| !rows_aligned(operand)) {
            Some(Self::Misaligned(operand))
        } else if let Some(operand) = staged.into_iter().find(|&operand| {
            !buffers
                .start_bytes
                .get(operand)
                .is_multiple_of(TMA_ALIGNMENT_BYTES as u64)
        }) {
            Some(Self::UnalignedStart(operand))
        } else if let Some(operand) = staged.into_iter().find(|&operand| !landed(operand)) {
            Some(Self::Unlanded(operand))
        } else if Delivery::Tma
            .moves(&stage.space().subspace(&[M, N, K]))
            .is_err()
        {
            Some(Self::BoxPastDescriptor)
        } else {
            None
        }
    }
}

impl CubeStagedTmaRefusal {
    /// Why a stage of a product in the block-split space `blocks`, over buffers as `buffers`
    /// states them, held as `storage`, is not one the engine lands for `microkernel`; `None`
    /// where it is. A float lhs lands through a descriptor over the matrix its blocks split, a
    /// stage's rows its blocks of `K` side by side; the packed rhs's words through one over the
    /// tiles they are stored in, a stage of whole tiles along its lines, at most one tile deep or
    /// whole tiles deep; the rhs's scales are copied beside them by the units.
    fn of_blocks(
        item: &MatmulWorkItem,
        blocks: &MatmulBlocks,
        buffers: &MatmulBuffers,
        stage: &MatmulStage,
        storage: &StageStorage,
        microkernel: &PlaneFragmentMicrokernel,
    ) -> Option<Self> {
        let axes = stage.axes();
        let landed = |operand: MatmulOperand, rows: Axis| {
            stage.landed_space(operand, microkernel.orders, &[rows, KB, KI])
        };
        let (lhs, rhs) = (
            landed(MatmulOperand::Lhs, axes.m),
            landed(MatmulOperand::Rhs, axes.n),
        );
        let bits = |operand: MatmulOperand| item.elements.get(operand).size_bits();
        let depth = rhs.extent(KB) * rhs.extent(KI);
        // A stage's depth in packed words is whole 16-byte lines.
        let words_bytes = depth * bits(MatmulOperand::Rhs) / 8;
        let rhs_tiles = buffers.stored_tiles.rhs.is_some_and(|tile| {
            rhs.extent(axes.n).is_multiple_of(tile.rows)
                && (tile.columns.is_multiple_of(depth) || depth.is_multiple_of(tile.columns))
                && words_bytes.is_multiple_of(TMA_ALIGNMENT_BYTES)
        });
        if microkernel.k_stated != PlaneFragmentKStatement::Blocks {
            Some(Self::Decoded)
        } else if blocks.packings.lhs.is_some() {
            Some(Self::PackedLhs)
        } else if blocks.packings.rhs.is_none() || !rhs_tiles {
            Some(Self::StoredInTiles(MatmulOperand::Rhs))
        } else if !buffers
            .alignments
            .lhs
            .admits(TMA_ALIGNMENT_BYTES * 8 / bits(MatmulOperand::Lhs))
        {
            Some(Self::Misaligned(MatmulOperand::Lhs))
        } else if let Some(operand) = MatmulOperand::INPUTS.into_iter().find(|&operand| {
            !buffers
                .start_bytes
                .get(operand)
                .is_multiple_of(TMA_ALIGNMENT_BYTES as u64)
        }) {
            Some(Self::UnalignedStart(operand))
        } else if storage
            .tma_swizzle(&lhs, 1, bits(MatmulOperand::Lhs))
            .is_err()
        {
            Some(Self::Unlanded(MatmulOperand::Lhs))
        } else if storage
            .tma_swizzle(
                &rhs,
                u32::BITS as usize / bits(MatmulOperand::Rhs),
                bits(MatmulOperand::Rhs),
            )
            .is_err()
        {
            Some(Self::Unlanded(MatmulOperand::Rhs))
        } else if [&lhs, &rhs]
            .into_iter()
            .any(|space| Delivery::Tma.moves(space).is_err())
        {
            Some(Self::BoxPastDescriptor)
        } else {
            None
        }
    }
}

impl CubeStagedFillingRefusal {
    /// Why a stage `delivery` fills through `ring`, its fragments draining
    /// through `scratch` and its planes split `planes_along_k` ways along `K`, keeps every plane
    /// contracting; `None` where planes may be set aside to fill it. The barriers the two sets
    /// meet on are the TMA engine's, which a TMA delivery already asked of the device.
    pub fn new(
        delivery: CubeStagedDelivery,
        ring: &StageRing,
        scratch: Scratch,
        planes_along_k: NonZeroUsize,
    ) -> Option<Self> {
        if delivery != CubeStagedDelivery::Tma {
            Some(Self::CopiedByUnits)
        } else if ring.stages_in_flight.get() < 2 {
            Some(Self::OneStage)
        } else if scratch != Scratch::None {
            Some(Self::Bounces)
        } else if planes_along_k.get() > 1 {
            Some(Self::SplitAlongK)
        } else {
            None
        }
    }
}

impl CubeStagedFill {
    /// Every unit copies its share of each stage, and every plane contracts.
    pub const BY_UNITS: Self = Self {
        delivery: CubeStagedDelivery::Copy,
        filling_planes: 0,
    };

    /// This fill as `hw` runs it for `item` over `buffers`, a cube covering `cube_space` staging
    /// `stage` through `ring` into `microkernel`'s fragments, which drain through `scratch`, its
    /// planes split `planes_along_k` ways along `K`: the TMA engine where nothing refuses it a
    /// stage held as `landed` ([`CubeStagedTmaRefusal`]), the units elsewhere; the planes set aside
    /// where nothing refuses them ([`CubeStagedFillingRefusal`]), none elsewhere.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        self,
        hw: &Hardware,
        item: &MatmulWorkItem,
        buffers: &MatmulBuffers,
        cube_space: &MatmulCubeSpace,
        stage: &MatmulStage,
        landed: &StageStorage,
        ring: &StageRing,
        microkernel: &PlaneFragmentMicrokernel,
        scratch: Scratch,
        planes_along_k: NonZeroUsize,
    ) -> Self {
        let delivery = match self.delivery {
            CubeStagedDelivery::Tma => {
                match CubeStagedTmaRefusal::new(
                    hw,
                    item,
                    buffers,
                    cube_space,
                    stage,
                    landed,
                    ring,
                    microkernel,
                ) {
                    None => CubeStagedDelivery::Tma,
                    Some(_) => CubeStagedDelivery::Copy,
                }
            }
            CubeStagedDelivery::AsyncCopy => match CubeStagedAsyncCopyRefusal::new(hw, ring) {
                None => CubeStagedDelivery::AsyncCopy,
                Some(_) => CubeStagedDelivery::Copy,
            },
            CubeStagedDelivery::Copy => CubeStagedDelivery::Copy,
        };
        let filling_planes =
            match CubeStagedFillingRefusal::new(delivery, ring, scratch, planes_along_k) {
                None => self.filling_planes,
                Some(_) => 0,
            };
        Self {
            delivery,
            filling_planes,
        }
    }

    /// This fill once `item`'s operands bind `widths` wide: a stage asked of the copy engine
    /// whose lines it does not copy whole ([`CubeStagedAsyncCopyRefusal::LineSize`]) is copied by
    /// the units, whose loads bind the same widths, so the widths stand.
    pub fn at_widths(self, widths: &MatmulOperands<usize>, item: &MatmulWorkItem) -> Self {
        self.at_widths_of(&MatmulOperand::INPUTS, widths, item)
    }

    /// [`at_widths`](Self::at_widths) where the engine copies only the `copied` operands' stages,
    /// the units loading the others: a product whose fill decodes a packed input loads its words
    /// itself
    /// ([`MatmulMicrokernel::words_delivery`](crate::matmul::MatmulMicrokernel::words_delivery)).
    pub fn at_widths_of(
        self,
        copied: &[MatmulOperand],
        widths: &MatmulOperands<usize>,
        item: &MatmulWorkItem,
    ) -> Self {
        match self.delivery {
            CubeStagedDelivery::AsyncCopy
                if CubeStagedAsyncCopyRefusal::of_widths(copied, widths, item).is_some() =>
            {
                Self {
                    delivery: CubeStagedDelivery::Copy,
                    ..self
                }
            }
            _ => self,
        }
    }

    /// What its delivery asks of the operands' widths: a descriptor moves whole elements, so a
    /// TMA-filled stage reads both operands one element wide.
    pub fn vectorize(&self, vectorization: &mut Vectorization<MatmulOperand>) {
        if self.delivery == CubeStagedDelivery::Tma {
            for operand in MatmulOperand::INPUTS {
                vectorization.require(VectorRequirement::Elementwise { operand });
            }
        }
    }
}

/// Nothing for a fill by every unit, so the settings that hold one keep their names.
impl core::fmt::Display for CubeStagedFill {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.delivery {
            CubeStagedDelivery::Copy => {}
            CubeStagedDelivery::AsyncCopy => write!(f, " delivery=async")?,
            CubeStagedDelivery::Tma => write!(f, " delivery=tma")?,
        }
        match self.filling_planes {
            0 => Ok(()),
            planes => write!(f, " filling_planes={planes}"),
        }
    }
}
