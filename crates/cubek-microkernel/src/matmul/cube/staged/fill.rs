use crate::matmul::cube::{CubeStagedMicrokernel, MatmulCubeSpace};
use crate::matmul::plane::{PlaneFragmentKStatement, PlaneFragmentMicrokernel};
use crate::matmul::{
    MatmulBlocks, MatmulBuffers, MatmulLayout, MatmulOperand, MatmulOperands, MatmulStage,
    MatmulWorkItem,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::{Delivery, Refusal, VectorRequirement, Vectorization};
use cubecl::tile::stage::Prefetch;
use cubecl::tile::{Axis, Scratch, Space, StageStorage};
use cubek_std::matmul_axes::{K, KB, KI, M, N};
use cubek_std::microkernel::StageRing;
use cubek_std::{Hardware, TMA_ALIGNMENT_BYTES};

/// Who fills the cube microkernel's stages: the cube's units, the copy engine their lines are
/// handed to, or the TMA engine, and how many of the cube's planes do nothing but fill them. The
/// routine's structure, never a setting: each routine fills one way and refuses a problem it
/// cannot fill so.
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

/// Why the TMA engine cannot fill a cube microkernel's stages.
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
    /// A block-split product's packed input states no tile a descriptor lands
    /// ([`MatmulBuffers::stored_tiles`]): a stack of matrices in rows, rows whose step is not
    /// whole 16-byte lines, or tiles laid block by block, whose tile rows are no row's run.
    NoStoredTile(MatmulOperand),
    /// Planes share each step of `K` over a packed input's stage deeper than one of its stored
    /// tiles, which a descriptor lands tile after tile: each plane's slice of `K` would read rows
    /// of the whole stage where the tiles hold them apart.
    SplitAcrossTiles(MatmulOperand),
}

/// Why the copy engine cannot fill a cube microkernel's stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeStagedAsyncCopyRefusal {
    /// The device offers no copy engine, or no barrier for its copies to land on
    /// ([`Hardware::unit_async_copy`]).
    NoCopyEngine,
    /// The ring holds its next stage in the units' registers, which the engine never passes
    /// through.
    PrefetchInRegisters,
    /// An operand's lines are not 4, 8 or 16 bytes, the only copies the engine makes. Known once
    /// the widths are ([`of_widths`](Self::of_widths)).
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
    pub fn of_widths(
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

/// Why a cube microkernel cannot set planes aside to only fill its stages.
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
    /// Why `hw` cannot deliver the stages `microkernel` walks through the space `cube_space` of
    /// `item`'s product, over buffers as `buffers` states them, by the TMA engine, its stages held
    /// as its instruction reads a landed one
    /// (`PlaneFragmentMicrokernel::landed_stage_storage`);
    /// `None` where it can.
    pub fn new(
        hw: &Hardware,
        item: &MatmulWorkItem,
        buffers: &MatmulBuffers,
        cube_space: &MatmulCubeSpace,
        microkernel: &CubeStagedMicrokernel,
    ) -> Option<Self> {
        let stage = &microkernel.stage(cube_space);
        let storage = &microkernel.plane.landed_stage_storage(stage);
        let ring = &microkernel.ring;
        let planes_along_k = microkernel.planes_along_k;
        let microkernel = &microkernel.plane;
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
            Self::of_blocks(
                item,
                blocks,
                buffers,
                stage,
                storage,
                microkernel,
                planes_along_k,
            )
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
    /// where it is. A packed input's words land through a descriptor over the tiles they are
    /// stored in, a stage of whole tiles along its lines, at most one tile deep or whole tiles
    /// deep, its scales copied beside them by the units: the rhs's always, the lhs's where it is
    /// packed too, as a block-scaled product reads it. A float lhs lands through a descriptor over
    /// the matrix its blocks split, a stage's rows its blocks of `K` side by side. Planes sharing
    /// each step of `K` (`planes_along_k` past one) read a packed input's stage only where it
    /// lies within one of its tiles along `K`.
    fn of_blocks(
        item: &MatmulWorkItem,
        blocks: &MatmulBlocks,
        buffers: &MatmulBuffers,
        stage: &MatmulStage,
        storage: &StageStorage,
        microkernel: &PlaneFragmentMicrokernel,
        planes_along_k: NonZeroUsize,
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
        let packed = |operand: MatmulOperand| blocks.packings.get(operand).is_some();
        // Whole tiles of a packed input's words along the stage's rows, and a stage depth of whole
        // tiles or a part dividing one, in whole 16-byte lines.
        let lands_tiles = |operand: MatmulOperand, space: &Space, rows: Axis| {
            let depth = space.extent(KB) * space.extent(KI);
            let words_bytes = depth * bits(operand) / 8;
            buffers.stored_tiles.get(operand).is_some_and(|tile| {
                space.extent(rows).is_multiple_of(tile.rows)
                    && (tile.columns.is_multiple_of(depth) || depth.is_multiple_of(tile.columns))
                    && words_bytes.is_multiple_of(TMA_ALIGNMENT_BYTES)
            })
        };
        // A packed input's descriptor moves a word of its values a cell, a float one a value.
        let cell = |operand: MatmulOperand| match packed(operand) {
            true => u32::BITS as usize / bits(operand),
            false => 1,
        };
        // A stage deeper than a tile lands tile after tile, each tile's rows one after another,
        // which planes splitting `K` read as rows of the whole stage.
        let tiles_deep = |operand: MatmulOperand, space: &Space| {
            let depth = space.extent(KB) * space.extent(KI);
            buffers
                .stored_tiles
                .get(operand)
                .is_some_and(|tile| depth > tile.columns)
        };
        let packed_refusal = |operand: MatmulOperand, space: &Space, rows: Axis| {
            if buffers.stored_tiles.get(operand).is_none() {
                Some(Self::NoStoredTile(operand))
            } else if !lands_tiles(operand, space, rows) {
                Some(Self::StoredInTiles(operand))
            } else if planes_along_k.get() > 1 && tiles_deep(operand, space) {
                Some(Self::SplitAcrossTiles(operand))
            } else {
                None
            }
        };
        if microkernel.k_stated != PlaneFragmentKStatement::Blocks {
            Some(Self::Decoded)
        } else if !packed(MatmulOperand::Rhs) {
            Some(Self::StoredInTiles(MatmulOperand::Rhs))
        } else if let Some(refusal) = packed_refusal(MatmulOperand::Rhs, &rhs, axes.n) {
            Some(refusal)
        } else if let Some(refusal) = packed(MatmulOperand::Lhs)
            .then(|| packed_refusal(MatmulOperand::Lhs, &lhs, axes.m))
            .flatten()
        {
            Some(refusal)
        } else if !packed(MatmulOperand::Lhs)
            && !buffers
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
        } else if let Some(operand) = [(MatmulOperand::Lhs, &lhs), (MatmulOperand::Rhs, &rhs)]
            .into_iter()
            .find(|&(operand, space)| {
                storage
                    .tma_swizzle(space, cell(operand), bits(operand))
                    .is_err()
            })
            .map(|(operand, _)| operand)
        {
            Some(Self::Unlanded(operand))
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

impl From<CubeStagedTmaRefusal> for Refusal {
    fn from(refusal: CubeStagedTmaRefusal) -> Self {
        let why = match refusal {
            CubeStagedTmaRefusal::NoBulkTensorCopy => {
                "this device offers no bulk tensor copy for the TMA engine to fill the stages with"
            }
            CubeStagedTmaRefusal::PrefetchInRegisters => {
                "the ring holds its next stage in registers, which a bulk copy never passes through"
            }
            CubeStagedTmaRefusal::Batched => {
                "a descriptor addresses one matrix, and the problem holds several"
            }
            CubeStagedTmaRefusal::StridedWeight => {
                "the rhs is contiguous along neither axis, and a descriptor's innermost stride is one element"
            }
            CubeStagedTmaRefusal::StoredInTiles(_) => {
                "an operand is stored in tiles a descriptor cannot land a stage of row by row"
            }
            CubeStagedTmaRefusal::Misaligned(_) => {
                "an operand's rows are not whole 16-byte lines contiguous along their own axis"
            }
            CubeStagedTmaRefusal::UnalignedStart(_) => {
                "an operand's view starts at an address a descriptor cannot encode"
            }
            CubeStagedTmaRefusal::Unlanded(_) => {
                "one box cannot land an operand's stage as its instruction reads it"
            }
            CubeStagedTmaRefusal::BoxPastDescriptor => {
                "a stage's edge is past what one descriptor's box holds"
            }
            CubeStagedTmaRefusal::Decoded => {
                "the stage holds a packed operand decoded, and the TMA engine lands words as stored"
            }
            CubeStagedTmaRefusal::SplitAcrossTiles(_) => {
                "planes split K over a stage deeper than a stored tile, which lands tile after tile"
            }
            CubeStagedTmaRefusal::NoStoredTile(_) => {
                "a packed operand states no tile a descriptor lands its words from"
            }
        };
        Refusal::Declined { why }
    }
}

impl From<CubeStagedAsyncCopyRefusal> for Refusal {
    fn from(refusal: CubeStagedAsyncCopyRefusal) -> Self {
        let why = match refusal {
            CubeStagedAsyncCopyRefusal::NoCopyEngine => {
                "this device offers no copy engine to fill the stages with"
            }
            CubeStagedAsyncCopyRefusal::PrefetchInRegisters => {
                "the ring holds its next stage in registers, which the copy engine never passes through"
            }
            CubeStagedAsyncCopyRefusal::LineSize(_) => {
                "an operand's lines are not 4, 8 or 16 bytes, the only copies the copy engine makes"
            }
        };
        Refusal::Declined { why }
    }
}

impl From<CubeStagedFillingRefusal> for Refusal {
    fn from(refusal: CubeStagedFillingRefusal) -> Self {
        let why = match refusal {
            CubeStagedFillingRefusal::CopiedByUnits => {
                "planes set aside to fill need the TMA engine: the units' fill spreads a stage over every unit"
            }
            CubeStagedFillingRefusal::OneStage => {
                "planes set aside to fill need two stages in flight to run ahead of the reads"
            }
            CubeStagedFillingRefusal::Bounces => {
                "the fragments bounce through a scratch, whose cube barriers the filling planes never reach"
            }
            CubeStagedFillingRefusal::SplitAlongK => {
                "the planes split K and merge behind a cube barrier, which the filling planes never reach"
            }
        };
        Refusal::Declined { why }
    }
}
