use crate::matmul::plane::PlaneFragmentRowsAlong;
use crate::matmul::{MatmulAxes, MatmulDim, MatmulLayout, MatmulOperands};
use cubecl::features::{MmaConfig, WgmmaConfig};
use cubecl::ir::ElemType;
use cubecl::tile::launch::Refusal;
use cubecl::tile::ops::matmul::MmaIo;
use cubecl::tile::stage::RowChunks;
use cubecl::tile::{
    Axis, BlockScaledOperands, Instruction, MmaOperands, ShapedInstruction, StageStorage,
};
use cubecl::wgmma::PLANE_GROUP_UNITS;
use cubek_std::BlockScaledMma;
use cubek_std::layout::BindingOrder;
use cubek_std::microkernel::{SWIZZLED_STAGE_UNREAD, StageLayout};

/// What a plane reads off the call the tile picks for its fragments: the order it binds its
/// operands in, and how a stage it reads lies. The call itself, its instruction and the shape of
/// one call, is the tile's ([`ShapedInstruction`]).
pub trait PlaneCall: Sized {
    /// The call's `m × n × k` laid with its rows along `rows`: its own where they lie along `m`,
    /// its rows' and columns' edges exchanged where they lie along `n`.
    fn shape_along(self, rows: PlaneFragmentRowsAlong) -> [(MatmulDim, usize); 3];

    /// The order the call reads operands laid out as `layouts` in: a col operand in its buffer's
    /// order where the fragment loads it col-major, so the stage is filled along the operand's own
    /// lines. Both matrix transports load `A` and `B` either way; a plane group MMA loads only `B`
    /// col-major. The others as the product names them.
    ///
    /// Measured on an L4 for a weight gradient's `f32` col lhs at tf32: staged in the product's
    /// order, a unit reads one value at a time across the lhs's lines, and the cmma candidates
    /// lose to the unit routines.
    fn orders(self, layouts: MatmulOperands<MatmulLayout>) -> MatmulOperands<BindingOrder>;

    /// The layout a stage the call reads lies in: `setting`'s, or the call's own where it states
    /// none.
    ///
    /// Both matrix instructions' own keep the rows a fragment load reads at once on distinct
    /// banks, each the way its transport can read. `cmma` reads a tile off a pointer and a stride,
    /// so its rows are padded by a chunk: a padded row is still a run a stride from the next. The
    /// manual `mma` hands every row its own address, so its rows' chunks are swizzled instead,
    /// which spends no shared memory.
    ///
    /// Measured against padding: an L4's shared loads go from conflicted to conflict-free at a
    /// level time; gfx1151's bank conflicts fall to zero, 8 to 20% faster on four of six boxes,
    /// the others within noise (a stage 64 deep along a col rhs's `K` up to about 10% slower).
    ///
    /// A block-scaled instruction's stage holds the stored words in the same swizzled tiles: a
    /// row of its tile is the instruction's `k` of words, read eight rows at a time as a 16-bit
    /// instruction's are.
    ///
    /// # Errors
    ///
    /// Swizzled tiles under `cmma`, which has no stride to read them by.
    fn stage_layout(self, setting: Option<StageLayout>) -> Result<StageLayout, Refusal>;

    /// The tile a fragment loads from a stage the call reads: the call's shape, its rows along
    /// `rows`, and one of every batch axis in `batches`, so a fragment loads one contiguous tile
    /// where the stage is blocked.
    fn stage_tile(
        self,
        batches: &[Axis],
        axes: MatmulAxes,
        rows: PlaneFragmentRowsAlong,
    ) -> Vec<(Axis, usize)>;

    /// How a stage the call reads holds its cells where one TMA box lands each operand's stage
    /// whole, `depth` of `K` deep: in tiles of a call along `m` and `n`, each the whole stage deep
    /// along `K`, so the tiles stack whole rows one after another as a box lands them, its rows
    /// along `rows`, and one of every batch axis in `batches`. The manual `mma`'s rows swizzled as
    /// the descriptor swizzles them, which keeps the rows a fragment load reads at once on distinct
    /// banks as its own [`stage_layout`](Self::stage_layout) does; `cmma`'s in order, since a box
    /// lands no padding. The plane group MMA's swizzled as the manual `mma`'s are: a row one
    /// swizzle span long lies as the instruction's descriptor reads it.
    fn landed_stage_storage(
        self,
        batches: &[Axis],
        axes: MatmulAxes,
        depth: usize,
        rows: PlaneFragmentRowsAlong,
    ) -> StageStorage;

    /// Whether it contracts a fragment deeper than its own step of `K`, one step after another
    /// into the same sum ([`PlaneFragmentDepth::Stage`](super::PlaneFragmentDepth::Stage)): the
    /// manual `mma` over plain operands alone. The block-scaled instruction reads one step's
    /// codes and scales as they lie, and the device offers it at one depth.
    fn contracts_deeper_fragments(self) -> bool;

    /// Whether a stored word of `form`'s block scales is one step of this call: the scales a step
    /// reads, one a block of its `k`, fill the word, which is then its scale register.
    fn scale_word_is_a_step(self, form: BlockScaledOperands) -> bool;
}

/// The fragment API's call of `shape`.
pub fn cmma_call(shape: MmaConfig) -> ShapedInstruction {
    ShapedInstruction {
        instruction: Instruction::Cmma,
        shape: edges(shape),
    }
}

/// The manual-mma call of `shape` over plain operands, through the tile's own transport.
pub fn mma_call(shape: MmaConfig) -> ShapedInstruction {
    ShapedInstruction {
        instruction: Instruction::Mma {
            io: MmaIo::default(),
            operands: MmaOperands::Plain,
        },
        shape: edges(shape),
    }
}

/// The device's block-scaled instruction `instruction`, through the manual-mma transport.
pub fn block_scaled_call(instruction: BlockScaledMma) -> ShapedInstruction {
    ShapedInstruction {
        instruction: Instruction::Mma {
            io: MmaIo::default(),
            operands: MmaOperands::BlockScaled(instruction.form),
        },
        shape: edges(instruction.shape),
    }
}

/// The plane group MMA of `config`, `n` wide: a group of four planes contracts one `64 × n` tile
/// together, `config.k` of `K` a step, reading both operands straight out of a stage the TMA
/// engine landed. `n` is the plane microkernel's columns times the form's granularity.
pub fn wgmma_call(config: WgmmaConfig, n: usize) -> ShapedInstruction {
    ShapedInstruction {
        instruction: Instruction::Wgmma,
        shape: [config.m as usize, n, config.k as usize],
    }
}

/// `shape`'s `m × n × k`.
fn edges(shape: MmaConfig) -> [usize; 3] {
    [shape.m, shape.n, shape.k].map(|edge| edge as usize)
}

/// Units in each plane of the group that issues a plane group MMA: the only plane width its MMA
/// runs at.
pub fn plane_group_plane_units() -> usize {
    PLANE_GROUP_UNITS / Instruction::Wgmma.planes()
}

impl PlaneCall for ShapedInstruction {
    fn shape_along(self, rows: PlaneFragmentRowsAlong) -> [(MatmulDim, usize); 3] {
        let [m, n, k] = self.shape;
        match rows {
            PlaneFragmentRowsAlong::M => [(MatmulDim::M, m), (MatmulDim::N, n), (MatmulDim::K, k)],
            PlaneFragmentRowsAlong::N => [(MatmulDim::M, n), (MatmulDim::N, m), (MatmulDim::K, k)],
        }
    }

    fn orders(self, layouts: MatmulOperands<MatmulLayout>) -> MatmulOperands<BindingOrder> {
        let in_buffer = |layout: MatmulLayout, col_major_load: bool| match layout {
            MatmulLayout::Col if col_major_load => BindingOrder::InBufferOrder,
            _ => BindingOrder::Stated,
        };
        let fragments = !matches!(self.instruction, Instruction::Wgmma);
        MatmulOperands {
            lhs: in_buffer(layouts.lhs, fragments),
            rhs: in_buffer(layouts.rhs, true),
            ..MatmulOperands::all(BindingOrder::Stated)
        }
    }

    fn stage_layout(self, setting: Option<StageLayout>) -> Result<StageLayout, Refusal> {
        let cmma = self.instruction == Instruction::Cmma;
        match (cmma, setting) {
            (true, Some(StageLayout::SwizzledBlocks)) => Err(SWIZZLED_STAGE_UNREAD),
            (_, Some(layout)) => Ok(layout),
            (true, None) => Ok(StageLayout::PaddedBlocks),
            (false, None) => Ok(StageLayout::SwizzledBlocks),
        }
    }

    fn stage_tile(
        self,
        batches: &[Axis],
        axes: MatmulAxes,
        rows: PlaneFragmentRowsAlong,
    ) -> Vec<(Axis, usize)> {
        batches
            .iter()
            .map(|&axis| (axis, 1))
            .chain(axes.name(self.shape_along(rows)))
            .collect()
    }

    fn landed_stage_storage(
        self,
        batches: &[Axis],
        axes: MatmulAxes,
        depth: usize,
        rows: PlaneFragmentRowsAlong,
    ) -> StageStorage {
        let chunks = match self.instruction {
            Instruction::Cmma => RowChunks::InOrder,
            _ => RowChunks::Swizzled,
        };
        let [m, n, _] = self.shape_along(rows);
        StageStorage::Tiled {
            block: batches
                .iter()
                .map(|&axis| (axis, 1))
                .chain(axes.name([m, n, (MatmulDim::K, depth)]))
                .collect(),
            chunks,
        }
    }

    fn contracts_deeper_fragments(self) -> bool {
        matches!(
            self.instruction,
            Instruction::Mma {
                operands: MmaOperands::Plain,
                ..
            }
        )
    }

    fn scale_word_is_a_step(self, form: BlockScaledOperands) -> bool {
        let scales_per_word = u32::BITS as usize / ElemType::Float(form.scales).size_bits();
        form.block * scales_per_word == self.depth()
    }
}
