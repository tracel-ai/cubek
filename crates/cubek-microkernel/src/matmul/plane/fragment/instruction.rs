use crate::matmul::plane::PlaneFragmentRowsAlong;
use crate::matmul::{MatmulAxes, MatmulDim, MatmulLayout, MatmulOperands};
use cubecl::features::{MmaConfig, WgmmaConfig};
use cubecl::ir::ElemType;
use cubecl::tile::launch::Refusal;
use cubecl::tile::ops::matmul::MmaIo;
use cubecl::tile::stage::RowChunks;
use cubecl::tile::{Axis, BlockScaledOperands, Instruction, MmaOperands, StageStorage};
use cubecl::wgmma::WARPGROUP_UNITS;
use cubek_std::layout::BindingOrder;
use cubek_std::microkernel::{ArithmeticInstruction, SWIZZLED_STAGE_UNREAD, StageLayout};
use cubek_std::{BlockScaledMma, Hardware};

/// What a plane's fragments contract through: one of the device's matrix instructions, reached
/// by either transport, the one the tile layer picks for it.
///
/// Each is a tile of `m × n × k` the plane's units hold between them, which the plane
/// microkernel's settings multiply into the tile one plane owns, so a setting means the same
/// whichever instruction it multiplies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneFragmentInstruction {
    /// A matrix instruction of the device, through cubecl's fragment operations (`wmma` on CUDA;
    /// on HIP, cubecl's LLVM lowering to the AMDGPU `wmma` intrinsics).
    Cmma { shape: MmaConfig },
    /// A matrix instruction of the device, through the tile DSL's own fragment transport, as the
    /// tile DSL chooses it ([`MmaIo::default`]): an operand through `ldmatrix` where the device
    /// offers it and the stage serves it, and every unit its own cells elsewhere and for the store.
    /// `operands` says what it reads: values as they are, or packed codes under their block scales
    /// as they lie, through the device's block-scaled instruction of that form.
    Mma {
        shape: MmaConfig,
        operands: MmaOperands,
    },
    /// The device's warpgroup MMA (Hopper's `wgmma.mma_async`): a group of four planes contracts
    /// one `64 × n` tile together, `config.k` of `K` a step, reading both operands straight out
    /// of a stage the TMA engine landed, and accumulating asynchronously. `n` is the plane
    /// microkernel's columns times the form's granularity.
    Wgmma { config: WgmmaConfig, n: usize },
}

impl PlaneFragmentInstruction {
    /// Planes one warpgroup is made of, which issue its MMA together.
    pub(crate) const WARPGROUP_PLANES: usize = 4;

    /// Units in each plane of a warpgroup: the only plane width its MMA runs at.
    pub const WARPGROUP_PLANE_UNITS: usize = WARPGROUP_UNITS / Self::WARPGROUP_PLANES;

    /// The instruction's `m × n × k` laid with its rows along `rows`: its own where they lie
    /// along `m`, its rows' and columns' edges exchanged where they lie along `n`.
    pub fn shape_along(self, rows: PlaneFragmentRowsAlong) -> [(MatmulDim, usize); 3] {
        let [(_, m), (_, n), k] = self.shape();
        match rows {
            PlaneFragmentRowsAlong::M => [(MatmulDim::M, m), (MatmulDim::N, n), k],
            PlaneFragmentRowsAlong::N => [(MatmulDim::M, n), (MatmulDim::N, m), k],
        }
    }

    /// The instruction's `m × n × k`, over the whole plane.
    pub fn shape(self) -> [(MatmulDim, usize); 3] {
        match self {
            Self::Cmma { shape } | Self::Mma { shape, .. } => [
                (MatmulDim::M, shape.m as usize),
                (MatmulDim::N, shape.n as usize),
                (MatmulDim::K, shape.k as usize),
            ],
            Self::Wgmma { config, n } => [
                (MatmulDim::M, config.m as usize),
                (MatmulDim::N, n),
                (MatmulDim::K, config.k as usize),
            ],
        }
    }

    /// Planes that issue one instruction together: a warpgroup's
    /// `WARPGROUP_PLANES`, one elsewhere. The warpgroup MMA runs only
    /// where planes are [`WARPGROUP_PLANE_UNITS`](Self::WARPGROUP_PLANE_UNITS) wide
    /// ([`PlaneFragmentMicrokernel`](super::PlaneFragmentMicrokernel)'s candidates).
    pub fn planes(self) -> usize {
        match self {
            Self::Wgmma { .. } => Self::WARPGROUP_PLANES,
            Self::Cmma { .. } | Self::Mma { .. } => 1,
        }
    }

    /// Units one plane runs the instruction on: the width the device issues a matrix instruction
    /// at.
    ///
    /// # Errors
    ///
    /// A device that does not commit to one width.
    pub fn units(self, hw: &Hardware) -> Result<u32, Refusal> {
        hw.plane_dim()
    }

    /// The instruction as the tile DSL names it: what the tile layer picks for a plane's
    /// fragment of this shape, where this is the tile layer's pick.
    pub fn tile_instruction(self) -> Instruction {
        match self {
            Self::Cmma { .. } => Instruction::Cmma,
            Self::Mma { operands, .. } => Instruction::Mma {
                io: MmaIo::default(),
                operands,
            },
            Self::Wgmma { .. } => Instruction::Wgmma,
        }
    }

    /// How the instruction's accumulator reaches its destination.
    pub fn drain(self) -> PlaneFragmentDrain {
        match self {
            Self::Cmma { .. } => PlaneFragmentDrain::Bounce,
            Self::Mma { .. } | Self::Wgmma { .. } => PlaneFragmentDrain::CellByCell,
        }
    }

    /// Which of the device's arithmetic rates the instruction runs at.
    pub fn arithmetic(self) -> ArithmeticInstruction {
        ArithmeticInstruction::Matrix
    }

    /// The order this instruction reads operands laid out as `layouts` in: a col operand in its
    /// buffer's order where the fragment loads it col-major, so the stage is filled along the
    /// operand's own lines. Both matrix transports load `A` and `B` either way; a warpgroup MMA
    /// loads only `B` col-major. The others as the product names them.
    ///
    /// Measured on an L4 for a weight gradient's `f32` col lhs at tf32: staged in the product's
    /// order, a unit reads one value at a time across the lhs's lines, and the cmma candidates
    /// lost to the software instruction a plane once ran.
    pub fn orders(self, layouts: MatmulOperands<MatmulLayout>) -> MatmulOperands<BindingOrder> {
        let lhs = match (self, layouts.lhs) {
            (Self::Cmma { .. } | Self::Mma { .. }, MatmulLayout::Col) => {
                BindingOrder::InBufferOrder
            }
            (Self::Cmma { .. } | Self::Mma { .. } | Self::Wgmma { .. }, _) => BindingOrder::Stated,
        };
        let rhs = match (self, layouts.rhs) {
            (Self::Cmma { .. } | Self::Mma { .. } | Self::Wgmma { .. }, MatmulLayout::Col) => {
                BindingOrder::InBufferOrder
            }
            (Self::Cmma { .. } | Self::Mma { .. } | Self::Wgmma { .. }, _) => BindingOrder::Stated,
        };
        MatmulOperands {
            lhs,
            rhs,
            ..MatmulOperands::all(BindingOrder::Stated)
        }
    }

    /// The layout a stage the instruction reads lies in: `setting`'s, or the instruction's own
    /// where it states none.
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
    pub fn stage_layout(self, setting: Option<StageLayout>) -> Result<StageLayout, Refusal> {
        match (self, setting) {
            (Self::Cmma { .. }, Some(StageLayout::SwizzledBlocks)) => Err(SWIZZLED_STAGE_UNREAD),
            (_, Some(layout)) => Ok(layout),
            (Self::Cmma { .. }, None) => Ok(StageLayout::PaddedBlocks),
            (Self::Mma { .. } | Self::Wgmma { .. }, None) => Ok(StageLayout::SwizzledBlocks),
        }
    }

    /// The tile a fragment loads from a stage the instruction reads: the instruction's shape, its
    /// rows along `rows`, and one of every batch axis in `batches`, so a fragment loads one
    /// contiguous tile where the stage is blocked.
    pub fn stage_tile(
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

    /// How a stage the instruction reads holds its cells where one TMA box lands each operand's
    /// stage whole, `depth` of `K` deep: in tiles of a matrix instruction along `m` and `n`, each
    /// the whole stage deep along `K`, so the tiles stack whole rows one after another as a box
    /// lands them, its rows along `rows`, and one of every batch axis in `batches`. The manual
    /// `mma`'s rows swizzled as the descriptor swizzles them, which keeps the rows a fragment load
    /// reads at once on distinct banks as its own [`stage_layout`](Self::stage_layout) does;
    /// `cmma`'s in order, since a box lands no padding; the software instruction's as the operand
    /// lies, as ever. The warpgroup MMA's swizzled as the manual `mma`'s are: a row one swizzle
    /// span long lies as the instruction's descriptor reads it.
    pub fn landed_stage_storage(
        self,
        batches: &[Axis],
        axes: MatmulAxes,
        depth: usize,
        rows: PlaneFragmentRowsAlong,
    ) -> StageStorage {
        let chunks = match self {
            Self::Cmma { .. } => RowChunks::InOrder,
            Self::Mma { .. } | Self::Wgmma { .. } => RowChunks::Swizzled,
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

    /// Whether this is a block-scaled instruction
    /// ([`block_scaled_mma`](cubek_std::block_scaled_mma)): one that reads packed codes and their
    /// scales as they lie.
    pub fn block_scaled(self) -> bool {
        self.form().is_some()
    }

    /// Whether it contracts a fragment deeper than its own step of `K`, one step after another
    /// into the same sum ([`PlaneFragmentDepth::Stage`](super::PlaneFragmentDepth::Stage)): the
    /// manual `mma` over plain operands alone. The block-scaled instruction reads one step's
    /// codes and scales as they lie, and the device offers it at one depth.
    pub fn contracts_deeper_fragments(self) -> bool {
        matches!(
            self,
            Self::Mma {
                operands: MmaOperands::Plain,
                ..
            }
        )
    }

    /// The form a block-scaled instruction's operands take: each input's codes, the scales' type
    /// and their block. `None` for any other.
    pub fn form(self) -> Option<BlockScaledOperands> {
        match self {
            Self::Mma { operands, .. } => operands.block_scaled(),
            Self::Cmma { .. } | Self::Wgmma { .. } => None,
        }
    }

    /// Whether a stored word of `form`'s block scales is one step of this instruction: the scales
    /// a step reads, one a block of its `k`, fill the word, which is then its scale register.
    pub fn scale_word_is_a_step(self, form: BlockScaledOperands) -> bool {
        let [_, _, (_, k)] = self.shape();
        let scales_per_word = u32::BITS as usize / ElemType::Float(form.scales).size_bits();
        form.block * scales_per_word == k
    }

    /// The plain manual-mma instruction of `shape`.
    pub fn mma(shape: MmaConfig) -> Self {
        Self::Mma {
            shape,
            operands: MmaOperands::Plain,
        }
    }

    /// The device's block-scaled instruction `instruction`, through the manual-mma transport.
    pub fn block_scaled_mma(instruction: BlockScaledMma) -> Self {
        Self::Mma {
            shape: instruction.shape,
            operands: MmaOperands::BlockScaled(instruction.form),
        }
    }
}

/// Which transport an attention's planes reach the device's matrix instruction through: its score
/// holder in disguise, a fragment of registers through `mma` or a window of shared memory through
/// `cmma`. A matmul's plane takes the instruction the tile layer picks for its product, and lists
/// both transports, so an attention picks its own from them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneFragmentTransport {
    /// The vendor's fragment API ([`PlaneFragmentInstruction::Cmma`]).
    Cmma,
    /// The tile DSL's own transport ([`PlaneFragmentInstruction::Mma`]).
    Mma,
}

/// How an instruction's accumulator reaches its destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneFragmentDrain {
    /// Through the fragment's own store, which writes its whole window, where the destination
    /// replaces and the window lies inside it; elsewhere through a window of shared memory, since
    /// a cmma fragment hides which unit holds a cell: each fragment is spilled there and its cells
    /// written on from it.
    Bounce,
    /// Each unit writing its own cells through the destination's write, one element at a time,
    /// so the destination is bound one element wide: a manual mma fragment knows which unit
    /// holds a cell, and its store reaches memory no other way.
    CellByCell,
}

impl core::fmt::Display for PlaneFragmentInstruction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let [(_, m), (_, n), (_, k)] = self.shape();
        let name = match self {
            Self::Cmma { .. } => "cmma",
            Self::Mma { .. } => "mma",
            Self::Wgmma { .. } => "wgmma",
        };
        write!(f, "{name}={m}x{n}x{k}")
    }
}
