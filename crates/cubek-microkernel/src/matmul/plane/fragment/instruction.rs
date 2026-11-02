use crate::matmul::plane::PlaneFragmentRowsAlong;
use crate::matmul::{MatmulAxes, MatmulDim, MatmulLayout, MatmulOperands, MatmulWorkItem};
use cubecl::features::{MmaConfig, WgmmaConfig};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::launch::Refusal;
use cubecl::tile::ops::matmul::MmaIo;
use cubecl::tile::stage::RowChunks;
use cubecl::tile::{
    Axis, BlockScaledOperands, Instruction, MmaOperands, RegisterBlock, StageStorage,
};
use cubecl::wgmma::WARPGROUP_UNITS;
use cubek_std::layout::BindingOrder;
use cubek_std::microkernel::{ArithmeticInstruction, SWIZZLED_STAGE_UNREAD, StageLayout};
use cubek_std::{BlockScaledMma, Hardware};

/// What a plane's fragments contract through: one of the device's matrix instructions, reached
/// by either transport, or the software instruction every device runs.
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
    /// The software instruction: [`SOFTWARE_UNITS`](Self::SOFTWARE_UNITS) units, each holding
    /// one cell of the tile in registers and contracting
    /// [`SOFTWARE_DEPTH`](Self::SOFTWARE_DEPTH) of `K` into it with vector arithmetic, in the
    /// element `step_sum` says. Asks nothing of the hardware.
    Registers { step_sum: PlaneFragmentStepSum },
    /// The device's warpgroup MMA (Hopper's `wgmma.mma_async`): a group of four planes contracts
    /// one `64 × n` tile together, `config.k` of `K` a step, reading both operands straight out
    /// of a stage the TMA engine landed, and accumulating asynchronously. `n` is the plane
    /// microkernel's columns times the form's granularity.
    Wgmma { config: WgmmaConfig, n: usize },
}

impl PlaneFragmentInstruction {
    /// How the software instruction's units lie over its tile: along `m`, then along `n`.
    ///
    /// One accumulator cell a unit, the smallest a unit can hold: the plane's settings multiply
    /// the instruction, and every cell they multiply it by is a register each unit keeps live
    /// through the whole walk, so the instruction itself adds none a setting did not ask for.
    /// The units share no instruction, so their count is this one's and never the device's
    /// plane width: the instruction runs where the device reports none.
    pub const SOFTWARE_UNITS: [(MatmulDim, usize); 2] = [(MatmulDim::M, 4), (MatmulDim::N, 8)];

    /// How much of `K` one software instruction contracts. It costs no register: the
    /// contraction walks it one line at a time. It is also as deep as a step may contract in
    /// `f16` before its rounding shows ([`PlaneFragmentStepSum::Narrow`]).
    pub const SOFTWARE_DEPTH: usize = 16;

    /// Planes one warpgroup is made of, which issue its MMA together.
    pub const WARPGROUP_PLANES: usize = 4;

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
            Self::Registers { .. } => {
                let [(_, m), (_, n)] = Self::SOFTWARE_UNITS;
                [
                    (MatmulDim::M, m),
                    (MatmulDim::N, n),
                    (MatmulDim::K, Self::SOFTWARE_DEPTH),
                ]
            }
            Self::Wgmma { config, n } => [
                (MatmulDim::M, config.m as usize),
                (MatmulDim::N, n),
                (MatmulDim::K, config.k as usize),
            ],
        }
    }

    /// Planes that issue one instruction together: a warpgroup's
    /// [`WARPGROUP_PLANES`](Self::WARPGROUP_PLANES), one elsewhere. The warpgroup MMA runs only
    /// where planes are [`WARPGROUP_PLANE_UNITS`](Self::WARPGROUP_PLANE_UNITS) wide
    /// ([`PlaneFragmentMicrokernel`](super::PlaneFragmentMicrokernel)'s candidates).
    pub fn planes(self) -> usize {
        match self {
            Self::Wgmma { .. } => Self::WARPGROUP_PLANES,
            Self::Cmma { .. } | Self::Mma { .. } | Self::Registers { .. } => 1,
        }
    }

    /// Units one plane runs the instruction on: the width the device issues a matrix instruction
    /// at, or the software instruction's own count.
    ///
    /// # Errors
    ///
    /// A matrix instruction on a device that does not commit to one width.
    pub fn units(self, hw: &Hardware) -> Result<u32, Refusal> {
        match self {
            Self::Cmma { .. } | Self::Mma { .. } | Self::Wgmma { .. } => hw.plane_dim(),
            Self::Registers { .. } => Ok(Self::software_units() as u32),
        }
    }

    /// The instruction as the tile DSL runs it, a unit holding `cells` accumulator cells.
    ///
    /// The software instruction takes each lhs line's values at fixed positions
    /// ([`RegisterBlock::component_fanout`]): a position read at runtime sends the whole line
    /// through local memory on NVPTX at every step of `K`. The plane microkernel runs on GPUs
    /// alone, where that is the right walk.
    pub fn tile_instruction(self, cells: usize) -> Instruction {
        match self {
            Self::Cmma { .. } => Instruction::Cmma,
            Self::Mma { operands, .. } => Instruction::Mma {
                io: MmaIo::default(),
                operands,
            },
            Self::Registers { .. } => Instruction::Registers {
                config: RegisterBlock::new(cells).component_fanout(),
            },
            Self::Wgmma { .. } => Instruction::Wgmma,
        }
    }

    /// How the instruction's accumulator reaches its destination.
    pub fn drain(self) -> PlaneFragmentDrain {
        match self {
            Self::Cmma { .. } => PlaneFragmentDrain::Bounce,
            Self::Mma { .. } | Self::Wgmma { .. } => PlaneFragmentDrain::CellByCell,
            Self::Registers { .. } => PlaneFragmentDrain::InLines,
        }
    }

    /// Which of the device's arithmetic rates the instruction runs at.
    pub fn arithmetic(self) -> ArithmeticInstruction {
        match self {
            Self::Cmma { .. } | Self::Mma { .. } | Self::Wgmma { .. } => {
                ArithmeticInstruction::Matrix
            }
            Self::Registers { .. } => ArithmeticInstruction::Vector,
        }
    }

    /// The order this instruction reads operands laid out as `layouts` in: a col operand in its
    /// buffer's order where the fragment loads it col-major, so the stage is filled along the
    /// operand's own lines. Both matrix transports load `A` and `B` either way; a warpgroup MMA
    /// loads only `B` col-major. The others as the product names them. The software instruction
    /// reads the role's own order, and under it a transposed tile would be a wrong answer rather
    /// than a slow one.
    ///
    /// Measured on an L4 for a weight gradient's `f32` col lhs at tf32: staged in the product's
    /// order, a unit reads one value at a time across the lhs's lines, and the cmma candidates
    /// lose to the software instruction.
    pub fn orders(self, layouts: MatmulOperands<MatmulLayout>) -> MatmulOperands<BindingOrder> {
        let lhs = match (self, layouts.lhs) {
            (Self::Cmma { .. } | Self::Mma { .. }, MatmulLayout::Col) => {
                BindingOrder::InBufferOrder
            }
            (
                Self::Cmma { .. } | Self::Mma { .. } | Self::Registers { .. } | Self::Wgmma { .. },
                _,
            ) => BindingOrder::Stated,
        };
        let rhs = match (self, layouts.rhs) {
            (Self::Cmma { .. } | Self::Mma { .. } | Self::Wgmma { .. }, MatmulLayout::Col) => {
                BindingOrder::InBufferOrder
            }
            (
                Self::Cmma { .. } | Self::Mma { .. } | Self::Registers { .. } | Self::Wgmma { .. },
                _,
            ) => BindingOrder::Stated,
        };
        MatmulOperands {
            lhs,
            rhs,
            ..MatmulOperands::all(BindingOrder::Stated)
        }
    }

    /// The layout a stage the instruction reads lies in: `setting`'s, or the instruction's own
    /// where it states none; strided for the software instruction, whose units read their cells
    /// where they are, whatever it states.
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
            (Self::Registers { .. }, _) => Ok(StageLayout::Strided),
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
            Self::Registers { .. } => return StageStorage::Strided,
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

    /// The form a block-scaled instruction's operands take: each input's codes, the scales' type
    /// and their block. `None` for any other.
    pub fn form(self) -> Option<BlockScaledOperands> {
        match self {
            Self::Mma { operands, .. } => operands.block_scaled(),
            Self::Cmma { .. } | Self::Registers { .. } | Self::Wgmma { .. } => None,
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

    /// The element a step contracts in: a matrix instruction's own, which sums straight into
    /// the accumulator.
    pub fn step_sum(self) -> PlaneFragmentStepSum {
        match self {
            Self::Cmma { .. } | Self::Mma { .. } | Self::Wgmma { .. } => PlaneFragmentStepSum::Wide,
            Self::Registers { step_sum } => step_sum,
        }
    }

    fn software_units() -> usize {
        Self::SOFTWARE_UNITS
            .iter()
            .map(|&(_, units)| units)
            .product()
    }
}

/// Which transport a plane reaches the device's matrix instruction through first, where the device
/// offers one under both: the other follows it, and the software instruction follows both.
///
/// A setting rather than a candidate's order, an exception to candidates being searched and never
/// tuned: the two emit the same instruction, so the hardware does not settle which is faster.
/// Each transport loads the fragments its own way, and which way a layout of the rhs favours
/// is a measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneFragmentTransport {
    /// The vendor's fragment API first ([`PlaneFragmentInstruction::Cmma`]).
    Cmma,
    /// The tile DSL's own transport first ([`PlaneFragmentInstruction::Mma`]).
    Mma,
    /// The warpgroup MMA alone ([`PlaneFragmentInstruction::Wgmma`]), where it runs, and nothing
    /// where it does not: a setting that asks for it is laid out for it, a plane group's one
    /// accumulator a row of `columns` granules, which no other instruction is measured at.
    Wgmma,
}

/// The element one instruction step of the software instruction contracts in.
///
/// An explicit opt-in, never a tune candidate in a race: [`Narrow`](Self::Narrow) is faster where
/// the device runs packed `f16` quickly and changes what the product computes, and a race weighs
/// time alone. The caller that states it owns the trade: its inputs must keep a step's
/// partial inside `f16`'s range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneFragmentStepSum {
    /// Straight into the accumulator, in the element it sums in.
    Wide,
    /// In the operands' element, into a partial emptied at every step and added into the
    /// accumulator at its end: on `f16` operands, packed `f16` arithmetic. Its rounding is one
    /// step's, [`SOFTWARE_DEPTH`](PlaneFragmentInstruction::SOFTWARE_DEPTH) products deep, and
    /// never the walk's; its range is `f16`'s, so a step whose products sum past 65504 is an
    /// infinity where the wide step is exact.
    Narrow,
}

impl PlaneFragmentStepSum {
    /// This setting where `item` admits it: a narrow step only for `f16` operands summed in a
    /// wider element, since `bf16`'s mantissa is too short for any step and an `f32` step is
    /// already wide. Elsewhere the wide step, whose kernel a narrow setting then repeats.
    pub fn serving(self, item: &MatmulWorkItem) -> Self {
        let f16 = ElemType::Float(FloatKind::F16);
        let narrow_serves = item.elements.lhs == f16 && item.elements.rhs == f16 && item.sum != f16;
        match (self, narrow_serves) {
            (Self::Narrow, true) => Self::Narrow,
            (Self::Narrow, false) | (Self::Wide, _) => Self::Wide,
        }
    }
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
    /// Through the destination's write in the register block's lines, which are the rhs's, so
    /// the rhs is bound no wider than the destination.
    InLines,
}

impl core::fmt::Display for PlaneFragmentInstruction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let [(_, m), (_, n), (_, k)] = self.shape();
        let name = match self {
            Self::Cmma { .. } => "cmma",
            Self::Mma { .. } => "mma",
            Self::Registers {
                step_sum: PlaneFragmentStepSum::Wide,
            } => "registers",
            Self::Registers {
                step_sum: PlaneFragmentStepSum::Narrow,
            } => "registers-narrow",
            Self::Wgmma { .. } => "wgmma",
        };
        write!(f, "{name}={m}x{n}x{k}")
    }
}
