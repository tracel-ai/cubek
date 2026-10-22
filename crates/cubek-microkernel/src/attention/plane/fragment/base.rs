use super::microkernel::PlaneFragmentAttention;
use crate::attention::CubeAttentionStageLayout;
use crate::attention::plane::{InstructionShape, PlaneFragmentAttentionMicrokernelSettings};
use crate::attention::{
    AttentionMicrokernel, AttentionOperand, AttentionPartitioning, AttentionWorkItem,
    CubePlaneKeys, CubePlaneLayout, CubePlaneMicrokernel, CubePlaneReads, CubeStagedPlane, D, DV,
    G, Q, S,
};
use crate::matmul::dense::plane::{
    PlaneFragmentDrain, PlaneFragmentInstruction, PlaneFragmentMicrokernel,
    PlaneFragmentMicrokernelSettings, PlaneFragmentTransport,
};
use crate::matmul::{MatmulDim, MatmulMicrokernel, MatmulOperand, MatmulOrder, MatmulWorkItem};
use core::num::NonZeroUsize;
use cubecl::ir::ElemType;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Axis, Instruction, Level, Levels, Space, StageStorage};
use cubecl::{self as cubecl, features::MmaConfig, prelude::*};
use cubek_std::Hardware;
use cubek_std::manual_mma;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// A box of rows of one head a plane owns: their scores against a stage of keys in the fragments
/// of a matrix instruction, the softmax taken where the plane keeps the score, and their outputs
/// over the whole value dim in fragments held across the walk.
///
/// Its two multiplications are the shared plane fragment microkernel's: the score, `query · keyᵀ`,
/// fragments of its rows by the stage's keys walking the head dim, and the value,
/// `probabilities · value`, fragments of its rows by the value dim walking the stage's keys. The
/// keys are the columns of the one and the contracted axis of the other, cut by the score's `n` in
/// the one and by the value's `k` in the other. Where the plane keeps the score between them is
/// the score's [`ScoreHolder`], which also says whether the two cuts are one grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentAttentionMicrokernel {
    pub settings: PlaneFragmentAttentionMicrokernelSettings,
    /// The element its query, keys and values reach the instruction in: as stored where the
    /// device has an instruction for it, else tf32 for `f32`
    /// ([`readings`](AttentionWorkItem::readings)).
    pub read: ElemType,
    /// The score's fragments: its rows by a stage's keys, walking the whole head dim.
    pub score: PlaneFragmentMicrokernel,
    /// The value's fragments: its rows by the whole value dim, walking a stage's keys.
    pub value: PlaneFragmentMicrokernel,
}

/// What a routine whose planes' rows merge with another holder's declines where a plane keeps its
/// score in fragments, each unit holding the state of its own rows.
pub const ROW_STATE_HELD_BY_UNIT: Refusal = Refusal::Declined {
    why: "planes whose rows merge with another holder's read the score through their \
              windows: a score kept in fragments leaves each unit its own rows' state",
};

/// What a cube declines where its stages are swizzled and its planes read the score through
/// windows: a cmma fragment loads a tile off a pointer and a stride.
pub const SWIZZLED_STAGE_UNREAD: Refusal = Refusal::Declined {
    why: "a cmma fragment reads a stage tile by a stride, which swizzled tiles do not keep",
};

/// Where a plane keeps a stage's score between its two multiplications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScoreHolder {
    /// The plane's windows of shared memory: the score drained into one, the softmax read there,
    /// the probabilities cast into the other and loaded again as fragments. What a cmma fragment,
    /// whose cells lie where the instruction keeps them, is read through.
    Window,
    /// The score's own manual-mma fragments: the softmax reads them where the plane's units hold
    /// them, and the value contracts them, cast, in the same registers. Each unit keeps the state
    /// of the rows it holds cells of, so the plane's state merges with no other holder's.
    Fragments,
}

impl PlaneFragmentAttentionMicrokernel {
    /// Where the plane keeps the score: in its fragments where the instruction lays them out for
    /// the program to read, through its windows otherwise.
    ///
    /// The score's fragments are the value's `A` only where they are one grid, the score's `n` the
    /// value's `k`: [`candidates`](Microkernel::candidates) pairs a manual mma only so. Windows
    /// decouple the two, so a cmma instruction of any shape runs both products.
    pub fn score_holder(&self) -> ScoreHolder {
        match self.score.instruction {
            PlaneFragmentInstruction::Mma { .. } => ScoreHolder::Fragments,
            PlaneFragmentInstruction::Cmma { .. }
            | PlaneFragmentInstruction::Registers { .. }
            | PlaneFragmentInstruction::Wgmma { .. } => ScoreHolder::Window,
        }
    }

    /// The edge a fragment of the output spans along the value dim, which a drain writes at once.
    pub fn out_edge(&self) -> usize {
        self.value.leaf_extent(MatmulDim::N)
    }

    /// The value's instruction as the tile DSL runs it, which the plane's sum is opened in.
    pub fn tile_instruction(&self) -> Instruction {
        self.value.tile_instruction()
    }

    /// The score's own levels where the plane reads it through its windows, which its accumulator
    /// is placed in: the walker's level of planes, `planes` its counts, and below it the score's
    /// fragments, its rows by the stage's keys by the steps of the head dim. The counts are what
    /// index the accumulator's scratch, a window for every plane of the cube; the steps keep the
    /// plane's box the whole head dim, as the query's and the keys' windows are.
    pub fn score_levels(&self, planes: &[(Axis, usize)]) -> Vec<Level> {
        let [(_, m), (_, n), (_, k)] = self.score.instruction.shape();
        let settings = self.score.settings;
        Levels::leaf(&[(Q, m), (S, n), (D, k)])
            .walk(&[
                (Q, settings.along_m.get()),
                (S, settings.along_n.get()),
                (D, settings.depth.get()),
            ])
            .planes(planes)
            .build()
    }

    /// 32-bit registers one unit keeps the softmax's running max and sum in, beside its share of
    /// both multiplications' fragments: at most every row's, which a window's state keeps.
    fn softmax_registers(&self, item: &AttentionWorkItem) -> u32 {
        (2 * self.seq_q() * item.sum.size()).div_ceil(4) as u32
    }

    /// The score as the shared microkernel is handed it: summed into the plane's window.
    fn score_item(item: &AttentionWorkItem) -> MatmulWorkItem {
        item.score(item.sum)
    }

    /// The value as the shared microkernel is handed it: out of probabilities cast to the
    /// operands' element, which its fragments load beside the values.
    fn value_item(item: &AttentionWorkItem) -> MatmulWorkItem {
        item.value(item.elements.query)
    }

    /// [`candidates`](Microkernel::candidates) with the operands read in `read`.
    fn reading(
        hw: &Hardware,
        item: &AttentionWorkItem,
        read: ElemType,
        settings: PlaneFragmentAttentionMicrokernelSettings,
    ) -> Vec<Self> {
        let item = &item.read_as(read);
        let (head, value) = (item.head_dim, item.val_dim);
        // The grids and walks are sized once the instruction is named, from its shape.
        let unsized_grid = PlaneFragmentMicrokernelSettings {
            along_m: settings.seq_q,
            along_n: settings.seq_kv,
            transport: settings.transport,
            ..PlaneFragmentMicrokernelSettings::DEFAULT
        };
        let value_item = Self::value_item(item);
        let mut scores: Vec<_> =
            PlaneFragmentMicrokernel::candidates(hw, &Self::score_item(item), &unsized_grid)
                .collect();
        scores.sort_by_key(|score| Self::deferred(settings, score.instruction));
        scores
            .into_iter()
            .filter_map(move |score| {
                let value_instruction = Self::value_instruction(hw, score.instruction)?;
                let [_, (_, keys), (_, depth)] = score.instruction.shape();
                let [_, (_, width), (_, value_depth)] = value_instruction.shape();
                let head_steps = Self::steps(head, depth)?;
                let value_steps = Self::steps(value, width)?;
                let key_steps = Self::steps(settings.seq_kv.get() * keys, value_depth)?;
                Some(Self {
                    settings,
                    read,
                    score: PlaneFragmentMicrokernel {
                        settings: PlaneFragmentMicrokernelSettings {
                            depth: head_steps,
                            ..score.settings
                        },
                        ..score
                    },
                    value: PlaneFragmentMicrokernel {
                        instruction: value_instruction,
                        orders: value_instruction.orders(value_item.layouts),
                        settings: PlaneFragmentMicrokernelSettings {
                            along_n: value_steps,
                            depth: key_steps,
                            ..unsized_grid
                        },
                        axes: value_item.axes,
                        blocks: None,
                    },
                })
            })
            .collect()
    }

    /// The value's instruction beside `score`, where the device offers it: the score's own
    /// instruction where it is a cmma one, of any shape, read through the windows; a manual `mma`
    /// as deep as the score is wide, which reads the score's fragments in their registers, on
    /// 16-bit operands, where the device's `A` lays its cells where the accumulator does
    /// ([`Hardware::manual_mma_accumulator_as_a`]; a `tf32` `A` never does).
    fn value_instruction(
        hw: &Hardware,
        score: PlaneFragmentInstruction,
    ) -> Option<PlaneFragmentInstruction> {
        match score {
            PlaneFragmentInstruction::Cmma { .. } => Some(score),
            PlaneFragmentInstruction::Mma { shape } if shape.a_type.size() == 2 => {
                let value = MmaConfig {
                    k: shape.n,
                    ..shape
                };
                (hw.accumulator_as_a && manual_mma(hw).any(|shape| shape == value))
                    .then_some(PlaneFragmentInstruction::Mma { shape: value })
            }
            PlaneFragmentInstruction::Mma { .. }
            | PlaneFragmentInstruction::Registers { .. }
            | PlaneFragmentInstruction::Wgmma { .. } => None,
        }
    }

    /// Whether `score` comes after another of the settings' preference: one through the other
    /// transport, or a cmma one of the other shape. The shared microkernel already lists the
    /// settings' transport first.
    fn deferred(
        settings: PlaneFragmentAttentionMicrokernelSettings,
        score: PlaneFragmentInstruction,
    ) -> (bool, bool) {
        match score {
            PlaneFragmentInstruction::Cmma { .. } => (
                settings.transport != PlaneFragmentTransport::Cmma,
                !settings.shape.holds(score),
            ),
            PlaneFragmentInstruction::Mma { .. } => {
                (settings.transport != PlaneFragmentTransport::Mma, false)
            }
            PlaneFragmentInstruction::Wgmma { .. } => {
                (settings.transport != PlaneFragmentTransport::Wgmma, false)
            }
            PlaneFragmentInstruction::Registers { .. } => (true, false),
        }
    }

    /// `extent` in steps of `step`, where they divide it.
    fn steps(extent: usize, step: usize) -> Option<NonZeroUsize> {
        NonZeroUsize::new(extent / step).filter(|_| extent.is_multiple_of(step))
    }
}

/// Its instruction and fragments.
impl core::fmt::Display for PlaneFragmentAttentionMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} {}", self.score.instruction, self.settings)
    }
}

impl Microkernel for PlaneFragmentAttentionMicrokernel {
    type WorkItem = AttentionWorkItem;
    type Settings = PlaneFragmentAttentionMicrokernelSettings;
    type Operand = AttentionOperand;
    type Partitioning = AttentionPartitioning;

    const DECLINE: &'static str = "the plane microkernel runs on a cmma instruction, or a 16-bit \
         manual mma whose accumulator the device lays out as its A operand, beside one as deep as \
         it is wide, of the operands' element summing in f32, \
         whose k divides the head dim and whose n divides the value dim and the stage's keys, on a \
         device that commits to one plane width";

    /// A star off the default: each other box of fragments, the whole grid bounced at once, and
    /// the score read through the windows on a square and on a rectangular cmma instruction.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![
            Self::Settings::DEFAULT,
            Self::Settings::fragments(1, 1),
            Self::Settings::fragments(4, 2),
            Self::Settings::fragments(2, 4),
            Self::Settings::WHOLE_GRID,
            Self::Settings::through_windows(InstructionShape::Square),
            Self::Settings::through_windows(InstructionShape::Rectangular),
        ]
    }

    /// The shared plane fragment microkernel's candidates for the score, at each element the
    /// operands may be read in, the exact one first ([`readings`](AttentionWorkItem::readings));
    /// within one, through the settings' transport first and, among cmma ones, of the settings'
    /// shape first, that have a value instruction beside them (`value_instruction`), whose depth
    /// divides the head dim, and whose value's width divides the value dim and depth the stage's
    /// keys. Their fragment grids and the steps each walks follow from the settings and the
    /// shapes.
    fn candidates(
        hw: &Hardware,
        item: &AttentionWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let candidates: Vec<_> = item
            .readings()
            .flat_map(|read| Self::reading(hw, item, read, *settings))
            .collect();
        candidates.into_iter()
    }

    /// Each multiplication's own, on the operands it binds: the score's fragments read the query
    /// and the keys, the value's the values. Its windows of scores and probabilities are its own,
    /// and its drain writes a fragment's row of the output at a time, or a manual fragment's cells
    /// one at a time.
    fn vectorize(
        &self,
        _partitioning: &AttentionPartitioning,
        vectorization: &mut Vectorization<AttentionOperand>,
    ) {
        let mut score = Vectorization::new();
        self.score.vectorize(&(), &mut score);
        vectorization.restate(score, |operand| match operand {
            MatmulOperand::Lhs => Some(AttentionOperand::Query),
            MatmulOperand::Rhs => Some(AttentionOperand::Key),
            MatmulOperand::Out => None,
        });
        let mut value = Vectorization::new();
        self.value.vectorize(&(), &mut value);
        vectorization.restate(value, |operand| match operand {
            MatmulOperand::Lhs => None,
            MatmulOperand::Rhs => Some(AttentionOperand::Value),
            MatmulOperand::Out => Some(AttentionOperand::Out),
        });
        vectorization.require(VectorRequirement::Access {
            operand: AttentionOperand::Out,
            elements: self.out_edge(),
        });
        // A manual fragment drains one cell at a time, wherever its unit holds it.
        if self.value.instruction.drain() == PlaneFragmentDrain::CellByCell {
            vectorization.require(VectorRequirement::Elementwise {
                operand: AttentionOperand::Out,
            });
        }
    }

    /// Each multiplication's shared microkernel sets the work of its fragments; the plane's units
    /// hold both at once, beside the softmax's state of every row, on the matrix instruction both
    /// run. Its stages are its routine's to count.
    fn estimate(
        &self,
        item: &AttentionWorkItem,
        _partitioning: &AttentionPartitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.score
            .estimate(&Self::score_item(item), &(), planes, estimate);
        let score = estimate.work.take();
        self.value
            .estimate(&Self::value_item(item), &(), planes, estimate);
        let softmax = UnitWork {
            instruction: ArithmeticInstruction::Matrix,
            registers: self.softmax_registers(item),
        };
        estimate.work = score
            .zip(estimate.work)
            .map(|(score, value)| score.beside(value).beside(softmax));
    }
}

impl AttentionMicrokernel for PlaneFragmentAttentionMicrokernel {
    /// The score's: a fragment loads a block of keys col-major where it lies.
    fn key_order(&self) -> MatmulOrder {
        self.score.orders().rhs
    }

    /// Its operands in the element its instruction reads them in.
    fn served(&self, item: &AttentionWorkItem) -> AttentionWorkItem {
        item.read_as(self.read)
    }
}

impl CubePlaneMicrokernel for PlaneFragmentAttentionMicrokernel {
    const NAME: &'static str = "fragment";

    /// Its fragments load each stage from shared memory.
    const READS: CubePlaneReads = CubePlaneReads::Stage;

    /// Its planes merge their softmax states and sum their products in shared memory.
    const KEYS: CubePlaneKeys = CubePlaneKeys::Split;

    /// The element its instruction reads: an `f32` operand read at tf32 is bound as its words,
    /// and a stage of it at `f32` would be fragments the device does not offer.
    fn decoded_stage(&self) -> Option<ElemType> {
        Some(self.read)
    }

    /// Its fragments, as its [`ScoreHolder`] cuts them. Where the score is kept in fragments they
    /// are one grid of both multiplications: its rows along the queries of one head, its keys, and
    /// the head and value dims whole, each multiplication walking its own of them and holding the
    /// other's. Where it is read through windows, the value's fragments, its rows by the value dim
    /// and the steps of the stage's keys it contracts, each spanning the head dim whole: the score
    /// states its own grid below the plane ([`score_levels`](Self::score_levels)). The walk
    /// through `S` is its walker's: the plane routine's under each plane, the cube microkernel's
    /// above its planes.
    fn tiling(&self) -> Levels {
        match self.score_holder() {
            ScoreHolder::Fragments => Levels::leaf(&[
                (Q, self.score.leaf_extent(MatmulDim::M)),
                (G, 1),
                (S, self.score.leaf_extent(MatmulDim::N)),
                (D, self.score.leaf_extent(MatmulDim::K)),
                (DV, self.out_edge()),
            ])
            .walk(&[
                (Q, self.settings.seq_q.get()),
                (S, self.settings.seq_kv.get()),
                (D, self.score.settings.depth.get()),
                (DV, self.value.settings.along_n.get()),
            ]),
            ScoreHolder::Window => self
                .value
                .tiling_holding(&[(G, 1), (D, self.score.tile_extent(D))]),
        }
    }

    fn seq_q(&self) -> usize {
        self.value.tile_extent(Q)
    }

    fn seq_kv(&self) -> usize {
        self.value.tile_extent(S)
    }

    /// Its window of scores and of probabilities, and the fragments of its output the
    /// correction bounces through at once; none where it keeps the score in its fragments.
    fn window_bytes(&self, item: &AttentionWorkItem) -> usize {
        if self.score_holder() == ScoreHolder::Fragments {
            return 0;
        }
        let cells = self.seq_q() * self.seq_kv();
        let bounced = self
            .value
            .scratch_bytes(&Self::value_item(item), self.settings.scratch);
        cells * (item.sum.size() + item.elements.query.size()) + bounced
    }

    /// A window of every row of the cube's box over the value dim, its sums drain into, and two
    /// of one cell a row, its softmax state's max and sum.
    fn partial_bytes(&self, item: &AttentionWorkItem, layout: CubePlaneLayout) -> usize {
        layout.row_groups() * self.seq_q() * (item.val_dim + 2) * item.sum.size()
    }

    /// The sum of the products of the planes holding the same rows, and every plane's softmax
    /// state, met beside the stages: the compiler does not hand a freed stage's bytes to what is
    /// declared after it.
    fn merge_bytes(&self, item: &AttentionWorkItem, layout: CubePlaneLayout) -> usize {
        match layout.planes_along_keys {
            1 => 0,
            _ => {
                let rows = layout.row_groups() * self.seq_q();
                (rows * item.val_dim + layout.planes() * self.seq_q()) * item.sum.size()
            }
        }
    }

    /// In the instructions' tiles where the layout blocks it: the keys' rows by the score's `n`
    /// and the value's `k`, whichever is longer, the head dim by the score's `k`, and the value
    /// dim by the value's `n`.
    fn stage_storage(&self, layout: CubeAttentionStageLayout, stage: &Space) -> StageStorage {
        layout.storage(
            stage,
            [
                (
                    S,
                    self.score
                        .leaf_extent(MatmulDim::N)
                        .max(self.value.leaf_extent(MatmulDim::K)),
                ),
                (D, self.score.leaf_extent(MatmulDim::K)),
                (DV, self.out_edge()),
            ],
        )
    }

    /// Whether its rows' softmax state merges with another plane's of the same rows, as planes that
    /// split a cube's keys merge it.
    ///
    /// # Errors
    ///
    /// Where it keeps the score in its fragments, each unit holding the state of its own rows:
    /// the routine reads the score through windows instead, with the cmma transport.
    fn merges_rows(&self) -> Result<(), Refusal> {
        match self.score_holder() {
            ScoreHolder::Window => Ok(()),
            ScoreHolder::Fragments => Err(ROW_STATE_HELD_BY_UNIT),
        }
    }

    /// Whether its fragments read a stage laid out as `layout`.
    ///
    /// # Errors
    ///
    /// Swizzled tiles where it reads the score through its windows: a cmma fragment loads a tile
    /// off a pointer and a stride, which a swizzled row does not have.
    fn reads_stage(&self, layout: CubeAttentionStageLayout) -> Result<(), Refusal> {
        match (layout, self.score_holder()) {
            (CubeAttentionStageLayout::SwizzledBlocks, ScoreHolder::Window) => {
                Err(SWIZZLED_STAGE_UNREAD)
            }
            (
                CubeAttentionStageLayout::Strided
                | CubeAttentionStageLayout::PaddedRows
                | CubeAttentionStageLayout::Blocks
                | CubeAttentionStageLayout::PaddedBlocks
                | CubeAttentionStageLayout::SwizzledBlocks,
                _,
            ) => Ok(()),
        }
    }
}

impl CubeStagedPlane for PlaneFragmentAttentionMicrokernel {
    type Attention<EA: Float, E: Float, ES: Float> = PlaneFragmentAttention<EA, ES>;
}
