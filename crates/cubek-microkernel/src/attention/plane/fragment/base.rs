use crate::attention::plane::{PlaneFragmentAttentionMicrokernelSettings, ScoreHolder};
use crate::attention::{
    AttentionMicrokernel, AttentionOperand, AttentionRead, AttentionWorkItem, B, D, DV, G, KV, Q, S,
};
use crate::matmul::plane::{
    PlaneFragmentDrain, PlaneFragmentMicrokernel, PlaneFragmentMicrokernelSettings,
};
use crate::matmul::{MatmulDim, MatmulMicrokernel, MatmulOperand, MatmulWorkItem};
use cubecl::ir::ElemType;
use cubecl::tile::launch::{Refusal, VectorRequirement, Vectorization};
use cubecl::tile::space::ComputeScope;
use cubecl::tile::{Axis, FragmentSource, Instruction, Level, Levels, Space};
use cubek_std::Hardware;
use cubek_std::layout::BindingOrder;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, SWIZZLED_STAGE_UNREAD,
    StageLayout, UnitWork,
};

/// The rows of one head a plane owns: their scores against a stage of keys in the fragments
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
    /// The elements its query and keys reach the score's instruction in and its values the
    /// value's ([`readings`](AttentionWorkItem::readings)): each as stored, or converted as its
    /// stage is filled.
    pub read: AttentionRead,
    /// The element a stage of its keys and values decodes into where one of them is converted:
    /// the values' read element where they are, else the score's.
    pub cache_stage: ElemType,
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

/// The bytes a row of a stage block spans at least, where the dim holds them: a cache line.
pub const BLOCK_ROW_BYTES: usize = 128;

impl PlaneFragmentAttentionMicrokernel {
    /// Where the plane keeps the score: its settings', which the tile picked its instructions for.
    /// Kept in fragments, the score is the value's `A`, read in runs of its cells; windows
    /// decouple the two products.
    pub fn score_holder(&self) -> ScoreHolder {
        self.settings.score
    }

    /// The edge a fragment of the output spans along the value dim, which a drain writes at once.
    pub fn out_edge(&self) -> usize {
        self.value.leaf_extent(MatmulDim::N)
    }

    /// The score's own levels where the plane reads it through its windows, which its accumulator
    /// is placed in: the walker's level of planes, `planes` its counts, and below it the plane's
    /// score, its rows by the stage's keys by the whole head dim, which the tile cuts and walks.
    /// The counts are what index the accumulator's scratch, a window for every plane of the cube.
    pub fn score_levels(&self, planes: &[(Axis, usize)]) -> Vec<Level> {
        Levels::leaf(&[
            (Q, self.score.tile_extent(Q)),
            (S, self.score.tile_extent(S)),
            (D, self.score.tile_extent(D)),
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
    /// values' element, which its fragments load beside the values.
    fn value_item(item: &AttentionWorkItem) -> MatmulWorkItem {
        item.value(item.elements.value)
    }

    /// [`candidates`](Microkernel::candidates) with the operands read in `read`: for each place
    /// the plane may keep its score, the tile's picks for both products over the plane's tile.
    ///
    /// The plane's tile is its rows by a stage's keys, counted in the device's matrix unit for
    /// the score's operands, by the whole head dim, and its rows by the whole value dim. The tile
    /// picks each product's instruction from the sources its kernel opens the sum on, cuts the
    /// tile into its calls and walks the contracted dim in their steps: a score in windows is a
    /// sum drained there, from the query and the keys; a score in fragments a sum contracted
    /// again ([`Instruction::for_stage_accumulator`]), which the value reads in runs of its cells
    /// as it lies. Where the tile picks no product for a holder, the plane keeps its score
    /// nowhere else.
    fn reading(
        hw: &Hardware,
        item: &AttentionWorkItem,
        read: AttentionRead,
        settings: PlaneFragmentAttentionMicrokernelSettings,
    ) -> Vec<Self> {
        // A stage of the cache converts a side it reads at another element as it fills, as an
        // `f32` read at tf32 or `e4m3` values read at `f16` are: its fragments load the values it
        // decodes. Both sides converted so are converted to one element alike.
        let converted = |stored: ElemType, read: ElemType| stored != read;
        let (keys_converted, values_converted) = (
            converted(item.elements.key, read.score),
            converted(item.elements.value, read.value),
        );
        if keys_converted && values_converted && read.score != read.value {
            return Vec::new();
        }
        let cache_stage = match values_converted {
            true => read.value,
            false => read.score,
        };
        let item = &item.read_as(read);
        let (score_item, value_item) = (Self::score_item(item), Self::value_item(item));
        let grid = PlaneFragmentMicrokernelSettings {
            along_m: settings.seq_q,
            along_n: settings.seq_kv,
            ..PlaneFragmentMicrokernelSettings::DEFAULT
        };
        let Some([rows, keys]) = PlaneFragmentMicrokernel::unit_tile(hw, &score_item, &grid) else {
            return Vec::new();
        };
        let score_tile = [rows, keys, item.head_dim];
        let value_tile = [rows, item.val_dim, keys];
        let (query, key, value) = (
            FragmentSource::Values(score_item.elements.lhs),
            FragmentSource::Values(score_item.elements.rhs),
            FragmentSource::Values(value_item.elements.rhs),
        );
        let (device, target, sum) = (&hw.properties, &hw.target, item.sum);
        let plane = ComputeScope::Plane;
        let holders = match settings.score {
            ScoreHolder::Fragments => vec![ScoreHolder::Fragments, ScoreHolder::Window],
            ScoreHolder::Window => vec![ScoreHolder::Window],
        };
        holders
            .into_iter()
            .filter(|_| hw.plane_dim().is_ok())
            .filter_map(|holder| {
                let (score, value) = match holder {
                    ScoreHolder::Fragments => {
                        let score = Instruction::for_stage_accumulator(
                            device, target, query, query, sum, score_tile,
                        )?;
                        let held = FragmentSource::Fragments {
                            instruction: score.instruction,
                            cell: score.cell(),
                        };
                        let value = Instruction::for_accumulator(
                            device, target, plane, held, value, sum, value_tile,
                        )?;
                        (score, value)
                    }
                    ScoreHolder::Window => (
                        Instruction::for_accumulator(
                            device, target, plane, query, key, sum, score_tile,
                        )?,
                        Instruction::for_accumulator(
                            device, target, plane, value, value, sum, value_tile,
                        )?,
                    ),
                };
                Some(Self {
                    settings: settings.held_in(holder),
                    read,
                    cache_stage,
                    score: PlaneFragmentMicrokernel::holding(&score_item, grid, score_tile, score)?,
                    value: PlaneFragmentMicrokernel::holding(&value_item, grid, value_tile, value)?,
                })
            })
            .collect()
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
    type CubeSpace = Space;

    const DECLINE: &'static str = "the plane microkernel runs on a cmma instruction, or a 16-bit \
         manual mma whose accumulator the device lays out as its A operand, beside one a whole \
         number of its fragments deep that the plane's keys divide into, of the operands' element \
         summing in f32, \
         whose k divides the head dim and whose n divides the value dim and the stage's keys, on a \
         device that commits to one plane width";

    /// A star off the default: each other tile shape, the whole grid bounced at once, and
    /// the score read through the windows.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![
            Self::Settings::DEFAULT,
            Self::Settings::fragments(1, 1),
            Self::Settings::fragments(4, 2),
            Self::Settings::fragments(2, 4),
            Self::Settings::WHOLE_GRID,
            Self::Settings::THROUGH_WINDOWS,
        ]
    }

    /// The shared plane fragment microkernel's candidates for the score, at each element the
    /// operands may be read in, the exact one first ([`readings`](AttentionWorkItem::readings)),
    /// that have a value instruction beside them (`value_instruction`), whose depth divides the
    /// head dim, and whose value's width divides the value dim and depth the stage's keys. Within
    /// one reading, a score kept in its fragments comes first where the settings ask for it, and
    /// through the windows where the tile layer picks no manual `mma` to keep it in; a score the
    /// settings read through the windows is read nowhere else. Their fragment grids and the steps
    /// each walks follow from the settings and the shapes.
    fn candidates(
        hw: &Hardware,
        item: &AttentionWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let candidates: Vec<_> = item
            .readings()
            .flat_map(|read| {
                let mut reading = Self::reading(hw, item, read, *settings);
                reading.sort_by_key(|candidate| candidate.score_holder() != settings.score);
                reading
            })
            .collect();
        candidates.into_iter()
    }

    /// Each multiplication's own, on the operands it binds: the score's fragments read the query
    /// and the keys, the value's the values. Its windows of scores and probabilities are its own,
    /// and its drain writes a fragment's row of the output at a time, or a manual fragment's cells
    /// one at a time.
    fn vectorize(&self, _cube_space: &Space, vectorization: &mut Vectorization<AttentionOperand>) {
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
        _cube_space: &Space,
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
    fn key_order(&self) -> BindingOrder {
        self.score.orders().rhs
    }

    /// Its operands in the element its instruction reads them in.
    fn served(&self, item: &AttentionWorkItem) -> AttentionWorkItem {
        item.read_as(self.read)
    }
}

/// What a cube holding it on its planes reads of it.
impl PlaneFragmentAttentionMicrokernel {
    /// What the routines whose cubes hold it are named after.
    pub const NAME: &'static str = "fragment";

    /// The element a stage of its query decodes into where the query is converted: the score's,
    /// as an `f16` query multiplied with `e4m3` keys on the FP8 instruction, or an `f32` one read
    /// at tf32, whose stage at `f32` would be fragments the device does not offer.
    pub fn decoded_query(&self) -> Option<ElemType> {
        Some(self.read.score)
    }

    /// The element a stage of its keys and values decodes into where either is converted
    /// ([`cache_stage`](Self::cache_stage)); one bound as stored is staged as it lies.
    pub fn decoded_stage(&self) -> Option<ElemType> {
        Some(self.cache_stage)
    }

    /// The tile one plane holds: its rows along the queries of one head, a stage's keys, and the
    /// head and value dims whole. The tile cuts each product's part of it into the calls of the
    /// instruction it picks and walks the dim each contracts: the score the head dim, the value
    /// the keys, a run of the score's cells a call where it keeps them. The shared microkernel
    /// steps the level below a stage, so the tile states one step there, and each product is one
    /// call a stage. The walk through `S` is its walker's: the plane routine's under each plane,
    /// the cube microkernel's above its planes.
    pub fn tiling(&self) -> Levels {
        Levels::leaf(&[
            (Q, self.score.tile_extent(Q)),
            (G, 1),
            (S, self.score.tile_extent(S)),
            (D, self.score.tile_extent(D)),
            (DV, self.value.tile_extent(DV)),
        ])
        .walk(&[(D, 1)])
    }

    /// Whether the value contracts a stage's keys in more than one call, a run of the score's
    /// cells each, where the plane keeps its score in fragments. Never where it reads the score
    /// through its windows.
    pub fn reads_keys_in_runs(&self) -> bool {
        match self.score_holder() {
            ScoreHolder::Fragments => self.value.leaf_extent(MatmulDim::K) < self.seq_kv(),
            ScoreHolder::Window => false,
        }
    }

    pub fn seq_q(&self) -> usize {
        self.value.tile_extent(Q)
    }

    pub fn seq_kv(&self) -> usize {
        self.value.tile_extent(S)
    }

    /// Its window of scores and of probabilities, and the fragments of its output the
    /// correction bounces through at once; none where it keeps the score in its fragments.
    pub fn window_bytes(&self, item: &AttentionWorkItem) -> usize {
        if self.score_holder() == ScoreHolder::Fragments {
            return 0;
        }

        let cells = self.seq_q() * self.seq_kv();
        let bounced = self
            .value
            .scratch_bytes(&Self::value_item(item), self.settings.scratch);
        cells * (item.sum.size() + item.elements.value.size()) + bounced
    }

    /// The tile one fragment of the score reads of the staged query: one batch, one head of keys
    /// and one of its queries, the score's rows by its depth along the head dim.
    pub fn query_tile(&self) -> Vec<(Axis, usize)> {
        vec![
            (B, 1),
            (KV, 1),
            (G, 1),
            (Q, self.score.leaf_extent(MatmulDim::M)),
            (D, self.score.leaf_extent(MatmulDim::K)),
        ]
    }

    /// The block a stage of keys and values is laid out in: one batch and one head of keys, the
    /// keys' rows by the score's `n` and the value's `k`, whichever is longer, and along the head
    /// and value dims a run of the steps the score and the value take along them
    /// (`block_edge`).
    pub fn stage_tile(&self) -> Vec<(Axis, usize)> {
        vec![
            (B, 1),
            (KV, 1),
            (
                S,
                self.score
                    .leaf_extent(MatmulDim::N)
                    .max(self.value.leaf_extent(MatmulDim::K)),
            ),
            (
                D,
                self.block_edge(
                    self.score.leaf_extent(MatmulDim::K),
                    self.score.tile_extent(D),
                    self.read.score,
                ),
            ),
            (
                DV,
                self.block_edge(self.out_edge(), self.value.tile_extent(DV), self.read.value),
            ),
        ]
    }

    /// A stage block's edge along a dim of `extent` its fragments take steps of `step` elements
    /// of `element` along: the fewest whole steps that cut the dim and whose row spans
    /// [`BLOCK_ROW_BYTES`], else the whole dim.
    /// A block one fragment wide holds a sliver of each source row, which the copy brings a few
    /// bytes at a time, splitting sectors and lines; a whole line a row is fetched in whole
    /// sectors, and a swizzle permutes its chunks within it. On an RTX PRO 6000, a 6016 by 8192
    /// causal `f16` prompt (32/8 heads, head dim 128) on eight planes a cube takes 2392 µs with
    /// 32-byte rows and 2047 µs with 128 on two-by-four tiles, 2143 and 1884 µs on one-by-four
    /// tiles.
    fn block_edge(&self, step: usize, extent: usize, element: ElemType) -> usize {
        (1..)
            .map(|count| count * step)
            .take_while(|edge| *edge <= extent)
            .filter(|edge| extent.is_multiple_of(*edge))
            .find(|edge| edge * element.size() >= BLOCK_ROW_BYTES)
            .unwrap_or(extent)
    }

    /// Whether its rows' softmax state merges with another plane's of the same rows, as planes that
    /// split a cube's keys merge it.
    ///
    /// # Errors
    ///
    /// Where it keeps the score in its fragments, each unit holding the state of its own rows:
    /// the routine keeps the score in windows instead ([`ScoreHolder::Window`]).
    pub fn merges_rows(&self) -> Result<(), Refusal> {
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
    pub fn reads_stage(&self, layout: StageLayout) -> Result<(), Refusal> {
        match (layout, self.score_holder()) {
            (StageLayout::SwizzledBlocks, ScoreHolder::Window) => Err(SWIZZLED_STAGE_UNREAD),
            (
                StageLayout::Strided
                | StageLayout::PaddedRows
                | StageLayout::Blocks
                | StageLayout::PaddedBlocks
                | StageLayout::SwizzledBlocks,
                _,
            ) => Ok(()),
        }
    }
}
