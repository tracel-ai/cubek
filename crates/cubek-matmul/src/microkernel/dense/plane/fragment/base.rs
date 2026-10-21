use crate::microkernel::{
    MatmulAxes, MatmulBlocks, MatmulDim, MatmulLayout, MatmulMicrokernel, MatmulOperand,
    MatmulOperands, MatmulOrder, MatmulRows, MatmulRowsKind, MatmulStep, MatmulWorkItem,
};
use cubek_std::{Hardware, Instructions, manual_mma};
use cubek_tile::launch::Refusal;

use crate::microkernel::dense::plane::{
    PlaneFragmentDepth, PlaneFragmentDrain, PlaneFragmentInstruction,
    PlaneFragmentMicrokernelSettings, PlaneFragmentStepSum, PlaneFragmentTransport,
};
use crate::stored::Operand;
use core::num::NonZeroUsize;
use cubecl::features::MmaConfig;
use cubecl::ir::{ElemType, FloatKind};
use cubek_std::matmul_axes::{K, M, N, blocked};
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel, UnitWork};
use cubek_tile::launch::{VectorRequirement, Vectorization};
use cubek_tile::{Axis, Instruction, Levels, Scratch, Space};

/// A tile of the output one plane owns: a grid of instructions its units hold between them,
/// contracted a stage of `K` at a time out of shared memory, as its settings size it. The
/// instruction is the device's matrix instruction where it offers one, and the software
/// instruction where it does not.
///
/// Against a packed weight ([`blocks`](Self::blocks)) the plane reads the weight straight from
/// memory and holds every row: one fragment of rows, its settings' fragments along `n`, each a
/// window of `K` its steps deep, so each unit reads its run of `K` in whole loads and the weight
/// decodes into the activation's element as each fragment loads (cubek's lane runs); the fragments
/// share the activation the plane loads ([`step_levels`](Self::step_levels)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentMicrokernel {
    /// What its fragments contract through.
    pub instruction: PlaneFragmentInstruction,
    /// The order its tiles of each operand are laid in: a col weight's buffer order where the
    /// instruction's fragment loads `B` col-major, the order the product names otherwise.
    pub orders: MatmulOperands<MatmulOrder>,
    pub settings: PlaneFragmentMicrokernelSettings,
    /// The caller's axes its levels are stated along.
    pub axes: MatmulAxes,
    /// The block-split space a packed weight it reads straight from memory puts the product in:
    /// none for a float weight, or one decoded into a stage before the plane reads it.
    pub blocks: Option<MatmulBlocks>,
}

impl PlaneFragmentMicrokernel {
    /// The most rows a plane holds against a packed weight: one fragment's, the decoding
    /// instruction's `m`, every row held whole.
    pub const MAX_PACKED_ROWS: usize = 16;

    /// The one instruction whose fragments decode a packed weight as they load, as
    /// `(m, n, k)`: the 16-bit `16 × 8 × 16`, whose `B` fragment's lanes cubek's lane runs are
    /// laid over.
    const DECODING_SHAPE: (u32, u32, u32) = (16, 8, 16);

    /// Units of a plane the decoding instruction's lane runs are laid over: the only width a
    /// plane against a packed weight runs at.
    pub const DECODING_PLANE_UNITS: usize = 32;

    /// The scale block along `K` a packed weight's fragments decode under: NVFP4's, sixteen
    /// values.
    const DECODED_BLOCK: usize = 16;

    /// What a packed item of no rows, or of more than [`MAX_PACKED_ROWS`](Self::MAX_PACKED_ROWS),
    /// is declined with.
    pub const ROWS_OUT_OF_RANGE: Refusal = Refusal::Declined {
        why: "the plane fragment holds from one row to `PlaneFragmentMicrokernel::MAX_PACKED_ROWS` against a packed weight, the rows of one fragment",
    };

    /// What a packed item whose operands are not a 16-bit activation against a weight of
    /// sixteen-value blocks along `K`, read along `K` alone, is declined with.
    pub const OPERANDS_UNSERVED: Refusal = Refusal::Declined {
        why: "the plane fragment contracts a packed weight against a 16-bit activation, the weight read along K alone, in blocks of sixteen",
    };

    /// The decoding instruction over an activation of `lhs`, summed in `sum`.
    fn decoding_instruction(lhs: ElemType, sum: ElemType) -> MmaConfig {
        let (m, n, k) = Self::DECODING_SHAPE;
        MmaConfig {
            a_type: lhs,
            b_type: lhs,
            cd_type: sum,
            m,
            n,
            k,
        }
    }

    /// Whether `hw` runs a plane against a packed weight for an activation of `lhs`, summed in
    /// `sum`: it offers the decoding instruction over `lhs` on a plane of thirty-two, which no
    /// element but a 16-bit float has.
    fn decodes(hw: &Hardware, lhs: ElemType, sum: ElemType) -> bool {
        manual_mma(hw).any(|shape| shape == Self::decoding_instruction(lhs, sum))
            && hw.plane_size().ok() == Some(Self::DECODING_PLANE_UNITS as u32)
    }

    /// Whether `hw` runs a plane against a packed weight for any activation: for one of the
    /// 16-bit floats, summed in `f32`.
    pub fn decodes_on(hw: &Hardware) -> bool {
        let f32 = ElemType::Float(FloatKind::F32);
        [FloatKind::F16, FloatKind::BF16]
            .into_iter()
            .any(|kind| Self::decodes(hw, ElemType::Float(kind), f32))
    }

    /// Whether a packed `item`'s operands are what the fragments contract: a 16-bit activation,
    /// and a packed weight read along `K` alone, under scales of sixteen-value blocks along `K`.
    fn serves(item: &MatmulWorkItem, space: &MatmulBlocks) -> bool {
        let rhs = &space.operand(MatmulOperand::Rhs);
        let sixteen_bits = matches!(
            item.elements.lhs,
            ElemType::Float(FloatKind::F16 | FloatKind::BF16)
        );
        sixteen_bits
            && rhs.lined() == K
            && rhs.read(N) == 1
            && rhs.grouping(K, space.k) == Some(Self::DECODED_BLOCK)
    }

    /// The planes against the packed weight of `space`: one fragment of the decoding
    /// instruction's rows, the settings' fragments along `n` and then every narrower count down
    /// to one, each where its columns close `n`, each fragment a window of the settings' steps of
    /// `K` through cubek's transport. A window is whole windows of `K`, and a unit's run of it,
    /// one of the four quads of `k` its fragment's units split, whole reads of the weight. Only
    /// settings stating such a grid size one ([`windowed`](PlaneFragmentMicrokernelSettings::windowed)),
    /// on a device offering the decoding instruction over the activation's element.
    fn packed_candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: PlaneFragmentMicrokernelSettings,
        space: MatmulBlocks,
    ) -> impl Iterator<Item = Self> {
        let shape = Self::decoding_instruction(item.elements.lhs, item.sum);
        let window = settings.depth.get() * shape.k as usize;
        let runs_whole = (window / 4).is_multiple_of(space.operand(MatmulOperand::Rhs).read(K));
        let windowed = settings.along_m == NonZeroUsize::MIN
            && settings.transport == PlaneFragmentTransport::Mma
            && settings.fragment_depth == PlaneFragmentDepth::Stage;
        let offered = windowed
            && runs_whole
            && space.k.is_multiple_of(window)
            && Self::decodes(hw, item.elements.lhs, item.sum);
        let instruction = PlaneFragmentInstruction::Mma { shape };
        (1..=settings.along_n.get())
            .rev()
            .filter(move |&along_n| offered && space.n.is_multiple_of(along_n * shape.n as usize))
            .filter_map(move |along_n| {
                Some(Self {
                    instruction,
                    orders: instruction.orders(item.layouts),
                    settings: PlaneFragmentMicrokernelSettings {
                        along_n: NonZeroUsize::new(along_n)?,
                        ..settings
                    },
                    axes: item.axes,
                    blocks: Some(space),
                })
            })
    }

    /// The planes against a float weight: every matrix instruction this device offers for the
    /// item's elements, through the transport the settings ask for first, then the software
    /// instruction ([`candidates`](Microkernel::candidates)).
    fn float_candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &PlaneFragmentMicrokernelSettings,
    ) -> impl Iterator<Item = Self> {
        let (lhs, rhs, sum) = item.instruction_elements();
        let committed = hw.plane_dim().is_ok();
        let cmma = hw
            .properties
            .features
            .matmul
            .cmma
            .iter()
            .copied()
            .for_triple(lhs, rhs, sum)
            .largest_first()
            .map(|shape| PlaneFragmentInstruction::Cmma { shape });
        let mma = manual_mma(hw)
            .for_triple(lhs, rhs, sum)
            .deepest()
            .largest_first()
            .map(|shape| PlaneFragmentInstruction::Mma { shape });
        let settings = *settings;
        let software = PlaneFragmentInstruction::Registers {
            step_sum: settings.step_sum.serving(item),
        };
        let instructions: Vec<_> = match settings.transport {
            PlaneFragmentTransport::Cmma => cmma
                .chain(mma)
                .filter(|_| committed)
                .chain([software])
                .collect(),
            PlaneFragmentTransport::Mma => mma
                .chain(cmma)
                .filter(|_| committed)
                .chain([software])
                .collect(),
            PlaneFragmentTransport::Wgmma => Self::wgmma(hw, item, &settings).into_iter().collect(),
        };
        // Only the manual-mma instruction contracts a fragment deeper than itself.
        let deep = settings.fragment_depth == PlaneFragmentDepth::Stage;
        let step = MatmulStep::new(hw, &Operand::plain(), item.elements.lhs);
        instructions
            .into_iter()
            .filter(move |instruction| {
                !deep || matches!(instruction, PlaneFragmentInstruction::Mma { .. })
            })
            .filter_map(move |instruction| {
                let [(_, m), (_, n), (_, k)] = instruction.shape();
                let step = step.clone().under(Space::new(&[(M, m), (N, n), (K, k)]));
                Some(Self {
                    instruction,
                    orders: instruction.orders(item.layouts),
                    settings: PlaneFragmentMicrokernelSettings {
                        along_m: settings
                            .along_m
                            .checked_mul(NonZeroUsize::new(step.calls(M))?)?,
                        along_n: settings
                            .along_n
                            .checked_mul(NonZeroUsize::new(step.calls(N))?)?,
                        depth: settings
                            .depth
                            .checked_mul(NonZeroUsize::new(step.calls(K))?)?,
                        ..settings
                    },
                    axes: item.axes,
                    blocks: None,
                })
            })
    }

    /// The planes reading both packed inputs of `space` as they are stored: each block-scaled
    /// instruction the device offers for their values, the largest first, whose step of `K` is
    /// a word of each input's block scales, the blocks one read of the scales covers
    /// ([`Operand::scale_read`]). A fragment's window spans those blocks along `KB` and the whole
    /// of each along `KI`, and the instruction reads the values and the word of scales beside
    /// them as they lie.
    fn block_scaled_candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: PlaneFragmentMicrokernelSettings,
        space: MatmulBlocks,
    ) -> Vec<Self> {
        let committed = hw.plane_dim().is_ok();
        let rhs = space.operand(MatmulOperand::Rhs);
        let step = rhs
            .grouping(K, space.k)
            .map(|block| block * rhs.scale_read(K));
        manual_mma(hw)
            .for_triple(item.elements.lhs, item.elements.rhs, item.sum)
            .largest_first()
            .map(|shape| PlaneFragmentInstruction::Mma { shape })
            .filter(|instruction| {
                let [_, _, (_, k)] = instruction.shape();
                committed && instruction.block_scaled() && step == Some(k)
            })
            .map(|instruction| Self {
                instruction,
                orders: instruction.orders(item.layouts),
                settings,
                axes: item.axes,
                blocks: Some(space),
            })
            .collect()
    }

    /// The axes `K` is stated over, each with what one fragment `depth` instruction steps deep
    /// spans of it: the instruction's whole step along the caller's `K` axis, or, where a
    /// block-scaled instruction reads packed inputs in their blocks, a word of block scales' blocks
    /// along `KB` beside the whole of each block along `KI`.
    fn k_axes(&self, depth: usize) -> Vec<(Axis, usize)> {
        let [_, _, (_, k)] = self.instruction.shape();
        match (self.blocks, self.instruction.block_scaled()) {
            (Some(space), true) => {
                let rhs = space.operand(MatmulOperand::Rhs);
                let block = rhs
                    .grouping(K, space.k)
                    .expect("a block-scaled plane reads inputs grouped along K");
                let (blocks, inside) = blocked(K);
                vec![(blocks, rhs.scale_read(K) * depth), (inside, block)]
            }
            _ => vec![(self.axes.k, k * depth)],
        }
    }

    /// The axis its steps of `K` walk: `K` itself, or the blocks a block-scaled plane steps across.
    pub fn walked(&self) -> Axis {
        match (self.blocks, self.instruction.block_scaled()) {
            (Some(_), true) => blocked(K).0,
            _ => self.axes.k,
        }
    }

    /// What one step of the plane covers against a packed weight: its fragments over a window of
    /// `K`, split into the weight's scale blocks, the instruction's rows walked over every row so
    /// a decode's few overhang the fragment rather than shrink it, and a unit level that cuts
    /// nothing, so the plane contracts its whole grid of fragments at once and loads the
    /// activation once for all of them. A cube whose planes share a step states them around this.
    /// `None` for a float weight.
    pub fn step_levels(&self) -> Option<Levels> {
        let space = self.blocks?;
        let block = space.operand(MatmulOperand::Rhs).grouping(K, space.k)?;
        let MatmulAxes { m, n, .. } = self.axes;
        let (blocks, inside) = blocked(K);
        let [(_, rows), (_, columns), _] = self.instruction.shape();
        let window = self.leaf_extent(MatmulDim::K);
        let leaf = Levels::leaf(&[
            (m, rows),
            (n, columns),
            (blocks, window / block),
            (inside, block),
        ])
        .walk_every(&[m]);
        let fragments = match self.settings.along_n.get() {
            1 => leaf,
            along_n => leaf.walk(&[(n, along_n)]),
        };
        Some(fragments.units(&[]))
    }

    /// The levels inside one plane's tile, which a routine states the rest of the partitioning
    /// around: the steps of `K` it walks through one stage, and below them the instructions the
    /// plane holds — a grid of fragments, one matrix instruction each, or each unit's block of the
    /// software instruction's cells, one cell of every instruction it holds.
    pub fn tiling(&self) -> Levels {
        self.tiling_levels(&[], &[])
    }

    /// [`tiling`](Self::tiling), its leaf also spanning `held`: axes of the caller's that this
    /// product does not name, each whole at every leaf, as attention's value holds the head dim
    /// its score contracts.
    pub fn tiling_holding(&self, held: &[(Axis, usize)]) -> Levels {
        self.tiling_levels(held, &[])
    }

    /// [`tiling`](Self::tiling) through a stage `blocks` deep: its steps of `K` walk every block
    /// along `blocks.0` too, each step one grid of instructions inside one block. What a stage of
    /// several scale blocks of `K` is contracted in, the instruction stepping inside a block.
    pub fn tiling_over(&self, blocks: (Axis, usize)) -> Levels {
        self.tiling_levels(&[], &[blocks])
    }

    /// The levels inside one plane's tile, its leaf also spanning `held`, its steps of `K`
    /// walking `outer` before the instruction's own steps.
    fn tiling_levels(&self, held: &[(Axis, usize)], outer: &[(Axis, usize)]) -> Levels {
        let settings = self.settings;
        let MatmulAxes { m, n, k } = self.axes;
        let (along_m, along_n) = (settings.along_m.get(), settings.along_n.get());
        // A fragment as deep as the stage's steps holds them all: the plane walks one step.
        let (depth, steps) = match settings.fragment_depth {
            PlaneFragmentDepth::Instruction => (1, settings.depth.get()),
            PlaneFragmentDepth::Stage => (settings.depth.get(), 1),
        };
        let tiling = match self.instruction {
            PlaneFragmentInstruction::Cmma { .. } | PlaneFragmentInstruction::Mma { .. } => {
                let [(_, rows), (_, columns), _] = self.instruction.shape();
                let leaf: Vec<_> = [(m, rows), (n, columns)]
                    .into_iter()
                    .chain(self.k_axes(depth))
                    .chain(held.iter().copied())
                    .collect();
                Levels::leaf(&leaf).walk(&[(m, along_m), (n, along_n)])
            }
            PlaneFragmentInstruction::Registers { .. } => {
                let leaf: Vec<_> = [
                    (m, along_m),
                    (n, along_n),
                    (k, PlaneFragmentInstruction::SOFTWARE_DEPTH),
                ]
                .into_iter()
                .chain(held.iter().copied())
                .collect();
                Levels::leaf(&leaf).units(&self.axes.name(PlaneFragmentInstruction::SOFTWARE_UNITS))
            }
            // The steps along `n` are the instruction's own width: one instruction a group.
            PlaneFragmentInstruction::Wgmma { .. } => {
                let leaf: Vec<_> = self
                    .axes
                    .name(self.instruction.shape())
                    .into_iter()
                    .chain(held.iter().copied())
                    .collect();
                Levels::leaf(&leaf).walk(&[(m, 1), (n, 1)])
            }
        };
        let steps: Vec<(Axis, usize)> = outer
            .iter()
            .copied()
            .chain([(self.walked(), steps)])
            .collect();
        tiling.walk(&steps)
    }

    /// The extent along `axis`, one of the caller's, of the tile one plane holds: its
    /// instructions along `m` and `n`, and the instruction steps of `K` it walks through one
    /// stage.
    pub fn tile_extent(&self, axis: Axis) -> usize {
        let settings = self.settings;
        let [(_, m), (_, n), (_, k)] = self.instruction.shape();
        let (along_m, along_n) = match self.instruction {
            PlaneFragmentInstruction::Wgmma { .. } => (1, 1),
            _ => (settings.along_m.get(), settings.along_n.get()),
        };
        let k_axes = self.k_axes(settings.depth.get());
        if let Some(&(_, extent)) = k_axes.iter().find(|&&(k_axis, _)| k_axis == axis) {
            return extent;
        }
        match self.axes.dim(axis) {
            Some(MatmulDim::M) => along_m * m,
            Some(MatmulDim::N) => along_n * n,
            Some(MatmulDim::K) => settings.depth.get() * k,
            _ => 1,
        }
    }

    /// The shape of one of its fragments, `depth` of the instruction's steps of `K` deep.
    fn fragment_shape(&self, depth: usize) -> [(MatmulDim, usize); 3] {
        let [m, n, (dim, k)] = self.instruction.shape();
        [m, n, (dim, k * depth)]
    }

    /// The extent along `dim` of the leaf a unit reads: one fragment's tile, as deep as its
    /// [`PlaneFragmentDepth`] says, or a unit's block of the software instruction.
    pub fn leaf_extent(&self, dim: MatmulDim) -> usize {
        let depth = match self.settings.fragment_depth {
            PlaneFragmentDepth::Instruction => 1,
            PlaneFragmentDepth::Stage => self.settings.depth.get(),
        };
        let [(_, m), (_, n), (_, k)] = match self.instruction {
            PlaneFragmentInstruction::Cmma { .. } | PlaneFragmentInstruction::Mma { .. } => {
                self.fragment_shape(depth)
            }
            PlaneFragmentInstruction::Wgmma { .. } => self.instruction.shape(),
            PlaneFragmentInstruction::Registers { .. } => [
                (MatmulDim::M, self.settings.along_m.get()),
                (MatmulDim::N, self.settings.along_n.get()),
                (MatmulDim::K, PlaneFragmentInstruction::SOFTWARE_DEPTH),
            ],
        };
        match dim {
            MatmulDim::M => m,
            MatmulDim::N => n,
            MatmulDim::K => k,
        }
    }

    /// The instruction as cubek's tile engine runs it, each unit holding its cells of every
    /// instruction the plane holds.
    pub fn tile_instruction(&self) -> Instruction {
        let cells = self.settings.along_m.get() * self.settings.along_n.get();
        self.instruction.tile_instruction(cells)
    }

    /// 32-bit registers one unit of a `plane_width`-wide plane keeps this microkernel's work in:
    /// its share of every accumulator, and of one step's operands. A lower bound: the compiler
    /// adds addressing and loop state around them.
    pub fn registers_per_unit(&self, item: &MatmulWorkItem, plane_width: usize) -> u32 {
        let (along_m, along_n) = (self.settings.along_m.get(), self.settings.along_n.get());
        let bytes = match self.instruction {
            PlaneFragmentInstruction::Cmma { .. } | PlaneFragmentInstruction::Mma { .. } => {
                let [(_, m), (_, n), (_, k)] = self.instruction.shape();
                // Bits rather than bytes: an `e2m1` value is half of one.
                let plane = along_m * along_n * m * n * item.sum.size()
                    + along_m * m * k * item.elements.lhs.size_bits() / 8
                    + along_n * k * n * item.elements.rhs.size_bits() / 8;
                plane.div_ceil(plane_width.max(1))
            }
            // One cell of every instruction, and one line of `K` of the operands at a time; a
            // narrow step holds its partial of every cell beside the sum.
            PlaneFragmentInstruction::Registers { step_sum } => {
                let partial = match step_sum {
                    PlaneFragmentStepSum::Wide => 0,
                    PlaneFragmentStepSum::Narrow => along_m * along_n * item.elements.lhs.size(),
                };
                along_m * along_n * item.sum.size()
                    + partial
                    + along_m * item.elements.lhs.size()
                    + along_n * item.elements.rhs.size()
            }
            // The accumulator alone, spread over the group's planes: its operands stay in
            // shared memory.
            PlaneFragmentInstruction::Wgmma { .. } => {
                let [(_, m), (_, n), _] = self.instruction.shape();
                let units = self.instruction.planes() * plane_width.max(1);
                (m * n * item.sum.size()).div_ceil(units)
            }
        };
        bytes.div_ceil(4) as u32
    }

    /// Bytes of the window one plane's fragments bounce through where `scratch` holds one: as many
    /// fragments as it holds of `item`'s sum. None for an instruction whose units write their own
    /// cells.
    pub fn scratch_bytes(&self, item: &MatmulWorkItem, scratch: Scratch) -> usize {
        match self.instruction.drain() {
            PlaneFragmentDrain::Bounce => {
                let fragments = self.settings.along_m.get() * self.settings.along_n.get();
                let [(_, m), (_, n), _] = self.instruction.shape();
                scratch.slots(fragments) * m * n * item.sum.size()
            }
            PlaneFragmentDrain::CellByCell | PlaneFragmentDrain::InLines => 0,
        }
    }

    /// The warpgroup MMA for `item` at `settings`, where every rule it runs under holds: the
    /// device offers its form and a bulk tensor copy to land its stages; its planes are as wide
    /// as a warpgroup's ([`PlaneFragmentInstruction::WARPGROUP_PLANE_UNITS`]); the weight lies
    /// `{n, k}`, as both operands must, K-major; the settings hold one step along `m`, a form `along_n`
    /// granules wide the form spans, and a stage of `depth` steps of `K` whose rows, each
    /// operand's, are one 32, 64 or 128-byte swizzle span, which the instruction's descriptors
    /// read. The cube walking it adds what only its fill decides
    /// ([`CubeStagedMicrokernel::resolve`](crate::microkernel::dense::cube::CubeStagedMicrokernel::resolve)).
    fn wgmma(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &PlaneFragmentMicrokernelSettings,
    ) -> Option<PlaneFragmentInstruction> {
        let (lhs, rhs, sum) = (item.elements.lhs, item.elements.rhs, item.sum);
        let config = hw
            .properties
            .features
            .matmul
            .wgmma
            .iter()
            .copied()
            .find(|form| form.elems.a == lhs && form.elems.b == rhs && form.elems.cd == sum)?;
        let n = settings.along_n.get() * config.n_granularity as usize;
        // Both operands' stage along_m are one swizzle span: each operand's descriptor reads its own.
        let spans_a_swizzle = |elem: ElemType| {
            matches!(
                settings.depth.get() * config.k as usize * elem.size(),
                32 | 64 | 128
            )
        };
        let runs = hw.bulk_tensor_copy()
            && hw.plane_dim() == Ok(PlaneFragmentInstruction::WARPGROUP_PLANE_UNITS as u32)
            && item.layouts.rhs == MatmulLayout::Col
            && settings.along_m.get() == 1
            && config.supports_n(n as u32)
            && spans_a_swizzle(lhs)
            && spans_a_swizzle(rhs);
        runs.then_some(PlaneFragmentInstruction::Wgmma { config, n })
    }
}

impl Microkernel for PlaneFragmentMicrokernel {
    type WorkItem = MatmulWorkItem;
    type Settings = PlaneFragmentMicrokernelSettings;
    type Operand = MatmulOperand;
    type Partitioning = ();

    const DECLINE: &'static str = "the plane microkernel's software instruction runs everywhere";

    /// The default grid, then one deeper and larger, and two smaller ones, which hold fewer
    /// registers and leave more planes room.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![
            Self::Settings::DEFAULT,
            Self::Settings::grid(4, 4, 2),
            Self::Settings::grid(2, 4, 1),
            Self::Settings::grid(2, 2, 1),
        ]
    }

    /// Any float item. A packed one ([`MatmulWorkItem::blocks`]) of from one row to
    /// [`MAX_PACKED_ROWS`](Self::MAX_PACKED_ROWS), every one held whole, against operands it
    /// contracts ([`OPERANDS_UNSERVED`](Self::OPERANDS_UNSERVED)).
    fn admits(item: &MatmulWorkItem) -> Result<(), Refusal> {
        let Some(space) = &item.blocks else {
            return Ok(());
        };
        match item.rows {
            MatmulRows::Whole(rows) if (1..=Self::MAX_PACKED_ROWS).contains(&rows) => {}
            MatmulRows::Whole(_) | MatmulRows::Tiled => return Err(Self::ROWS_OUT_OF_RANGE),
        }
        match Self::serves(item, space) {
            true => Ok(()),
            false => Err(Self::OPERANDS_UNSERVED),
        }
    }

    /// Every matrix instruction this device offers for these elements and this sum through
    /// cubecl's fragment operations, then through cubek's own transport, each the one that contracts the
    /// most per call first; then the software instruction, which every device runs. A matrix
    /// instruction spreads its fragment across the plane, so it is offered only where the device
    /// commits to one plane width. Where the settings ask for the warpgroup MMA, that alone, where
    /// it runs. Against a packed weight, the planes that decode it as they load, at the grids
    /// [`windowed`](PlaneFragmentMicrokernelSettings::windowed) names, narrowed until their
    /// columns close `n`.
    fn candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let packed: Vec<Self> = match item.blocks {
            None => Vec::new(),
            Some(space) if item.stages_scales_beside() => {
                Self::block_scaled_candidates(hw, item, *settings, space)
            }
            Some(space) => Self::packed_candidates(hw, item, *settings, space).collect(),
        };
        let float = item
            .blocks
            .is_none()
            .then(|| Self::float_candidates(hw, item, settings));
        packed.into_iter().chain(float.into_iter().flatten())
    }

    /// The leaf a unit reads is one instruction's tile of each operand, or the unit's block of the
    /// software instruction: it reads the lhs in stretches of the leaf's `k` and the rhs in
    /// stretches of the leaf's extent along the axis its tiles lie along. It writes only its
    /// accumulator; the drain into the output is its walker's.
    ///
    /// Nothing against a packed weight: every width follows from its packing.
    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<MatmulOperand>) {
        if self.blocks.is_some() {
            return;
        }
        vectorization.require(VectorRequirement::Access {
            operand: MatmulOperand::Lhs,
            elements: self.leaf_extent(MatmulDim::K),
        });
        vectorization.require(VectorRequirement::Access {
            operand: MatmulOperand::Rhs,
            elements: self.leaf_extent(self.orders.innermost(MatmulOperand::Rhs)),
        });
        // The software instruction's register block lines its accumulator along `n` with the rhs.
        if let PlaneFragmentInstruction::Registers { .. } = self.instruction {
            vectorization.require(VectorRequirement::Together {
                operand: MatmulOperand::Rhs,
                with: MatmulOperand::Out,
            });
        }
    }

    /// Its units hold their share of its accumulators, contracted by its instruction: the work,
    /// since it walks nothing of the launch's.
    fn estimate(
        &self,
        item: &MatmulWorkItem,
        _: &(),
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        estimate.work = Some(UnitWork {
            instruction: self.instruction.arithmetic(),
            registers: self.registers_per_unit(item, planes.width),
        });
    }
}

/// The instruction it contracts through, and the order it binds the rhs in.
impl core::fmt::Display for PlaneFragmentMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} rhs={}", self.instruction, self.orders.rhs)
    }
}

impl MatmulMicrokernel for PlaneFragmentMicrokernel {
    /// A plane owns a tile of the along_m.
    const ROWS_KIND: MatmulRowsKind = MatmulRowsKind::Tiled;

    fn orders(&self) -> MatmulOperands<MatmulOrder> {
        self.orders
    }
}
