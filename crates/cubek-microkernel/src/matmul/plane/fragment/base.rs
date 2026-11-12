use crate::matmul::plane::{
    PlaneFragmentDepth, PlaneFragmentDrain, PlaneFragmentInstruction,
    PlaneFragmentMicrokernelSettings, PlaneFragmentRowsAlong,
};
use crate::matmul::{
    MatmulAxes, MatmulBlocks, MatmulDim, MatmulLayout, MatmulMicrokernel, MatmulOperand,
    MatmulOperands, MatmulRows, MatmulStage, MatmulStep, MatmulWorkItem,
};
use core::num::NonZeroUsize;
use cubecl::features::MmaConfig;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::quant::scheme::QuantValue;
use cubecl::tile::launch::{Refusal, VectorRequirement, Vectorization};
use cubecl::tile::space::ComputeScope;
use cubecl::tile::{Axis, FragmentSource, Instruction, Levels, Scratch, Space, StageStorage};
use cubek_std::layout::BindingOrder;
use cubek_std::layout::stored::Operand;
use cubek_std::matmul_axes::{K, M, N, blocked};
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel, StageLayout, UnitWork};
use cubek_std::{Hardware, Instructions, block_scaled_for, manual_mma};

/// A tile of the output one plane owns: a grid of instructions its units hold between them,
/// contracted a stage of `K` at a time out of shared memory, as its settings size it. The
/// instruction is the device's matrix instruction the tile layer picks; a device that offers none
/// for the elements runs no plane.
///
/// Against a packed rhs ([`blocks`](Self::blocks)) the plane reads the rhs straight from
/// memory and holds every row: one fragment of rows, its settings' fragments along `n`, each a
/// window of `K` its steps deep, so each unit reads its run of `K` in whole loads and the rhs
/// decodes into the lhs's element as each fragment loads (the tile DSL's lane runs); the
/// fragments share the lhs the plane loads (`step_levels`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentMicrokernel {
    /// What its fragments contract through.
    pub instruction: PlaneFragmentInstruction,
    /// The order its tiles of each operand are laid in: a col rhs's buffer order where the
    /// instruction's fragment loads `B` col-major, the order the product names otherwise.
    pub orders: MatmulOperands<BindingOrder>,
    pub settings: PlaneFragmentMicrokernelSettings,
    /// The caller's axes its levels are stated along.
    pub axes: MatmulAxes,
    /// The block-split space a packed rhs it reads straight from memory puts the product in:
    /// none for a float rhs, or one decoded into a stage before the plane reads it.
    pub blocks: Option<MatmulBlocks>,
    /// How the tiles it reads state `K` (`k_axes`).
    pub k_stated: PlaneFragmentKStatement,
}

/// How the tiles a [`PlaneFragmentMicrokernel`] reads state `K`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneFragmentKStatement {
    /// Along the caller's `K` axis: a float operand's, or a packed rhs read straight from memory.
    Caller,
    /// In a packed rhs's blocks, `KB` beside `KI`: what the block-scaled instruction reads, and a
    /// stage holding the rhs as it is stored.
    Blocks,
}

impl PlaneFragmentMicrokernel {
    /// The most rows a plane holds against a packed rhs: one fragment's, the decoding
    /// instruction's `m`, every row held whole.
    pub const MAX_PACKED_ROWS: usize = 16;

    /// The one instruction whose fragments decode a packed rhs as they load, as `(m, n, k)`: the
    /// 16-bit `16 × 8 × 16`, whose `B` fragment's lanes the tile DSL's lane runs are laid over.
    const DECODING_SHAPE: (u32, u32, u32) = (16, 8, 16);

    /// Units of a plane the decoding instruction's lane runs are laid over: the only width a
    /// plane against a packed rhs runs at.
    pub(crate) const DECODING_PLANE_UNITS: usize = 32;

    /// The values of `K` a fragment's lane run looks one scale up for: NVFP4's block. A packed
    /// rhs's fragments decode under scale blocks a whole number of them long, NVFP4's sixteen
    /// values or MXFP4's thirty-two.
    const DECODED_BLOCK: usize = 16;

    /// What a packed item of no rows, or of more than [`MAX_PACKED_ROWS`](Self::MAX_PACKED_ROWS),
    /// is declined with.
    pub const ROWS_OUT_OF_RANGE: Refusal = Refusal::Declined {
        why: "the plane fragment holds from one row to `PlaneFragmentMicrokernel::MAX_PACKED_ROWS` against a packed rhs, the rows of one fragment",
    };

    /// What a packed item whose operands are not a 16-bit lhs against an rhs of
    /// sixteen-value blocks along `K`, read along `K` alone, is declined with.
    pub(crate) const OPERANDS_UNSERVED: Refusal = Refusal::Declined {
        why: "the plane fragment contracts a packed rhs against a 16-bit lhs, the rhs read along K alone, in blocks of sixteen",
    };

    /// The decoding instruction over an lhs of element `lhs`, summed in `sum`.
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

    /// Whether `hw` runs a plane against a packed rhs for an lhs of element `lhs`, summed in
    /// `sum`: it offers the decoding instruction over `lhs` on a plane of thirty-two, which no
    /// element but a 16-bit float has.
    fn decodes(hw: &Hardware, lhs: ElemType, sum: ElemType) -> bool {
        manual_mma(hw).any(|shape| shape == Self::decoding_instruction(lhs, sum))
            && hw.plane_size().ok() == Some(Self::DECODING_PLANE_UNITS as u32)
    }

    /// Whether `hw` runs a plane against a packed rhs for any lhs: for one of the
    /// 16-bit floats, summed in `f32`.
    pub fn decodes_on(hw: &Hardware) -> bool {
        let f32 = ElemType::Float(FloatKind::F32);
        [FloatKind::F16, FloatKind::BF16]
            .into_iter()
            .any(|kind| Self::decodes(hw, ElemType::Float(kind), f32))
    }

    /// Whether a packed `item`'s operands are what the fragments contract: a 16-bit lhs,
    /// and a packed rhs of `e2m1` codes, the only values a lane run decodes, read along `K` alone,
    /// under scale blocks along `K` a whole number of [`DECODED_BLOCK`](Self::DECODED_BLOCK) long.
    fn serves(item: &MatmulWorkItem, space: &MatmulBlocks) -> bool {
        let rhs = &space.operand(MatmulOperand::Rhs);
        let sixteen_bits = matches!(
            item.elements.lhs,
            ElemType::Float(FloatKind::F16 | FloatKind::BF16)
        );
        let e2m1 = space
            .packings
            .rhs
            .is_some_and(|packing| packing.scheme.value == QuantValue::E2M1);
        sixteen_bits
            && e2m1
            && rhs.lined() == K
            && rhs.read(N) == 1
            && rhs
                .grouping(K, space.k)
                .is_some_and(|block| block.is_multiple_of(Self::DECODED_BLOCK))
    }

    /// The planes against the packed rhs of `space`: one fragment of the decoding instruction's
    /// rows where the item holds its rows whole, the settings' fragments along `m` where a stage
    /// tiles them; the settings' fragments along `n` and then every narrower count down to one,
    /// each where its columns close `n`, each fragment a window of the settings' steps of `K`
    /// through the tile DSL's transport. A window is whole windows of `K`, and a unit's run of it,
    /// one of the four quads of `k` its fragment's units split, whole reads of the rhs. Only
    /// settings stating such a grid size one
    /// ([`windowed`](PlaneFragmentMicrokernelSettings::windowed)), on a device offering the
    /// decoding instruction over the lhs's element.
    fn packed_candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: PlaneFragmentMicrokernelSettings,
        space: MatmulBlocks,
    ) -> impl Iterator<Item = Self> {
        let shape = Self::decoding_instruction(item.elements.lhs, item.sum);
        let window = settings.depth.get() * shape.k as usize;
        let rhs = space.operand(MatmulOperand::Rhs);
        let runs_whole = (window / 4).is_multiple_of(rhs.read(K));
        // A window states `K` as whole scale blocks and the position inside one.
        let blocks_whole = rhs
            .grouping(K, space.k)
            .is_some_and(|block| window.is_multiple_of(block));
        // A plane holding every row whole holds one fragment of them; one whose rows are tiled, a
        // stage's, holds the settings' fragments along `M`, each against the rhs fragment
        // decoded once a step.
        let rows_fit = settings.along_m == NonZeroUsize::MIN || item.rows == MatmulRows::Tiled;
        let windowed = rows_fit && settings.fragment_depth == PlaneFragmentDepth::Stage;
        let offered = windowed
            && runs_whole
            && blocks_whole
            && space.k.is_multiple_of(window)
            && Self::decodes(hw, item.elements.lhs, item.sum);
        let instruction = PlaneFragmentInstruction::mma(shape);
        let [_, (_, columns), _] = instruction.shape_along(settings.rows_along);
        (1..=settings.along_n.get())
            .rev()
            .filter(move |&along_n| offered && space.n.is_multiple_of(along_n * columns))
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
                    k_stated: PlaneFragmentKStatement::Caller,
                })
            })
    }

    /// The matrix unit `hw` offers a product of `lhs` by `rhs` summed in `sum`: the instruction
    /// that contracts the most per call, the squarest among those that contract as much
    /// ([`largest_first`](Instructions::largest_first)), over both of the device's sets. A float
    /// plane's settings count their box in it: `8 × 8 × 8` on an M2, `16 × 16 × 16` for `f16` on
    /// NVIDIA and `16 × 16 × 8` for its tf32. `None` where the device offers none.
    pub fn matrix_unit(
        hw: &Hardware,
        lhs: ElemType,
        rhs: ElemType,
        sum: ElemType,
    ) -> Option<[usize; 3]> {
        let unit = hw
            .properties
            .features
            .matmul
            .cmma
            .iter()
            .copied()
            .chain(manual_mma(hw))
            .for_triple(lhs, rhs, sum)
            .largest_first()
            .next()?;
        Some([unit.m as usize, unit.n as usize, unit.k as usize])
    }

    /// The planes against a float rhs: every matrix instruction this device offers for the
    /// item's elements that the tile layer picks for a plane's fragment of its shape
    /// ([`Instruction::for_accumulator`]), the fragment API's first, each the one that contracts
    /// the most per call first ([`candidates`](Microkernel::candidates)), each cutting the same box:
    /// the settings' grid of the device's [`matrix_unit`](Self::matrix_unit), so the instruction
    /// the tile picks never changes what a setting means. An instruction that does not cut the box
    /// in whole steps ([`MatmulStep`]) is not a candidate. The warpgroup MMA alone where the
    /// settings deal the tiles to warpgroups. None where the device offers no matrix instruction
    /// for the elements: a plane holds a fragment.
    fn float_candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &PlaneFragmentMicrokernelSettings,
    ) -> impl Iterator<Item = Self> {
        let (MatmulOperands { lhs, rhs, .. }, sum) = (item.elements, item.sum);
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
            .map(PlaneFragmentInstruction::mma);
        let settings = *settings;
        let step = MatmulStep::new(hw, MatmulOperands::all(Operand::plain()), item.elements);
        // The box one plane holds, in elements along `M`, `N` and `K`: the settings' grid of the
        // unit's steps, the fewest of its calls every load fills whole. Its rows lie along the
        // axis the settings lay the instruction's rows along, so the unit's edges are exchanged
        // where they lie along `n`.
        let tile = Self::matrix_unit(hw, lhs, rhs, sum).map(|[m, n, k]| {
            let (m, n) = match settings.rows_along {
                PlaneFragmentRowsAlong::M => (m, n),
                PlaneFragmentRowsAlong::N => (n, m),
            };
            let unit = step.clone().under(Space::new(&[(M, m), (N, n), (K, k)]));
            [
                settings.along_m.get() * m * unit.calls(M),
                settings.along_n.get() * n * unit.calls(N),
                settings.depth.get() * k * unit.calls(K),
            ]
        });
        let tiles_pick = move |instruction: &PlaneFragmentInstruction| {
            let [(_, m), (_, n), (_, k)] = instruction.shape_along(settings.rows_along);
            let depth = match (settings.fragment_depth, tile) {
                (PlaneFragmentDepth::Stage, Some([_, _, box_k])) => box_k,
                _ => k,
            };
            let picked = Instruction::for_accumulator(
                &hw.properties,
                &hw.target,
                ComputeScope::Plane,
                FragmentSource::Values(lhs),
                FragmentSource::Values(rhs),
                sum,
                [m, n, depth],
            );
            picked == Some(instruction.tile_instruction())
        };
        let instructions: Vec<_> = match settings.holder {
            ComputeScope::WARPGROUP => Self::wgmma(hw, item, &settings).into_iter().collect(),
            _ => cmma
                .chain(mma)
                .filter(|_| committed)
                .filter(tiles_pick)
                .collect(),
        };
        let deep = settings.fragment_depth == PlaneFragmentDepth::Stage;
        instructions
            .into_iter()
            .filter(move |instruction| !deep || instruction.contracts_deeper_fragments())
            .filter_map(move |instruction| {
                let [(_, m), (_, n), (_, k)] = instruction.shape_along(settings.rows_along);
                let step = step.clone().under(Space::new(&[(M, m), (N, n), (K, k)]));
                // The warpgroup MMA's settings size its one instruction, not a grid of units.
                let (along_m, along_n, depth) = match (instruction, tile) {
                    (PlaneFragmentInstruction::Wgmma { .. }, _) => {
                        (settings.along_m, settings.along_n, settings.depth)
                    }
                    (_, Some([box_m, box_n, box_k])) => (
                        Self::calls_cutting(box_m, m, step.calls(M))?,
                        Self::calls_cutting(box_n, n, step.calls(N))?,
                        Self::calls_cutting(box_k, k, step.calls(K))?,
                    ),
                    (_, None) => return None,
                };
                Some(Self {
                    instruction,
                    orders: instruction.orders(item.layouts),
                    settings: PlaneFragmentMicrokernelSettings {
                        along_m,
                        along_n,
                        depth,
                        ..settings
                    },
                    axes: item.axes,
                    blocks: None,
                    k_stated: PlaneFragmentKStatement::Caller,
                })
            })
    }

    /// The instruction's calls of `extent` each that cut `edge` of a box, where they cut it in
    /// whole steps of `calls` of them.
    ///
    /// A division, against "multiply, never divide": cubek guesses here the instruction the tile
    /// will pick, to lay the plane's fragments out for it. It goes once the tile cuts the plane's
    /// tile into the calls of the instruction it picks itself.
    fn calls_cutting(edge: usize, extent: usize, calls: usize) -> Option<NonZeroUsize> {
        let step = extent * calls;
        NonZeroUsize::new(edge / extent).filter(|_| edge.is_multiple_of(step))
    }

    /// The planes reading both packed inputs of `space` as they are stored: each block-scaled
    /// instruction the device offers for their formats ([`block_scaled_for`]), the largest first,
    /// whose step of `K` is whole blocks. A fragment's window spans its steps' blocks along `KB`
    /// and the whole of each along `KI`, and the instruction reads the values as they lie, and
    /// their scales as a word where a word of them is its step, one a block otherwise
    /// ([`scale_word_is_a_step`](PlaneFragmentInstruction::scale_word_is_a_step)).
    fn block_scaled_candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: PlaneFragmentMicrokernelSettings,
        space: MatmulBlocks,
    ) -> Vec<Self> {
        let committed = hw.plane_dim().is_ok();
        let deep = settings.fragment_depth == PlaneFragmentDepth::Stage;
        let block = space.operand(MatmulOperand::Rhs).grouping(K, space.k);
        // Each input under its own scheme, at its own block along `K`.
        let input = |input: MatmulOperand| {
            let packing = space.packings.get(input)?;
            Some((packing.scheme, space.operand(input).grouping(K, space.k)?))
        };
        let offered: Vec<_> = match (input(MatmulOperand::Lhs), input(MatmulOperand::Rhs)) {
            (Some((lhs, lhs_block)), Some((rhs, rhs_block))) => {
                block_scaled_for(hw, (&lhs, lhs_block), (&rhs, rhs_block))
                    .filter(|instruction| instruction.shape.cd_type == item.sum)
                    .map(PlaneFragmentInstruction::block_scaled_mma)
                    .collect()
            }
            _ => Vec::new(),
        };
        // Along `K`, what one read of each packed input's scales covers: the tile DSL cuts a step's
        // window of them on whole reads.
        let scale_reads: Vec<usize> = MatmulOperand::INPUTS
            .into_iter()
            .filter(|&input| space.packings.get(input).is_some())
            .filter_map(|input| space.operand(input).scale_read_space())
            .map(|read| read.extent(K))
            .collect();
        offered
            .into_iter()
            // A step of the instruction is whole reads of each input's scales, so whole blocks too,
            // or its window of them would start or end inside a read. MXFP4's step of 64 covers
            // half a read of four scales of thirty-two, which leaves it to the planes that decode
            // as they load.
            .filter(|instruction| {
                let [_, _, (_, k)] = instruction.shape();
                committed
                    && block.is_some()
                    && scale_reads.iter().all(|&read| k.is_multiple_of(read))
            })
            // It contracts a fragment its own step deep, so a plane whose fragments span a
            // stage's steps of `K` is not one it serves.
            .filter(|instruction| !deep || instruction.contracts_deeper_fragments())
            .map(|instruction| Self {
                instruction,
                orders: instruction.orders(item.layouts),
                settings,
                axes: item.axes,
                blocks: Some(space),
                k_stated: PlaneFragmentKStatement::Blocks,
            })
            .collect()
    }

    /// Its instruction's `m × n × k`, its rows along the axis the settings lay them along.
    pub(crate) fn shape(&self) -> [(MatmulDim, usize); 3] {
        self.instruction.shape_along(self.settings.rows_along)
    }

    /// Whether its instruction lays its rows where its settings ask: along `m` everywhere; along
    /// `n` only for the manual-mma fragments as deep as a stage, a 16-bit instruction's, which
    /// load both factors in lane runs and so fill either side of it. The block-scaled
    /// instruction and the step-at-a-time decode read their packed factor as a `B` alone.
    fn lays_its_rows(&self) -> bool {
        match self.settings.rows_along {
            PlaneFragmentRowsAlong::M => true,
            PlaneFragmentRowsAlong::N => {
                matches!(self.instruction, PlaneFragmentInstruction::Mma { .. })
                    && !self.instruction.block_scaled()
                    && self.settings.fragment_depth == PlaneFragmentDepth::Stage
            }
        }
    }

    /// The axes `K` is stated over, each with what one fragment `depth` instruction steps deep
    /// spans of it: the instruction's whole step along the caller's `K` axis, or, where the plane
    /// reads a packed rhs as stored (the block-scaled instruction, or fragments that decode it as
    /// they load), the blocks its steps cover along `KB` beside the whole of each block along
    /// `KI`.
    fn k_axes(&self, depth: usize) -> Vec<(Axis, usize)> {
        let [_, _, (_, k)] = self.shape();
        match (self.blocks, self.k_stated) {
            (Some(space), PlaneFragmentKStatement::Blocks) => {
                let block = space
                    .operand(MatmulOperand::Rhs)
                    .grouping(K, space.k)
                    .expect("a plane reading a packed rhs as stored reads it grouped along K");
                let (blocks, inside) = blocked(K);
                vec![(blocks, k * depth / block), (inside, block)]
            }
            _ => vec![(self.axes.k, k * depth)],
        }
    }

    /// How a stage it reads holds its cells, filled by its units, laid out as `layout`: where it
    /// blocks the stage, in its instruction's tiles ([`PlaneFragmentInstruction::stage_tile`]),
    /// each one step deep along every axis `K` is stated over.
    pub(crate) fn stage_storage(&self, layout: StageLayout, stage: &MatmulStage) -> StageStorage {
        let beside: Vec<Axis> = stage.beside().collect();
        let tile = Self::over_k(
            self.instruction
                .stage_tile(&beside, stage.axes(), self.settings.rows_along),
            &self.k_axes(1),
        );
        layout.storage(stage.space(), tile)
    }

    /// How a stage it reads holds its cells where one TMA box lands it whole
    /// ([`PlaneFragmentInstruction::landed_stage_storage`]): each tile the whole stage deep along
    /// every axis `K` is stated over.
    pub(crate) fn landed_stage_storage(&self, stage: &MatmulStage) -> StageStorage {
        let (beside, axes): (Vec<Axis>, _) = (stage.beside().collect(), stage.axes());
        let whole_k: Vec<_> = self
            .k_axes(1)
            .into_iter()
            .map(|(axis, _)| (axis, stage.extent(axis)))
            .collect();
        match self.instruction.landed_stage_storage(
            &beside,
            axes,
            stage.extent(axes.k),
            self.settings.rows_along,
        ) {
            StageStorage::Tiled { block, chunks } => StageStorage::Tiled {
                block: Self::over_k(block, &whole_k),
                chunks,
            },
            storage => storage,
        }
    }

    /// `tile` with its extents along `K` restated as `k_axes`: an instruction tiles `K` as one
    /// axis, where a block-scaled plane states it as its scale blocks and the position inside
    /// one, which the stage lays beside its other axes.
    fn over_k(tile: Vec<(Axis, usize)>, k_axes: &[(Axis, usize)]) -> Vec<(Axis, usize)> {
        tile.into_iter()
            .map(|(axis, extent)| {
                k_axes
                    .iter()
                    .copied()
                    .find(|&(k_axis, _)| k_axis == axis)
                    .unwrap_or((axis, extent))
            })
            .collect()
    }

    /// The axis its steps of `K` walk: `K` itself, or the blocks a plane reading a packed rhs as
    /// stored steps across.
    pub(crate) fn walked(&self) -> Axis {
        match (self.blocks, self.k_stated) {
            (Some(_), PlaneFragmentKStatement::Blocks) => blocked(K).0,
            _ => self.axes.k,
        }
    }

    /// This plane reading a stage that holds its packed rhs as it is stored, `K` in the rhs's
    /// blocks ([`PlaneFragmentKStatement::Blocks`]).
    pub(crate) fn reading_a_stage_as_stored(self) -> Self {
        Self {
            k_stated: PlaneFragmentKStatement::Blocks,
            ..self
        }
    }

    /// What one step of the plane covers against a packed rhs: its fragments over a window of
    /// `K`, split into the rhs's scale blocks, the instruction's rows walked over every row so
    /// a decode's few overhang the fragment rather than shrink it, and a unit level that cuts
    /// nothing, so the plane contracts its whole grid of fragments at once and loads the
    /// lhs once for all of them. A cube whose planes share a step states them around this.
    /// `None` for a float rhs.
    pub(crate) fn step_levels(&self) -> Option<Levels> {
        let space = self.blocks?;
        let block = space.operand(MatmulOperand::Rhs).grouping(K, space.k)?;
        let MatmulAxes { m, n, .. } = self.axes;
        let (blocks, inside) = blocked(K);
        let [(_, rows), (_, columns), _] = self.shape();
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

    /// [`tiling`](Self::tiling), its leaf also spanning `held`: axes of the caller's that this
    /// product does not name, each whole at every leaf, as attention's value holds the head dim
    /// its score contracts.
    pub fn tiling_holding(&self, held: &[(Axis, usize)]) -> Levels {
        self.tiling_levels(held, &[])
    }

    /// [`tiling`](Self::tiling) through a stage `blocks` deep: its steps of `K` walk every block
    /// along `blocks.0` too, each step one grid of instructions inside one block. What a stage of
    /// several scale blocks of `K` is contracted in, the instruction stepping inside a block.
    pub(crate) fn tiling_over(&self, blocks: (Axis, usize)) -> Levels {
        self.tiling_levels(&[], &[blocks])
    }

    /// The levels inside one plane's tile, its leaf also spanning `held`, its steps of `K`
    /// walking `outer` before the instruction's own steps.
    fn tiling_levels(&self, held: &[(Axis, usize)], outer: &[(Axis, usize)]) -> Levels {
        let settings = self.settings;
        let MatmulAxes { m, n, .. } = self.axes;
        let (along_m, along_n) = (settings.along_m.get(), settings.along_n.get());
        // A fragment as deep as the stage's steps holds them all: the plane walks one step.
        let (depth, steps) = match settings.fragment_depth {
            PlaneFragmentDepth::Instruction => (1, settings.depth.get()),
            PlaneFragmentDepth::Stage => (settings.depth.get(), 1),
        };
        let tiling = match self.instruction {
            PlaneFragmentInstruction::Cmma { .. } | PlaneFragmentInstruction::Mma { .. } => {
                let [(_, rows), (_, columns), _] = self.shape();
                let leaf: Vec<_> = [(m, rows), (n, columns)]
                    .into_iter()
                    .chain(self.k_axes(depth))
                    .chain(held.iter().copied())
                    .collect();
                Levels::leaf(&leaf).walk(&[(m, along_m), (n, along_n)])
            }
            // The steps along `n` are the instruction's own width: one instruction a group.
            PlaneFragmentInstruction::Wgmma { .. } => {
                let leaf: Vec<_> = self
                    .axes
                    .name(self.shape())
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
        let [(_, m), (_, n), (_, k)] = self.shape();
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
        let [m, n, (dim, k)] = self.shape();
        [m, n, (dim, k * depth)]
    }

    /// The extent along `dim` of the leaf a unit reads: one fragment's tile, as deep as its
    /// [`PlaneFragmentDepth`] says.
    pub fn leaf_extent(&self, dim: MatmulDim) -> usize {
        let depth = match self.settings.fragment_depth {
            PlaneFragmentDepth::Instruction => 1,
            PlaneFragmentDepth::Stage => self.settings.depth.get(),
        };
        let [(_, m), (_, n), (_, k)] = match self.instruction {
            PlaneFragmentInstruction::Cmma { .. } | PlaneFragmentInstruction::Mma { .. } => {
                self.fragment_shape(depth)
            }
            PlaneFragmentInstruction::Wgmma { .. } => self.shape(),
        };
        match dim {
            MatmulDim::M => m,
            MatmulDim::N => n,
            MatmulDim::K => k,
        }
    }

    /// The instruction as the tile DSL names it: the tile layer's pick for this plane's
    /// fragments, which its candidates keep alone.
    pub fn tile_instruction(&self) -> Instruction {
        self.instruction.tile_instruction()
    }

    /// 32-bit registers one unit of a `plane_width`-wide plane keeps this microkernel's work in:
    /// its share of every accumulator, and of one step's operands. A lower bound: the compiler
    /// adds addressing and loop state around them.
    pub fn registers_per_unit(&self, item: &MatmulWorkItem, plane_width: usize) -> u32 {
        let (along_m, along_n) = (self.settings.along_m.get(), self.settings.along_n.get());
        let bytes = match self.instruction {
            PlaneFragmentInstruction::Cmma { .. } | PlaneFragmentInstruction::Mma { .. } => {
                let [(_, m), (_, n), (_, k)] = self.shape();
                // Bits rather than bytes: an `e2m1` value is half of one.
                let plane = along_m * along_n * m * n * item.sum.size()
                    + along_m * m * k * item.elements.lhs.size_bits() / 8
                    + along_n * k * n * item.elements.rhs.size_bits() / 8;
                plane.div_ceil(plane_width.max(1))
            }
            // The accumulator alone, spread over the group's planes: its operands stay in
            // shared memory.
            PlaneFragmentInstruction::Wgmma { .. } => {
                let [(_, m), (_, n), _] = self.shape();
                let units = self.instruction.planes() * plane_width.max(1);
                (m * n * item.sum.size()).div_ceil(units)
            }
        };
        bytes.div_ceil(4) as u32
    }

    /// Bytes of the window one plane's fragments bounce through where `scratch` holds one: as many
    /// fragments as it holds of `item`'s sum. None for an instruction whose units write their own
    /// cells.
    pub(crate) fn scratch_bytes(&self, item: &MatmulWorkItem, scratch: Scratch) -> usize {
        match self.instruction.drain() {
            PlaneFragmentDrain::Bounce => {
                let fragments = self.settings.along_m.get() * self.settings.along_n.get();
                let [(_, m), (_, n), _] = self.shape();
                scratch.slots(fragments) * m * n * item.sum.size()
            }
            PlaneFragmentDrain::CellByCell => 0,
        }
    }

    /// The warpgroup MMA for `item` at `settings`, where every rule it runs under holds: the device
    /// offers its form and a bulk tensor copy to land its stages; its planes are as wide as a
    /// warpgroup's ([`PlaneFragmentInstruction::WARPGROUP_PLANE_UNITS`]); the rhs lies `{n, k}`, as
    /// both operands must, K-major; the settings hold one step along `m`, a form `along_n` granules
    /// wide the form spans, and a stage of `depth` steps of `K` whose rows, each operand's, are one
    /// 32, 64 or 128-byte swizzle span, which the instruction's descriptors read. The cube walking
    /// it adds what only its fill decides
    /// ([`CubeStagedMicrokernel::resolve`](crate::matmul::cube::CubeStagedMicrokernel::resolve)).
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
        // Both operands' stage along_m are one swizzle span: each operand's descriptor reads its
        // own.
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
    type CubeSpace = ();

    const DECLINE: &'static str = "the plane microkernel runs where the device offers a matrix \
         instruction for the product's elements, at a plane width it commits to";

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
    /// contracts (`OPERANDS_UNSERVED`).
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

    /// Every matrix instruction this device offers for these elements and this sum at a shape the
    /// tile layer picks it for ([`Instruction::for_accumulator`]), the one that
    /// contracts the most per call first; none where the device offers none, as a plane holds a
    /// fragment. A matrix instruction spreads its fragment across the plane, so it is offered only
    /// where the device commits to one plane width. Where the settings deal the tiles to
    /// warpgroups, the warpgroup MMA alone, where it runs. Against a packed rhs, the planes that
    /// decode it as they load, at the grids
    /// [`windowed`](PlaneFragmentMicrokernelSettings::windowed) names, narrowed until their columns
    /// close `n`.
    fn candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let packed: Vec<Self> = match item.blocks {
            None => Vec::new(),
            Some(space) if item.both_stored_as_codes() => {
                Self::block_scaled_candidates(hw, item, *settings, space)
            }
            Some(space) => Self::packed_candidates(hw, item, *settings, space).collect(),
        };
        let float = item
            .blocks
            .is_none()
            .then(|| Self::float_candidates(hw, item, settings));
        packed
            .into_iter()
            .chain(float.into_iter().flatten())
            .filter(Self::lays_its_rows)
    }

    /// The leaf a unit reads is one instruction's tile of each operand, or the unit's block of the
    /// software instruction: it reads the lhs in stretches of the leaf's `k` and the rhs in
    /// stretches of the leaf's extent along the axis its tiles lie along. It writes only its
    /// accumulator; the drain into the output is its walker's.
    ///
    /// Nothing against a packed rhs: every width follows from its packing.
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
    /// The levels inside one plane's tile, which a routine states the rest of the partitioning
    /// around: the steps of `K` it walks through one stage, and below them the instructions the
    /// plane holds — a grid of fragments, one matrix instruction each, or each unit's block of the
    /// software instruction's cells, one cell of every instruction it holds.
    fn tiling(&self) -> Levels {
        self.tiling_levels(&[], &[])
    }

    fn orders(&self) -> MatmulOperands<BindingOrder> {
        self.orders
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::{MatmulDestination, MatmulReading};
    use cubecl::quant::scheme::{QuantScheme, QuantStore, ScaleDtype};
    use cubek_std::layout::stored::Packing;
    use cubek_std::layout::tile_order;

    /// An `f16` activation against an `e2m1` weight packed down `k`, under scale blocks `block`
    /// long along `k`.
    fn against_blocks_of(block: usize) -> (MatmulWorkItem, MatmulBlocks) {
        against(tile_order::nvfp4_scheme(), block)
    }

    /// An `f16` activation against a weight packed down `k` under `scheme`, eight values a word
    /// and four scales, under scale blocks `block` long along `k`.
    fn against(scheme: QuantScheme, block: usize) -> (MatmulWorkItem, MatmulBlocks) {
        let f16 = ElemType::Float(FloatKind::F16);
        let packings = MatmulOperands {
            lhs: None,
            rhs: Some(Packing {
                operand: Operand::packed(K, 8, K, block, 4),
                scheme,
            }),
            out: None,
        };
        let space = MatmulBlocks {
            packings,
            m: 16,
            n: 4096,
            k: 4096,
        };
        let item = MatmulWorkItem {
            elements: MatmulOperands {
                lhs: f16,
                rhs: ElemType::Float(FloatKind::E2M1),
                out: f16,
            },
            sum: ElemType::Float(FloatKind::F32),
            layouts: MatmulOperands::all(MatmulLayout::Row),
            blocks: Some(space),
            rows: MatmulRows::Whole(16),
            destination: MatmulDestination::Buffer,
            axes: MatmulAxes::MNK,
            reading: MatmulReading::AsStored,
        };
        (item, space)
    }

    /// The fragments decode a packed rhs under NVFP4's sixteen-value blocks and MXFP4's
    /// thirty-two, each a whole number of the sixteen a lane run looks one scale up for, and under
    /// no block shorter than that.
    #[test]
    fn the_fragments_decode_under_blocks_a_whole_number_of_sixteen_long() {
        let serves = |block| {
            let (item, space) = against_blocks_of(block);
            PlaneFragmentMicrokernel::serves(&item, &space)
        };
        assert!(serves(16));
        assert!(serves(32));
        assert!(!serves(8));
    }

    /// The fragments decode `e2m1` codes alone: an MXFP8 weight's `e4m3` values under the same
    /// thirty-two-value blocks as MXFP4's are a lane run's to refuse, so no plane is offered for
    /// them, rather than one whose kernel fails to expand.
    #[test]
    fn the_fragments_decode_e2m1_codes_alone() {
        let mxfp8 = QuantScheme::default()
            .with_value(QuantValue::E4M3)
            .per_block([32, 1], ScaleDtype::UE8M0)
            .with_store(QuantStore::PackedU32(1));
        let (item, space) = against(mxfp8, 32);
        assert!(!PlaneFragmentMicrokernel::serves(&item, &space));
    }

    /// A plane over `instruction` at `settings`, its rows along `rows`.
    fn plane(
        instruction: PlaneFragmentInstruction,
        settings: PlaneFragmentMicrokernelSettings,
        rows: PlaneFragmentRowsAlong,
    ) -> PlaneFragmentMicrokernel {
        PlaneFragmentMicrokernel {
            instruction,
            orders: MatmulOperands::all(BindingOrder::Stated),
            settings: settings.rows_along(rows),
            axes: MatmulAxes::MNK,
            blocks: None,
            k_stated: PlaneFragmentKStatement::Caller,
        }
    }

    /// The instruction's rows lie along `n` only for the manual-mma fragments as deep as a stage,
    /// which load both factors in lane runs and so fill either side of it; everywhere else a
    /// plane asked for them is not offered, rather than one whose kernel the tile DSL refuses or
    /// contracts by position. Along `m`, every plane is offered as before.
    #[test]
    fn rows_lie_along_n_only_for_fragments_a_stage_deep() {
        let f16 = ElemType::Float(FloatKind::F16);
        let mma = PlaneFragmentInstruction::mma(MmaConfig {
            a_type: f16,
            b_type: f16,
            cd_type: ElemType::Float(FloatKind::F32),
            m: 16,
            n: 8,
            k: 16,
        });
        let stage_deep = PlaneFragmentMicrokernelSettings::windowed(2, 4);
        let instruction_deep = PlaneFragmentMicrokernelSettings::grid(2, 2, 1);
        let (m, n) = (PlaneFragmentRowsAlong::M, PlaneFragmentRowsAlong::N);

        assert!(plane(mma, stage_deep, n).lays_its_rows());
        assert!(!plane(mma, instruction_deep, n).lays_its_rows());
        for (instruction, settings) in [(mma, stage_deep), (mma, instruction_deep)] {
            assert!(plane(instruction, settings, m).lays_its_rows());
        }
    }

    /// Rows along `n` exchange the instruction's edges along `m` and `n`, so a plane's box and its
    /// stages' tiles follow: a `16 × 8` instruction is eight rows of the product by sixteen of its
    /// columns.
    #[test]
    fn rows_along_n_exchange_the_instructions_edges() {
        let f16 = ElemType::Float(FloatKind::F16);
        let mma = PlaneFragmentInstruction::mma(MmaConfig {
            a_type: f16,
            b_type: f16,
            cd_type: ElemType::Float(FloatKind::F32),
            m: 16,
            n: 8,
            k: 16,
        });
        assert_eq!(
            mma.shape_along(PlaneFragmentRowsAlong::M),
            [(MatmulDim::M, 16), (MatmulDim::N, 8), (MatmulDim::K, 16)]
        );
        assert_eq!(
            mma.shape_along(PlaneFragmentRowsAlong::N),
            [(MatmulDim::M, 8), (MatmulDim::N, 16), (MatmulDim::K, 16)]
        );
    }

    /// A plane at its default settings contracts through the instruction the tile layer picks for
    /// a drained sum at its fragment's shape ([`Instruction::for_accumulator`]), on every mocked device and
    /// float element, and refuses where the tile layer picks none: a plane holds a fragment.
    #[test]
    fn a_plane_takes_the_tiles_pick_or_refuses() {
        use cubecl::tile::FragmentSource;
        use cubecl::tile::space::ComputeScope;
        use cubek_std::devices;

        let f32 = ElemType::Float(FloatKind::F32);
        let devices = [
            ("strix_halo", devices::strix_halo()),
            ("wgpu_no_subgroups", devices::wgpu_no_subgroups()),
            ("rdna_range", devices::rdna_range()),
            ("cdna_wave64", devices::cdna_wave64()),
            ("m2_pro", devices::m2_pro()),
            ("h100", devices::h100()),
            ("rtx_pro_6000", devices::rtx_pro_6000()),
            ("cpu_12", devices::cpu_12()),
        ];
        for (name, hw) in devices {
            for kind in [FloatKind::F16, FloatKind::BF16, FloatKind::F32] {
                let input = ElemType::Float(kind);
                let item = MatmulWorkItem {
                    elements: MatmulOperands {
                        lhs: input,
                        rhs: input,
                        out: input,
                    },
                    sum: f32,
                    layouts: MatmulOperands::all(MatmulLayout::Row),
                    blocks: None,
                    rows: MatmulRows::Tiled,
                    destination: MatmulDestination::Buffer,
                    axes: MatmulAxes::MNK,
                    reading: MatmulReading::AsStored,
                };
                let plane = PlaneFragmentMicrokernel::best(
                    &hw,
                    &item,
                    &PlaneFragmentMicrokernelSettings::DEFAULT,
                );
                let Ok(plane) = plane else {
                    let values = FragmentSource::Values(input);
                    let any = [(8, 8, 8), (16, 16, 16), (16, 8, 16), (16, 16, 8)]
                        .into_iter()
                        .any(|(m, n, k)| {
                            Instruction::for_accumulator(
                                &hw.properties,
                                &hw.target,
                                ComputeScope::Plane,
                                values,
                                values,
                                f32,
                                [m, n, k],
                            )
                            .is_some()
                        });
                    assert!(!any, "{name}, {kind:?}: refused, though the tile picks one");
                    continue;
                };
                let [(_, m), (_, n), (_, k)] = plane.instruction.shape();
                let picked = Instruction::for_accumulator(
                    &hw.properties,
                    &hw.target,
                    ComputeScope::Plane,
                    FragmentSource::Values(input),
                    FragmentSource::Values(input),
                    f32,
                    [m, n, k],
                );
                assert_eq!(picked, Some(plane.tile_instruction()), "{name}, {kind:?}");
            }
        }
    }

    /// A setting is one box on a device whichever instruction cuts it: on an H100 offering `f16`
    /// three cmma forms and the manual `16 × 8 × 16`, every candidate at the default's four by
    /// four units of `16 × 16 × 16` holds `64 × 64` of the output and walks `16` of `K` a stage,
    /// each in its own instruction's fragments. A unit that an instruction does not cut whole is
    /// not that instruction's: `8 × 32` cuts no box one unit high.
    #[test]
    fn every_instruction_cuts_the_box_the_setting_names() {
        use cubek_std::devices;

        let (f16, f32) = (
            ElemType::Float(FloatKind::F16),
            ElemType::Float(FloatKind::F32),
        );
        let shape = |m, n, k| MmaConfig {
            a_type: f16,
            b_type: f16,
            cd_type: f32,
            m,
            n,
            k,
        };
        let hw = devices::offering_mma(
            devices::offering_cmma(
                devices::h100(),
                [shape(16, 16, 16), shape(32, 8, 16), shape(8, 32, 16)],
            ),
            [shape(16, 8, 16)],
        );
        let item = MatmulWorkItem {
            elements: MatmulOperands {
                lhs: f16,
                rhs: f16,
                out: f16,
            },
            sum: f32,
            layouts: MatmulOperands::all(MatmulLayout::Row),
            blocks: None,
            rows: MatmulRows::Tiled,
            destination: MatmulDestination::Buffer,
            axes: MatmulAxes::MNK,
            reading: MatmulReading::AsStored,
        };
        assert_eq!(
            PlaneFragmentMicrokernel::matrix_unit(&hw, f16, f16, f32),
            Some([16, 16, 16])
        );
        let boxes = |settings: &PlaneFragmentMicrokernelSettings| -> Vec<_> {
            PlaneFragmentMicrokernel::candidates(&hw, &item, settings)
                .map(|plane| {
                    let [(_, m), (_, n), (_, k)] = plane.instruction.shape();
                    ((m, n, k), [M, N, K].map(|axis| plane.tile_extent(axis)))
                })
                .collect()
        };
        let default = boxes(&PlaneFragmentMicrokernelSettings::DEFAULT);
        assert_eq!(default.len(), 4, "{default:?}");
        for (instruction, tile) in default {
            assert_eq!(tile, [64, 64, 16], "{instruction:?}");
        }
        let one_unit = boxes(&PlaneFragmentMicrokernelSettings::grid(1, 1, 1));
        assert!(
            one_unit
                .iter()
                .all(|(instruction, _)| *instruction != (8, 32, 16)),
            "{one_unit:?}"
        );
    }
}
