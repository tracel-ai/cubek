use crate::matmul::plane::{
    PlaneFragmentDepth, PlaneFragmentInstruction, PlaneFragmentMicrokernelSettings,
    PlaneFragmentRowsAlong,
};
use crate::matmul::{
    MatmulBlocks, MatmulDim, MatmulDimensions, MatmulLayout, MatmulMicrokernel, MatmulOperand,
    MatmulOperands, MatmulRows, MatmulStage, MatmulWorkItem,
};
use core::num::NonZeroUsize;
use cubecl::features::MmaConfig;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::quant::scheme::QuantValue;
use cubecl::tile::launch::{Decline, Delivery, VectorRequirement, Vectorization};
use cubecl::tile::space::ComputeScope;
use cubecl::tile::stage::RowChunks;
use cubecl::tile::{
    BaseRegion, Dimension, FragmentSource, Instruction, LaneMap, Levels, MmaOperands, ProductStep,
    Scratch, ShapedInstruction, Space, StageOrder, StageStorage, StepCost,
};
use cubek_std::layout::BindingOrder;
use cubek_std::layout::stored::{Operand, OuterScale};
use cubek_std::matmul_dimensions::{K, M, N, blocked};
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
    /// The Step it hands the tile, opened on the sources its kernel opens the sum on: its whole
    /// tile where it states its tile; none where it states its instruction.
    pub step: Option<ProductStep>,
    /// What computing its Step spends where its sum is drained, as the tile answers it
    /// ([`ProductStep::cost`]); for a plane that states its instruction, what that instruction
    /// spends ([`PlaneFragmentInstruction::cost`]).
    pub cost: StepCost,
    /// The order its tiles of each operand are laid in: a col rhs's buffer order where the
    /// instruction's fragment loads `B` col-major, the order the product names otherwise.
    pub orders: MatmulOperands<BindingOrder>,
    pub settings: PlaneFragmentMicrokernelSettings,
    /// The caller's dimensions its levels are stated along.
    pub dimensions: MatmulDimensions,
    /// The block-split space a packed rhs it reads straight from memory puts the product in:
    /// none for a float rhs, or one decoded into a stage before the plane reads it.
    pub blocks: Option<MatmulBlocks>,
    /// How the tiles it reads state `K` (`k_dimensions`).
    pub k_stated: PlaneFragmentKStatement,
    /// What one count of its settings stands for ([`PlaneUnit`]).
    pub unit: PlaneUnit,
}

/// What one count of a [`PlaneFragmentMicrokernel`]'s settings stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneUnit {
    /// The device's matrix unit for the product's elements, `m × n × k` along the caller's
    /// dimensions, and the calls of it every load fills whole along each ([`BaseRegion`]): the
    /// plane states its tile as a grid of them, each one unit deep, and the tile cuts it into the
    /// calls of the instruction it picks.
    Matrix { unit: [usize; 3], calls: [usize; 3] },
    /// Elements: the plane states its tile as it is, `m × n × k` along the caller's dimensions, a
    /// problem's extents included, and the tile cuts it into the calls of the instruction it picks
    /// and walks its `k` in their steps. Its settings' counts are not read.
    Elements { tile: [usize; 3] },
    /// Its instruction: the plane states its fragments, a grid of the instruction's calls.
    Instruction,
}

/// How the tiles a [`PlaneFragmentMicrokernel`] reads state `K`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneFragmentKStatement {
    /// Along the caller's `K` dimension: a float operand's, or a packed rhs read straight from
    /// memory.
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
    const DECODING_SHAPE: (u32, u32, u32) =
        (16, 8, Self::DECODING_RUNS * Self::DECODING_RUN as u32);

    /// The quads of `k` the units of one of its fragments split, one a unit.
    const DECODING_RUNS: u32 = 4;

    /// The values of `k` one of its instructions has a unit read: its quad.
    const DECODING_RUN: usize = 4;

    /// The values of `K` a fragment's lane run looks one scale up for: NVFP4's block. A packed
    /// rhs's fragments decode under scale blocks a whole number of them long, NVFP4's sixteen
    /// values or MXFP4's thirty-two.
    const DECODED_BLOCK: usize = 16;

    /// What a packed item of no rows, or of more than [`MAX_PACKED_ROWS`](Self::MAX_PACKED_ROWS),
    /// is declined with.
    pub const ROWS_OUT_OF_RANGE: Decline = Decline::Unserved {
        why: "the plane fragment holds from one row to `PlaneFragmentMicrokernel::MAX_PACKED_ROWS` against a packed rhs, the rows of one fragment",
    };

    /// What a packed item whose operands are not a 16-bit lhs against an rhs of
    /// sixteen-value blocks along `K`, read along `K` alone, is declined with.
    pub(crate) const OPERANDS_UNSERVED: Decline = Decline::Unserved {
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
    /// `sum`: it offers the decoding instruction over `lhs`, which no element but a 16-bit float
    /// has, its target lays that instruction's lanes out as lane runs read them ([`LaneMap`]),
    /// and its planes are the width that lane map is laid over.
    fn decodes(hw: &Hardware, lhs: ElemType, sum: ElemType) -> bool {
        let (m, n, k) = Self::DECODING_SHAPE;
        let laid_out = LaneMap::try_from(&*hw.target).is_ok_and(|map| {
            map.edges() == (m as usize, n as usize) && map.depths().contains(&(k as usize))
        });
        manual_mma(hw).any(|shape| shape == Self::decoding_instruction(lhs, sum))
            && laid_out
            && hw.plane_size() == Ok(LaneMap::PLANE_UNITS as u32)
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
    /// through the tile DSL's transport. A window is whole windows of `K` and whole reads of the
    /// rhs's scales, and a unit's run of it, one of the four quads of `k` its fragment's units
    /// split, whole reads of the rhs. Only
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
        // A unit's run of the window, one of the four quads of `k` its fragment's units split, is
        // whole reads of the rhs, and the window whole reads of its scales: each read as widely as
        // they hold whole ([`MatmulBlocks::within`]). MXFP4 in tile order, bound for a 32-value
        // load, is read two words at a time in a run of 16, its scales one at a time in a window
        // of 64.
        let planned = space.within(settings.depth.get() * Self::DECODING_RUN, window);
        let reads_whole = planned.is_some();
        let space = planned.unwrap_or(space);
        // A window states `K` as whole scale blocks and the position inside one.
        let blocks_whole = space
            .operand(MatmulOperand::Rhs)
            .grouping(K, space.k)
            .is_some_and(|block| window.is_multiple_of(block));
        // A plane holding every row whole holds one fragment of them; one whose rows are tiled, a
        // stage's, holds the settings' fragments along `M`, each against the rhs fragment
        // decoded once a step.
        let rows_fit = settings.along_m == NonZeroUsize::MIN || item.rows == MatmulRows::Tiled;
        let windowed = rows_fit && settings.fragment_depth == PlaneFragmentDepth::Stage;
        let offered = windowed
            && Self::serves(item, &space)
            && reads_whole
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
                    step: None,
                    cost: instruction.cost(),
                    orders: instruction.orders(item.layouts),
                    settings: PlaneFragmentMicrokernelSettings {
                        along_n: NonZeroUsize::new(along_n)?,
                        ..settings
                    },
                    dimensions: item.dimensions,
                    blocks: Some(space),
                    k_stated: PlaneFragmentKStatement::Caller,
                    unit: PlaneUnit::Instruction,
                })
            })
    }

    /// The matrix unit `hw` offers a product of `lhs` by `rhs` summed in `sum`: the first of the
    /// Steps the device allows a plane ([`ProductStep::sizes`]), the one that contracts the most,
    /// the squarest among those that contract as much. A float plane's settings count their tile's
    /// shape in it: `8 × 8 × 8` on an M2, `16 × 16 × 16` for `f16` on NVIDIA and `16 × 16 × 8` for
    /// its tf32. `None` where the device offers none.
    pub fn matrix_unit(
        hw: &Hardware,
        lhs: ElemType,
        rhs: ElemType,
        sum: ElemType,
    ) -> Option<[usize; 3]> {
        Self::matrix_unit_within(hw, lhs, rhs, sum, None)
    }

    /// [`matrix_unit`](Self::matrix_unit), among the shapes whose `k` divides `block` where one
    /// is given: a plane walking decoded stages a scale block at a time cannot take a deeper call.
    fn matrix_unit_within(
        hw: &Hardware,
        lhs: ElemType,
        rhs: ElemType,
        sum: ElemType,
        block: Option<usize>,
    ) -> Option<[usize; 3]> {
        ProductStep::sizes(
            &hw.properties,
            ComputeScope::Plane,
            FragmentSource::Values(lhs),
            FragmentSource::Values(rhs),
            sum,
        )
        .into_iter()
        .find(|&[_, _, k]| block.is_none_or(|block| block.is_multiple_of(k)))
    }

    /// The plane against decoded stages of `item`, cut by the tile as a float plane is, its unit
    /// the device's largest whose `k` divides the scale `block`.
    pub(crate) fn decoding_plane(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &PlaneFragmentMicrokernelSettings,
        block: usize,
    ) -> Option<Self> {
        let values = item
            .elements
            .map(|_, element| FragmentSource::Values(element));
        Self::cut_by_the_tile(hw, item, settings, values.lhs, values.rhs, Some(block))
            .filter(|_| hw.plane_dim().is_ok())
            .filter(Self::lays_its_rows)
    }

    /// The planes against a float rhs. A plane's settings count the device's
    /// [`matrix_unit`](Self::matrix_unit) for the elements, the fewest of its calls every load
    /// fills whole ([`BaseRegion`]), so the instruction never changes what a setting means. With
    /// fragments one instruction deep, held by a plane, the one plane holding that tile, which the
    /// tile cuts into the calls of the instruction it picks for it
    /// ([`Instruction::for_accumulator`]); otherwise the planes that state their fragments
    /// ([`stated_candidates`](Self::stated_candidates)).
    fn float_candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &PlaneFragmentMicrokernelSettings,
    ) -> impl Iterator<Item = Self> {
        let planes: Vec<Self> = match (settings.holder, settings.fragment_depth) {
            (ComputeScope::Plane, PlaneFragmentDepth::Instruction) => {
                let values = item
                    .elements
                    .map(|_, element| FragmentSource::Values(element));
                Self::cut_by_the_tile(hw, item, settings, values.lhs, values.rhs, None)
                    .filter(|_| hw.plane_dim().is_ok())
                    .into_iter()
                    .collect()
            }
            _ => Self::stated_candidates(hw, item, settings),
        };
        planes.into_iter()
    }

    /// The planes against a float rhs that state their fragments, a grid of their instruction's
    /// calls cutting the tile its settings name in the device's matrix unit: every matrix
    /// instruction the device offers for the elements that the tile picks for a fragment of its
    /// own shape, as deep as the settings say, the fragment API's first, each the one that
    /// contracts the most per call first; the plane group MMA alone where the settings deal the
    /// tiles to plane groups. What a caller that walks its fragments itself reads, as attention's
    /// products do, and the planes whose fragments are a stage deep.
    pub(crate) fn stated_candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &PlaneFragmentMicrokernelSettings,
    ) -> Vec<Self> {
        if settings.holder == ComputeScope::MMA_PLANE_GROUP {
            return Self::wgmma(hw, item, settings)
                .map(|instruction| {
                    Self::stating(instruction, PlaneUnit::Instruction, item, *settings)
                })
                .into_iter()
                .collect();
        }
        let Some((unit, calls)) = Self::unit_of(hw, item, settings, None) else {
            return Vec::new();
        };
        let (MatmulOperands { lhs, rhs, .. }, sum) = (item.elements, item.sum);
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
        let deep = settings.fragment_depth == PlaneFragmentDepth::Stage;
        cmma.chain(mma)
            .filter(|_| hw.plane_dim().is_ok())
            .filter(|instruction| !deep || instruction.contracts_deeper_fragments())
            .filter_map(|instruction| {
                Self::stating(
                    instruction,
                    PlaneUnit::Matrix { unit, calls },
                    item,
                    *settings,
                )
                .stated(hw, item.elements)
            })
            .filter(|plane| {
                let values = item
                    .elements
                    .map(|_, element| FragmentSource::Values(element));
                plane.tiles_pick(hw, item.sum, values.lhs, values.rhs)
            })
            .collect()
    }

    /// A plane stating `instruction` against the float `item`, its settings counting `unit`: it
    /// hands the tile no Step, and spends what the instruction does.
    fn stating(
        instruction: PlaneFragmentInstruction,
        unit: PlaneUnit,
        item: &MatmulWorkItem,
        settings: PlaneFragmentMicrokernelSettings,
    ) -> Self {
        Self::over(instruction, None, instruction.cost(), unit, item, settings)
    }

    /// A plane over `instruction` against the float `item`, handing the tile `step`, which spends
    /// `cost`, its settings counting `unit`.
    fn over(
        instruction: PlaneFragmentInstruction,
        step: Option<ProductStep>,
        cost: StepCost,
        unit: PlaneUnit,
        item: &MatmulWorkItem,
        settings: PlaneFragmentMicrokernelSettings,
    ) -> Self {
        Self {
            instruction,
            step,
            cost,
            orders: instruction.orders(item.layouts),
            settings,
            dimensions: item.dimensions,
            blocks: None,
            k_stated: PlaneFragmentKStatement::Caller,
            unit,
        }
    }

    /// The device's matrix unit for `item`'s elements and the calls of it every load fills whole
    /// ([`PlaneUnit::Matrix`]), along the caller's dimensions: its edges exchanged where the
    /// settings lay the instruction's rows along `n`.
    fn unit_of(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &PlaneFragmentMicrokernelSettings,
        block: Option<usize>,
    ) -> Option<([usize; 3], [usize; 3])> {
        let [m, n, k] =
            Self::matrix_unit_within(hw, item.elements.lhs, item.elements.rhs, item.sum, block)?;
        let (m, n) = match settings.rows_along {
            PlaneFragmentRowsAlong::M => (m, n),
            PlaneFragmentRowsAlong::N => (n, m),
        };
        let region = Self::base_region(hw, &item.elements, [m, n, k]);
        Some((
            [m, n, k],
            [M, N, K].map(|dimension| region.instructions(dimension)),
        ))
    }

    /// The base region of a product of float inputs of `elements` under an instruction of
    /// `[m, n, k]` along the caller's dimensions.
    fn base_region(
        hw: &Hardware,
        elements: &MatmulOperands<ElemType>,
        [m, n, k]: [usize; 3],
    ) -> BaseRegion {
        let float = MatmulOperands::all(Operand::plain());
        BaseRegion::new(
            &float.vector_region(hw, elements),
            Space::new(&[(M, m), (N, n), (K, k)]),
        )
    }

    /// The edges along the caller's `m` and `n` of the tile a plane holds where its settings count
    /// the device's matrix unit for `item` ([`PlaneUnit::Matrix`]): its settings' counts times the
    /// unit and the calls of it every load fills whole. `None` where the device offers no unit, or
    /// commits to no single plane width, which a plane's units need.
    #[cfg(feature = "attention")]
    pub(crate) fn unit_tile(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &PlaneFragmentMicrokernelSettings,
    ) -> Option<[usize; 2]> {
        hw.plane_dim().ok()?;
        let ([m, n, _], [calls_m, calls_n, _]) = Self::unit_of(hw, item, settings, None)?;
        Some([
            settings.along_m.get() * m * calls_m,
            settings.along_n.get() * n * calls_n,
        ])
    }

    /// The plane whose tile, one unit deep, the tile cuts into the calls of the instruction it
    /// picks, its operands' fragments loaded from `lhs` and `rhs`, its settings counting the
    /// device's matrix unit, where the tile picks the fragment API or the crate's own transport
    /// over plain values: the instruction's shape is the cell the tile cuts the plane's tile into.
    fn cut_by_the_tile(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &PlaneFragmentMicrokernelSettings,
        lhs: FragmentSource,
        rhs: FragmentSource,
        block: Option<usize>,
    ) -> Option<Self> {
        let (unit, calls) = Self::unit_of(hw, item, settings, block)?;
        let ([m, n, k], [calls_m, calls_n, _]) = (unit, calls);
        let tile = [
            settings.along_m.get() * m * calls_m,
            settings.along_n.get() * n * calls_n,
        ];
        let elements = MmaConfig {
            a_type: item.elements.lhs,
            b_type: item.elements.rhs,
            cd_type: item.sum,
            m: 0,
            n: 0,
            k: k as u32,
        };
        let (instruction, step, cost) = Self::tiles_cut(hw, settings, tile, elements, lhs, rhs)?;
        let unit = PlaneUnit::Matrix { unit, calls };
        Some(Self::over(
            instruction,
            Some(step),
            cost,
            unit,
            item,
            *settings,
        ))
    }

    /// The Step of a plane's tile of `tile` along the caller's `m` and `n`, `elements.k` deep, its
    /// operands' fragments loaded from `lhs` and `rhs`, what computing it spends, and the
    /// instruction the tile picks for it, `elements` cut into the cell each call fills, where the
    /// tile picks the fragment API or the crate's own transport over plain values.
    fn tiles_cut(
        hw: &Hardware,
        settings: &PlaneFragmentMicrokernelSettings,
        tile: [usize; 2],
        elements: MmaConfig,
        lhs: FragmentSource,
        rhs: FragmentSource,
    ) -> Option<(PlaneFragmentInstruction, ProductStep, StepCost)> {
        let [rows_edge, columns_edge] = match settings.rows_along {
            PlaneFragmentRowsAlong::M => tile,
            PlaneFragmentRowsAlong::N => [tile[1], tile[0]],
        };
        let (rows, columns) = rows_first(settings.rows_along, lhs, rhs);
        let step = ProductStep {
            shape: [rows_edge, columns_edge, elements.k as usize],
            holder: settings.holder,
            lhs: rows,
            rhs: columns,
            sum: elements.cd_type,
        };
        let (device, target) = (&hw.properties, &hw.target);
        let call = Instruction::for_accumulator(
            device,
            target,
            step.holder,
            step.lhs,
            step.rhs,
            step.sum,
            step.shape,
        )?;
        let instruction = Self::picked(call, [elements.a_type, elements.b_type, elements.cd_type])?;
        Some((instruction, step, step.cost(device, target)?))
    }

    /// The tile's `call` as one of its fragments, of the lhs, rhs and sum `types`, at the call's
    /// shape. `None` for any instruction but the fragment API and the crate's own transport over
    /// plain values.
    fn picked(
        call: ShapedInstruction,
        [a_type, b_type, cd_type]: [ElemType; 3],
    ) -> Option<PlaneFragmentInstruction> {
        let [m, n, k] = call.shape.map(|edge| edge as u32);
        let shape = MmaConfig {
            a_type,
            b_type,
            cd_type,
            m,
            n,
            k,
        };
        match call.instruction {
            Instruction::Cmma => Some(PlaneFragmentInstruction::Cmma { shape }),
            Instruction::Mma {
                operands: MmaOperands::Plain,
                ..
            } => Some(PlaneFragmentInstruction::mma(shape)),
            _ => None,
        }
    }

    /// A plane holding `step`'s tile of `item`, `m × n × k` in elements along the caller's
    /// dimensions ([`PlaneUnit::Elements`]), its sum drained into an output
    /// ([`Instruction::for_accumulator`]), `step` stating the sources its kernel opens the sum on.
    /// `None` where the tile picks neither the fragment API nor the crate's own transport over
    /// plain values.
    pub fn draining(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: PlaneFragmentMicrokernelSettings,
        step: ProductStep,
    ) -> Option<Self> {
        let (device, target) = (&hw.properties, &hw.target);
        let pick = Instruction::for_accumulator(
            device,
            target,
            step.holder,
            step.lhs,
            step.rhs,
            step.sum,
            step.shape,
        )?;
        Self::holding(item, settings, step, pick, step.cost(device, target)?)
    }

    /// [`draining`](Self::draining), its sum's rows scaled between its products, as an online
    /// softmax's output is ([`Instruction::for_rescaled_accumulator`]): held in the crate's own
    /// transport where it offers the fragment API's call, it bounces nothing through a scratch.
    pub fn rescaling(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: PlaneFragmentMicrokernelSettings,
        step: ProductStep,
    ) -> Option<Self> {
        let (device, target) = (&hw.properties, &hw.target);
        let pick = Instruction::for_rescaled_accumulator(
            device, target, step.lhs, step.rhs, step.sum, step.shape,
        )?;
        Self::holding(
            item,
            settings,
            step,
            pick,
            step.rescaled_cost(device, target)?,
        )
    }

    /// [`draining`](Self::draining), its sum kept where it lies for the next product to read as
    /// its `A` ([`Instruction::for_stage_accumulator`]): never drained, it spends nothing a drain
    /// does.
    pub fn keeping(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: PlaneFragmentMicrokernelSettings,
        step: ProductStep,
    ) -> Option<Self> {
        let pick = Instruction::for_stage_accumulator(
            &hw.properties,
            &hw.target,
            step.lhs,
            step.rhs,
            step.sum,
            step.shape,
        )?;
        let kept = StepCost { bounced_cell: None };
        Self::holding(item, settings, step, pick, kept)
    }

    /// [`keeping`](Self::keeping), its lhs held in registers across the products
    /// ([`Instruction::for_held_factor_accumulator`]).
    pub fn keeping_held_lhs(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: PlaneFragmentMicrokernelSettings,
        step: ProductStep,
    ) -> Option<Self> {
        let pick = Instruction::for_held_factor_accumulator(
            &hw.properties,
            &hw.target,
            step.lhs,
            step.rhs,
            step.sum,
            step.shape,
        )?;
        let kept = StepCost { bounced_cell: None };
        Self::holding(item, settings, step, pick, kept)
    }

    /// A plane holding `step`'s tile on the tile's `pick` for it, spending `cost`.
    fn holding(
        item: &MatmulWorkItem,
        settings: PlaneFragmentMicrokernelSettings,
        step: ProductStep,
        pick: ShapedInstruction,
        cost: StepCost,
    ) -> Option<Self> {
        let instruction = Self::picked(pick, [item.elements.lhs, item.elements.rhs, item.sum])?;
        let unit = PlaneUnit::Elements { tile: step.shape };
        Some(Self::over(
            instruction,
            Some(step),
            cost,
            unit,
            item,
            settings,
        ))
    }

    /// Where a product reading this plane's kept sum as its `A` finds it: in its fragments, runs of
    /// the cells each of its calls fills ([`keeping`](Self::keeping)).
    pub fn kept_sum(&self) -> FragmentSource {
        let [(_, m), (_, n), _] = self.instruction.shape();
        FragmentSource::Fragments {
            instruction: self.tile_instruction(),
            cell: [m, n],
        }
    }

    /// This plane as the tile opens its sum on fragments loaded from `lhs` and `rhs`, as its
    /// kernel states them: the tile's pick for the plane's tile where the tile cuts it, this plane
    /// where the tile picks its stated instruction for them. `None` where it picks neither, so a
    /// kernel opening its sum there would find no instruction, or another one than the plane was
    /// sized for.
    pub fn opened_on(
        self,
        hw: &Hardware,
        item: &MatmulWorkItem,
        lhs: FragmentSource,
        rhs: FragmentSource,
    ) -> Option<Self> {
        let elements = match (self.unit, self.instruction) {
            (
                PlaneUnit::Matrix { .. },
                PlaneFragmentInstruction::Cmma { shape }
                | PlaneFragmentInstruction::Mma { shape, .. },
            ) => shape,
            (
                PlaneUnit::Elements { tile: [_, _, k] },
                PlaneFragmentInstruction::Cmma { shape }
                | PlaneFragmentInstruction::Mma { shape, .. },
            ) => MmaConfig {
                k: k as u32,
                ..shape
            },
            // The tile cuts only into the fragment API's or its own transport's calls.
            (PlaneUnit::Instruction, _) | (_, PlaneFragmentInstruction::Wgmma { .. }) => {
                return self.tiles_pick(hw, item.sum, lhs, rhs).then_some(self);
            }
        };
        let (instruction, step, cost) =
            Self::tiles_cut(hw, &self.settings, self.tile_edges(), elements, lhs, rhs)?;
        Some(Self {
            instruction,
            step: Some(step),
            cost,
            orders: instruction.orders(item.layouts),
            ..self
        })
    }

    /// Whether the tile picks this plane's instruction for one of its fragments, as deep as its
    /// settings say, its operands loaded from `lhs` and `rhs` and summed in `sum`, and fills it in
    /// one call: the fragments the plane states are the tile's own.
    fn tiles_pick(
        &self,
        hw: &Hardware,
        sum: ElemType,
        lhs: FragmentSource,
        rhs: FragmentSource,
    ) -> bool {
        let [(_, m), (_, n), _] = self.instruction.shape_along(self.settings.rows_along);
        let (rows, columns) = rows_first(self.settings.rows_along, lhs, rhs);
        let shape = [m, n, self.leaf_extent(MatmulDim::K)];
        let picked = Instruction::for_accumulator(
            &hw.properties,
            &hw.target,
            self.settings.holder,
            rows,
            columns,
            sum,
            shape,
        );
        picked
            == Some(ShapedInstruction {
                instruction: self.tile_instruction(),
                shape,
            })
    }

    /// This plane stating its fragments, a grid of its instruction's calls, where its settings
    /// count the device's matrix unit ([`stated_candidates`](Self::stated_candidates)). `None`
    /// where the instruction does not cut the plane's tile in whole [`BaseRegion`]s.
    fn stated(self, hw: &Hardware, elements: MatmulOperands<ElemType>) -> Option<Self> {
        let PlaneUnit::Matrix {
            unit: [unit_m, unit_n, unit_k],
            calls: [calls_m, calls_n, calls_k],
        } = self.unit
        else {
            return Some(self);
        };
        let settings = self.settings;
        let [(_, m), (_, n), (_, k)] = self.shape();
        let region = Self::base_region(hw, &elements, [m, n, k]);
        Some(Self {
            settings: PlaneFragmentMicrokernelSettings {
                along_m: Self::calls_cutting(
                    settings.along_m.get() * unit_m * calls_m,
                    &region,
                    M,
                )?,
                along_n: Self::calls_cutting(
                    settings.along_n.get() * unit_n * calls_n,
                    &region,
                    N,
                )?,
                depth: Self::calls_cutting(settings.depth.get() * unit_k * calls_k, &region, K)?,
                ..settings
            },
            unit: PlaneUnit::Instruction,
            ..self
        })
    }

    /// The instructions of `region` along `dimension` that cut `edge` of a plane's tile, where
    /// whole regions do.
    fn calls_cutting(
        edge: usize,
        region: &BaseRegion,
        dimension: Dimension,
    ) -> Option<NonZeroUsize> {
        let regions = Space::new(&[(dimension, edge)]).count_along(dimension, &region.space())?;
        NonZeroUsize::new(regions * region.instructions(dimension))
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
        offered
            .into_iter()
            .filter(|_| committed && block.is_some())
            // It contracts a fragment its own step deep, so a plane whose fragments span a
            // stage's steps of `K` is not one it serves.
            .filter(|instruction| !deep || instruction.contracts_deeper_fragments())
            // A step of the instruction holds whole reads of each input, values and scales, or its
            // window of them would start or end inside a read: each input is read as widely as
            // the step holds whole ([`MatmulBlocks::within`]). MXFP4's step of 64 holds half a
            // word of four scales of thirty-two, so its scales are read one at a time.
            .filter_map(|instruction| {
                let [_, _, (_, k)] = instruction.shape();
                Some(Self {
                    instruction,
                    step: None,
                    cost: instruction.cost(),
                    orders: instruction.orders(item.layouts),
                    settings,
                    dimensions: item.dimensions,
                    blocks: Some(space.within(k, k)?),
                    k_stated: PlaneFragmentKStatement::Blocks,
                    unit: PlaneUnit::Instruction,
                })
            })
            .collect()
    }

    /// Its instruction's `m × n × k`, its rows along the dimension the settings lay them along.
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

    /// The dimensions `K` is stated over, each with what one fragment `depth` instruction steps
    /// deep spans of it: the instruction's whole step along the caller's `K` dimension, or, where
    /// the plane reads a packed rhs as stored (the block-scaled instruction, or fragments that
    /// decode it as they load), the blocks its steps cover along `KB` beside the whole of each
    /// block along `KI`, or one block along `KB` and the part of it they cover along `KI`.
    fn k_dimensions(&self, depth: usize) -> Vec<(Dimension, usize)> {
        let [_, _, (_, k)] = self.shape();
        match (self.blocks, self.k_stated) {
            (Some(space), PlaneFragmentKStatement::Blocks) => {
                let block = space
                    .operand(MatmulOperand::Rhs)
                    .grouping(K, space.k)
                    .expect("a plane reading a packed rhs as stored reads it grouped along K");
                // A span of whole blocks, or, shorter than one, part of a single block: an `f16`
                // step of 16 inside MXFP4's block of 32.
                let (blocks, inside) = blocked(K);
                let span = Space::new(&[(K, k * depth)]);
                match span.count_along(K, &Space::new(&[(K, block)])) {
                    Some(count) => vec![(blocks, count), (inside, block)],
                    None => vec![(blocks, 1), (inside, span.along(K))],
                }
            }
            _ => vec![(self.dimensions.k, k * depth)],
        }
    }

    /// How a stage it reads holds its cells, laid out as `layout` and filled by `delivery`'s
    /// units: where it blocks the stage, in its instruction's tiles
    /// ([`PlaneFragmentInstruction::stage_tile`]), each one step deep along every dimension `K` is
    /// stated over, their rows arranged as its instruction reads them (`read_by`).
    pub fn stage_storage(
        &self,
        layout: StageLayout,
        delivery: Delivery,
        stage: &MatmulStage,
    ) -> StageStorage {
        let beside: Vec<Dimension> = stage.beside().collect();
        let tile = Self::over_k(
            self.instruction
                .stage_tile(&beside, stage.dimensions(), self.settings.rows_along),
            &self.k_dimensions(1),
        );
        self.read_by(layout.storage(stage.space(), tile), delivery)
    }

    /// How a stage it reads holds its cells where one TMA box lands it whole
    /// ([`PlaneFragmentInstruction::landed_stage_tile`]): each tile the whole stage deep along
    /// every dimension `K` is stated over, its rows swizzled where its instruction reads them so and in
    /// order where it cannot, since a box lands no padding (`read_by`).
    pub(crate) fn landed_stage_storage(&self, stage: &MatmulStage) -> StageStorage {
        let (beside, dimensions): (Vec<Dimension>, _) =
            (stage.beside().collect(), stage.dimensions());
        let whole_k: Vec<_> = self
            .k_dimensions(1)
            .into_iter()
            .map(|(dimension, _)| (dimension, stage.extent(dimension)))
            .collect();
        let tile = self.instruction.landed_stage_tile(
            &beside,
            dimensions,
            stage.extent(dimensions.k),
            self.settings.rows_along,
        );
        let swizzled = StageStorage::Tiled {
            block: Self::over_k(tile, &whole_k),
            chunks: RowChunks::Swizzled,
            order: StageOrder::AsSource,
        };
        self.read_by(swizzled, Delivery::Tma)
    }

    /// `stated` as a stage this plane reads and `delivery` fills lays it, its rows arranged by the
    /// tile for the instruction its fragments load through ([`StageStorage::read_by`]): the
    /// tile's pick for how its kernel opens the sum, a kept sum's included, which its Step alone
    /// would not say.
    pub fn read_by(&self, stated: StageStorage, delivery: Delivery) -> StageStorage {
        stated.read_by(self.tile_instruction(), delivery)
    }

    /// `tile` with its extents along `K` restated as `k_dimensions`: an instruction tiles `K` as
    /// one dimension, where a block-scaled plane states it as its scale blocks and the position
    /// inside one, which the stage lays beside its other dimensions.
    fn over_k(
        tile: Vec<(Dimension, usize)>,
        k_dimensions: &[(Dimension, usize)],
    ) -> Vec<(Dimension, usize)> {
        tile.into_iter()
            .map(|(dimension, extent)| {
                k_dimensions
                    .iter()
                    .copied()
                    .find(|&(k_dimension, _)| k_dimension == dimension)
                    .unwrap_or((dimension, extent))
            })
            .collect()
    }

    /// The dimension its steps of `K` walk: `K` itself, or the blocks a plane reading a packed rhs
    /// as stored steps across.
    pub(crate) fn walked(&self) -> Dimension {
        match (self.blocks, self.k_stated) {
            (Some(_), PlaneFragmentKStatement::Blocks) => blocked(K).0,
            _ => self.dimensions.k,
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
        let MatmulDimensions { m, n, .. } = self.dimensions;
        let (blocks, inside) = blocked(K);
        let [(_, rows), (_, columns), _] = self.shape();
        let window = Space::new(&[(K, self.leaf_extent(MatmulDim::K))]);
        let leaf = Levels::leaf(&[
            (m, rows),
            (n, columns),
            (blocks, window.count_along(K, &Space::new(&[(K, block)]))?),
            (inside, block),
        ])
        .walk_every(&[m]);
        let fragments = match self.settings.along_n.get() {
            1 => leaf,
            along_n => leaf.walk(&[(n, along_n)]),
        };
        Some(fragments.units(&[]))
    }

    /// Its own levels below a walker's planes, `planes` their counts: the plane's tile, its `m` by
    /// its `n` by its whole `k`, which the tile cuts and walks, as the backward's planes state each
    /// product they hold.
    #[cfg(feature = "attention-backward")]
    pub(crate) fn tile_levels(&self, planes: &[(Dimension, usize)]) -> Vec<cubecl::tile::Level> {
        let dimensions = self.dimensions;
        Levels::leaf(&[
            (dimensions.m, self.tile_extent(dimensions.m)),
            (dimensions.n, self.tile_extent(dimensions.n)),
            (dimensions.k, self.tile_extent(dimensions.k)),
        ])
        .planes(planes)
        .build()
    }

    /// [`tiling`](Self::tiling), its leaf also spanning `held`: dimensions of the caller's that
    /// this product does not name, each whole at every leaf, as attention's value holds the head
    /// dim its score contracts.
    pub fn tiling_holding(&self, held: &[(Dimension, usize)]) -> Levels {
        self.tiling_levels(held, &[])
    }

    /// [`tiling`](Self::tiling) through a stage `blocks` deep: its steps of `K` walk every block
    /// along `blocks.0` too, each step one grid of instructions inside one block. What a stage of
    /// several scale blocks of `K` is contracted in, the instruction stepping inside a block.
    pub(crate) fn tiling_over(&self, blocks: (Dimension, usize)) -> Levels {
        self.tiling_levels(&[], &[blocks])
    }

    /// The levels inside one plane's tile, its leaf also spanning `held`, its steps of `K`
    /// walking `outer` before the instruction's own steps.
    fn tiling_levels(&self, held: &[(Dimension, usize)], outer: &[(Dimension, usize)]) -> Levels {
        let settings = self.settings;
        let MatmulDimensions { m, n, .. } = self.dimensions;
        let (along_m, along_n) = (settings.along_m.get(), settings.along_n.get());
        // A fragment as deep as the stage's steps holds them all: the plane walks one step.
        let (depth, steps) = match settings.fragment_depth {
            PlaneFragmentDepth::Instruction => (1, self.steps_of_k()),
            PlaneFragmentDepth::Stage => (settings.depth.get(), 1),
        };
        let tiling = match (self.instruction, self.unit) {
            // The plane's tile, one step of `K` deep: the tile cuts it into its instruction's
            // calls.
            (
                PlaneFragmentInstruction::Cmma { .. } | PlaneFragmentInstruction::Mma { .. },
                PlaneUnit::Matrix { .. },
            ) => {
                let [tile_m, tile_n] = self.tile_edges();
                let leaf: Vec<_> = [(m, tile_m), (n, tile_n)]
                    .into_iter()
                    .chain(self.k_dimensions(depth))
                    .chain(held.iter().copied())
                    .collect();
                Levels::leaf(&leaf)
            }
            // The plane's tile as it states it: the tile cuts it and walks its `K`.
            (
                PlaneFragmentInstruction::Cmma { .. } | PlaneFragmentInstruction::Mma { .. },
                PlaneUnit::Elements {
                    tile: [tile_m, tile_n, tile_k],
                },
            ) => {
                let leaf: Vec<_> = [(m, tile_m), (n, tile_n), (self.dimensions.k, tile_k)]
                    .into_iter()
                    .chain(held.iter().copied())
                    .collect();
                Levels::leaf(&leaf)
            }
            (PlaneFragmentInstruction::Cmma { .. } | PlaneFragmentInstruction::Mma { .. }, _) => {
                let [(_, rows), (_, columns), _] = self.shape();
                let leaf: Vec<_> = [(m, rows), (n, columns)]
                    .into_iter()
                    .chain(self.k_dimensions(depth))
                    .chain(held.iter().copied())
                    .collect();
                Levels::leaf(&leaf).walk(&[(m, along_m), (n, along_n)])
            }
            // The steps along `n` are the instruction's own width: one instruction a group.
            (PlaneFragmentInstruction::Wgmma { .. }, _) => {
                let leaf: Vec<_> = self
                    .dimensions
                    .name(self.shape())
                    .into_iter()
                    .chain(held.iter().copied())
                    .collect();
                Levels::leaf(&leaf).walk(&[(m, 1), (n, 1)])
            }
        };
        let steps: Vec<(Dimension, usize)> = outer
            .iter()
            .copied()
            .chain([(self.walked(), steps)])
            .collect();
        tiling.walk(&steps)
    }

    /// The extent along `dimension`, one of the caller's, of the tile one plane holds: its edges
    /// along `m` and `n` (`tile_edges`), and the steps of `K` it walks through
    /// one stage; a tile stated in elements, as it states it.
    pub fn tile_extent(&self, dimension: Dimension) -> usize {
        match self.unit {
            PlaneUnit::Elements { tile } => along(tile, self.dimensions.dim(dimension)),
            PlaneUnit::Matrix { .. } | PlaneUnit::Instruction => self.stated_extent(dimension),
        }
    }

    /// [`tile_extent`](Self::tile_extent) where the plane states its tile in units: its edges
    /// along `m` and `n`, and the steps of `K` it walks through one stage.
    fn stated_extent(&self, dimension: Dimension) -> usize {
        let [(_, _), (_, _), (_, k)] = self.shape();
        let steps = self.steps_of_k();
        let k_dimensions = self.k_dimensions(steps);
        if let Some(&(_, extent)) = k_dimensions
            .iter()
            .find(|&&(k_dimension, _)| k_dimension == dimension)
        {
            return extent;
        }
        let [tile_m, tile_n] = self.tile_edges();
        match self.dimensions.dim(dimension) {
            Some(MatmulDim::M) => tile_m,
            Some(MatmulDim::N) => tile_n,
            Some(MatmulDim::K) => steps * k,
            _ => 1,
        }
    }

    /// The edges along `m` and `n` of the tile one plane holds: its settings' grid of the device's
    /// matrix unit, or of its instruction's calls; the plane group MMA's one instruction.
    fn tile_edges(&self) -> [usize; 2] {
        let settings = self.settings;
        match (self.instruction, self.unit) {
            (PlaneFragmentInstruction::Wgmma { .. }, _) => {
                let [(_, m), (_, n), _] = self.shape();
                [m, n]
            }
            (
                _,
                PlaneUnit::Matrix {
                    unit: [m, n, _],
                    calls: [calls_m, calls_n, _],
                },
            ) => [
                settings.along_m.get() * m * calls_m,
                settings.along_n.get() * n * calls_n,
            ],
            (
                _,
                PlaneUnit::Elements {
                    tile: [tile_m, tile_n, _],
                },
            ) => [tile_m, tile_n],
            (_, PlaneUnit::Instruction) => {
                let [(_, m), (_, n), _] = self.shape();
                [settings.along_m.get() * m, settings.along_n.get() * n]
            }
        }
    }

    /// The instruction steps of `K` one stage walks: its settings' depth in the device's matrix
    /// unit's calls, or in its instruction's; one, for a tile stated in elements, whose `K` the
    /// tile walks.
    fn steps_of_k(&self) -> usize {
        let depth = self.settings.depth.get();
        match self.unit {
            PlaneUnit::Matrix {
                calls: [_, _, calls_k],
                ..
            } => depth * calls_k,
            // The tile walks the plane's whole `K`: the plane states one step of it.
            PlaneUnit::Elements { .. } => 1,
            PlaneUnit::Instruction => depth,
        }
    }

    /// The shape of one of its fragments, `depth` of the instruction's steps of `K` deep.
    fn fragment_shape(&self, depth: usize) -> [(MatmulDim, usize); 3] {
        let [m, n, (dim, k)] = self.shape();
        [m, n, (dim, k * depth)]
    }

    /// The values of `K` one unit reads in one run where it decodes a packed rhs as it loads its
    /// fragments: a quarter of its window, the settings' steps of its instruction. A load of the
    /// rhs is no wider. `None` for a plane reading no packed rhs so: a float one, or the
    /// block-scaled instruction's.
    pub fn decoding_run(&self) -> Option<usize> {
        let decodes = self.blocks.is_some() && !self.instruction.block_scaled();
        decodes.then(|| self.settings.depth.get() * Self::DECODING_RUN)
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
        let bytes = match self.instruction {
            PlaneFragmentInstruction::Cmma { .. } | PlaneFragmentInstruction::Mma { .. } => {
                let [tile_m, tile_n] = self.tile_edges();
                let [_, _, (_, k)] = self.shape();
                // Bits rather than bytes: an `e2m1` value is half of one.
                let plane = tile_m * tile_n * item.sum.size()
                    + tile_m * k * item.elements.lhs.size_bits() / 8
                    + tile_n * k * item.elements.rhs.size_bits() / 8;
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

    /// Bytes of the window one plane's sum bounces through where `scratch` holds one: one call's
    /// cell or its whole tile of `item`'s sum ([`StepCost::scratch_cells`]). None where its units
    /// write their own cells.
    pub(crate) fn scratch_bytes(&self, item: &MatmulWorkItem, scratch: Scratch) -> usize {
        self.cost.scratch_cells(self.tile_edges(), scratch) * item.sum.size()
    }

    /// The plane group MMA for `item` at `settings`, where every rule it runs under holds: the
    /// device offers its form and a bulk tensor copy to land its stages; its planes are as wide as
    /// a plane group's ([`PlaneFragmentInstruction::plane_group_plane_units`]); the rhs lies
    /// `{n, k}`, as both operands must, K-major; the settings hold one step along `m`, a form
    /// `along_n` granules wide the form spans, and a stage of `depth` steps of `K` whose rows, each
    /// operand's, are one 32, 64 or 128-byte swizzle span, which the instruction's descriptors
    /// read. The cube walking it adds what only its fill decides
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
            && hw.plane_dim() == Ok(PlaneFragmentInstruction::plane_group_plane_units() as u32)
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
    fn admits(item: &MatmulWorkItem) -> Result<(), Decline> {
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
    /// plane groups, the plane group MMA alone, where it runs. Against a packed rhs, the planes
    /// that decode it as they load, at the grids
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

    /// The leaf a unit reads is one instruction's tile of each operand: it reads the lhs in
    /// stretches of the leaf's `k` and the rhs in stretches of the leaf's extent along the
    /// dimension its tiles lie along. It writes only its accumulator; the drain into the output is
    /// its walker's.
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
    /// plane holds, a grid of fragments, one matrix instruction each.
    fn tiling(&self) -> Levels {
        self.tiling_levels(&[], &[])
    }

    fn orders(&self) -> MatmulOperands<BindingOrder> {
        self.orders
    }

    fn outer_scale(&self) -> OuterScale {
        OuterScale::of(self.tile_instruction())
    }
}

/// The extent of `[m, n, k]` along `dim`; one along an dimension the product does not name.
fn along([m, n, k]: [usize; 3], dim: Option<MatmulDim>) -> usize {
    match dim {
        Some(MatmulDim::M) => m,
        Some(MatmulDim::N) => n,
        Some(MatmulDim::K) => k,
        None => 1,
    }
}

/// `lhs` and `rhs` in the order a plane's kernel opens its sum on them: the factor its
/// instruction's rows lie along first.
fn rows_first<T>(rows_along: PlaneFragmentRowsAlong, lhs: T, rhs: T) -> (T, T) {
    match rows_along {
        PlaneFragmentRowsAlong::M => (lhs, rhs),
        PlaneFragmentRowsAlong::N => (rhs, lhs),
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
            dimensions: MatmulDimensions::MNK,
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
            step: None,
            cost: instruction.cost(),
            orders: MatmulOperands::all(BindingOrder::Stated),
            settings: settings.rows_along(rows),
            dimensions: MatmulDimensions::MNK,
            blocks: None,
            k_stated: PlaneFragmentKStatement::Caller,
            unit: PlaneUnit::Instruction,
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

    /// Rows along `n` exchange the instruction's edges along `m` and `n`, so a plane's tile and its
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
    /// a drained sum over the plane's tile ([`Instruction::for_accumulator`]), its fragments the
    /// cells the tile cuts the tile into, on every mocked device and float element, and refuses
    /// where the tile layer picks none at any shape the device offers: a plane holds a fragment.
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
                    dimensions: MatmulDimensions::MNK,
                    reading: MatmulReading::AsStored,
                };
                let plane = PlaneFragmentMicrokernel::best(
                    &hw,
                    &item,
                    &PlaneFragmentMicrokernelSettings::DEFAULT,
                );
                let Ok(plane) = plane else {
                    let values = FragmentSource::Values(input);
                    let matmul = &hw.properties.features.matmul;
                    let any = matmul.cmma.iter().chain(&matmul.mma).any(|shape| {
                        Instruction::for_accumulator(
                            &hw.properties,
                            &hw.target,
                            ComputeScope::Plane,
                            values,
                            values,
                            f32,
                            [shape.m, shape.n, shape.k].map(|extent| extent as usize),
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
                    [plane.tile_extent(M), plane.tile_extent(N), k],
                );
                assert_eq!(
                    picked.map(|call| (call.instruction, call.cell())),
                    Some((plane.tile_instruction(), [m, n])),
                    "{name}, {kind:?}"
                );
            }
        }
    }

    /// A setting is one tile shape on a device, which the tile cuts: on an H100 offering `f16`
    /// three cmma forms and the manual `16 × 8 × 16`, the default's four by four units of
    /// `16 × 16 × 16` hold `64 × 64` of the output and walk `16` of `K` a stage, in the one
    /// instruction the tile picks for that tile, cut into its calls. The instruction is the tile's
    /// decision, never a candidate of its own.
    #[test]
    fn the_tile_cuts_the_tile_the_setting_names() {
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
            dimensions: MatmulDimensions::MNK,
            reading: MatmulReading::AsStored,
        };
        assert_eq!(
            PlaneFragmentMicrokernel::matrix_unit(&hw, f16, f16, f32),
            Some([16, 16, 16])
        );
        let tiles = |settings: &PlaneFragmentMicrokernelSettings| -> Vec<_> {
            PlaneFragmentMicrokernel::candidates(&hw, &item, settings)
                .map(|plane| {
                    let [(_, m), (_, n), (_, k)] = plane.instruction.shape();
                    (
                        (m, n, k),
                        [M, N, K].map(|dimension| plane.tile_extent(dimension)),
                    )
                })
                .collect()
        };
        assert_eq!(
            tiles(&PlaneFragmentMicrokernelSettings::DEFAULT),
            [((16, 16, 16), [64, 64, 16])]
        );
        assert_eq!(
            tiles(&PlaneFragmentMicrokernelSettings::grid(1, 2, 2)),
            [((16, 16, 16), [16, 32, 32])]
        );
    }
}
