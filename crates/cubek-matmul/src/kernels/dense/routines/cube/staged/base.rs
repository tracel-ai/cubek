use super::kernel::cube_staged_matmul_kernel;
use crate::kernels::MatmulBinding;
use crate::kernels::select::packed::nvfp4_activation_scheme;
use crate::kernels::{CUBE_STRIPS, cube_order_name};
use core::num::NonZeroUsize;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::Space;
use cubecl::tile::launch::Refusal;
use cubecl::tile::space::CubeOrder;
use cubecl::tile::stage::RowChunks;
use cubek_microkernel::matmul::MatmulStep;
use cubek_microkernel::matmul::stored::Operand;
use cubek_microkernel::matmul::stored::Packing;
use cubek_microkernel::matmul::tile_order;
use cubek_microkernel::matmul::{
    MatmulDim, MatmulOperand, MatmulOperands, MatmulPartitioning, MatmulWorkItem,
};
use cubek_microkernel::staging::StageRing;
use cubek_std::matmul_axes::{K, KB, KI, MI, NI};
use cubek_std::microkernel::{CubeStaging, GlobalStream};
use cubek_std::{Hardware, Instructions};
use cubek_std::{UNIT_REGISTERS, manual_mma};

use crate::kernels::{
    MATMUL_INDEX_UNREAD_BY_TMA, MatmulBlueprint, MatmulOutput, MatmulRoutine, MatmulRoutineProblem,
    MatmulTensors, MatmulView,
};
use cubecl::tile::launch::Vectorization;
use cubecl::tile::{
    StageStorage, TileArgLaunch,
    launch::{DestinationLaunch, TmaStored},
};
use cubek_microkernel::matmul::dense::cube::{
    CubeStagedDelivery, CubeStagedFill, CubeStagedMicrokernel, CubeStagedMicrokernelSettings,
    CubeStagedRows, CubeStagedStore,
};
use cubek_microkernel::matmul::dense::plane::{
    PlaneFragmentMicrokernelSettings, PlaneFragmentTransport,
};
use cubek_microkernel::matmul::staged_walk::MatmulPlaneLayout;
use cubek_std::matmul_axes::{M, N};
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// `out = lhs · rhs` with every cube owning a box of the output: it stages the box for its planes,
/// which each own a tile of it, or split its contraction between them
/// ([`CubeStagedMicrokernel`]).
///
/// The cubes cover the output, one batch each.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeStagedMatmulRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedMatmulRoutineSettings {
    pub microkernel: CubeStagedMicrokernelSettings,
    pub cube_order: CubeOrder,
}

impl CubeStagedMatmulRoutineSettings {
    pub const DEFAULT: Self = Self {
        microkernel: CubeStagedMicrokernelSettings::DEFAULT,
        cube_order: CubeOrder::RowMajor,
    };

    /// One tile a cube, four planes splitting its `K`
    /// ([`CubeStagedMicrokernelSettings::SPLIT_ALONG_K`]), its cubes in the grid's order.
    pub const SPLIT_ALONG_K: Self = Self {
        microkernel: CubeStagedMicrokernelSettings::SPLIT_ALONG_K,
        cube_order: CubeOrder::RowMajor,
    };
}

/// The shape of a setting the routine races, counted in steps of the product
/// ([`block_scaled_step`](CubeStagedMatmulRoutine::block_scaled_step)): a plane's steps of `M`, `N` and `K` (a stage's depth), the layout of
/// its planes, the planes sharing each step of `K`, the stages in flight and who stores the box.
#[derive(Debug, Clone, Copy)]
struct StagedShape {
    steps: (usize, usize, usize),
    planes: MatmulPlaneLayout,
    planes_along_k: usize,
    ring: StageRing,
    store: CubeStagedStore,
}

impl CubeStagedMatmulRoutine {
    /// The settings an NVFP4 product read as stored is raced at on `hw`, counted in steps of the
    /// product ([`block_scaled_step`](Self::block_scaled_step)): a plane's steps of `M`, `N` and `K`, the layout of its planes,
    /// the planes sharing each step of `K`, the stages in flight and who stores the box, and who
    /// fills the stages.
    ///
    /// For a prompt's rows, filled by the TMA engine through the tile DSL's transport: four by
    /// eight steps on four by two planes, the box a long weight fills the device with; two by
    /// eight, the one a narrower weight at a prompt's chunk of rows leaves fewer cores idle with;
    /// two by four on two by two planes, the small box the narrowest projections take; each two
    /// stages in flight, its planes storing their fragments. Then four by four on two by four
    /// planes, and two by four on four by two or two by four planes, three stages in flight and the
    /// box stored by the TMA engine in one bulk copy, which a contraction of a few thousand
    /// finishes sooner with. A stage one step deep is one tile of a weight in [tile
    /// order](cubek_microkernel::matmul::tile_order), which a box lands as it lands rows. Filled by
    /// the cube's units asynchronously: four by four on two by four planes, three stages in flight,
    /// and two by four on four by two planes, two.
    ///
    /// For a decode's rows, one step of rows (the instruction's sixteen, every row a decode
    /// holds) by one or two steps of columns, so a weight of a few thousand columns is hundreds
    /// of boxes and every core reads its share of it. Filled by the TMA engine, a stage one step
    /// deep, the most one box lands, with three or four stages in flight to keep enough of the
    /// weight in the air. Filled by the units asynchronously, deeper stages: four steps on four
    /// planes side by side, or planes sharing each step of `K`, which a projection with few
    /// columns and a long `K` keeps busy with, one step deep on four planes where `K` divides no
    /// deeper stage.
    ///
    /// Each is kept where it prepares on a product of its own box, its stages within the
    /// device's shared memory and a unit's share of its planes within [`UNIT_REGISTERS`], the
    /// compiler's addressing beside it. Then the default, which every device serves; a device
    /// with no block-scaled instruction has no step, and races the default alone.
    ///
    /// Every box is the one measured best for some Qwen3-4B projection on an RTX PRO 6000, the
    /// weight read back to back from memory as a model step reads it: the prompt's at 512 and
    /// 2048 rows, the decode's at one and eight, in rows and in tile order. A setting asking for
    /// the TMA engine whose stage one box cannot land is filled by the units instead, so each
    /// list holds only what its fill lands.
    ///
    /// Apart from the float list ([`settings`](Routine::settings)): a block-scaled step contracts
    /// four times the float one's `K` from the same bytes, so the boxes that keep it fed are not
    /// the float product's, and the float table races none of them.
    pub fn block_scaled_settings(&self, hw: &Hardware) -> Vec<CubeStagedMatmulRoutineSettings> {
        let Some(step) = Self::block_scaled_step(hw) else {
            return vec![CubeStagedMatmulRoutineSettings::DEFAULT];
        };
        let layout = MatmulPlaneLayout::new;
        let (double, triple) = (StageRing::DOUBLE, StageRing::TRIPLE);
        let (fragments, by_tma_store) = (CubeStagedStore::Fragments, CubeStagedStore::Tma);
        let prompt = [
            ((4, 8, 1), layout(4, 2), 1, double, fragments),
            ((2, 8, 1), layout(4, 2), 1, double, fragments),
            ((2, 4, 1), layout(2, 2), 1, double, fragments),
            ((4, 4, 1), layout(2, 4), 1, triple, by_tma_store),
            ((2, 4, 1), layout(4, 2), 1, triple, by_tma_store),
            ((2, 4, 1), layout(2, 4), 1, triple, by_tma_store),
        ];
        let prompt_by_units = [
            ((4, 4, 1), layout(2, 4), 1, triple, fragments),
            ((2, 4, 1), layout(4, 2), 1, double, fragments),
        ];
        let quad = StageRing {
            stages_in_flight: NonZeroUsize::new(4).expect("four stages"),
            ..StageRing::DOUBLE
        };
        // Filled by the TMA engine, a stage one step deep: a box lands a stage row of at most a
        // swizzle's 128 bytes, two steps of `K` sharing it at most.
        let decode_by_tma = [
            ((1, 1, 1), layout(1, 4), 1, quad, fragments),
            ((1, 1, 1), layout(1, 4), 1, triple, fragments),
            ((1, 2, 1), layout(1, 4), 1, triple, fragments),
            ((1, 2, 1), layout(1, 2), 2, triple, fragments),
            ((1, 1, 1), layout(1, 2), 2, quad, fragments),
        ];
        // Filled by the units, as deep as `K` divides.
        let decode_by_units = [
            ((1, 1, 4), layout(1, 4), 1, double, fragments),
            ((1, 2, 4), layout(1, 4), 1, double, fragments),
            ((1, 1, 4), layout(1, 2), 4, double, fragments),
            ((1, 1, 1), layout(1, 1), 4, double, fragments),
            ((1, 2, 4), layout(1, 2), 2, double, fragments),
        ];
        let by_tma = prompt
            .into_iter()
            .chain(decode_by_tma)
            .map(|shape| (shape, CubeStagedDelivery::Tma));
        let by_units = prompt_by_units
            .into_iter()
            .chain(decode_by_units)
            .map(|shape| (shape, CubeStagedDelivery::AsyncCopy));
        let mut settings: Vec<_> = by_tma
            .chain(by_units)
            .map(|((steps, planes, planes_along_k, ring, store), delivery)| {
                let shape = StagedShape {
                    steps,
                    planes,
                    planes_along_k,
                    ring,
                    store,
                };
                Self::in_steps(&step, shape, delivery)
            })
            .filter(|settings| Self::fits(hw, &step, settings))
            .collect();
        settings.push(CubeStagedMatmulRoutineSettings::DEFAULT);
        settings
    }

    /// The step of an NVFP4 product on `hw` ([`MatmulStep`]): the least box every load of
    /// both operands fills whole — a row of the TMA engine's box, a word of scales — under the
    /// device's block-scaled instruction. `None` on a device that offers none. A weight in rows and
    /// one in [tile order](cubek_microkernel::matmul::tile_order) step alike, and a load on such a device
    /// stores in this step.
    pub fn block_scaled_step(hw: &Hardware) -> Option<MatmulStep> {
        let e2m1 = ElemType::Float(FloatKind::E2M1);
        let instruction = manual_mma(hw)
            .for_triple(e2m1, e2m1, ElemType::Float(FloatKind::F32))
            .next()?;
        let operand = Self::nvfp4_weight().operand;
        let shape = Space::new(&[
            (M, instruction.m as usize),
            (N, instruction.n as usize),
            (K, instruction.k as usize),
        ]);
        Some(
            MatmulStep::new(hw, &operand, e2m1)
                .filled_by_tma(&operand, e2m1)
                .under(shape),
        )
    }

    /// An NVFP4 weight's packing over `[K, N]`, its words along `K`, as a checkpoint packs it and as
    /// tile order stores it.
    fn nvfp4_weight() -> Packing {
        let scheme = tile_order::scheme();
        Packing {
            operand: Operand::of_scheme(&scheme, K, [K, N]).expect("NVFP4 is one block along k"),
            scheme,
        }
    }

    /// `m` rows of an NVFP4 activation against an NVFP4 weight `k` deep and `n` wide, into an `f16`
    /// output: what [`block_scaled_settings`](Self::block_scaled_settings) prepares its boxes on.
    fn block_scaled_problem(m: usize, n: usize, k: usize) -> MatmulRoutineProblem {
        let scheme = nvfp4_activation_scheme();
        let activation = Packing {
            operand: Operand::of_scheme(&scheme, K, [M, K]).expect("NVFP4 is one block along k"),
            scheme,
        };
        MatmulRoutineProblem::packed(
            m,
            n,
            k,
            ElemType::Float(FloatKind::F16),
            MatmulOperands {
                lhs: Some(activation),
                rhs: Some(Self::nvfp4_weight()),
                out: None,
            },
        )
    }

    /// Registers the compiler adds around a plane's lower bound
    /// ([`registers_per_unit`](cubek_microkernel::matmul::dense::plane::PlaneFragmentMicrokernel::registers_per_unit)):
    /// addressing, loop state and the scale registers. A margin rather than a derivation, since
    /// the compiler decides it: the four by eight box's bound of 160 compiled to 182 to 246 on
    /// `sm_120`.
    const ADDRESSING_REGISTERS: u32 = 64;

    /// A setting of `shape`, its stages filled by `delivery` and its planes reaching them through
    /// the tile DSL's transport, which reads the rows a box lands swizzled.
    fn in_steps(
        step: &MatmulStep,
        shape: StagedShape,
        delivery: CubeStagedDelivery,
    ) -> CubeStagedMatmulRoutineSettings {
        let calls = |count: usize, axis| {
            NonZeroUsize::new(count * step.calls(axis)).expect("a step is at least one call")
        };
        let (rows, columns, depth) = shape.steps;
        CubeStagedMatmulRoutineSettings {
            microkernel: CubeStagedMicrokernelSettings {
                plane: PlaneFragmentMicrokernelSettings {
                    along_m: calls(rows, M),
                    along_n: calls(columns, N),
                    depth: calls(depth, K),
                    transport: PlaneFragmentTransport::Mma,
                    ..PlaneFragmentMicrokernelSettings::DEFAULT
                },
                planes: shape.planes,
                planes_along_k: NonZeroUsize::new(shape.planes_along_k)
                    .expect("at least one plane"),
                ring: shape.ring,
                store: shape.store,
                fill: CubeStagedFill {
                    delivery,
                    filling_planes: 0,
                },
                ..CubeStagedMicrokernelSettings::DEFAULT
            },
            cube_order: CubeOrder::RowMajor,
        }
    }

    /// Whether `settings` prepare a product of their own box on `hw`, two stages deep, and a
    /// unit's share of their planes fits its registers.
    fn fits(hw: &Hardware, step: &MatmulStep, settings: &CubeStagedMatmulRoutineSettings) -> bool {
        let (plane, planes) = (&settings.microkernel.plane, &settings.microkernel.planes);
        let extent = |axis, fragments: NonZeroUsize, count: NonZeroUsize| {
            fragments.get() / step.calls(axis) * step.along(axis) * count.get()
        };
        let problem = Self::block_scaled_problem(
            extent(M, plane.along_m, planes.along_m),
            extent(N, plane.along_n, planes.along_n),
            2 * plane.depth.get() / step.calls(K) * step.along(K),
        );
        let Ok((blueprint, grid)) = CubeStagedMatmulRoutine.prepare(hw, &problem, settings) else {
            return false;
        };
        let item = MatmulView::of::<CubeStagedMicrokernel>(&problem)
            .expect(MatmulView::PREPARED)
            .work_item()
            .as_stored()
            .expect("an NVFP4 product is read as stored");
        let registers = blueprint
            .microkernel
            .plane
            .registers_per_unit(&item, grid.planes().width);
        registers + Self::ADDRESSING_REGISTERS <= UNIT_REGISTERS
    }

    /// The settings a product against a packed weight races on `hw`
    /// ([`CubeStagedMicrokernel::packed_settings`]), its cubes in the grid's order.
    pub fn packed_settings(&self, hw: &Hardware) -> Vec<CubeStagedMatmulRoutineSettings> {
        CubeStagedMicrokernel::packed_settings(hw)
            .into_iter()
            .map(Self::on_grid)
            .collect()
    }

    /// The settings a product of a handful of rows against a packed weight races
    /// ([`CubeStagedMicrokernel::packed_one_step_settings`]), its cubes in the grid's order.
    pub fn packed_one_step_settings(&self) -> Vec<CubeStagedMatmulRoutineSettings> {
        CubeStagedMicrokernel::packed_one_step_settings()
            .into_iter()
            .map(Self::on_grid)
            .collect()
    }

    /// `microkernel`'s settings, its cubes in the grid's order.
    fn on_grid(microkernel: CubeStagedMicrokernelSettings) -> CubeStagedMatmulRoutineSettings {
        CubeStagedMatmulRoutineSettings {
            microkernel,
            cube_order: CubeOrder::RowMajor,
        }
    }

    /// [`Routine::estimate`] over a block-split space: each cube reads its box's rows of the lhs
    /// and its columns of the rhs over the whole of `K`, a packed input its words and a float one
    /// its values, a stage's depth of `K` at a time, and writes its box of the output once. Then
    /// its planes, for the work, on the item they contract: the packed inputs decoded where its
    /// stages decode them, read as stored where it stages them as they lie.
    fn estimate_in_blocks(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &CubeStagedMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let item = MatmulView::of::<CubeStagedMicrokernel>(problem)
            .expect(MatmulView::PREPARED)
            .work_item();
        let contracted = blueprint.microkernel.contracted(&item);
        let cube = blueprint.partitioning.cube_box();
        let (rows, columns, block) = (cube.extent(MI), cube.extent(NI), cube.extent(KI));
        let stage = blueprint.partitioning.stage();
        let stage_depth: usize = [KB, KI]
            .into_iter()
            .filter(|&axis| stage.space().contains(axis))
            .map(|axis| stage.extent(axis))
            .product();
        let k = problem.k;
        let size = |operand: MatmulOperand| item.elements.get(operand).size();
        let input = |operand: MatmulOperand, extent: usize| match problem.packings.get(operand) {
            Some(packing) => GlobalStream {
                bytes: packing.values_bytes(extent * k) as u64,
                run_bytes: packing.values_bytes(block),
            },
            None => GlobalStream {
                bytes: (extent * k * size(operand)) as u64,
                run_bytes: block * size(operand),
            },
        };
        estimate.staging = Some(CubeStaging {
            operations: 2.0 * (rows * columns * k) as f64,
            reads: vec![
                input(MatmulOperand::Lhs, rows),
                input(MatmulOperand::Rhs, columns),
            ],
            writes: vec![GlobalStream {
                bytes: (rows * columns * size(MatmulOperand::Out)) as u64,
                run_bytes: columns * size(MatmulOperand::Out),
            }],
            stages: k.div_ceil(stage_depth * blueprint.microkernel.planes_along_k.get()) as u64,
            stages_in_flight: blueprint.microkernel.ring.stages_in_flight.get() as u32,
            drain_atomics: 0,
        });
        blueprint
            .microkernel
            .plane
            .estimate(&contracted, &(), grid.planes(), estimate);
    }

    /// The blueprint of a float product: each operand at the widest width its accesses, its
    /// buffer and the device admit, the fill narrowed to the widths it lands.
    fn float_blueprint(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        item: &MatmulWorkItem,
        microkernel: CubeStagedMicrokernel,
        partitioning: MatmulPartitioning,
        storage: StageStorage,
    ) -> CubeStagedMatmulRoutineBlueprint {
        let mut vectorization = Vectorization::new();
        self.vectorize(&microkernel, &partitioning, &mut vectorization);
        let mut blueprint = MatmulBlueprint::new(
            hw,
            problem,
            microkernel,
            partitioning,
            vectorization,
            storage,
        );
        blueprint.microkernel.fill = blueprint
            .microkernel
            .fill
            .at_widths(&blueprint.widths, item);
        blueprint
    }

    /// The blueprint of a product over a block-split space, whose inputs are restated contiguous
    /// along `K`, so no buffer fact narrows them: a packed input its fill decodes binds at
    /// [`decoded_width`](Self::decoded_width); every other input a device's load of the element its planes contract it at,
    /// at most a block; the output one value wide.
    fn blueprint_in_blocks(
        hw: &Hardware,
        contracted: &MatmulWorkItem,
        microkernel: CubeStagedMicrokernel,
        partitioning: MatmulPartitioning,
        storage: StageStorage,
    ) -> CubeStagedMatmulRoutineBlueprint {
        let block = partitioning.cube_box().extent(KI);
        let width = |operand: MatmulOperand| match microkernel.reads.get(operand).staged() {
            Some(_) => Self::decoded_width(
                microkernel.plane.leaf_extent(MatmulDim::K),
                &storage,
                contracted.elements.get(operand),
            ),
            None => hw.load_values(contracted.elements.get(operand)).min(block),
        };
        MatmulBlueprint {
            widths: MatmulOperands {
                lhs: width(MatmulOperand::Lhs),
                rhs: width(MatmulOperand::Rhs),
                out: 1,
            },
            microkernel,
            partitioning,
            shared_memory: storage,
        }
    }

    /// The width a packed input the fill decodes binds at: `step`, one instruction step of `K`, a
    /// line of the stage it decodes into, so every unit of the cube decodes one; at most a chunk
    /// of the stage's rows where `storage` swizzles them. A swizzled stage is read a chunk a unit,
    /// and a line of `decoded` values wider than one would be read back value by value.
    fn decoded_width(step: usize, storage: &StageStorage, decoded: ElemType) -> usize {
        match storage {
            StageStorage::Tiled {
                chunks: RowChunks::Swizzled,
                ..
            } => step.min(RowChunks::CHUNK_BYTES / decoded.size()),
            StageStorage::Tiled { .. } | StageStorage::Strided | StageStorage::Lines { .. } => step,
        }
    }
}

impl core::fmt::Display for CubeStagedMatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cube-staged-matmul {}{}",
            self.microkernel,
            cube_order_name(self.cube_order)
        )
    }
}

/// The cube routine's blueprint: how a stage its microkernel walks holds its cells is the one
/// layout of shared memory that depends on the problem; the rest is the microkernel's.
pub type CubeStagedMatmulRoutineBlueprint = MatmulBlueprint<CubeStagedMicrokernel, StageStorage>;

impl Routine for CubeStagedMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = CubeStagedMicrokernel;
    type Settings = CubeStagedMatmulRoutineSettings;
    type Blueprint = CubeStagedMatmulRoutineBlueprint;
    type Partitioning = MatmulPartitioning;

    /// Its microkernel's settings ([`CubeStagedMicrokernel::settings`]), its cubes in the grid's
    /// order, except the tile DSL's multi-row box
    /// ([`MULTI_ROW`](CubeStagedMicrokernelSettings::MULTI_ROW)), whose cubes walk in strips along
    /// `m`. Then its own knob around the boxes it suits: the default, the default's grid on two by
    /// two planes with its stage's rows in order, and the planes splitting `K`, in strips.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let [along_m, _] = CUBE_STRIPS;
        let on_grid = CubeStagedMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| Self::Settings {
                microkernel,
                cube_order: match microkernel.same_box(&CubeStagedMicrokernelSettings::MULTI_ROW) {
                    true => along_m,
                    false => CubeOrder::RowMajor,
                },
            });
        let rows_in_order_on_2x2 = CubeStagedMicrokernelSettings {
            stage_rows: CubeStagedRows::InOrder,
            ..CubeStagedMicrokernelSettings::boxed(
                PlaneFragmentMicrokernelSettings::DEFAULT,
                MatmulPlaneLayout::new(2, 2),
            )
        };
        let in_strips = [
            Self::Settings {
                cube_order: along_m,
                ..Self::Settings::DEFAULT
            },
            Self::Settings {
                microkernel: rows_in_order_on_2x2,
                cube_order: along_m,
            },
        ]
        .into_iter()
        .chain(CUBE_STRIPS.map(|cube_order| Self::Settings {
            cube_order,
            ..Self::Settings::SPLIT_ALONG_K
        }));
        on_grid.chain(in_strips).collect()
    }

    /// Nothing of its own: it only walks the cubes, and its microkernel makes every access.
    fn vectorize(
        &self,
        microkernel: &CubeStagedMicrokernel,
        partitioning: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        microkernel.vectorize(partitioning, vectorization);
    }

    /// Nothing of its own: it only distributes the cubes, and its microkernel walks the whole of
    /// every cube's box. Over a block-split space, its own staging and its planes' work
    /// (`estimate_in_blocks`).
    fn estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &CubeStagedMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let item = MatmulView::of::<CubeStagedMicrokernel>(problem)
            .expect(MatmulView::PREPARED)
            .work_item();
        match item.blocks {
            None => blueprint.microkernel.estimate(
                &item,
                &blueprint.partitioning,
                grid.planes(),
                estimate,
            ),
            Some(_) => self.estimate_in_blocks(problem, blueprint, grid, estimate),
        }
    }

    /// A plane is as many units as its instruction runs on: the width a matrix instruction issues
    /// at, or the software instruction's own count. One cube a box of the output, over the
    /// problem's space or, against packed inputs, its block-split space; its microkernel resolved
    /// against the item its planes contract ([`CubeStagedMicrokernel::contracted`]) and the box it
    /// is placed in ([`CubeStagedMicrokernel::resolve`]), then launched beside the planes its fill
    /// sets aside. A microkernel decoding a packed input reads it in whole steps
    /// ([`MatmulView::check_steps`]). Its blueprint's widths are a float product's
    /// (`float_blueprint`) or a block-split space's
    /// (`blueprint_in_blocks`).
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let view = MatmulView::of::<CubeStagedMicrokernel>(problem)?;
        let item = view.work_item();
        let microkernel = CubeStagedMicrokernel::best(hw, &item, &settings.microkernel)?;
        if microkernel.walk().decoded().is_some() {
            view.check_steps(hw)?;
        }
        let contracted = microkernel.contracted(&item);
        let place = |tiling| {
            view.partitioning(
                tiling,
                &[item.axes.m, item.axes.n],
                settings.cube_order,
                MatmulView::ONE_CUBE_A_BOX,
            )
        };
        let resolved = microkernel.resolve(
            hw,
            &contracted,
            &problem.buffers(microkernel.plane.orders),
            <Self as MatmulRoutine<CubeStagedMicrokernel, StageStorage>>::WRITE,
            &place(microkernel.tiling()),
        )?;
        let (microkernel, storage) = (resolved.microkernel, resolved.storage);
        if problem.index.is_some() && microkernel.fill.delivery == CubeStagedDelivery::Tma {
            return Err(MATMUL_INDEX_UNREAD_BY_TMA);
        }
        let partitioning = place(microkernel.launch_tiling());
        let cube = CubeResources {
            plane_width: resolved.plane_width,
            shared_memory: resolved.shared_memory,
        };
        let grid = Grid::new(hw, partitioning.as_tile(), cube)?;
        let blueprint = match item.blocks {
            None => self.float_blueprint(hw, problem, &item, microkernel, partitioning, storage),
            Some(_) => {
                Self::blueprint_in_blocks(hw, &contracted, microkernel, partitioning, storage)
            }
        };
        Ok((blueprint, grid))
    }
}

impl MatmulRoutine<CubeStagedMicrokernel, StageStorage> for CubeStagedMatmulRoutine {
    fn dispatch<O: DestinationLaunch, Out: MatmulOutput>(
        &self,
        tensors: &MatmulTensors<Out>,
        binding: MatmulBinding<O::Operand, TileArgLaunch, TileArgLaunch>,
        blueprint: CubeStagedMatmulRoutineBlueprint,
        grid: Grid,
    ) {
        // The elements of the item its planes contract: packed inputs it decodes at the element
        // they decode into, ones it reads as stored at the element their values name.
        let elements = blueprint
            .microkernel
            .contracted(&tensors.work_item())
            .kernel_elements();
        let client = tensors.client();
        let stage = blueprint.partitioning.stage();
        let out_width = blueprint.widths.out;
        // The fill's delivery is who moves the operands into the stages: the cube's units from the
        // bound operands, or the TMA engine from descriptors over the same tensors.
        let (lhs, rhs) = match blueprint.microkernel.fill.delivery {
            CubeStagedDelivery::Copy | CubeStagedDelivery::AsyncCopy => (binding.lhs, binding.rhs),
            CubeStagedDelivery::Tma => tensors.tma_operands(
                &stage,
                &blueprint.shared_memory,
                blueprint.microkernel.plane.orders,
            ),
        };
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, or by a descriptor, whose copies fill what lies
        // past the operand with zeros and whose stores leave what lies past it unwritten; the
        // kernel indexes no buffer but through its tiles. Its shared memory is sized by the planes
        // of the grid it launches on: the stage fill counts `grid.launch.cube_dim`'s units, the
        // scratch windows `grid.planes()`.
        match blueprint.microkernel.store {
            CubeStagedStore::Fragments => unsafe {
                cube_staged_matmul_kernel::launch_unchecked::<O>(
                    client,
                    grid.launch.cube_count.clone(),
                    grid.launch.cube_dim,
                    out_width,
                    lhs,
                    rhs,
                    O::arg(binding.out),
                    binding.launcher.partitioning_arg(),
                    blueprint.microkernel.walk(),
                    blueprint.shared_memory,
                    elements,
                );
            },
            CubeStagedStore::Tma => unsafe {
                cube_staged_matmul_kernel::launch_unchecked::<TmaStored>(
                    client,
                    grid.launch.cube_count.clone(),
                    grid.launch.cube_dim,
                    out_width,
                    lhs,
                    rhs,
                    tensors.tma_output(&stage).arg(),
                    binding.launcher.partitioning_arg(),
                    blueprint.microkernel.walk(),
                    blueprint.shared_memory,
                    elements,
                );
            },
        }
    }
}
