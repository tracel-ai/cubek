//! What every cube staged routine shares: its blueprint, how it places and resolves its
//! microkernel, its widths, its estimate and its launch. The routines differ only in who fills
//! the stages and who stores the box, which the microkernel carries, and in the problems those
//! engines refuse, which each routine checks between [`CubeStagedPlacement::new`] and
//! [`CubeStagedPlacement::prepare`].

use super::kernel::cube_staged_matmul_kernel;
use crate::kernels::{
    CubeStagedMatmulRoutine, MATMUL_INDEX_UNREAD_BY_TMA, MATMUL_ONE_ELEMENT_ROWS_UNMOVED_BY_TMA,
    MATMUL_PACKED_WALK_PAST_K, MatmulBinding, MatmulBlueprint, MatmulOutput, MatmulPartitioning,
    MatmulRoutine, MatmulRoutineProblem, MatmulTensors, MatmulView,
};
use cubecl::ir::ElemType;
use cubecl::tile::launch::{
    DestinationLaunch, Refusal, TmaStored, VectorRequirement, Vectorization,
};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::space::CubeOrder;
use cubecl::tile::stage::RowChunks;
use cubecl::tile::{FragmentSource, Instruction, StageStorage, TileTensorLaunch};
use cubek_microkernel::matmul::cube::{
    CubeStagedAsyncCopyRefusal, CubeStagedDelivery, CubeStagedFill, CubeStagedMicrokernel,
    CubeStagedMicrokernelSettings, CubeStagedResolved, CubeStagedStore, CubeStagedTmaRefusal,
    CubeStagedTmaStoreRefusal, MatmulCubeSpace, MatmulCubeWrite,
};
use cubek_microkernel::matmul::{
    MatmulBlocks, MatmulDim, MatmulMicrokernel, MatmulOperand, MatmulOperands, MatmulWorkItem,
};
use cubek_std::Hardware;
use cubek_std::layout::BindingOrder;
use cubek_std::matmul_axes::{K, KB, KI, MI, N, NI};
use cubek_std::microkernel::{CubeStaging, GlobalStream, KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid};

/// The cube routines' blueprint: how a stage its microkernel walks holds its cells is the one
/// layout of shared memory that depends on the problem; the rest is the microkernel's.
pub type CubeStagedMatmulRoutineBlueprint = MatmulBlueprint<CubeStagedMicrokernel, StageStorage>;

/// A cube staged microkernel picked for a problem, placed one cube a box over it and resolved:
/// what each cube staged routine's `prepare` computes alike, before it checks its own engines.
pub(crate) struct CubeStagedPlacement<'a> {
    problem: &'a MatmulRoutineProblem,
    view: MatmulView<'a>,
    item: MatmulWorkItem,
    cube_order: CubeOrder,
    /// The item its planes contract ([`CubeStagedMicrokernel::contracted`]).
    pub(crate) contracted: MatmulWorkItem,
    /// Its microkernel, resolved against the box it was placed in.
    pub(crate) resolved: CubeStagedResolved,
    /// The box one cube covers.
    pub(crate) cube_space: MatmulCubeSpace,
}

impl<'a> CubeStagedPlacement<'a> {
    /// The first of `settings`' microkernels on `hw` for `problem`, its stages filled by `fill`
    /// and its box stored by `store`, placed one cube a box in `cube_order` and resolved. A
    /// microkernel decoding a packed input reads it in whole steps ([`MatmulView::check_steps`]).
    /// A box the TMA engine stores is checked before resolving, which sizes the shared memory it
    /// is drained into ([`CubeStagedTmaStoreRefusal`]).
    /// A stage an engine fills lands as it lies, so there a float product's planes read their
    /// stages undecoded, never rounded by the units on the way in.
    pub(crate) fn new(
        hw: &Hardware,
        problem: &'a MatmulRoutineProblem,
        settings: &CubeStagedMicrokernelSettings,
        cube_order: CubeOrder,
        fill: CubeStagedFill,
        store: CubeStagedStore,
    ) -> Result<Self, Refusal> {
        let view = MatmulView::new(problem, CubeStagedMatmulRoutine::ROWS_KIND)?;
        let item = view.work_item();
        CubeStagedMicrokernel::admits(&item)?;
        let by_units = fill.delivery == CubeStagedDelivery::Copy;
        let candidates: Vec<_> = CubeStagedMicrokernel::candidates(hw, &item, settings)
            .filter(|candidate| {
                by_units || item.blocks.is_some() || candidate.walk().decoded().is_none()
            })
            .collect();
        // An operand the TMA engine lands is read in the instruction the tile layer picks for
        // landed operands: the planes take the first candidate it picks, and the first one
        // otherwise, as the kernel's sum will.
        let tiles_pick = |candidate: &&CubeStagedMicrokernel| {
            let plane = &candidate.plane;
            let [(_, m), (_, n), (_, k)] = plane.instruction.shape_along(plane.settings.rows_along);
            let picked = Instruction::for_accumulator(
                &hw.properties,
                &hw.target,
                plane.settings.holder,
                FragmentSource::Landed(item.elements.lhs),
                FragmentSource::Landed(item.elements.rhs),
                item.sum,
                [m, n, k],
            );
            picked == Some(plane.tile_instruction())
        };
        let landed = fill.delivery == CubeStagedDelivery::Tma;
        let microkernel = candidates
            .iter()
            .find(|candidate| landed && tiles_pick(candidate))
            .or(candidates.first())
            .copied()
            .ok_or(Refusal::Declined {
                why: CubeStagedMicrokernel::decline(settings),
            })?;
        let microkernel = CubeStagedMicrokernel {
            fill,
            store,
            ..microkernel
        };
        if microkernel.walk().decoded().is_some() {
            view.check_steps(hw)?;
        }
        let contracted = microkernel.contracted(&item);
        let cube_space = view
            .partitioning(
                microkernel.tiling(),
                &[item.axes.m, item.axes.n],
                cube_order,
                MatmulView::ONE_CUBE_A_BOX,
            )
            .cube_space();
        if store == CubeStagedStore::Tma {
            check_tma_store(hw, problem, &contracted, &cube_space, &microkernel)?;
        }
        let resolved = microkernel.resolve(hw, &contracted, &cube_space)?;
        Ok(Self {
            problem,
            view,
            item,
            cube_order,
            contracted,
            resolved,
            cube_space,
        })
    }

    fn microkernel(&self) -> &CubeStagedMicrokernel {
        &self.resolved.microkernel
    }

    /// Whether the TMA engine can fill its stages: a box lands each stage as its instruction
    /// reads it ([`CubeStagedTmaRefusal`]), the operands are read through no index, and no
    /// input's rows are one element wide.
    pub(crate) fn check_tma_fill(&self, hw: &Hardware) -> Result<(), Refusal> {
        let microkernel = self.microkernel();
        let orders = microkernel.plane.orders;
        if let Some(refusal) = CubeStagedTmaRefusal::new(
            hw,
            &self.contracted,
            &self.problem.buffers(orders),
            &self.cube_space,
            microkernel,
        ) {
            return Err(refusal.into());
        }
        if self.problem.index.is_some() {
            return Err(MATMUL_INDEX_UNREAD_BY_TMA);
        }
        let one_element_rows = MatmulOperand::INPUTS.into_iter().any(|operand| {
            self.problem
                .rows_one_element_wide(operand, orders.get(operand))
        });
        match one_element_rows {
            true => Err(MATMUL_ONE_ELEMENT_ROWS_UNMOVED_BY_TMA),
            false => Ok(()),
        }
    }

    /// Whether the copy engine can fill its stages through its ring
    /// ([`CubeStagedAsyncCopyRefusal::new`]); the lines it copies are checked once the blueprint
    /// binds their widths ([`check_copy_engine_lines`]).
    pub(crate) fn check_copy_engine(&self, hw: &Hardware) -> Result<(), Refusal> {
        match CubeStagedAsyncCopyRefusal::new(hw, &self.microkernel().ring) {
            Some(refusal) => Err(refusal.into()),
            None => Ok(()),
        }
    }

    /// Its launch: launched beside the planes its fill sets aside, on a grid of the box's cubes,
    /// its blueprint's widths a float product's or a block-split space's.
    pub(crate) fn prepare(
        self,
        hw: &Hardware,
    ) -> Result<(CubeStagedMatmulRoutineBlueprint, Grid), Refusal> {
        let Self {
            problem,
            view,
            item,
            cube_order,
            resolved,
            ..
        } = self;
        let microkernel = resolved.microkernel;
        problem.check_block_scales(
            microkernel.block_scales(),
            microkernel.fill.delivery == CubeStagedDelivery::Tma,
        )?;
        let partitioning = view.partitioning(
            microkernel.launch_tiling(),
            &[item.axes.m, item.axes.n],
            cube_order,
            MatmulView::ONE_CUBE_A_BOX,
        );
        // Only a block-split space has `KB` and `KI` to ask about: a float product's has `K`, and
        // masks its reads past it.
        let past_k = || {
            [KB, KI]
                .into_iter()
                .any(|axis| partitioning.as_tile().overhangs(axis))
        };
        if item.blocks.is_some() && past_k() {
            return Err(MATMUL_PACKED_WALK_PAST_K);
        }
        let cube = CubeResources {
            plane_width: resolved.plane_width,
            shared_memory: resolved.shared_memory,
        };
        let grid = Grid::new(hw, partitioning.as_tile(), cube)?;
        let storage = resolved.storage;
        let blueprint = match item.blocks {
            None => float_blueprint(hw, problem, microkernel, partitioning, storage),
            Some(_) => blueprint_in_blocks(hw, problem, &item, microkernel, partitioning, storage),
        };
        Ok((blueprint, grid))
    }
}

/// Whether the TMA engine can store `microkernel`'s box of `problem`, stated as `contracted`, in the
/// box `cube_space`, each cube writing its own box once ([`CubeStagedTmaStoreRefusal`]), the
/// output's rows more than one element wide.
fn check_tma_store(
    hw: &Hardware,
    problem: &MatmulRoutineProblem,
    contracted: &MatmulWorkItem,
    cube_space: &MatmulCubeSpace,
    microkernel: &CubeStagedMicrokernel,
) -> Result<(), Refusal> {
    let orders = microkernel.plane.orders;
    if let Some(refusal) = CubeStagedTmaStoreRefusal::new(
        hw,
        contracted,
        &problem.buffers(orders),
        cube_space,
        MatmulCubeWrite::Replace,
        microkernel.planes_along_k,
    ) {
        return Err(refusal.into());
    }
    match problem.rows_one_element_wide(MatmulOperand::Out, orders.out) {
        true => Err(MATMUL_ONE_ELEMENT_ROWS_UNMOVED_BY_TMA),
        false => Ok(()),
    }
}

/// Whether the copy engine copies whole every line `blueprint` binds of the inputs it hands it,
/// those its fill does not decode, of `contracted`: 4, 8 or 16 bytes
/// ([`CubeStagedAsyncCopyRefusal::LineSize`]).
pub(crate) fn check_copy_engine_lines(
    blueprint: &CubeStagedMatmulRoutineBlueprint,
    contracted: &MatmulWorkItem,
) -> Result<(), Refusal> {
    let copied: Vec<MatmulOperand> = MatmulOperand::INPUTS
        .into_iter()
        .filter(|&operand| blueprint.microkernel.reads.get(operand).staged().is_none())
        .collect();
    match CubeStagedAsyncCopyRefusal::of_widths(&copied, &blueprint.widths, contracted) {
        Some(refusal) => Err(refusal.into()),
        None => Ok(()),
    }
}

/// What every cube staged routine's cubes access: nothing of the routine's own, which only walks
/// the cubes; its microkernel makes every access.
pub(crate) fn vectorize(
    microkernel: &CubeStagedMicrokernel,
    partitioning: &MatmulPartitioning,
    vectorization: &mut Vectorization<MatmulOperand>,
) {
    assert_eq!(
        partitioning.as_tile().fillers() as usize,
        microkernel.fill.filling_planes,
        "a cube is launched over its launch tiling, beside the planes its fill sets aside"
    );
    microkernel.vectorize(&partitioning.cube_space(), vectorization);
}

/// What every cube staged routine's estimate is: nothing of its own, since it only distributes the
/// cubes and its microkernel walks the whole of every cube's box. Over a block-split space, its
/// own staging and its planes' work (`estimate_in_blocks`).
pub(crate) fn estimate(
    problem: &MatmulRoutineProblem,
    blueprint: &CubeStagedMatmulRoutineBlueprint,
    grid: &Grid,
    estimate: &mut KernelEstimate,
) {
    let item = MatmulView::new(problem, CubeStagedMatmulRoutine::ROWS_KIND)
        .expect(MatmulView::PREPARED)
        .work_item();
    match item.blocks {
        None => blueprint.microkernel.estimate(
            &item,
            &blueprint.partitioning.cube_space(),
            grid.planes(),
            estimate,
        ),
        Some(_) => estimate_in_blocks(problem, blueprint, grid, estimate),
    }
}

/// Every cube staged routine's launch: the operands the units fill the stages from, or the tensor
/// maps the TMA engine fills them from; the output its fragments store, or the box the TMA engine
/// stores.
pub(crate) fn dispatch<O: DestinationLaunch, Out: MatmulOutput>(
    tensors: &MatmulTensors<Out>,
    binding: MatmulBinding<O::Operand, TileTensorLaunch, TileTensorLaunch>,
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
        // A packed product's inputs through descriptors over its block-split space.
        CubeStagedDelivery::Tma if tensors.work_item().blocks.is_some() => tensors
            .tma_packed_operands(
                &binding.launcher,
                &stage,
                &blueprint.shared_memory,
                blueprint.microkernel.plane.orders,
                blueprint.widths,
                blueprint.microkernel.block_scales(),
            ),
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
                tensors.tma_output(&stage),
                binding.launcher.partitioning_arg(),
                blueprint.microkernel.walk(),
                blueprint.shared_memory,
                elements,
            );
        },
    }
}

/// A cube staged routine's estimate over a block-split space: each cube reads its box's rows of the lhs
/// and its columns of the rhs over the whole of `K`, a packed input its words and a float one
/// its values, a stage's depth of `K` at a time, and writes its box of the output once. Then
/// its planes, for the work, on the item they contract: the packed inputs decoded where its
/// stages decode them, read as stored where it stages them as they lie.
fn estimate_in_blocks(
    problem: &MatmulRoutineProblem,
    blueprint: &CubeStagedMatmulRoutineBlueprint,
    grid: &Grid,
    estimate: &mut KernelEstimate,
) {
    let item = MatmulView::new(problem, CubeStagedMatmulRoutine::ROWS_KIND)
        .expect(MatmulView::PREPARED)
        .work_item();
    let contracted = blueprint.microkernel.contracted(&item);
    let cube = blueprint.partitioning.cube_space().space;
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
/// buffer and the device admit.
fn float_blueprint(
    hw: &Hardware,
    problem: &MatmulRoutineProblem,
    microkernel: CubeStagedMicrokernel,
    partitioning: MatmulPartitioning,
    storage: StageStorage,
) -> CubeStagedMatmulRoutineBlueprint {
    let mut vectorization = Vectorization::new();
    vectorize(&microkernel, &partitioning, &mut vectorization);
    MatmulBlueprint::new(
        hw,
        problem,
        microkernel,
        partitioning,
        vectorization,
        storage,
    )
}

/// The blueprint of a product over a block-split space, its operands at the widths
/// [`widths_in_blocks`] settles.
fn blueprint_in_blocks(
    hw: &Hardware,
    problem: &MatmulRoutineProblem,
    item: &MatmulWorkItem,
    microkernel: CubeStagedMicrokernel,
    partitioning: MatmulPartitioning,
    storage: StageStorage,
) -> CubeStagedMatmulRoutineBlueprint {
    let widths = widths_in_blocks(hw, problem, item, &microkernel, &partitioning, &storage);
    MatmulBlueprint {
        widths,
        microkernel,
        partitioning,
        shared_memory: storage,
    }
}

/// The widths a cube staged `microkernel` binds the operands of `problem`, stated as `item`, at
/// over its block-split space, each placed as `partitioning` places it and staged as `storage`
/// lays it:
/// what [`blueprint_in_blocks`] holds, and every routine whose cubes
/// run that microkernel over such a space. A packed input its fill decodes binds at
/// [`decoded_width`]; every other input at a device's load of the element
/// its planes contract it at, at most a block, a float one narrowed as its axes require
/// ([`float_width_in_blocks`]); the output one value wide.
pub(crate) fn widths_in_blocks(
    hw: &Hardware,
    problem: &MatmulRoutineProblem,
    item: &MatmulWorkItem,
    microkernel: &CubeStagedMicrokernel,
    partitioning: &MatmulPartitioning,
    storage: &StageStorage,
) -> MatmulOperands<usize> {
    let block = partitioning.cube_space().extent(KI);
    let blocks = item
        .blocks
        .expect("a product over a block-split space has blocks");
    let contracted = microkernel.contracted(item);
    let width = |operand: MatmulOperand| match microkernel.reads.get(operand).staged() {
        Some(_) => decoded_width(
            microkernel.plane.leaf_extent(MatmulDim::K),
            storage,
            contracted.elements.get(operand),
        ),
        None => {
            let widest = hw.load_values(contracted.elements.get(operand)).min(block);
            match problem.packings.get(operand) {
                Some(_) => widest,
                None => float_width_in_blocks(problem, &blocks, partitioning, operand, widest),
            }
        }
    };
    MatmulOperands {
        lhs: width(MatmulOperand::Lhs),
        rhs: width(MatmulOperand::Rhs),
        out: 1,
    }
}

/// The width float `operand` of `problem` binds at over its block-split space `blocks`, at most
/// `widest`. Restated there it lies as its own dense rows
/// ([`PlainTensor::restated`](crate::kernels::tensor::PlainTensor::restated)), along the inside
/// of its columns' axis: the lhs's `KI`, which a box holds whole blocks of, and the rhs's
/// `NI`, its buffer's rows, which a box may reach past. So it owes that axis what any operand
/// owes its innermost one ([`MatmulBlueprint::new`]): its buffer's alignment and its start's,
/// and, where the tiles overhang it, a width dividing its extent, the widest line the tile
/// engine masks at an edge.
fn float_width_in_blocks(
    problem: &MatmulRoutineProblem,
    blocks: &MatmulBlocks,
    partitioning: &MatmulPartitioning,
    operand: MatmulOperand,
    widest: usize,
) -> usize {
    let columns = match operand {
        MatmulOperand::Lhs => K,
        _ => N,
    };
    let [_, (inside, _)] = blocks.axes(columns);
    let mut vectorization = Vectorization::new();
    vectorization.require(VectorRequirement::Aligned {
        operand,
        alignment: problem.alignment(operand, BindingOrder::Stated),
    });
    vectorization.require(VectorRequirement::Aligned {
        operand,
        alignment: ElemAlignment::at_start(
            problem.start_bytes.get(operand),
            problem.elements.get(operand).size(),
        ),
    });
    if partitioning.overhangs(inside) {
        vectorization.require(VectorRequirement::masked(
            operand,
            partitioning.space(),
            inside,
        ));
    }
    vectorization.width(operand, |_| widest)
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
