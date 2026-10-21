use super::cube_staged_matmul_kernel;
use crate::kernels::MatmulBinding;
use crate::kernels::{CUBE_STRIPS, cube_order_name};
use crate::microkernel::staged_walk::MatmulCubeWrite;
use crate::microkernel::{MatmulDim, MatmulOperand, MatmulOperands, MatmulPartitioning};
use crate::stored::Packing;
use cubecl::ir::ElemType;
use cubek_std::Hardware;
use cubek_std::matmul_axes::{KI, MI, NI};
use cubek_std::microkernel::{CubeStaging, GlobalStream};
use cubek_tile::launch::Refusal;
use cubek_tile::space::CubeOrder;
use cubek_tile::stage::RowChunks;

use crate::kernels::{
    MATMUL_Q_RHS_NOT_PACKED, MatmulBlueprint, MatmulOutput, MatmulRoutine, MatmulRoutineProblem,
    MatmulTensors, MatmulView,
};
use crate::microkernel::dense::cube::{
    CubeStagedDelivery, CubeStagedDepth, CubeStagedMicrokernel, CubeStagedMicrokernelSettings,
    CubeStagedRows, CubeStagedStore,
};
use crate::microkernel::dense::plane::PlaneFragmentMicrokernelSettings;
use crate::microkernel::staged_walk::MatmulPlaneLayout;
use cubek_std::matmul_axes::{M, N};
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};
use cubek_tile::launch::Vectorization;
use cubek_tile::{
    StageStorage, TileArgLaunch,
    launch::{DestinationLaunch, TmaStored},
};

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

impl CubeStagedMatmulRoutine {
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

    /// [`Routine::estimate`] against `packing`, a packed weight: each cube reads its box's rows
    /// of the activation and its columns of the weight's words over the whole of `K`, its
    /// microkernel's blocks a stage, and writes its box of the output once. Then its planes, for
    /// the work, on the stages decoded.
    fn packed_estimate(
        &self,
        problem: &MatmulRoutineProblem,
        packing: &Packing,
        blueprint: &CubeStagedMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let item = MatmulView::of::<CubeStagedMicrokernel>(problem)
            .expect(MatmulView::PREPARED)
            .work_item();
        let cube = blueprint.partitioning.cube_box();
        let (rows, columns, block) = (cube.extent(MI), cube.extent(NI), cube.extent(KI));
        let stage_blocks = match blueprint.microkernel.depth {
            CubeStagedDepth::ScaleBlocks(blocks) => blocks.get(),
            CubeStagedDepth::Steps => unreachable!("a packed staged gemm stages scale blocks"),
        };
        let k = problem.k;
        let size = |operand: MatmulOperand| item.elements.get(operand).size();
        let words = packing.values_bytes(k * columns);
        estimate.staging = Some(CubeStaging {
            operations: 2.0 * (rows * columns * k) as f64,
            reads: vec![
                GlobalStream {
                    bytes: (rows * k * size(MatmulOperand::Lhs)) as u64,
                    run_bytes: block * size(MatmulOperand::Lhs),
                },
                GlobalStream {
                    bytes: words as u64,
                    run_bytes: packing.values_bytes(block),
                },
            ],
            writes: vec![GlobalStream {
                bytes: (rows * columns * size(MatmulOperand::Out)) as u64,
                run_bytes: columns * size(MatmulOperand::Out),
            }],
            stages: k.div_ceil(block * stage_blocks * blueprint.microkernel.planes_along_k.get())
                as u64,
            stages_in_flight: blueprint.microkernel.ring.stages_in_flight.get() as u32,
            drain_atomics: 0,
        });
        blueprint
            .microkernel
            .plane
            .estimate(&item.decoded(), &(), grid.planes(), estimate);
    }

    /// [`Routine::prepare`] against a packed weight: one cube a box of the output over the
    /// problem's block-split space. The activation binds a device's load of its element along
    /// `K`, at most a block; the weight at [`packed_rhs_width`](Self::packed_rhs_width); the
    /// output one value wide.
    fn prepare_packed(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &CubeStagedMatmulRoutineSettings,
    ) -> Result<(CubeStagedMatmulRoutineBlueprint, Grid), Refusal> {
        let MatmulOperands {
            lhs: None,
            rhs: Some(_),
            ..
        } = problem.packings
        else {
            return Err(MATMUL_Q_RHS_NOT_PACKED);
        };
        let view = MatmulView::of::<CubeStagedMicrokernel>(problem)?;
        view.check_steps(hw)?;
        let item = view.work_item();
        let microkernel = CubeStagedMicrokernel::best(hw, &item, &settings.microkernel)?;
        let partitioning = view.partitioning(
            microkernel.tiling(),
            &[MI, NI],
            settings.cube_order,
            MatmulView::ONE_CUBE_A_BOX,
        );
        let resolved = microkernel.resolve(
            hw,
            &item.decoded(),
            &problem.buffers(microkernel.plane.orders),
            MatmulCubeWrite::Replace,
            &partitioning,
        )?;
        let cube = CubeResources {
            plane_width: resolved.plane_width,
            shared_memory: resolved.shared_memory,
        };
        let grid = Grid::new(hw, partitioning.as_cubek(), cube)?;
        let block = partitioning.cube_box().extent(KI);
        let blueprint = MatmulBlueprint {
            microkernel: resolved.microkernel,
            partitioning,
            widths: MatmulOperands {
                lhs: hw.load_values(item.elements.lhs).min(block),
                rhs: Self::packed_rhs_width(
                    resolved.microkernel.plane.leaf_extent(MatmulDim::K),
                    &resolved.storage,
                    item.elements.lhs,
                ),
                out: 1,
            },
            shared_memory: resolved.storage,
        };
        Ok((blueprint, grid))
    }

    /// The width a packed weight binds at: `step`, one instruction step of `K`, a line of the
    /// stage it decodes into, so every unit of the cube decodes one; at most a chunk of the
    /// stage's rows where `storage` swizzles them. A swizzled stage is read a chunk a unit, and a
    /// line of `decoded` values wider than one would be read back value by value.
    fn packed_rhs_width(step: usize, storage: &StageStorage, decoded: ElemType) -> usize {
        match storage {
            StageStorage::Tiled {
                chunks: RowChunks::Swizzled,
                ..
            } => step.min(RowChunks::CHUNK_BYTES / decoded.size()),
            StageStorage::Tiled { .. } | StageStorage::Strided | StageStorage::Lines { .. } => step,
        }
    }

    /// [`Routine::prepare`] against a float weight.
    fn prepare_float(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &CubeStagedMatmulRoutineSettings,
    ) -> Result<(CubeStagedMatmulRoutineBlueprint, Grid), Refusal> {
        problem.check_plain()?;
        let view = MatmulView::of::<CubeStagedMicrokernel>(problem)?;
        let item = view.work_item();
        let place = |tiling| {
            view.partitioning(
                tiling,
                &[M, N],
                settings.cube_order,
                MatmulView::ONE_CUBE_A_BOX,
            )
        };
        let microkernel = CubeStagedMicrokernel::best(hw, &item, &settings.microkernel)?;
        let resolved = microkernel.resolve(
            hw,
            &item,
            &problem.buffers(microkernel.plane.orders),
            <Self as MatmulRoutine<CubeStagedMicrokernel, StageStorage>>::WRITE,
            &place(microkernel.tiling()),
        )?;
        let (microkernel, storage) = (resolved.microkernel, resolved.storage);
        let partitioning = place(microkernel.launch_tiling());
        let cube = CubeResources {
            plane_width: resolved.plane_width,
            shared_memory: resolved.shared_memory,
        };
        let grid = Grid::new(hw, partitioning.as_cubek(), cube)?;
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
            .at_widths(&blueprint.widths, &item);
        Ok((blueprint, grid))
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
    /// order, except cubek's multi-row box
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
    /// every cube's box. Against a packed weight, its own staging and its planes' work
    /// ([`packed_estimate`](Self::packed_estimate)).
    fn estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &CubeStagedMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        if let Some(packing) = &problem.packings.rhs {
            return self.packed_estimate(problem, packing, blueprint, grid, estimate);
        }
        blueprint.microkernel.estimate(
            &MatmulView::of::<CubeStagedMicrokernel>(problem)
                .expect(MatmulView::PREPARED)
                .work_item(),
            &blueprint.partitioning,
            grid.planes(),
            estimate,
        );
    }

    /// A plane is as many units as its instruction runs on: the width a matrix instruction issues
    /// at, or the software instruction's own count. One cube a box, its microkernel resolved
    /// against the box it is placed in ([`CubeStagedMicrokernel::resolve`]), then launched beside
    /// the planes its fill sets aside. Against a packed weight, over its block-split space
    /// ([`prepare_packed`](Self::prepare_packed)).
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        match problem.packings == MatmulOperands::all(None) {
            true => self.prepare_float(hw, problem, settings),
            false => self.prepare_packed(hw, problem, settings),
        }
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
        let item = tensors.work_item();
        // Against a packed weight the stages hold it decoded into the activation's element.
        let elements = match item.blocks {
            None => item.kernel_elements(),
            Some(_) => CubeStagedMicrokernel::packed_kernel_elements(&item),
        };
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
