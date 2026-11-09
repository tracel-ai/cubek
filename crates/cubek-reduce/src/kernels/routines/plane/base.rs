use super::kernel::{plane_reduce_kernel, plane_reduce_with_indices_kernel};
use crate::kernels::{
    ReduceBinding, ReduceRoutine, ReduceRoutineBlueprint, ReduceRoutineProblem, ReduceTensors,
    reduce_widths,
};
use cubecl::tile::launch::{Refusal, Vectorization};
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubecl::tile::{Levels, Partitioning};
use cubek_microkernel::reduce::ReduceOperand;
use cubek_microkernel::reduce::plane::{PlaneReduceMicrokernel, PlaneReduceMicrokernelSettings};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Every owner a plane of the launch, which reduces its rows with the plane microkernel; the
/// launch groups `planes` of them in a cube along the innermost row axis.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlaneReduceRoutine;

/// How many of the launch's planes a cube holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneReduceRoutineSettings {
    pub microkernel: PlaneReduceMicrokernelSettings,
    pub planes: usize,
}

impl core::fmt::Display for PlaneReduceRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} planes={}", self.microkernel, self.planes)
    }
}

pub type PlaneReduceRoutineBlueprint = ReduceRoutineBlueprint<PlaneReduceMicrokernel>;

impl Routine for PlaneReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = PlaneReduceMicrokernel;
    type Settings = PlaneReduceRoutineSettings;
    type Blueprint = PlaneReduceRoutineBlueprint;
    type Partitioning = Partitioning;

    /// Four planes a cube by default, then one and eight.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        PlaneReduceMicrokernel::settings(hw)
            .into_iter()
            .flat_map(|microkernel| {
                [4, 1, 8].map(|planes| PlaneReduceRoutineSettings {
                    microkernel,
                    planes,
                })
            })
            .collect()
    }

    /// The routine's own code reads nothing: every access is its microkernel's.
    fn vectorize(
        &self,
        microkernel: &PlaneReduceMicrokernel,
        _partitioning: &Partitioning,
        vectorization: &mut Vectorization<ReduceOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    fn prepare(
        &self,
        hw: &Hardware,
        problem: &ReduceRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let microkernel =
            PlaneReduceMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let mut vectorization = Vectorization::new();
        microkernel.vectorize(&(), &mut vectorization);
        let widths = reduce_widths(hw, problem, vectorization);
        let owners = problem.owner_axes();
        let innermost = *owners
            .last()
            .expect("a problem places its owners along an axis");
        let partitioning = Partitioning::new(
            problem.space(),
            Levels::leaf(&problem.owner_box(widths[0]))
                .launch_planes(
                    &owners,
                    PlaneOrder::new(&[(innermost, settings.planes)], CubeOrder::RowMajor),
                )
                .build(),
        );
        let cube = CubeResources {
            plane_width: microkernel.lanes as u32,
            shared_memory: 0,
        };
        let grid = Grid::new(hw, &partitioning, cube)?;
        Ok((
            ReduceRoutineBlueprint::new(microkernel, partitioning, widths),
            grid,
        ))
    }

    fn estimate(
        &self,
        problem: &ReduceRoutineProblem,
        blueprint: &Self::Blueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        blueprint
            .microkernel
            .estimate(&problem.work_item(), &(), grid.planes(), estimate);
    }
}

impl ReduceRoutine<PlaneReduceMicrokernel> for PlaneReduceRoutine {
    fn dispatch(
        &self,
        tensors: &ReduceTensors,
        binding: ReduceBinding,
        blueprint: PlaneReduceRoutineBlueprint,
        grid: Grid,
    ) {
        let elements = tensors.problem().elements;
        // SAFETY: both operands are bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles; it holds no shared memory.
        match (binding.indices, tensors.problem().indices) {
            (Some(indices), Some(index)) => unsafe {
                plane_reduce_with_indices_kernel::launch_unchecked(
                    tensors.client(),
                    grid.launch.cube_count,
                    grid.launch.cube_dim,
                    blueprint.width,
                    binding.input.arg(),
                    binding.out.arg(),
                    indices.arg(),
                    binding.launcher.partitioning_arg(),
                    blueprint.microkernel,
                    [elements.input, elements.accumulator, elements.out, index],
                )
            },
            _ => unsafe {
                plane_reduce_kernel::launch_unchecked(
                    tensors.client(),
                    grid.launch.cube_count,
                    grid.launch.cube_dim,
                    blueprint.width,
                    binding.input.arg(),
                    binding.out.arg(),
                    binding.launcher.partitioning_arg(),
                    blueprint.microkernel,
                    [elements.input, elements.accumulator, elements.out],
                )
            },
        }
    }
}
