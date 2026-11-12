use super::ConvCandidates;
use super::kernel::cube_staged_conv_kernel;
use crate::kernels::tiled::{
    ConvBinding, ConvPartitioning, ConvProblem, ConvRoutine, ConvSettings, ConvStaging, ConvTiles,
    TileConvBlueprint,
};
use cubecl::tile::launch::{Refusal, Vectorization};
use cubek_microkernel::matmul::cube::{
    CubeStagedBox, CubeStagedDepth, CubeStagedMicrokernel, MatmulPlaneLayout,
};
use cubek_microkernel::matmul::plane::PlaneFragmentMicrokernelSettings as Fragments;
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// A cube owning a box of the output rows and channels, staging the box's window of the input and
/// of the rhs once for its planes, which each own a tile of it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeStagedConvRoutine;

impl CubeStagedConvRoutine {
    /// 2x2 fragments of 16 rows on each of 2x2 planes.
    pub const F2X2_P2X2_T16: ConvSettings =
        ConvSettings::new(16, Fragments::grid(2, 2, 1), MatmulPlaneLayout::new(2, 2));

    /// 1x2 fragments of 16 rows on each of 1x2 planes: one tile of rows.
    pub const F1X2_P1X2_T16: ConvSettings =
        ConvSettings::new(16, Fragments::grid(1, 2, 1), MatmulPlaneLayout::new(1, 2));

    /// 2x2 fragments of 16 rows on each of 1x2 planes: two tiles of rows.
    pub const F2X2_P1X2_T16: ConvSettings =
        ConvSettings::new(16, Fragments::grid(2, 2, 1), MatmulPlaneLayout::new(1, 2));

    /// 1x2 fragments of 16 rows on each of 2x2 planes: two tiles of rows.
    pub const F1X2_P2X2_T16: ConvSettings =
        ConvSettings::new(16, Fragments::grid(1, 2, 1), MatmulPlaneLayout::new(2, 2));

    /// Seven 8x32 instructions down each of two planes across the columns.
    pub const F7X1_P1X2_T8: ConvSettings =
        ConvSettings::new(8, Fragments::grid(7, 1, 1), MatmulPlaneLayout::new(1, 2));

    /// One 8x32 instruction on each of two planes across the columns.
    pub const F1X1_P1X2_T8: ConvSettings =
        ConvSettings::new(8, Fragments::grid(1, 1, 1), MatmulPlaneLayout::new(1, 2));

    /// One 8x32 plane for each of seven output rows, which share each staged weight block.
    pub const F1X1_P1X1_T8_A7: ConvSettings =
        ConvSettings::new(8, Fragments::grid(1, 1, 1), MatmulPlaneLayout::new(1, 1)).across(7);
}

impl Routine for CubeStagedConvRoutine {
    type Problem = ConvProblem;
    type Microkernel = CubeStagedMicrokernel;
    type Settings = ConvSettings;
    type Blueprint = TileConvBlueprint<CubeStagedMicrokernel>;
    type Partitioning = ConvPartitioning;

    /// The device's own: [`ConvCandidates`].
    fn settings(&self, hw: &Hardware) -> Vec<ConvSettings> {
        ConvCandidates::on(hw)
            .map(|candidates| candidates.list())
            .unwrap_or_default()
    }

    /// Nothing of its microkernel's: the stage fill's own reads are the partitioning's.
    fn vectorize(
        &self,
        _microkernel: &CubeStagedMicrokernel,
        partitioning: &ConvPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        partitioning.vectorize(vectorization);
    }

    /// The cube's staging, which its partitioning places, and its planes' work.
    fn estimate(
        &self,
        problem: &ConvProblem,
        blueprint: &Self::Blueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        estimate.staging = Some(
            blueprint
                .partitioning
                .staging(&problem.elements, blueprint.microkernel.ring),
        );
        blueprint
            .microkernel
            .plane
            .estimate(&problem.work_item(), &(), grid.planes(), estimate);
    }

    /// The cube microkernel over the instruction `settings` ask for, a cube a box of the output,
    /// one plane for every tile of it, the stages held as the partitioning lays them out.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &ConvProblem,
        settings: &ConvSettings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let plane = settings.plane(hw, problem)?;
        let tiles = ConvTiles::new(problem, &plane, settings)?;
        let partitioning = ConvPartitioning::cube_staged(hw, problem, &tiles);
        let roles = problem.roles();
        let mut microkernel = CubeStagedMicrokernel::new(
            &settings.microkernel,
            plane,
            CubeStagedDepth::Steps,
            MatmulOperands::AS_BOUND,
        );
        let bounces = microkernel.bounces_masked_output();
        let staging = match bounces {
            true => ConvStaging::Bounced,
            false => ConvStaging::Staged,
        };
        let placed = CubeStagedBox {
            stage: partitioning.block().stage(roles.axes()),
            contracting_planes: settings.planes() * plane.instruction.planes(),
            masks_output: bounces,
            storage: Some(partitioning.storage()),
        };
        microkernel.ring = partitioning.ring(settings.microkernel.ring);
        let resolved = microkernel.resolve_in(hw, &problem.work_item(), &placed)?;
        let mut vectorization = Vectorization::new();
        self.vectorize(&resolved.microkernel, &partitioning, &mut vectorization);
        let cube = CubeResources {
            plane_width: resolved.plane_width,
            shared_memory: resolved.shared_memory,
        };
        TileConvBlueprint::new(
            hw,
            problem,
            resolved.microkernel,
            partitioning,
            vectorization,
            staging,
            cube,
        )
    }
}

impl ConvRoutine<CubeStagedMicrokernel> for CubeStagedConvRoutine {
    /// Every cube staging for its planes, which read the stage.
    fn dispatch(
        &self,
        problem: &ConvProblem,
        binding: ConvBinding,
        blueprint: TileConvBlueprint<CubeStagedMicrokernel>,
    ) {
        let walk = blueprint.microkernel.walk();
        let ConvBinding {
            launcher,
            input,
            rhs,
            out,
        } = binding;
        // SAFETY: every operand is a tile bound through the launcher, which masks the accesses a
        // box reaches past its tensor; the kernel indexes no buffer but through those tiles.
        unsafe {
            cube_staged_conv_kernel::launch_unchecked(
                launcher.client(),
                launcher.cube_count(),
                launcher.cube_dim(),
                input,
                rhs,
                out,
                launcher.partitioning_arg(),
                walk,
                blueprint.partitioning.storage(),
                problem.kernel_elements(),
            );
        }
    }
}
