use super::kernel::plane_fragment_conv_kernel;
use crate::kernels::tiled::{
    ConvBinding, ConvBlueprint, ConvLines, ConvPartitioning, ConvProblem, ConvRoutine,
    ConvSettings, ConvTiles, STAGE_RING,
};
use cubecl::tile::launch::{Refusal, Vectorization};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_microkernel::matmul::plane::PlaneFragmentMicrokernel;
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{Grid, Routine};

/// A plane of the launch for every box of the output rows and channels, staging its own window of
/// the input and of the rhs: planes share nothing, so no barrier is cube-wide.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaneFragmentConvRoutine;

impl PlaneFragmentConvRoutine {
    /// A plane holds 2x2 fragments, a cube two planes along the channels.
    pub const GRID_2X2: ConvSettings = ConvSettings::new((2, 2), (1, 2), ConvLines::Widest);

    /// A plane holds one tile of rows by two of channels, which wastes less where the map is
    /// narrow.
    pub const GRID_1X2: ConvSettings = ConvSettings::new((1, 2), (1, 2), ConvLines::Widest);
}

impl Routine for PlaneFragmentConvRoutine {
    type Problem = ConvProblem;
    type Microkernel = PlaneFragmentMicrokernel;
    type Settings = ConvSettings;
    type Blueprint = ConvBlueprint<PlaneFragmentMicrokernel>;
    type Partitioning = ConvPartitioning;

    /// The 2x2 grid and the 1x2, through cmma and then through Mma.
    fn settings(&self, _hw: &Hardware) -> Vec<ConvSettings> {
        vec![
            Self::GRID_2X2,
            Self::GRID_1X2,
            Self::GRID_2X2.mma(),
            Self::GRID_1X2.mma(),
        ]
    }

    /// Nothing of its microkernel's: the stage fill's own reads are the partitioning's.
    fn vectorize(
        &self,
        _microkernel: &PlaneFragmentMicrokernel,
        partitioning: &ConvPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        partitioning.vectorize(vectorization);
    }

    /// The planes' staging, which the partitioning places, and the plane microkernel's work.
    fn estimate(
        &self,
        problem: &ConvProblem,
        blueprint: &Self::Blueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        estimate.staging = Some(blueprint.partitioning.staging(&problem.elements));
        blueprint
            .microkernel
            .estimate(&problem.work_item(), &(), grid.planes(), estimate);
    }

    /// The plane microkernel over the instruction `settings` ask for, a plane for every box of the
    /// output, a cube the planes of the settings' layout.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &ConvProblem,
        settings: &ConvSettings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let microkernel = settings.plane(hw, problem)?;
        let tiles = ConvTiles::new(problem, &microkernel, settings)?;
        let partitioning = ConvPartitioning::plane(problem, &tiles);
        let mut vectorization = Vectorization::new();
        self.vectorize(&microkernel, &partitioning, &mut vectorization);
        ConvBlueprint::prepared(
            hw,
            problem,
            (microkernel, &microkernel),
            partitioning,
            vectorization,
        )
    }
}

impl ConvRoutine<PlaneFragmentMicrokernel> for PlaneFragmentConvRoutine {
    fn dispatch(
        &self,
        problem: &ConvProblem,
        binding: ConvBinding,
        blueprint: ConvBlueprint<PlaneFragmentMicrokernel>,
    ) {
        let ConvBinding {
            launcher,
            input,
            rhs,
            out,
        } = binding;
        // SAFETY: every operand is a tile bound through the launcher, which masks the accesses a box
        // reaches past its tensor; the kernel indexes no buffer but through those tiles.
        unsafe {
            plane_fragment_conv_kernel::launch_unchecked(
                launcher.client(),
                launcher.cube_count(),
                launcher.cube_dim(),
                input,
                rhs,
                out,
                launcher.partitioning_arg(),
                blueprint.microkernel,
                blueprint.partitioning.storage(),
                STAGE_RING,
                problem.kernel_elements(),
            );
        }
    }
}
