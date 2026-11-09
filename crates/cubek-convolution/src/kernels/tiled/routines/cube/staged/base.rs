use super::kernel::cube_staged_conv_kernel;
use crate::kernels::tiled::{
    ConvBinding, ConvBlueprint, ConvLines, ConvPartitioning, ConvProblem, ConvRoutine,
    ConvSettings, ConvTiles, STAGE_RING,
};
use core::num::NonZeroUsize;
use cubecl::tile::Scratch;
use cubecl::tile::launch::{Refusal, Vectorization};
use cubek_microkernel::matmul::cube::{
    CubeStagedDelivery, CubeStagedFill, CubeStagedMicrokernel, CubeStagedStore, MatmulPlaneLayout,
};
use cubek_microkernel::matmul::plane::PlaneFragmentDrain;
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{Grid, Routine};

/// A cube owning a box of the output rows and channels, staging the box's window of the input and
/// of the rhs once for its planes, which each own a tile of it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeStagedConvRoutine;

impl CubeStagedConvRoutine {
    /// 2x2 fragments on each of 2x2 planes.
    pub const GRID_2X2: ConvSettings = ConvSettings::new((2, 2), (2, 2), ConvLines::Widest);

    /// One tile of rows, which wastes less where the map is narrow.
    pub const GRID_1X2: ConvSettings = ConvSettings::new((1, 2), (1, 2), ConvLines::Widest);

    /// An 8x32 instruction on two planes across the channels: 8 rows hold a map seven wide at 88%.
    pub const GRID_8X32: ConvSettings =
        ConvSettings::new((1, 1), (1, 2), ConvLines::Widest).eight_rows();
}

impl Routine for CubeStagedConvRoutine {
    type Problem = ConvProblem;
    type Microkernel = CubeStagedMicrokernel;
    type Settings = ConvSettings;
    type Blueprint = ConvBlueprint<CubeStagedMicrokernel>;
    type Partitioning = ConvPartitioning;

    /// Cmma over lines at 64 rows and at one tile of rows; then Mma, which serves the steps of
    /// taps and channels cmma declines: one element at a time at 64 rows, which a
    /// deep filter wants, then lines at both.
    fn settings(&self, _hw: &Hardware) -> Vec<ConvSettings> {
        vec![
            Self::GRID_2X2,
            Self::GRID_1X2,
            Self::GRID_2X2.mma().width_one(),
            Self::GRID_2X2.mma(),
            Self::GRID_1X2.mma(),
        ]
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
        estimate.staging = Some(blueprint.partitioning.staging(&problem.elements));
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
        let partitioning = ConvPartitioning::cube(problem, &tiles, hw.shared_memory_budget());
        let scratch = match plane.instruction.drain() {
            PlaneFragmentDrain::Bounce => Scratch::OneTile,
            PlaneFragmentDrain::CellByCell | PlaneFragmentDrain::InLines => Scratch::None,
        };
        let microkernel = CubeStagedMicrokernel {
            plane,
            planes: MatmulPlaneLayout::new(settings.planes.0, settings.planes.1),
            planes_along_k: NonZeroUsize::MIN,
            scratch,
            ring: STAGE_RING,
            fill: CubeStagedFill {
                delivery: CubeStagedDelivery::Copy,
                filling_planes: 0,
            },
            stage_layout: None,
            store: CubeStagedStore::Fragments,
            depth: cubek_microkernel::matmul::cube::CubeStagedDepth::Steps,
            reads: MatmulOperands::AS_BOUND,
        };
        let mut vectorization = Vectorization::new();
        self.vectorize(&microkernel, &partitioning, &mut vectorization);
        ConvBlueprint::prepared(
            hw,
            problem,
            (microkernel, &plane),
            partitioning,
            vectorization,
        )
    }
}

impl ConvRoutine<CubeStagedMicrokernel> for CubeStagedConvRoutine {
    /// Every cube staging for its planes, which read the stage.
    fn dispatch(
        &self,
        problem: &ConvProblem,
        binding: ConvBinding,
        blueprint: ConvBlueprint<CubeStagedMicrokernel>,
    ) {
        let walk = blueprint.microkernel.walk();
        let ConvBinding {
            launcher,
            input,
            rhs,
            out,
        } = binding;
        // SAFETY: every operand is a tile bound through the launcher, which masks the accesses a box
        // reaches past its tensor; the kernel indexes no buffer but through those tiles.
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
