use super::base::{self, CubeStagedMatmulRoutineBlueprint, CubeStagedPlacement};
use super::block_scaled::{self, QUAD, StagedShape};
use crate::kernels::{
    MatmulBinding, MatmulOutput, MatmulPartitioning, MatmulRoutine, MatmulRoutineProblem,
    MatmulTensors, cube_order_name,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::{DestinationLaunch, Refusal, Vectorization};
use cubecl::tile::space::CubeOrder;
use cubecl::tile::{StageStorage, TileTensorLaunch};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_microkernel::matmul::MatmulRowsKind;
use cubek_microkernel::matmul::cube::{
    CubeStagedDelivery, CubeStagedFill, CubeStagedMicrokernel, CubeStagedMicrokernelSettings,
    CubeStagedStore, MatmulPlaneLayout,
};
use cubek_microkernel::matmul::plane::PlaneFragmentRowsAlong;
use cubek_microkernel::matmul::plane::{PlaneFragmentMicrokernelSettings, PlaneFragmentTransport};
use cubek_std::Hardware;
use cubek_std::microkernel::KernelEstimate;
use cubek_std::microkernel::StageRing;
use cubek_std::routine::{Grid, Routine};

/// `out = lhs · rhs` with the TMA engine landing each stage, as
/// [`CubeStagedTmaMatmulRoutine`](crate::kernels::CubeStagedTmaMatmulRoutine) does, issued by
/// planes set aside to do nothing else: those walk the slots filling each a lap ahead, and the
/// rest walk them contracting, then each stores its own fragments. The two meet only on each
/// slot's barriers.
///
/// Refused where the TMA engine cannot land the stages, as the TMA routine is, and where planes
/// cannot be set aside: one stage in flight, a fragment that can bounce through a scratch, or
/// planes splitting `K`
/// ([`CubeStagedFillingRefusal`](cubek_microkernel::matmul::cube::CubeStagedFillingRefusal)).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeStagedTmaFillerMatmulRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedTmaFillerMatmulRoutineSettings {
    pub microkernel: CubeStagedMicrokernelSettings,
    /// Planes set aside to fill the stages, beside the ones that contract.
    pub filling_planes: NonZeroUsize,
    pub cube_order: CubeOrder,
}

impl CubeStagedTmaFillerMatmulRoutine {
    /// Boxes against a packed input staged as it is stored, in fragments of the decoding
    /// instruction a plane (along `m`, `n` and `k`) and planes a cube (along `m` and `n`), beside
    /// a plane set aside to fill: it issues the descriptors and hands the scales to the copy
    /// engine, and the planes that contract only decode and multiply. Their instruction lays its
    /// rows along `n`: the weight decodes straight into the `A` it fills, and the activation's
    /// lane runs load into `B` as they lie, where an `A` interleaves two of its rows a register at
    /// a time. Each box is four steps of `K` deep: its stages keep the packed input's scales beside
    /// the codes, read four to a word along `KB`, and a stage shallower than a word of them would
    /// cut one.
    ///
    /// Measured on Qwen3-4B's projections at 512 rows and Qwen3-8B's at 2048 on an RTX PRO 6000
    /// (sm_120): the fewest boxes and rings that come within 3% of the TMA engine's fastest on
    /// every projection. A TMA fill without a filling plane never was the fastest, nor its
    /// instruction's rows along `m`.
    const STORED: [CubeStagedMicrokernelSettings; 3] = [
        CubeStagedMicrokernelSettings::stored((4, 2, 4), (2, 2), StageRing::TRIPLE)
            .rows_along(PlaneFragmentRowsAlong::N),
        CubeStagedMicrokernelSettings::stored((8, 2, 4), (2, 4), StageRing::TRIPLE)
            .rows_along(PlaneFragmentRowsAlong::N),
        // Eight planes along `n`: no two planes decode the same rhs fragment.
        CubeStagedMicrokernelSettings::stored((8, 1, 4), (1, 8), StageRing::TRIPLE)
            .rows_along(PlaneFragmentRowsAlong::N),
    ];

    /// The settings a product against a packed input races on `hw`: the boxes staged as stored
    /// (`STORED`), one plane filling. None without a bulk tensor copy.
    pub fn packed_settings(&self, hw: &Hardware) -> Vec<CubeStagedTmaFillerMatmulRoutineSettings> {
        if !hw.bulk_tensor_copy() {
            return Vec::new();
        }
        Self::STORED
            .into_iter()
            .map(|microkernel| CubeStagedTmaFillerMatmulRoutineSettings {
                microkernel,
                filling_planes: NonZeroUsize::MIN,
                cube_order: CubeOrder::RowMajor,
            })
            .collect()
    }
}

impl CubeStagedTmaFillerMatmulRoutine {
    /// The settings a block-scaled product read as stored races on `hw`, one plane set aside to
    /// issue the descriptors and hand the scales to the copy engine: the TMA routine's prompt
    /// boxes. None without a bulk tensor copy.
    pub fn block_scaled_settings(
        &self,
        hw: &Hardware,
    ) -> Vec<CubeStagedTmaFillerMatmulRoutineSettings> {
        if !hw.bulk_tensor_copy() {
            return Vec::new();
        }
        let shape = |steps, planes, ring| StagedShape {
            steps,
            planes,
            planes_along_k: 1,
            ring,
        };
        let (double, triple) = (StageRing::DOUBLE, StageRing::TRIPLE);
        let shapes = [
            shape((4, 4, 1), (2, 4), triple),
            shape((4, 4, 1), (2, 4), double),
            shape((4, 4, 1), (2, 4), QUAD),
            shape((4, 8, 1), (2, 4), double),
            shape((4, 8, 1), (2, 4), triple),
            shape((4, 8, 1), (4, 2), double),
            shape((4, 4, 1), (2, 2), triple),
            shape((1, 1, 1), (1, 4), QUAD),
            shape((1, 2, 1), (1, 4), triple),
        ];
        let mut listed = Vec::new();
        for cube_order in [CubeOrder::RowMajor, CubeOrder::SwizzleRow(8)] {
            listed.extend(block_scaled::block_scaled_settings(
                hw,
                *self,
                &shapes,
                |microkernel| CubeStagedTmaFillerMatmulRoutineSettings {
                    microkernel,
                    filling_planes: NonZeroUsize::MIN,
                    cube_order,
                },
            ));
        }
        listed
    }
}

impl core::fmt::Display for CubeStagedTmaFillerMatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cube-staged-tma-filler-matmul {} filling_planes={}{}",
            self.microkernel,
            self.filling_planes,
            cube_order_name(self.cube_order)
        )
    }
}

impl Routine for CubeStagedTmaFillerMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = CubeStagedMicrokernel;
    type Settings = CubeStagedTmaFillerMatmulRoutineSettings;
    type Blueprint = CubeStagedMatmulRoutineBlueprint;
    type Partitioning = MatmulPartitioning;

    /// As [`CubeStagedTmaMatmulRoutine`](crate::kernels::CubeStagedTmaMatmulRoutine)'s boxes, from
    /// the same race: the `64 × 128` box beside a filling plane, which puts twice the cubes on a
    /// narrow projection a prompt's chunk of rows would leave cores idle under; and the `128 ×
    /// 128` box on two by four planes, each four by four fragments four steps deep, which the long
    /// projections of Qwen3-8B want: its output projection takes 73.7 µs against the deep box's
    /// 86.0, its down projection 184.3 against 227.3. One plane filling. None without a bulk
    /// tensor copy.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        if !hw.bulk_tensor_copy() {
            return Vec::new();
        }
        let boxed = |(along_m, along_n, depth), (planes_m, planes_n)| {
            CubeStagedMicrokernelSettings::boxed(
                PlaneFragmentMicrokernelSettings::grid(along_m, along_n, depth),
                MatmulPlaneLayout::new(planes_m, planes_n),
            )
        };
        [
            ((2, 8, 2), (2, 2), StageRing::DOUBLE),
            ((4, 4, 4), (2, 4), StageRing::TRIPLE),
        ]
        .map(|(grid, planes, ring)| Self::Settings {
            microkernel: boxed(grid, planes)
                .ringed(ring)
                .through(PlaneFragmentTransport::Mma),
            filling_planes: NonZeroUsize::MIN,
            cube_order: CubeOrder::RowMajor,
        })
        .into()
    }

    fn vectorize(
        &self,
        microkernel: &CubeStagedMicrokernel,
        partitioning: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        base::vectorize(microkernel, partitioning, vectorization);
    }

    fn estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &CubeStagedMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        base::estimate(problem, blueprint, grid, estimate);
    }

    /// As [`CubeStagedTmaMatmulRoutine`](crate::kernels::CubeStagedTmaMatmulRoutine)'s, its planes
    /// set aside where resolving admits them, launched beside the rest.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let fill = CubeStagedFill {
            delivery: CubeStagedDelivery::Tma,
            filling_planes: settings.filling_planes.get(),
        };
        let placement = CubeStagedPlacement::new(
            hw,
            problem,
            &settings.microkernel,
            settings.cube_order,
            fill,
            CubeStagedStore::Fragments,
        )?;
        placement.check_tma_fill(hw)?;
        placement.prepare(hw)
    }
}

impl MatmulRoutine<CubeStagedMicrokernel, StageStorage> for CubeStagedTmaFillerMatmulRoutine {
    /// A cube owns a box of the rows.
    const ROWS_KIND: MatmulRowsKind = MatmulRowsKind::Tiled;

    fn dispatch<O: DestinationLaunch, Out: MatmulOutput>(
        &self,
        tensors: &MatmulTensors<Out>,
        binding: MatmulBinding<O::Operand, TileTensorLaunch, TileTensorLaunch>,
        blueprint: CubeStagedMatmulRoutineBlueprint,
        grid: Grid,
    ) {
        base::dispatch::<O, Out>(tensors, binding, blueprint, grid);
    }
}
