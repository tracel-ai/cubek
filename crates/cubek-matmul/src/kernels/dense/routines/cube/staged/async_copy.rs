use super::base::{self, CubeStagedMatmulRoutineBlueprint, CubeStagedPlacement};
use super::block_scaled::{self, StagedShape};
use crate::kernels::CubeStagedMatmulRoutine;
use crate::kernels::{
    MatmulBinding, MatmulOutput, MatmulPartitioning, MatmulRoutine, MatmulRoutineProblem,
    MatmulTensors, cube_order_name,
};
use cubecl::tile::launch::{DestinationLaunch, Refusal, Vectorization};
use cubecl::tile::space::CubeOrder;
use cubecl::tile::{StageStorage, TileTensorLaunch};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_microkernel::matmul::MatmulRowsKind;
use cubek_microkernel::matmul::cube::{
    CubeStagedDelivery, CubeStagedFill, CubeStagedMicrokernel, CubeStagedMicrokernelSettings,
    CubeStagedStore, MatmulPlaneLayout,
};
use cubek_microkernel::matmul::plane::PlaneFragmentMicrokernelSettings;
use cubek_std::Hardware;
use cubek_std::microkernel::KernelEstimate;
use cubek_std::microkernel::StageRing;
use cubek_std::routine::{Grid, Routine};

/// `out = lhs · rhs` as [`CubeStagedMatmulRoutine`] runs it, each stage's lines handed by the
/// cube's units to the copy engine (`cp.async`) rather than loaded and stored, landing on the
/// slot's barrier after the fill returns ([`CubeStagedDelivery::AsyncCopy`]). Its planes store
/// their own fragments.
///
/// Refused where the device has no copy engine, where the ring holds its next stage in registers,
/// which the engine never passes through, and where a line it would copy is not 4, 8 or 16 bytes,
/// as the delivery itself answers ([`Delivery::runs_on`](cubecl::tile::launch::Delivery::runs_on),
/// [`holds`](cubecl::tile::launch::Delivery::holds),
/// [`moves_lines`](cubecl::tile::launch::Delivery::moves_lines)).
/// A packed input its fill decodes is loaded by the units; the engine copies the others.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeStagedAsyncCopyMatmulRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedAsyncCopyMatmulRoutineSettings {
    pub microkernel: CubeStagedMicrokernelSettings,
    pub cube_order: CubeOrder,
}

impl CubeStagedAsyncCopyMatmulRoutine {
    /// How it fills its stages: the units hand every line to the copy engine, and every plane
    /// contracts.
    const FILL: CubeStagedFill = CubeStagedFill {
        delivery: CubeStagedDelivery::AsyncCopy,
        filling_planes: 0,
    };

    /// Four by two steps on each of two by four planes, two stages in flight, the float input
    /// handed to the copy engine while the units decode the packed one into the other stage. On
    /// an RTX PRO 6000 at Qwen3-8B's projections it leads every other box against an NVFP4 weight
    /// on the `o` and `down` projections, by 11 to 19%, both at 512 rows and at 2048, and runs
    /// within 5% of the fastest on most of the others.
    const PACKED_FLOAT_ON_ENGINE: CubeStagedMicrokernelSettings =
        CubeStagedMicrokernelSettings::in_steps((4, 2), (2, 4), StageRing::DOUBLE);

    /// The settings a product against a packed input races on `hw`: the box handing its float
    /// input to the engine (`PACKED_FLOAT_ON_ENGINE`), then the boxes staged as stored
    /// (`CubeStagedMatmulRoutine::STORED`), which by the units never were the fastest. None
    /// without a copy engine. Its cubes in the grid's order.
    pub fn packed_settings(&self, hw: &Hardware) -> Vec<CubeStagedAsyncCopyMatmulRoutineSettings> {
        if !hw.unit_async_copy() {
            return Vec::new();
        }
        [Self::PACKED_FLOAT_ON_ENGINE]
            .into_iter()
            .chain(CubeStagedMatmulRoutine::STORED)
            .map(|microkernel| CubeStagedAsyncCopyMatmulRoutineSettings {
                microkernel,
                cube_order: CubeOrder::RowMajor,
            })
            .collect()
    }

    /// The settings a block-scaled product read as stored races on `hw`
    /// (`block_scaled`): for a prompt's rows, four by four steps on two by
    /// four planes, three stages in flight, and two by four on four by two planes, two; for a
    /// decode's, as deep as `K` divides. None without a copy engine.
    pub fn block_scaled_settings(
        &self,
        hw: &Hardware,
    ) -> Vec<CubeStagedAsyncCopyMatmulRoutineSettings> {
        if !hw.unit_async_copy() {
            return Vec::new();
        }
        let shape = |steps, planes, planes_along_k, ring| StagedShape {
            steps,
            planes,
            planes_along_k,
            ring,
        };
        let (double, triple) = (StageRing::DOUBLE, StageRing::TRIPLE);
        let shapes = [
            shape((4, 4, 1), (2, 4), 1, triple),
            shape((2, 4, 1), (4, 2), 1, double),
            shape((1, 1, 4), (1, 4), 1, double),
            shape((1, 2, 4), (1, 4), 1, double),
            shape((1, 1, 4), (1, 2), 4, double),
            shape((1, 1, 1), (1, 1), 4, double),
            shape((1, 2, 4), (1, 2), 2, double),
        ];
        block_scaled::block_scaled_settings(hw, *self, &shapes, |microkernel| {
            CubeStagedAsyncCopyMatmulRoutineSettings {
                microkernel,
                cube_order: CubeOrder::RowMajor,
            }
        })
    }
}

impl core::fmt::Display for CubeStagedAsyncCopyMatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cube-staged-async-copy-matmul {}{}",
            self.microkernel,
            cube_order_name(self.cube_order)
        )
    }
}

impl Routine for CubeStagedAsyncCopyMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = CubeStagedMicrokernel;
    type Settings = CubeStagedAsyncCopyMatmulRoutineSettings;
    type Blueprint = CubeStagedMatmulRoutineBlueprint;
    type Partitioning = MatmulPartitioning;

    /// The units routine's lead and named boxes, and the rings that fill in their slots, their
    /// stages handed to the engine: whether it beats the units' loads moves with the box, by a few
    /// percent either way on an L4. A ring that prefetches through registers is not listed, which
    /// the engine refuses. None without a copy engine.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        if !hw.unit_async_copy() {
            return Vec::new();
        }
        let boxed = |(along_m, along_n, depth), (planes_m, planes_n)| {
            CubeStagedMicrokernelSettings::boxed(
                PlaneFragmentMicrokernelSettings::grid(along_m, along_n, depth),
                MatmulPlaneLayout::new(planes_m, planes_n),
            )
        };
        let lead = CubeStagedMicrokernelSettings::DEFAULT;
        let deep = boxed((4, 4, 2), (2, 4));
        let small_on_2x2 = boxed((2, 4, 2), (2, 2));
        [
            lead,
            deep,
            boxed((4, 4, 2), (2, 2)),
            boxed((4, 4, 2), (4, 2)),
            boxed((2, 4, 2), (2, 1)),
            small_on_2x2,
            lead.ringed(StageRing::DOUBLE),
        ]
        .map(|microkernel| Self::Settings {
            microkernel,
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

    /// As [`CubeStagedMatmulRoutine`]'s, the copy engine checked against the ring, then against
    /// the widths the blueprint binds.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let placement = CubeStagedPlacement::new(
            hw,
            problem,
            &settings.microkernel,
            settings.cube_order,
            Self::FILL,
            CubeStagedStore::Fragments,
        )?;
        placement.check_copy_engine(hw)?;
        let contracted = placement.contracted;
        let (blueprint, grid) = placement.prepare(hw)?;
        base::check_copy_engine_lines(&blueprint, &contracted)?;
        Ok((blueprint, grid))
    }
}

impl MatmulRoutine<CubeStagedMicrokernel, StageStorage> for CubeStagedAsyncCopyMatmulRoutine {
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
