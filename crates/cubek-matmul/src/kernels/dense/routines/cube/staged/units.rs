use super::base::{self, CubeStagedMatmulRoutineBlueprint, CubeStagedPlacement};
use crate::kernels::{
    CUBE_STRIPS, MatmulBinding, MatmulOutput, MatmulPartitioning, MatmulRoutine,
    MatmulRoutineProblem, MatmulTensors, cube_order_name,
};
use cubecl::tile::launch::{DestinationLaunch, Refusal, Vectorization};
use cubecl::tile::space::CubeOrder;
use cubecl::tile::{StageStorage, TileTensorLaunch};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_microkernel::matmul::MatmulRowsKind;
use cubek_microkernel::matmul::cube::{
    CubeStagedFill, CubeStagedMicrokernel, CubeStagedMicrokernelSettings, CubeStagedStore,
    MatmulPlaneLayout,
};
use cubek_microkernel::matmul::plane::PlaneFragmentMicrokernelSettings;
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel, StageLayout, StageRing};
use cubek_std::routine::{Grid, Routine};

/// `out = lhs · rhs` with every cube owning a box of the output: its units fill each stage of the
/// box, and its planes each own a tile of it, or split its contraction between them
/// ([`CubeStagedMicrokernel`]); each plane stores its own fragments.
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

    /// `microkernel`'s settings, its cubes in the grid's order.
    pub const fn on_grid(microkernel: CubeStagedMicrokernelSettings) -> Self {
        Self {
            microkernel,
            cube_order: CubeOrder::RowMajor,
        }
    }
}

impl CubeStagedMatmulRoutine {
    /// Boxes against a packed input several steps of `M` a plane, `(steps, planes)` as
    /// [`in_steps`](CubeStagedMicrokernelSettings::in_steps) takes them. A cube decodes every
    /// block of the rhs it stages once, for all its rows, so the boxes more planes high give that
    /// decode more rows to serve, and the box four steps wide along `N` serves each row of the lhs
    /// it stages to more columns.
    const PACKED_BOXES: [((usize, usize), (usize, usize)); 9] = [
        ((4, 2), (2, 4)),
        ((4, 4), (2, 2)),
        ((8, 1), (1, 2)),
        ((4, 2), (1, 2)),
        ((4, 2), (2, 2)),
        ((4, 2), (4, 1)),
        ((8, 2), (4, 1)),
        ((8, 1), (2, 2)),
        ((8, 1), (4, 2)),
    ];

    /// Boxes against a packed input one step of `M` and `N` a plane, which holds every row of a
    /// product of a handful of rows, on planes side by side along `N`. A box no taller than the
    /// rows leaves no instruction idle, and a narrow one puts many cubes on the rhs, which so
    /// few rows stream once.
    const PACKED_ONE_STEP_BOXES: [((usize, usize), (usize, usize)); 3] =
        [((1, 1), (1, 1)), ((1, 1), (1, 2)), ((1, 1), (1, 4))];

    /// Planes sharing a one-step box along `K`. One plane walking all of `K` stalls on each
    /// stage's fill, and a projection with few columns has too few cubes to hide it; past eight,
    /// a stage of `K` that many planes deep outgrows shared memory.
    const PACKED_ONE_STEP_PLANES_ALONG_K: [usize; 2] = [4, 8];

    /// Boxes against a packed input staged as it is stored, which the copy engine's routine races
    /// on a device with a copy engine
    /// ([`CubeStagedAsyncCopyMatmulRoutine`](crate::kernels::CubeStagedAsyncCopyMatmulRoutine)),
    /// and the units elsewhere.
    pub(crate) const STORED: [CubeStagedMicrokernelSettings; 3] = [
        CubeStagedMicrokernelSettings::stored((4, 4, 4), (1, 4), StageRing::TRIPLE),
        CubeStagedMicrokernelSettings::stored((2, 4, 4), (2, 2), StageRing::TRIPLE),
        CubeStagedMicrokernelSettings::stored((4, 2, 4), (1, 8), StageRing::DOUBLE),
    ];

    /// The settings a product of a handful of rows against a packed input races: one step of `M`
    /// and `N` a plane, on one, two or four planes along `N`, split four or eight ways along `K`,
    /// one stage at a time or three in flight. Measured on Qwen3-4B's projections at one to sixteen
    /// rows on an RTX PRO 6000: these were the fastest of every box, split and ring tried. The
    /// first, one plane split four ways one stage at a time, is the lead a caller that measures
    /// nothing takes. Its cubes in the grid's order.
    pub fn packed_one_step_settings(&self) -> Vec<CubeStagedMatmulRoutineSettings> {
        let mut settings = Vec::new();
        for ring in [StageRing::SINGLE, StageRing::TRIPLE] {
            for (steps, planes) in Self::PACKED_ONE_STEP_BOXES {
                for along_k in Self::PACKED_ONE_STEP_PLANES_ALONG_K {
                    settings.push(CubeStagedMatmulRoutineSettings::on_grid(
                        CubeStagedMicrokernelSettings::in_steps(steps, planes, ring)
                            .split_along_k(along_k),
                    ));
                }
            }
        }
        settings
    }

    /// The settings a product against a packed input races on `hw`: the lead, the boxes several
    /// steps tall the float routine races, one stage at a time and with a second in flight, then
    /// the one-step boxes ([`packed_one_step_settings`](Self::packed_one_step_settings)), then, on
    /// a device without a copy engine, the boxes staged as stored (`STORED`). Planes splitting
    /// `K` only on a device whose shared memory takes the atomic `f32` add they merge through:
    /// elsewhere they would only be tried and declined. Its cubes in the grid's order.
    ///
    /// None through the `mma` transport: on an L4 its boxes lead `cmma`'s on Qwen3-4B's down
    /// projection by a few percent, and lose a third once the projection's residual add is fused
    /// into its store, which its fragments drain into cell by cell; the race sees only the
    /// product.
    pub fn packed_settings(&self, hw: &Hardware) -> Vec<CubeStagedMatmulRoutineSettings> {
        let boxed = [StageRing::SINGLE, StageRing::DOUBLE]
            .into_iter()
            .flat_map(|ring| {
                Self::PACKED_BOXES.map(|(steps, planes)| {
                    CubeStagedMicrokernelSettings::in_steps(steps, planes, ring)
                })
            });
        let one_step = self
            .packed_one_step_settings()
            .into_iter()
            .map(|settings| settings.microkernel);
        let stored = Self::STORED.into_iter().filter(|_| !hw.unit_async_copy());
        let mut settings = vec![CubeStagedMicrokernelSettings::PACKED];
        for candidate in boxed.chain(one_step).chain(stored) {
            let merges = candidate.planes_along_k.get() > 1;
            if (!merges || hw.shared_f32_atomic_add()) && !settings.contains(&candidate) {
                settings.push(candidate);
            }
        }
        settings
            .into_iter()
            .map(CubeStagedMatmulRoutineSettings::on_grid)
            .collect()
    }

    /// The cube order a box's cubes walk in: in strips along `m` for the tile DSL's multi-row box
    /// ([`MULTI_ROW`](CubeStagedMicrokernelSettings::MULTI_ROW)), in the grid's order otherwise.
    pub(crate) fn cube_order_of(microkernel: &CubeStagedMicrokernelSettings) -> CubeOrder {
        let [along_m, _] = CUBE_STRIPS;
        match microkernel.same_box(&CubeStagedMicrokernelSettings::MULTI_ROW) {
            true => along_m,
            false => CubeOrder::RowMajor,
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

impl Routine for CubeStagedMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = CubeStagedMicrokernel;
    type Settings = CubeStagedMatmulRoutineSettings;
    type Blueprint = CubeStagedMatmulRoutineBlueprint;
    type Partitioning = MatmulPartitioning;

    /// Its microkernel's settings ([`CubeStagedMicrokernel::settings`]), its cubes in the order
    /// their box walks (`cube_order_of`). Then its own knob around the
    /// boxes it suits: the default, the default's grid on two by two planes with its stage's rows
    /// in order, and the planes splitting `K`, in strips.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let [along_m, _] = CUBE_STRIPS;
        let on_grid = CubeStagedMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| Self::Settings {
                microkernel,
                cube_order: Self::cube_order_of(&microkernel),
            });
        let rows_in_order_on_2x2 = CubeStagedMicrokernelSettings {
            stage_layout: Some(StageLayout::Blocks),
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

    /// A plane is as many units as its instruction runs on. One cube a box of the output, over the
    /// problem's space or, against packed inputs, its block-split space, its stages filled by its
    /// units and its fragments stored by its planes (`CubeStagedPlacement`).
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        CubeStagedPlacement::new(
            hw,
            problem,
            &settings.microkernel,
            settings.cube_order,
            CubeStagedFill::BY_UNITS,
            CubeStagedStore::Fragments,
        )?
        .prepare(hw)
    }
}

impl MatmulRoutine<CubeStagedMicrokernel, StageStorage> for CubeStagedMatmulRoutine {
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
