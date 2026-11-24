use super::base::{self, CubeStagedMatmulRoutineBlueprint, CubeStagedPlacement};
use crate::kernels::{
    CUBE_STRIPS, MatmulBinding, MatmulOutput, MatmulPartitioning, MatmulRoutine,
    MatmulRoutineProblem, MatmulTensors, cube_order_name,
};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::launch::{Decline, DestinationLaunch, Vectorization};
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

    /// The lead against a packed input ([`CubeStagedMicrokernelSettings::PACKED`]), its cubes in
    /// the grid's order.
    pub const PACKED: Self = Self {
        microkernel: CubeStagedMicrokernelSettings::PACKED,
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

    /// Boxes against a packed input two steps of `M` a plane, one plane high: a group of a few dozen
    /// rows, as a prompt's routes leave each of many experts, fills one such box where a taller one
    /// computes mostly the padding of its group.
    const PACKED_GROUP_BOXES: [((usize, usize), (usize, usize)); 3] =
        [((2, 2), (1, 2)), ((2, 2), (1, 4)), ((2, 4), (1, 2))];

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

    /// Boxes of a product of 8-bit floats, activations stored in one
    /// ([`RowScaledTensor`](crate::kernels::tensor::RowScaledTensor)) against a packed weight the
    /// fill decodes into their element, whatever its format, a second stage in flight: four by
    /// four fragments on four by two planes, one step deep and two. Measured with MXFP4: on an
    /// R9700 the first was the fastest box of every grid raced on Qwen3.8 27B's projections at 512
    /// and 2048 rows (FP8-row against MXFP4 with row-factored scales, 156 to 187 TFLOP/s), but at
    /// its output projection at 512 rows, which the second took. They join the race of every
    /// packed product on such a device.
    const EIGHT_BIT_FLOAT_BOXES: [CubeStagedMicrokernelSettings; 2] = [
        CubeStagedMicrokernelSettings::boxed(
            PlaneFragmentMicrokernelSettings::grid(4, 4, 1),
            MatmulPlaneLayout::new(4, 2),
        )
        .ringed(StageRing::DOUBLE),
        CubeStagedMicrokernelSettings::boxed(
            PlaneFragmentMicrokernelSettings::grid(4, 4, 2),
            MatmulPlaneLayout::new(4, 2),
        )
        .ringed(StageRing::DOUBLE),
    ];

    /// The steps of `K` a stage decoding a packed input holds, each a step of the inputs' finest
    /// reads, which the stage then reads as widely as it holds whole: four, which over byte scales
    /// is a word of them, NVFP4's 64 of `K` and MXFP4's 128; and two, which on an RTX PRO 6000
    /// gave MXFP4's boxes with a second stage in flight room in shared memory at 64 of `K`, its
    /// 27B down projection at 512 rows then running as fast as NVFP4's (672 against 674 µs), where
    /// one block a stage lost a fifth to the barriers between twice as many stages. Tunables:
    /// each is raced, the first a caller that measures nothing takes.
    pub const PACKED_STAGE_STEPS: [usize; 2] = [4, 2];

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
                            .split_along_k(along_k)
                            .stage_steps(Self::PACKED_STAGE_STEPS[0]),
                    ));
                }
            }
        }
        settings
    }

    /// The settings a grouped product against a packed input races beside
    /// [`packed_settings`](Self::packed_settings): the group boxes, one stage at a time. Its cubes in
    /// the grid's order.
    pub fn packed_group_settings(&self) -> Vec<CubeStagedMatmulRoutineSettings> {
        Self::PACKED_GROUP_BOXES
            .map(|(steps, planes)| {
                CubeStagedMatmulRoutineSettings::on_grid(
                    CubeStagedMicrokernelSettings::in_steps(steps, planes, StageRing::SINGLE)
                        .stage_steps(Self::PACKED_STAGE_STEPS[0]),
                )
            })
            .to_vec()
    }

    /// The settings a product against a packed input races on `hw`: the lead, the boxes several
    /// steps tall the float routine races, one stage at a time, with a second in flight, and one
    /// stage with the next held in registers while it is contracted, each
    /// at every count of steps of `K` a stage holds ([`PACKED_STAGE_STEPS`](Self::PACKED_STAGE_STEPS)),
    /// then the one-step boxes ([`packed_one_step_settings`](Self::packed_one_step_settings)), then, on
    /// a device without a copy engine, the boxes staged as stored (`STORED`), then, on a device
    /// whose manual MMA multiplies 8-bit floats, the boxes of an 8-bit float product
    /// (`EIGHT_BIT_FLOAT_BOXES`). Planes splitting `K` only on a device whose shared memory takes the
    /// atomic `f32` add they merge through:
    /// elsewhere they would only be tried and declined. Its cubes in the grid's order.
    ///
    /// The instruction is the tile layer's, `cmma` for a sum written out: on an L4 the `mma`
    /// transport's boxes lead `cmma`'s on Qwen3-4B's down projection by a few percent, and lose a
    /// third once the projection's residual add is fused into its store, which its fragments
    /// drain into cell by cell.
    ///
    /// A step of `K` is planned on the inputs' finest reads, one scale block where scales are
    /// bytes ([`CubeStagedMicrokernel::planned`](cubek_microkernel::matmul::cube::CubeStagedMicrokernel::planned)).
    pub fn packed_settings(&self, hw: &Hardware) -> Vec<CubeStagedMatmulRoutineSettings> {
        let boxed = [
            StageRing::SINGLE,
            StageRing::DOUBLE,
            StageRing::SINGLE_THROUGH_REGISTERS,
        ]
        .into_iter()
        .flat_map(|ring| {
            Self::PACKED_BOXES
                .map(|(steps, planes)| CubeStagedMicrokernelSettings::in_steps(steps, planes, ring))
        })
        .flat_map(|settings| Self::PACKED_STAGE_STEPS.map(|steps| settings.stage_steps(steps)));
        let one_step = self
            .packed_one_step_settings()
            .into_iter()
            .map(|settings| settings.microkernel);
        let stored = Self::STORED.into_iter().filter(|_| !hw.unit_async_copy());
        let f32_sum = ElemType::Float(FloatKind::F32);
        let multiplies_8_bit_floats = [FloatKind::E4M3, FloatKind::E5M2]
            .map(ElemType::Float)
            .into_iter()
            .any(|element| hw.manual_mma(element, element, f32_sum));
        let eight_bit = Self::EIGHT_BIT_FLOAT_BOXES
            .into_iter()
            .filter(|_| multiplies_8_bit_floats);
        let mut settings = vec![CubeStagedMicrokernelSettings::PACKED];
        for candidate in boxed.chain(one_step).chain(stored).chain(eight_bit) {
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

    /// Its microkernel's settings ([`CubeStagedMicrokernel::settings`]), its cubes in the grid's
    /// order. Then its own knob around the
    /// boxes it suits: the default, the default's grid on two by two planes with its stage's rows
    /// in order, and the planes splitting `K`, in strips.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        let [along_m, _] = CUBE_STRIPS;
        let on_grid = CubeStagedMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| Self::Settings {
                microkernel,
                cube_order: CubeOrder::RowMajor,
            });
        let rows_in_order_on_2x2 = CubeStagedMicrokernelSettings {
            stage_layout: StageLayout::Blocks,
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
        microkernel: &CubeStagedMicrokernel,
        partitioning: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        base::vectorize(microkernel, partitioning, vectorization);
    }

    fn estimate(
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
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Decline> {
        // The problem its microkernel plans on, every packed input read as it reads it.
        let (read, picked) = CubeStagedPlacement::planned(
            hw,
            problem,
            &settings.microkernel,
            CubeStagedFill::BY_UNITS,
            CubeStagedStore::Fragments,
        )?;
        CubeStagedPlacement::new(
            hw,
            &read,
            &settings.microkernel,
            settings.cube_order,
            CubeStagedFill::BY_UNITS,
            CubeStagedStore::Fragments,
            picked,
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
