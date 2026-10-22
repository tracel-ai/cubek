use super::cube_shuffle_matmul_kernel;
use crate::kernels::MatmulBinding;
use crate::kernels::{
    MatmulBlueprint, MatmulOutput, MatmulRoutine, MatmulRoutineProblem, MatmulTensors, MatmulView,
};
use crate::microkernel::MatmulPartitioning;
use crate::microkernel::MatmulRead;
use crate::microkernel::dense::cube::{CubeShuffleMicrokernel, CubeShuffleMicrokernelSettings};
use crate::microkernel::dense::plane::PlaneShuffleMicrokernel;
use crate::microkernel::{MatmulOperand, MatmulOperands};
use cubecl::tile::TileArgLaunch;
use cubecl::tile::launch::DestinationLaunch;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubecl::tile::space::CubeOrder;
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, M, N};
use cubek_std::microkernel::{CubeStaging, GlobalStream, KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// `out = lhs · rhs` with every cube owning every row of a run of the output's columns, and its
/// planes splitting the contraction between them.
///
/// What a product whose runs of columns are too few to fill the device with planes wants: a
/// narrow `N` on a device of many compute units. Declines a row weight, a col weight whose `K` a
/// step of its planes does not tile, and a device that commits to no plane width.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeShuffleMatmulRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeShuffleMatmulRoutineSettings {
    pub microkernel: CubeShuffleMicrokernelSettings,
}

impl CubeShuffleMatmulRoutineSettings {
    pub const DEFAULT: Self = Self {
        microkernel: CubeShuffleMicrokernelSettings::DEFAULT,
    };
}

impl core::fmt::Display for CubeShuffleMatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "cube-shuffle-matmul {}", self.microkernel)
    }
}

pub type CubeShuffleMatmulRoutineBlueprint = MatmulBlueprint<CubeShuffleMicrokernel>;

impl Routine for CubeShuffleMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = CubeShuffleMicrokernel;
    type Settings = CubeShuffleMatmulRoutineSettings;
    type Blueprint = CubeShuffleMatmulRoutineBlueprint;
    type Partitioning = MatmulPartitioning;

    /// Its microkernel's settings ([`CubeShuffleMicrokernel::settings`]): it has no knob of its own.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        CubeShuffleMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| Self::Settings { microkernel })
            .collect()
    }

    /// Nothing of its own: it only walks the cubes, and its microkernel makes every access.
    fn vectorize(
        &self,
        microkernel: &CubeShuffleMicrokernel,
        _: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    /// Its walk: each group of units, across the cube's planes, reads the lhs's rows along the
    /// whole of `K` and the weight's columns it owns, straight from global memory, once; a step's
    /// planes read neighbouring slices, so a column is still read down all of `K`. Its drain,
    /// every plane adding its partial run into the accumulator atomically before the cube copies
    /// it out. Then its microkernel, for the work.
    fn estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &CubeShuffleMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let view = MatmulView::of::<CubeShuffleMicrokernel>(problem).expect(MatmulView::PREPARED);
        let (item, microkernel) = (view.work_item(), &blueprint.microkernel);
        let partitioning = &blueprint.partitioning;
        let (cube, leaf) = (&partitioning.cube_box(), partitioning.leaf());
        let [rows, columns, k] = [M, N, K].map(|axis| cube.extent(axis) as u64);
        let groups = columns / leaf.extent(N) as u64;
        let planes_along_k = microkernel.planes_along_k.get() as u64;
        let k_per_stage = microkernel.plane.step_depth(&leaf) as u64 * planes_along_k;
        let stream = |operand: MatmulOperand, elements: u64, run: u64| {
            let size = item.elements.get(operand).size() as u64;
            GlobalStream {
                bytes: elements * size,
                run_bytes: (run * size) as usize,
            }
        };
        estimate.staging = Some(CubeStaging {
            operations: 2.0 * (rows * columns * k) as f64,
            reads: vec![
                stream(MatmulOperand::Lhs, groups * rows * k, k),
                stream(MatmulOperand::Rhs, k * columns, k),
            ],
            // The copy out of the accumulator reads its atomics one element at a time, and the
            // output binds elementwise to match, so each write is one element.
            writes: vec![stream(MatmulOperand::Out, rows * columns, 1)],
            stages: k.div_ceil(k_per_stage),
            stages_in_flight: PlaneShuffleMicrokernel::STAGES_IN_FLIGHT as u32,
            drain_atomics: planes_along_k * rows * columns,
        });
        microkernel.estimate(&item, &(), grid.planes(), estimate);
    }

    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        problem.check_plain()?;
        let view = MatmulView::of::<CubeShuffleMicrokernel>(problem)?;
        let item = view.work_item();
        let microkernel = CubeShuffleMicrokernel::best(hw, &item, &settings.microkernel)?;
        let partitioning = view.partitioning(
            microkernel.tiling(),
            &[M, N],
            CubeOrder::RowMajor,
            MatmulView::ONE_CUBE_A_BOX,
        );
        let mut vectorization = Vectorization::new();
        self.vectorize(&microkernel, &partitioning, &mut vectorization);
        let blueprint =
            MatmulBlueprint::new(hw, problem, microkernel, partitioning, vectorization, ());
        microkernel.plane.check_vectors(&blueprint.widths)?;
        let cube = CubeResources {
            plane_width: microkernel.plane.plane_width() as u32,
            shared_memory: microkernel.shared_memory(&item, &blueprint.partitioning),
        };
        let grid = Grid::new(hw, blueprint.partitioning.as_cubek(), cube)?;
        Ok((blueprint, grid))
    }
}

impl MatmulRoutine<CubeShuffleMicrokernel> for CubeShuffleMatmulRoutine {
    fn dispatch<O: DestinationLaunch, Out: MatmulOutput>(
        &self,
        tensors: &MatmulTensors<Out>,
        binding: MatmulBinding<O::Operand, TileArgLaunch, TileArgLaunch>,
        blueprint: CubeShuffleMatmulRoutineBlueprint,
        grid: Grid,
    ) {
        let microkernel = blueprint.microkernel;
        let [sum, _, _, out] = tensors.work_item().kernel_elements();
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles; its shared memory is the accumulator cubek sizes to the cube's run.
        unsafe {
            cube_shuffle_matmul_kernel::launch_unchecked::<O>(
                tensors.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                blueprint.widths.out,
                binding.lhs,
                binding.rhs,
                O::arg(binding.out),
                binding.launcher.partitioning_arg(),
                microkernel.plane.tile_instruction(),
                microkernel.meet,
                microkernel.split(),
                microkernel.planes_along_k.get(),
                MatmulOperands::all(MatmulRead::AsBound),
                [sum, out],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::microkernel::MatmulLayout;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// The `K` of qwen3-1.7b's decode projections: 2048 for `qkv`, `o`, `gate` and `up`, 6144
    /// for `down`.
    const PROJECTION_KS: [usize; 2] = [2048, 6144];

    /// `m` rows against a `k × 2048` weight laid as `rhs_layout`.
    fn projection(m: usize, k: usize, rhs_layout: MatmulLayout) -> MatmulRoutineProblem {
        MatmulRoutineProblem::dense(m, 2048, k, F16, rhs_layout)
    }

    /// The default leads the list and no setting is listed twice.
    #[test]
    fn the_default_leads_and_no_setting_repeats() {
        let settings = CubeShuffleMatmulRoutine.settings(&devices::strix_halo());
        assert_eq!(settings[0], CubeShuffleMatmulRoutineSettings::DEFAULT);
        let names: std::collections::HashSet<_> =
            settings.iter().map(ToString::to_string).collect();
        assert_eq!(names.len(), settings.len());
    }

    /// Every setting prepares every decode projection's col weight where the device commits to a
    /// plane width, on a wave32 plane and on a wave64 one, its planes meeting atomically or in
    /// turns, and declines with its reason everywhere else.
    #[test]
    fn a_setting_runs_where_the_device_commits_to_a_plane_width() {
        for (hw, k) in [
            devices::strix_halo(),
            devices::cdna_wave64(),
            devices::m2_pro(),
            devices::h100(),
            devices::cpu_12(),
            devices::wgpu_no_subgroups(),
            devices::rdna_range(),
        ]
        .into_iter()
        .flat_map(|hw| PROJECTION_KS.map(|k| (hw.clone(), k)))
        {
            let problem = projection(1, k, MatmulLayout::Col);
            let adds = hw.plane_size().is_ok();
            for settings in CubeShuffleMatmulRoutine.settings(&hw) {
                match (
                    CubeShuffleMatmulRoutine.prepare(&hw, &problem, &settings),
                    adds,
                ) {
                    (Ok(_), true) => {}
                    (Err(Refusal::Declined { why: details }), false) => {
                        assert_eq!(details, CubeShuffleMicrokernel::DECLINE, "{settings}")
                    }
                    (prepared, _) => panic!("{settings} at k={k} on {hw:?}: {prepared:?}"),
                }
            }
        }
    }

    /// The cube's planes split `K`: a cube launches all of them, holds the accumulator over its
    /// run, and each walk step covers every plane's slice.
    #[test]
    fn the_planes_split_the_contraction_and_add_in_shared_memory() {
        let hw = devices::strix_halo();
        let (blueprint, grid) = CubeShuffleMatmulRoutine
            .prepare(
                &hw,
                &projection(8, 6144, MatmulLayout::Col),
                &CubeShuffleMatmulRoutineSettings::DEFAULT,
            )
            .expect("the default prepares on a committed wave32 with shared atomics");
        let (microkernel, partitioning) = (blueprint.microkernel, &blueprint.partitioning);
        let planes_along_k = microkernel.planes_along_k.get();
        assert_eq!(grid.planes().count, planes_along_k);
        assert_eq!(grid.planes().width, 32);
        let cube = partitioning.cube_box();
        assert_eq!(cube.extent(K), 6144);
        assert_eq!(
            grid.shared_memory,
            8 * cube.extent(N) * core::mem::size_of::<f32>()
        );
    }

    /// A col weight is read down `K` beside the lhs in vectors, and the output copied out of the
    /// accumulator an element at a time; a row weight is declined with its reason.
    #[test]
    fn a_col_weight_is_read_in_vectors_and_a_row_weight_declined() {
        let hw = devices::strix_halo();
        let settings = CubeShuffleMatmulRoutineSettings::DEFAULT;
        let (blueprint, _) = CubeShuffleMatmulRoutine
            .prepare(&hw, &projection(1, 2048, MatmulLayout::Col), &settings)
            .expect("the default prepares on a committed wave32 with shared atomics");
        let widths = blueprint.widths;
        assert_eq!((widths.lhs, widths.rhs, widths.out), (8, 8, 1));
        let row = projection(1, 2048, MatmulLayout::Row);
        assert!(matches!(
            CubeShuffleMatmulRoutine.prepare(&hw, &row, &settings),
            Err(error) if error == CubeShuffleMicrokernel::ROW_RHS_UNSUPPORTED
        ));
    }

    /// A product of no rows, or of more than a work item holds, is declined before anything is
    /// sized to it.
    #[test]
    fn the_rows_a_work_item_holds_bound_the_product() {
        let settings = CubeShuffleMatmulRoutineSettings::DEFAULT;
        for m in [0, PlaneShuffleMicrokernel::MAX_ROWS + 1] {
            assert!(matches!(
                CubeShuffleMatmulRoutine.prepare(
                    &devices::strix_halo(),
                    &projection(m, 2048, MatmulLayout::Col),
                    &settings
                ),
                Err(error) if error == PlaneShuffleMicrokernel::ROWS_OUT_OF_RANGE
            ));
        }
    }
}
