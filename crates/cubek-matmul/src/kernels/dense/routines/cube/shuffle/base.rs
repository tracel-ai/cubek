use super::cube_shuffle_matmul_kernel;
use crate::kernels::MatmulBinding;
use crate::kernels::{
    MatmulBlueprint, MatmulOutput, MatmulRoutine, MatmulRoutineProblem, MatmulTensors, MatmulView,
};
use cubecl::tile::TileArgLaunch;
use cubecl::tile::launch::DestinationLaunch;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubecl::tile::space::CubeOrder;
use cubek_microkernel::matmul::MatmulOperand;
use cubek_microkernel::matmul::MatmulPartitioning;
use cubek_microkernel::matmul::dense::cube::{
    CubeMeet, CubeShuffleMicrokernel, CubeShuffleMicrokernelSettings,
};
use cubek_microkernel::matmul::dense::plane::PlaneShuffleMicrokernel;
use cubek_microkernel::matmul::stored::Packing;
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, M, N};
use cubek_std::microkernel::{CubeStaging, GlobalStream, KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, ROUTINE_LISTS_NO_SETTING, Routine};

/// `out = lhs · rhs` with every cube owning every row of a run of the output's columns, and its
/// planes splitting the contraction between them: in software sums
/// ([`PlaneShuffleMicrokernel`]), or, against a packed weight, on the tensor cores, where an NVFP4
/// weight against a 16-bit activation holds up to a fragment's sixteen rows at the cost of
/// reading the weight
/// ([`PlaneFragmentMicrokernel`](cubek_microkernel::matmul::dense::plane::PlaneFragmentMicrokernel)).
///
/// What a product whose runs of columns are too few to fill the device with planes wants: a
/// narrow `N` on a device of many compute units. Declines a float row weight, a col weight whose
/// `K` a step of its planes does not tile, and a device that commits to no plane width; against a
/// packed weight, what its microkernel declines
/// ([`CubeShuffleMicrokernel::decline`](cubek_std::microkernel::Microkernel::decline)).
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

    /// The lead against a packed weight: the software planes one read a step, four planes a cube.
    pub const PACKED: Self = Self {
        microkernel: CubeShuffleMicrokernelSettings::PACKED,
    };
}

impl core::fmt::Display for CubeShuffleMatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "cube-shuffle-matmul {}", self.microkernel)
    }
}

pub type CubeShuffleMatmulRoutineBlueprint = MatmulBlueprint<CubeShuffleMicrokernel>;

impl CubeShuffleMatmulRoutine {
    /// The settings worth racing against a packed weight
    /// ([`CubeShuffleMicrokernel::packed_settings`]): what the packed few-rows table races, and
    /// what a caller naming the routine and no setting runs against a packed weight, in this
    /// order.
    pub fn packed_settings(&self, hw: &Hardware) -> Vec<CubeShuffleMatmulRoutineSettings> {
        CubeShuffleMicrokernel::packed_settings(hw)
            .into_iter()
            .map(|microkernel| CubeShuffleMatmulRoutineSettings { microkernel })
            .collect()
    }

    /// [`Routine::estimate`] against a float weight: each group of units, across the cube's
    /// planes, reads the lhs's rows along the whole of `K` and the weight's columns it owns,
    /// straight from global memory, once; a step's planes read neighbouring slices, so a column is
    /// still read down all of `K`. Its drain, every plane adding its partial run into the
    /// accumulator atomically before the cube copies it out. Then its microkernel, for the work.
    fn float_estimate(
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

    /// [`Routine::estimate`] against `packing`, a packed weight: each group of units, across the
    /// cube's planes, reads the lhs's rows along the whole of `K` and the words and scales of the
    /// columns it owns, straight from global memory, once; a turn's planes read neighbouring scale
    /// blocks. Its drain, every plane adding its partial run into the accumulator. Then its
    /// microkernel, for the work.
    fn packed_estimate(
        &self,
        problem: &MatmulRoutineProblem,
        packing: &Packing,
        blueprint: &CubeShuffleMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let view = MatmulView::of::<CubeShuffleMicrokernel>(problem).expect(MatmulView::PREPARED);
        let (item, microkernel) = (view.work_item(), &blueprint.microkernel);
        let [rows, k] = [view.rows(), problem.k].map(|extent| extent as u64);
        let leaf = blueprint.partitioning.leaf();
        let planes = microkernel.planes_along_k.get() as u64;
        let step_k = microkernel.plane.step_depth(&leaf) as u64 * planes;
        let columns = problem.n as u64 / grid.cubes().expect("a static grid").max(1);
        let groups = columns / microkernel.plane.columns() as u64;
        let size = |operand: MatmulOperand| item.elements.get(operand).size() as u64;
        let rhs = packing.bytes(k as usize * columns as usize) as u64;
        estimate.staging = Some(CubeStaging {
            operations: 2.0 * (rows * columns * k) as f64,
            reads: vec![
                GlobalStream {
                    bytes: groups * rows * k * size(MatmulOperand::Lhs),
                    run_bytes: (k * size(MatmulOperand::Lhs)) as usize,
                },
                GlobalStream {
                    bytes: rhs,
                    run_bytes: (rhs / columns.max(1)) as usize,
                },
            ],
            writes: vec![GlobalStream {
                bytes: rows * columns * size(MatmulOperand::Out),
                run_bytes: (columns * size(MatmulOperand::Out)) as usize,
            }],
            stages: k.div_ceil(step_k),
            stages_in_flight: 1,
            drain_atomics: match microkernel.meet {
                CubeMeet::Atomic => planes * rows * columns,
                CubeMeet::InSlots => 0,
            },
        });
        microkernel.estimate(&item, &(), grid.planes(), estimate);
    }

    /// [`Routine::prepare`] against a float weight.
    fn prepare_float(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &CubeShuffleMatmulRoutineSettings,
    ) -> Result<(CubeShuffleMatmulRoutineBlueprint, Grid), Refusal> {
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
        let grid = Grid::new(hw, blueprint.partitioning.as_tile(), cube)?;
        Ok((blueprint, grid))
    }

    /// [`Routine::prepare`] against `packing`, a packed weight: every row in each cube, the cubes
    /// side by side along the axis its planes' columns land on.
    fn prepare_packed(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        packing: &Packing,
        settings: &CubeShuffleMatmulRoutineSettings,
    ) -> Result<(CubeShuffleMatmulRoutineBlueprint, Grid), Refusal> {
        let view = MatmulView::of::<CubeShuffleMicrokernel>(problem)?;
        view.check_steps(hw)?;
        let item = view.work_item();
        let microkernel = CubeShuffleMicrokernel::best(hw, &item, &settings.microkernel)?;
        let columns = microkernel.plane.column_axis();
        let partitioning = view.partitioning(
            microkernel.tiling(),
            &[columns],
            CubeOrder::RowMajor,
            MatmulView::ONE_CUBE_A_BOX,
        );
        let cube = CubeResources {
            plane_width: microkernel.plane.plane_width() as u32,
            shared_memory: microkernel.shared_memory(&item, &partitioning),
        };
        let blueprint = MatmulBlueprint::packed(packing, microkernel, partitioning);
        let grid = Grid::new(hw, blueprint.partitioning.as_tile(), cube)?;
        Ok((blueprint, grid))
    }
}

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

    /// The first of the settings for the weight `problem` states that prepares a launch on this
    /// device: [`settings`](Routine::settings) against a float weight,
    /// [`packed_settings`](CubeShuffleMatmulRoutine::packed_settings) against a packed one.
    ///
    /// # Errors
    ///
    /// The last setting's decline, where none serves.
    fn prepare_first_served(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let settings = match problem.packings.rhs {
            Some(_) => self.packed_settings(hw),
            None => self.settings(hw),
        };
        let mut declined = ROUTINE_LISTS_NO_SETTING;
        for settings in settings {
            match self.prepare(hw, problem, &settings) {
                Ok(prepared) => return Ok(prepared),
                Err(why) => declined = why,
            }
        }
        Err(declined)
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

    /// Its walk's reads and its drain, against a float weight or a packed one; then its
    /// microkernel, for the work.
    fn estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &CubeShuffleMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        match &problem.packings.rhs {
            Some(packing) => self.packed_estimate(problem, packing, blueprint, grid, estimate),
            None => self.float_estimate(problem, blueprint, grid, estimate),
        }
    }

    /// Against a float weight, the cubes over `M` and `N`; against a packed one, over its
    /// block-split space, every row in each cube.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        problem.check_unindexed()?;
        match &problem.packings.rhs {
            Some(packing) => self.prepare_packed(hw, problem, packing, settings),
            None => self.prepare_float(hw, problem, settings),
        }
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
        let item = tensors.work_item();
        let [sum, _, _, out] = item.kernel_elements();
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles; its shared memory is the accumulator the tile DSL sizes to the cube's run.
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
                microkernel.plane.reads(&item),
                [sum, out],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind, UIntKind};
    use cubek_microkernel::matmul::dense::cube::CubeShufflePlaneSettings;
    use cubek_microkernel::matmul::dense::plane::PlaneFragmentMicrokernelSettings;
    use cubek_microkernel::matmul::stored::Operand;
    use cubek_microkernel::matmul::tile_order::{self, VALUES_AXES};
    use cubek_microkernel::matmul::{MatmulLayout, MatmulOperands};
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

    /// An H100 offering the one manual mma the tensor-core plane contracts on, the 16-bit
    /// `16 × 8 × 16`: the fixture offers none of its own.
    fn tensor_cores() -> Hardware {
        let shape = MmaConfig {
            a_type: F16,
            b_type: F16,
            cd_type: ElemType::Float(FloatKind::F32),
            m: 16,
            n: 8,
            k: 16,
        };
        devices::offering_mma(devices::h100(), [shape])
    }

    /// `m` rows against an NVFP4 `k × n` weight in tile order, read as `hw` loads it.
    fn nvfp4(hw: &Hardware, m: usize, n: usize, k: usize) -> MatmulRoutineProblem {
        let scheme = tile_order::scheme();
        let words = hw.load_values(ElemType::UInt(UIntKind::U32));
        let operand = Operand::of_scheme(&scheme, K, [K, N])
            .expect("one axis a block")
            .stored(
                N,
                words,
                Some((
                    &VALUES_AXES,
                    tile_order::values_storage((128, 8), words).tiles(),
                )),
                None,
            )
            .expect("tile order is read");
        let packing = Packing { operand, scheme };
        MatmulRoutineProblem::packed(
            m,
            n,
            k,
            F16,
            MatmulOperands {
                rhs: Some(packing),
                ..MatmulOperands::all(None)
            },
        )
    }

    /// The packed settings whose planes are tensor-core planes, on `hw`.
    fn fragment_settings(hw: &Hardware) -> Vec<CubeShuffleMatmulRoutineSettings> {
        CubeShuffleMatmulRoutine
            .packed_settings(hw)
            .into_iter()
            .filter(|settings| {
                matches!(
                    settings.microkernel.plane,
                    CubeShufflePlaneSettings::Fragment(_)
                )
            })
            .collect()
    }

    /// `planes_along_k` tensor-core planes of `along_n` fragments each.
    fn fragments(along_n: usize, planes_along_k: usize) -> CubeShuffleMatmulRoutineSettings {
        CubeShuffleMatmulRoutineSettings {
            microkernel: CubeShuffleMicrokernelSettings::new(
                CubeShufflePlaneSettings::Fragment(PlaneFragmentMicrokernelSettings::windowed(
                    along_n, 8,
                )),
                planes_along_k,
            ),
        }
    }

    /// Against a packed weight the packed lead leads, and no setting is listed twice, software or
    /// tensor-core.
    #[test]
    fn the_packed_lead_leads_and_no_packed_setting_repeats() {
        let packed = CubeShuffleMatmulRoutine.packed_settings(&tensor_cores());
        assert_eq!(packed[0], CubeShuffleMatmulRoutineSettings::PACKED);
        let names: std::collections::HashSet<_> = packed.iter().map(ToString::to_string).collect();
        assert_eq!(names.len(), packed.len());
    }

    /// Every tensor-core setting prepares on a device offering the 16-bit `16 × 8 × 16`
    /// instruction, from one row to sixteen, on each of a decode's projections; and declines with
    /// its plane's reason where the device offers no such instruction, the H100 fixture among
    /// them.
    #[test]
    fn a_fragment_setting_runs_where_the_device_offers_the_instruction() {
        let shapes = [(6144, 2560), (2560, 4096), (9728, 2560), (2560, 9728)];
        let hw = tensor_cores();
        let fragment_settings = fragment_settings(&hw);
        assert!(!fragment_settings.is_empty());
        for m in [1, 3, 8, 11, 16] {
            for (n, k) in shapes {
                for settings in &fragment_settings {
                    CubeShuffleMatmulRoutine
                        .prepare(&hw, &nvfp4(&hw, m, n, k), settings)
                        .unwrap_or_else(|error| panic!("{m}x{n}x{k} at {settings}: {error:?}"));
                }
            }
        }
        for other in [devices::rdna_range(), devices::h100()] {
            for settings in &fragment_settings {
                match CubeShuffleMatmulRoutine.prepare(
                    &other,
                    &nvfp4(&other, 8, 2560, 4096),
                    settings,
                ) {
                    Err(Refusal::Declined { why }) => {
                        assert_eq!(why, CubeShuffleMicrokernel::FRAGMENT_DECLINE, "{other:?}")
                    }
                    prepared => panic!("{other:?} at {settings}: {prepared:?}"),
                }
            }
        }
    }

    /// Past a fragment's sixteen rows a tensor-core setting declines before anything is sized to
    /// it, and from nine rows on only the tensor-core planes serve a packed weight.
    #[test]
    fn the_rows_of_a_fragment_bound_a_packed_product() {
        let hw = tensor_cores();
        let problem = |m| nvfp4(&hw, m, 2560, 4096);
        let fragment = fragments(1, 8);
        assert!(
            CubeShuffleMatmulRoutine
                .prepare(&hw, &problem(17), &fragment)
                .is_err()
        );
        assert!(
            CubeShuffleMatmulRoutine
                .prepare(&hw, &problem(12), &fragment)
                .is_ok()
        );
        assert!(
            CubeShuffleMatmulRoutine
                .prepare(&hw, &problem(12), &CubeShuffleMatmulRoutineSettings::PACKED)
                .is_err()
        );
    }

    /// Two fragments a plane narrow to one where `n` is an odd number of eights, and the planes
    /// along `K` to the windows a short `K` holds.
    #[test]
    fn the_fragment_settings_narrow_to_what_the_weight_closes() {
        let hw = tensor_cores();
        let (blueprint, _) = CubeShuffleMatmulRoutine
            .prepare(&hw, &nvfp4(&hw, 8, 8 * 3, 512), &fragments(2, 16))
            .expect("a narrow weight prepares");
        assert_eq!(blueprint.microkernel.plane.columns(), 8);
        assert_eq!(blueprint.microkernel.planes_along_k.get(), 4);
    }
}
