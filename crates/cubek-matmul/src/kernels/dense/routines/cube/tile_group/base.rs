use crate::kernels::dense::routines::cube::cube_shuffle_matmul_kernel;
use crate::kernels::{
    MATMUL_Q_RHS_NOT_PACKED, MatmulBinding, MatmulBlueprint, MatmulOutput, MatmulRoutine,
    MatmulRoutineProblem, MatmulTensors, MatmulView,
};
use cubecl::tile::TileArgLaunch;
use cubecl::tile::launch::{DestinationLaunch, Refusal, Vectorization};
use cubecl::tile::space::CubeOrder;
use cubek_microkernel::matmul::dense::cube::{
    CubeMeet, CubeTileGroupMicrokernel, CubeTileGroupMicrokernelSettings,
};
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands, MatmulPartitioning, MatmulRead};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubeStaging, GlobalStream, KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// `out = lhs · rhs` against a packed weight, with every cube owning every row of one stored tile
/// group's columns, its planes taking the group's steps along `K` in turns
/// ([`CubeTileGroupMicrokernel`]).
///
/// What a weight stored in tiles wants at a few rows: each cube reads its group front to back,
/// whole tiles a step. Declines what its microkernel does.
#[derive(Debug, Clone, Copy, Default)]
pub struct CubeTileGroupMatmulRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeTileGroupMatmulRoutineSettings {
    pub microkernel: CubeTileGroupMicrokernelSettings,
}

impl CubeTileGroupMatmulRoutineSettings {
    /// The microkernel's lead.
    pub const DEFAULT: Self = Self {
        microkernel: CubeTileGroupMicrokernelSettings::DEFAULT,
    };
}

impl core::fmt::Display for CubeTileGroupMatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "cube-tile-group-matmul {}", self.microkernel)
    }
}

pub type CubeTileGroupMatmulRoutineBlueprint = MatmulBlueprint<CubeTileGroupMicrokernel>;

impl Routine for CubeTileGroupMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = CubeTileGroupMicrokernel;
    type Settings = CubeTileGroupMatmulRoutineSettings;
    type Blueprint = CubeTileGroupMatmulRoutineBlueprint;
    type Partitioning = MatmulPartitioning;

    /// Its microkernel's settings: it has no knob of its own.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        CubeTileGroupMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| Self::Settings { microkernel })
            .collect()
    }

    /// Nothing of its own: it only walks the cubes, and its microkernel makes every access.
    fn vectorize(
        &self,
        microkernel: &CubeTileGroupMicrokernel,
        partitioning: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        microkernel.vectorize(partitioning, vectorization);
    }

    /// Each cube reads the lhs's rows along the whole of `K` once a column of its group, and its
    /// group's words and scales as one run; its drain, every plane adding its partial into the
    /// accumulator. Then its microkernel, for the work.
    fn estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &CubeTileGroupMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let view = MatmulView::of::<CubeTileGroupMicrokernel>(problem).expect(MatmulView::PREPARED);
        let packing = problem.packings.rhs.as_ref().expect(MatmulView::PREPARED);
        let (item, microkernel) = (view.work_item(), &blueprint.microkernel);
        let [rows, k] = [view.rows(), problem.k].map(|extent| extent as u64);
        let leaf = blueprint.partitioning.leaf();
        let planes = microkernel.planes_along_k.get() as u64;
        let step_k = microkernel.step_depth(&leaf) as u64 * planes;
        let cubes = grid.cubes().expect("a static grid").max(1);
        let columns = problem.n as u64 / cubes;
        let size = |operand: MatmulOperand| item.elements.get(operand).size() as u64;
        let rhs = packing.bytes(k as usize * columns as usize) as u64;
        estimate.staging = Some(CubeStaging {
            operations: 2.0 * (rows * columns * k) as f64,
            reads: vec![
                GlobalStream {
                    bytes: columns * rows * k * size(MatmulOperand::Lhs),
                    run_bytes: (k * size(MatmulOperand::Lhs)) as usize,
                },
                GlobalStream {
                    bytes: rhs,
                    run_bytes: rhs as usize,
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
        microkernel.estimate(&item, &blueprint.partitioning, grid.planes(), estimate);
    }

    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let Some(packing) = problem.packings.rhs else {
            return Err(MATMUL_Q_RHS_NOT_PACKED);
        };
        let view = MatmulView::of::<CubeTileGroupMicrokernel>(problem)?;
        view.check_steps(hw)?;
        let item = view.work_item();
        let microkernel = CubeTileGroupMicrokernel::best(hw, &item, &settings.microkernel)?;
        let columns = microkernel.plane.column_cut().0;
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
        let blueprint = MatmulBlueprint::packed(&packing, microkernel, partitioning);
        let grid = Grid::new(hw, blueprint.partitioning.as_tile(), cube)?;
        Ok((blueprint, grid))
    }
}

impl MatmulRoutine<CubeTileGroupMicrokernel> for CubeTileGroupMatmulRoutine {
    fn dispatch<O: DestinationLaunch, Out: MatmulOutput>(
        &self,
        tensors: &MatmulTensors<Out>,
        binding: MatmulBinding<O::Operand, TileArgLaunch, TileArgLaunch>,
        blueprint: CubeTileGroupMatmulRoutineBlueprint,
        grid: Grid,
    ) {
        let microkernel = &blueprint.microkernel;
        let item = MatmulView::of::<CubeTileGroupMicrokernel>(&tensors.problem())
            .expect(MatmulView::PREPARED)
            .work_item();
        let [sum, _, _, out] = item.kernel_elements();
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles; its shared memory is the accumulator the tile DSL sizes to the cube's group.
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
    use cubecl::ir::{ElemType, FloatKind, UIntKind};
    use cubek_microkernel::matmul::stored::{Operand, Packing};
    use cubek_microkernel::matmul::tile_order::{self, VALUES_AXES};
    use cubek_std::devices;
    use cubek_std::matmul_axes::{K, N};

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// Qwen3-8B's decode projections, `(n, k)`: gate and up fused, down, q k and v fused, o.
    const QWEN3_8B: [(usize, usize); 4] =
        [(24576, 4096), (4096, 12288), (6144, 4096), (4096, 4096)];

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

    /// No setting is listed twice, and the lead is the default.
    #[test]
    fn no_setting_repeats() {
        let settings = CubeTileGroupMatmulRoutine.settings(&devices::h100());
        assert_eq!(settings[0], CubeTileGroupMatmulRoutineSettings::DEFAULT);
        let names: std::collections::HashSet<_> =
            settings.iter().map(ToString::to_string).collect();
        assert_eq!(names.len(), settings.len());
    }

    /// On a device that stores NVFP4 in the block-scaled instruction's `128 × 8` step, a plane's
    /// units cover the tile's eight columns, four along `K`, so a step of the plane is two whole
    /// tiles; eight planes share a group on every Qwen3-8B decode projection, from one row to
    /// eight.
    #[test]
    fn a_plane_reads_whole_stored_tiles() {
        let hw = devices::offering_nvfp4(devices::h100());
        for m in [1, 3, 8] {
            for (n, k) in QWEN3_8B {
                let (blueprint, _) = CubeTileGroupMatmulRoutine
                    .prepare(
                        &hw,
                        &nvfp4(&hw, m, n, k),
                        &CubeTileGroupMatmulRoutineSettings::DEFAULT,
                    )
                    .unwrap_or_else(|error| panic!("{m}x{n}x{k}: {error:?}"));
                let microkernel = &blueprint.microkernel;
                assert_eq!(microkernel.plane.groups_along_n, 8, "{m}x{n}x{k}");
                assert_eq!(microkernel.plane.units_along_k, 4, "{m}x{n}x{k}");
                assert_eq!(microkernel.planes_along_k.get(), 8, "{m}x{n}x{k}");
                let leaf = blueprint.partitioning.leaf();
                assert_eq!(microkernel.step_depth(&leaf), 2 * 128, "{m}x{n}x{k}");
            }
        }
    }

    /// The planes along `K` narrow to the steps a short `K` holds: ten steps of two tiles take
    /// eight planes, and two steps take two.
    #[test]
    fn the_planes_narrow_to_the_steps_of_k() {
        let hw = devices::offering_nvfp4(devices::h100());
        let planes = |k| {
            let (blueprint, _) = CubeTileGroupMatmulRoutine
                .prepare(
                    &hw,
                    &nvfp4(&hw, 1, 256, k),
                    &CubeTileGroupMatmulRoutineSettings::DEFAULT,
                )
                .expect("a short weight prepares");
            blueprint.microkernel.planes_along_k.get()
        };
        assert_eq!(planes(10 * 256), 8);
        assert_eq!(planes(2 * 256), 2);
    }

    /// A device that commits to no plane width runs none of it.
    #[test]
    fn a_device_without_a_plane_width_declines() {
        let hw = devices::wgpu_no_subgroups();
        let refused = CubeTileGroupMatmulRoutine.prepare(
            &hw,
            &nvfp4(&hw, 1, 4096, 4096),
            &CubeTileGroupMatmulRoutineSettings::DEFAULT,
        );
        assert!(
            matches!(refused, Err(Refusal::Declined { .. })),
            "{refused:?}"
        );
    }
}
