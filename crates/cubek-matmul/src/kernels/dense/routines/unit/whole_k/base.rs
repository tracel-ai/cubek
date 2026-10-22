use super::kernel::unit_whole_k_matmul_kernel;
use crate::kernels::MatmulBinding;
use crate::kernels::{
    MatmulBlueprint, MatmulOutput, MatmulRoutine, MatmulRoutineProblem, MatmulTensors, MatmulView,
};
use core::num::NonZeroUsize;
use cubecl::tile::TileArgLaunch;
use cubecl::tile::launch::DestinationLaunch;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubek_microkernel::matmul::MatmulPartitioning;
use cubek_microkernel::matmul::dense::unit::{
    UnitWholeKMicrokernel, UnitWholeKMicrokernelSettings,
};
use cubek_microkernel::matmul::staged_walk::MatmulStagingCount;
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands, MatmulOrder};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, M, N};
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// `out = lhs · rhs` with every unit owning every row of a run of the output's columns, walking
/// all of `K` alone. The gemv of a device that commits to no plane width: a plane's units share
/// nothing, so any width is correct.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnitWholeKMatmulRoutine;

impl UnitWholeKMatmulRoutine {
    /// Blocks along `n` a plane's units take in turns under `partitioning`, as its units level
    /// states them.
    fn blocks_per_plane(partitioning: &MatmulPartitioning) -> usize {
        partitioning
            .distributed(N)
            .expect("the whole-K gemv distributes its blocks to a plane's units")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitWholeKMatmulRoutineSettings {
    pub microkernel: UnitWholeKMicrokernelSettings,
    /// Planes the launch groups in one cube, side by side along `n`.
    pub planes: NonZeroUsize,
    /// Blocks along `n` one plane's units take in turns, however many units the plane has.
    pub blocks_per_plane: NonZeroUsize,
}

impl UnitWholeKMatmulRoutineSettings {
    /// The setting that reads a col weight, the layout weights are stored in, fastest.
    pub const DEFAULT: Self = Self {
        microkernel: UnitWholeKMicrokernelSettings::DEFAULT,
        planes: NonZeroUsize::new(4).unwrap(),
        blocks_per_plane: NonZeroUsize::new(8).unwrap(),
    };
}

impl core::fmt::Display for UnitWholeKMatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "unit-whole-k-matmul {} planes={} blocks_per_plane={}",
            self.microkernel, self.planes, self.blocks_per_plane
        )
    }
}

pub type UnitWholeKMatmulRoutineBlueprint = MatmulBlueprint<UnitWholeKMicrokernel>;

impl Routine for UnitWholeKMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = UnitWholeKMicrokernel;
    type Settings = UnitWholeKMatmulRoutineSettings;
    type Blueprint = UnitWholeKMatmulRoutineBlueprint;
    type Partitioning = MatmulPartitioning;

    /// Empty where the device commits to a plane width, since the shuffles serve it. Otherwise a
    /// star off the default, which leads: [`launch_first_served_on`](MatmulRoutine::launch_first_served_on)
    /// runs the first setting that serves.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        if hw.plane_size().is_ok() {
            return Vec::new();
        }
        let lead = Self::Settings::DEFAULT;
        let count = |n: usize| NonZeroUsize::new(n).unwrap();
        let microkernels = UnitWholeKMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| Self::Settings {
                microkernel,
                ..lead
            });
        let planes = [1, 8].map(|planes| Self::Settings {
            planes: count(planes),
            ..lead
        });
        let shares = [16, 32].map(|blocks| Self::Settings {
            blocks_per_plane: count(blocks),
            ..lead
        });
        microkernels.chain(planes).chain(shares).collect()
    }

    /// Nothing of its own: it walks the launch's planes and their units, and its microkernel makes
    /// every access.
    fn vectorize(
        &self,
        microkernel: &UnitWholeKMicrokernel,
        _: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    /// Each unit reads every row of the lhs along all of `K`, and its own columns of the weight
    /// once, from global memory. Then its microkernel, for the work.
    fn estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &UnitWholeKMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let (item, partitioning) = (
            MatmulView::of::<UnitWholeKMicrokernel>(problem)
                .expect(MatmulView::PREPARED)
                .work_item(),
            &blueprint.partitioning,
        );
        let (cube, block) = (partitioning.cube_box(), partitioning.leaf());
        let plane_row = block.extent(N) * Self::blocks_per_plane(partitioning);
        let rhs_run = match blueprint.microkernel.orders.rhs {
            MatmulOrder::InBufferOrder => cube.extent(K),
            MatmulOrder::Stated => plane_row,
        };
        let staging = MatmulStagingCount {
            depth: block.extent(K),
            stages_in_flight: UnitWholeKMicrokernel::STAGES_IN_FLIGHT as u32,
            runs: MatmulOperands {
                lhs: cube.extent(K),
                rhs: rhs_run,
                out: plane_row,
            },
            readers: MatmulOperands {
                lhs: (cube.extent(N) / block.extent(N)) as u64,
                rhs: 1,
                out: 1,
            },
            drain_atomics: 0,
        };
        staging.count(&item, partitioning, estimate);
        blueprint
            .microkernel
            .estimate(&item, &(), grid.planes(), estimate);
    }

    /// As wide as the device dispatches a plane and the cube's unit budget allows, since any width
    /// is correct.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        problem.check_plain()?;
        problem.check_unindexed()?;
        let view = MatmulView::of::<UnitWholeKMicrokernel>(problem)?;
        let microkernel =
            UnitWholeKMicrokernel::best(hw, &view.work_item(), &settings.microkernel)?;
        let planes = settings.planes.get();
        let plane_width = (hw.properties.hardware.max_units_per_cube / planes as u32)
            .min(hw.dispatch_plane_size())
            .max(1);
        let partitioning = view.plane_partitioning(
            microkernel
                .tiling()
                .units_distributed(N, settings.blocks_per_plane.get()),
            &[M, N],
            PlaneOrder::new(&[(N, planes)], CubeOrder::RowMajor),
        );
        let mut vectorization = Vectorization::new();
        self.vectorize(&microkernel, &partitioning, &mut vectorization);
        let blueprint =
            MatmulBlueprint::new(hw, problem, microkernel, partitioning, vectorization, ());
        microkernel.check_vectors(&blueprint.widths)?;
        let cube = CubeResources {
            plane_width,
            shared_memory: 0,
        };
        let grid = Grid::new(hw, blueprint.partitioning.as_cubek(), cube)?;
        Ok((blueprint, grid))
    }
}

impl MatmulRoutine<UnitWholeKMicrokernel> for UnitWholeKMatmulRoutine {
    fn dispatch<O: DestinationLaunch, Out: MatmulOutput>(
        &self,
        tensors: &MatmulTensors<Out>,
        binding: MatmulBinding<O::Operand, TileArgLaunch, TileArgLaunch>,
        blueprint: UnitWholeKMatmulRoutineBlueprint,
        grid: Grid,
    ) {
        let [sum, _, _, out] = tensors.work_item().kernel_elements();
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles; it holds no shared memory.
        unsafe {
            unit_whole_k_matmul_kernel::launch_unchecked::<O>(
                tensors.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                blueprint.widths.out,
                binding.lhs,
                binding.rhs,
                O::arg(binding.out),
                binding.launcher.partitioning_arg(),
                blueprint.microkernel,
                [sum, out],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::matmul::MatmulLayout;
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// The devices that commit to no plane width, which are the only ones it lists settings on.
    fn uncommitted() -> [Hardware; 2] {
        [devices::rdna_range(), devices::wgpu_no_subgroups()]
    }

    /// A decode projection of `m` rows, `k` deep and `n` wide, its weight laid as `rhs_layout`.
    fn decode(m: usize, n: usize, k: usize, rhs_layout: MatmulLayout) -> MatmulRoutineProblem {
        MatmulRoutineProblem::dense(m, n, k, F16, rhs_layout)
    }

    /// A table on a device that commits to a plane width pays nothing for it: the shuffles serve
    /// there, so it lists no setting.
    #[test]
    fn it_lists_nothing_where_the_device_commits_to_a_width() {
        for hw in [
            devices::strix_halo(),
            devices::cdna_wave64(),
            devices::m2_pro(),
            devices::h100(),
            devices::cpu_12(),
        ] {
            assert!(UnitWholeKMatmulRoutine.settings(&hw).is_empty(), "{hw:?}");
        }
    }

    /// The default leads, and no setting repeats another.
    #[test]
    fn the_default_leads_and_no_setting_repeats() {
        for hw in uncommitted() {
            let settings = UnitWholeKMatmulRoutine.settings(&hw);
            assert_eq!(settings[0], UnitWholeKMatmulRoutineSettings::DEFAULT);
            for (index, setting) in settings.iter().enumerate() {
                assert!(!settings[index + 1..].contains(setting), "{setting}");
            }
        }
    }

    /// Where the shuffles decline, it serves every decode projection: one to eight rows of `qkv` to
    /// `down`, against a weight laid either way, at its default.
    #[test]
    fn the_default_prepares_every_few_row_projection_where_the_shuffles_decline() {
        let settings = UnitWholeKMatmulRoutineSettings::DEFAULT;
        for hw in uncommitted() {
            for m in 1..=UnitWholeKMicrokernel::MAX_ROWS {
                for (n, k) in [(4096, 2048), (2048, 2048), (6144, 2048), (2048, 6144)] {
                    for rhs in [MatmulLayout::Col, MatmulLayout::Row] {
                        let prepared =
                            UnitWholeKMatmulRoutine.prepare(&hw, &decode(m, n, k, rhs), &settings);
                        assert!(
                            prepared.is_ok(),
                            "{m}x{k}x{n}, rhs {rhs}, on {hw:?}: {prepared:?}"
                        );
                    }
                }
            }
        }
    }

    /// Every setting it lists serves a decode projection, against a weight laid either way.
    #[test]
    fn every_setting_serves_both_layouts() {
        for hw in uncommitted() {
            for settings in UnitWholeKMatmulRoutine.settings(&hw) {
                for rhs in [MatmulLayout::Row, MatmulLayout::Col] {
                    let prepared = UnitWholeKMatmulRoutine.prepare(
                        &hw,
                        &decode(4, 4096, 2048, rhs),
                        &settings,
                    );
                    assert!(
                        prepared.is_ok(),
                        "{settings}, rhs {rhs}, on {hw:?}: {prepared:?}"
                    );
                }
            }
        }
    }

    /// It holds from one row to its bound and declines past it, so the selector's float gemv
    /// bound is its own as much as the plane shuffle's.
    #[test]
    fn it_holds_its_rows_and_declines_past_them() {
        let hw = devices::rdna_range();
        let settings = UnitWholeKMatmulRoutineSettings::DEFAULT;
        let rows = |m| {
            UnitWholeKMatmulRoutine.prepare(
                &hw,
                &decode(m, 2048, 2048, MatmulLayout::Row),
                &settings,
            )
        };
        assert!(rows(UnitWholeKMicrokernel::MAX_ROWS).is_ok());
        assert_eq!(
            rows(UnitWholeKMicrokernel::MAX_ROWS + 1).err(),
            Some(UnitWholeKMicrokernel::ROWS_OUT_OF_RANGE)
        );
    }

    /// A col weight is read in vectors along `K`, folded with the lhs's, never one element at a
    /// time: the read the unit floor makes of the same weight.
    #[test]
    fn a_col_weight_is_read_in_vectors_along_k() {
        for hw in uncommitted() {
            let (blueprint, _) = UnitWholeKMatmulRoutine
                .prepare(
                    &hw,
                    &decode(1, 2048, 2048, MatmulLayout::Col),
                    &UnitWholeKMatmulRoutineSettings::DEFAULT,
                )
                .expect("the default serves a col weight");
            assert!(blueprint.widths.rhs > 1, "{hw:?}");
            assert_eq!(blueprint.widths.rhs, blueprint.widths.lhs, "{hw:?}");
        }
    }

    /// The plane width sizes the grid and never the blueprint: a device dispatching planes of 32
    /// or of 64 compiles the same kernel, and launches it that wide.
    #[test]
    fn the_plane_width_is_the_grids_alone() {
        let problem = decode(4, 4096, 2048, MatmulLayout::Row);
        let settings = UnitWholeKMatmulRoutineSettings::DEFAULT;
        let prepare = |hw: Hardware| {
            UnitWholeKMatmulRoutine
                .prepare(&hw, &problem, &settings)
                .expect("the default serves a row weight")
        };
        let (wave32, wave32_grid) = prepare(devices::strix_halo());
        let (wave64, wave64_grid) = prepare({
            let mut hw = devices::strix_halo();
            devices::topology(&mut hw).plane_size_min = 64;
            devices::topology(&mut hw).plane_size_max = 64;
            hw
        });
        assert_eq!(wave32, wave64);
        assert_eq!(wave32_grid.planes().width, 32);
        assert_eq!(wave64_grid.planes().width, 64);
    }
}
