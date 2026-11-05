use super::kernel::unit_whole_k_matmul_kernel;
use crate::kernels::{
    MatmulBinding, MatmulBlueprint, MatmulOutput, MatmulPartitioning, MatmulRoutine,
    MatmulRoutineProblem, MatmulTensors, MatmulView,
};
use core::num::NonZeroUsize;
use cubecl::tile::TileTensorLaunch;
use cubecl::tile::launch::{DestinationLaunch, Refusal, Vectorization};
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_microkernel::matmul::unit::{UnitRegistersMicrokernel, UnitRegistersMicrokernelSettings};
use cubek_microkernel::matmul::{
    MatmulLayout, MatmulOperand, MatmulOperands, MatmulRowsKind, MatmulStagingCount, MatmulWorkItem,
};
use cubek_std::Hardware;
use cubek_std::layout::BindingOrder;
use cubek_std::matmul_axes::{K, M, N};
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// `out = lhs · rhs` with every unit owning every row of a run of the output's columns, walking all
/// of `K` alone: the unit registers microkernel, its block holding every row. The routine that does
/// not stage on a device that commits to no plane width: a plane's units share nothing, so any
/// width is correct.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnitWholeKMatmulRoutine;

impl UnitWholeKMatmulRoutine {
    /// Steps of `K` a unit holds at once: it loads a step once the last is summed.
    const STAGES_IN_FLIGHT: u32 = 1;

    /// Blocks along `n` a plane's units take in turns under `partitioning`, as its units level
    /// states them.
    fn blocks_per_plane(partitioning: &MatmulPartitioning) -> usize {
        partitioning
            .distributed(N)
            .expect("the unit whole-K routine distributes its blocks to a plane's units")
    }
}

/// One unit's step, `columns × depth`, counted in reads of the rhs rather than elements, so one
/// setting loads the same bytes a step at every element size and on every device. The routine
/// sizes its microkernel's block in elements from it, as the rhs lies
/// ([`microkernel`](Self::microkernel)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitWholeKBlock {
    /// The output's columns one unit reads a step.
    pub columns: NonZeroUsize,
    /// The steps of `K` one unit reads a step, their loads issued together.
    pub depth: NonZeroUsize,
}

impl UnitWholeKBlock {
    pub const DEFAULT: Self = Self {
        columns: NonZeroUsize::MIN,
        depth: NonZeroUsize::new(8).unwrap(),
    };

    /// The default's columns, `depth` steps of `K` a step.
    pub const fn deep(depth: usize) -> Self {
        Self {
            depth: NonZeroUsize::new(depth).unwrap(),
            ..Self::DEFAULT
        }
    }

    /// The microkernel's block for `item` on `hw`: each read the widest vector the device reads
    /// the rhs in, along `K` for a col rhs and along `N` for a row rhs. Its rows are unread,
    /// since the item holds every row whole.
    fn microkernel(
        &self,
        hw: &Hardware,
        item: &MatmulWorkItem,
    ) -> UnitRegistersMicrokernelSettings {
        let vector = hw.vector_ceiling(item.elements.rhs);
        let (columns, depth) = match item.layouts.rhs {
            MatmulLayout::Col => (self.columns.get(), self.depth.get() * vector),
            MatmulLayout::Row | MatmulLayout::Strided => {
                (self.columns.get() * vector, self.depth.get())
            }
        };
        UnitRegistersMicrokernelSettings {
            rows: NonZeroUsize::MIN,
            columns: NonZeroUsize::new(columns).unwrap(),
            depth: NonZeroUsize::new(depth).unwrap(),
        }
    }
}

impl core::fmt::Display for UnitWholeKBlock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "block={}x{}", self.columns, self.depth)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitWholeKMatmulRoutineSettings {
    pub block: UnitWholeKBlock,
    /// Planes the launch groups in one cube, side by side along `n`.
    pub planes: NonZeroUsize,
    /// Blocks along `n` one plane's units take in turns, however many units the plane has.
    pub blocks_per_plane: NonZeroUsize,
}

impl UnitWholeKMatmulRoutineSettings {
    /// The setting that reads a col rhs, the layout an rhs is most often stored in, fastest.
    pub const DEFAULT: Self = Self {
        block: UnitWholeKBlock::DEFAULT,
        planes: NonZeroUsize::new(4).unwrap(),
        blocks_per_plane: NonZeroUsize::new(8).unwrap(),
    };
}

impl core::fmt::Display for UnitWholeKMatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "unit-whole-k-matmul {} planes={} blocks_per_plane={}",
            self.block, self.planes, self.blocks_per_plane
        )
    }
}

pub type UnitWholeKMatmulRoutineBlueprint = MatmulBlueprint<UnitRegistersMicrokernel>;

impl Routine for UnitWholeKMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = UnitRegistersMicrokernel;
    type Settings = UnitWholeKMatmulRoutineSettings;
    type Blueprint = UnitWholeKMatmulRoutineBlueprint;
    type Partitioning = MatmulPartitioning;

    /// Empty where the device commits to a plane width, since the shuffles serve it. Otherwise a
    /// star off the default, which leads (a shallower and a deeper block: a deeper step issues
    /// more loads together and holds more registers):
    /// [`launch_first_served_on`](MatmulRoutine::launch_first_served_on) runs the first setting
    /// that serves.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        if hw.plane_size().is_ok() {
            return Vec::new();
        }
        let lead = Self::Settings::DEFAULT;
        let count = |n: usize| NonZeroUsize::new(n).unwrap();
        let blocks = [UnitWholeKBlock::deep(4), UnitWholeKBlock::deep(16)]
            .map(|block| Self::Settings { block, ..lead });
        let planes = [1, 8].map(|planes| Self::Settings {
            planes: count(planes),
            ..lead
        });
        let shares = [16, 32].map(|blocks| Self::Settings {
            blocks_per_plane: count(blocks),
            ..lead
        });
        [lead]
            .into_iter()
            .chain(blocks)
            .chain(planes)
            .chain(shares)
            .collect()
    }

    /// Nothing of its own: it walks the launch's planes and their units, and its microkernel makes
    /// every access.
    fn vectorize(
        &self,
        microkernel: &UnitRegistersMicrokernel,
        _: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    /// Each unit reads every row of the lhs along all of `K`, and its own columns of the rhs
    /// once, from global memory. Then its microkernel, for the work.
    fn estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &UnitWholeKMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let (item, partitioning) = (
            MatmulView::new(problem, Self::ROWS_KIND)
                .expect(MatmulView::PREPARED)
                .work_item(),
            &blueprint.partitioning,
        );
        let (cube, block) = (partitioning.cube_space().space, partitioning.leaf());
        let plane_row = block.extent(N) * Self::blocks_per_plane(partitioning);
        let rhs_run = match blueprint.microkernel.orders.rhs {
            BindingOrder::InBufferOrder => cube.extent(K),
            BindingOrder::Stated => plane_row,
        };
        let staging = MatmulStagingCount {
            depth: block.extent(K),
            stages_in_flight: Self::STAGES_IN_FLIGHT,
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
        staging.count(&item, &cube, estimate);
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
        let view = MatmulView::new(problem, Self::ROWS_KIND)?;
        let item = view.work_item();
        let microkernel =
            UnitRegistersMicrokernel::best(hw, &item, &settings.block.microkernel(hw, &item))?;
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
        let cube = CubeResources {
            plane_width,
            shared_memory: 0,
        };
        let grid = Grid::new(hw, blueprint.partitioning.as_tile(), cube)?;
        Ok((blueprint, grid))
    }
}

impl MatmulRoutine<UnitRegistersMicrokernel> for UnitWholeKMatmulRoutine {
    /// A unit holds every row, compiled in, against each step of the rhs.
    const ROWS_KIND: MatmulRowsKind = MatmulRowsKind::Whole;

    fn dispatch<O: DestinationLaunch, Out: MatmulOutput>(
        &self,
        tensors: &MatmulTensors<Out>,
        binding: MatmulBinding<O::Operand, TileTensorLaunch, TileTensorLaunch>,
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
    use cubek_microkernel::matmul::plane::PlaneShuffleMicrokernel;
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// The devices that commit to no plane width, which are the only ones it lists settings on.
    fn uncommitted() -> [Hardware; 2] {
        [devices::rdna_range(), devices::wgpu_no_subgroups()]
    }

    /// A decode projection of `m` rows, `k` deep and `n` wide, its rhs laid as `rhs_layout`.
    fn decode(m: usize, n: usize, k: usize, rhs_layout: MatmulLayout) -> MatmulRoutineProblem {
        MatmulRoutineProblem::dense(m, n, k, F16).laid(MatmulOperand::Rhs, rhs_layout)
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
    /// `down`, against an rhs laid either way, at its default.
    #[test]
    fn the_default_prepares_every_few_row_projection_where_the_shuffles_decline() {
        let settings = UnitWholeKMatmulRoutineSettings::DEFAULT;
        for hw in uncommitted() {
            for m in 1..=PlaneShuffleMicrokernel::MAX_ROWS {
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

    /// Every setting it lists serves a decode projection, against an rhs laid either way.
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

    /// It holds from one row to its bound and declines past it, so the selector's float few-rows
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
        assert!(rows(PlaneShuffleMicrokernel::MAX_ROWS).is_ok());
        assert_eq!(
            rows(PlaneShuffleMicrokernel::MAX_ROWS + 1).err(),
            Some(UnitRegistersMicrokernel::WHOLE_ROWS_OUT_OF_RANGE)
        );
    }

    /// A col rhs is read in vectors along `K`, contracted with the lhs's, never one element at a
    /// time: the read the unit floor makes of the same rhs.
    #[test]
    fn a_col_rhs_is_read_in_vectors_along_k() {
        for hw in uncommitted() {
            let (blueprint, _) = UnitWholeKMatmulRoutine
                .prepare(
                    &hw,
                    &decode(1, 2048, 2048, MatmulLayout::Col),
                    &UnitWholeKMatmulRoutineSettings::DEFAULT,
                )
                .expect("the default serves a col rhs");
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
                .expect("the default serves a row rhs")
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
