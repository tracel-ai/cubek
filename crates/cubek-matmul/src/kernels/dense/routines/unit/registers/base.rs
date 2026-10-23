use super::kernel::unit_registers_matmul_kernel;
use crate::kernels::MatmulBinding;
use crate::kernels::{CUBE_STRIPS, cube_order_name};
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
    UnitRegistersMicrokernel, UnitRegistersMicrokernelSettings,
};
use cubek_microkernel::matmul::staged_walk::{MatmulPlaneLayout, MatmulStagingCount};
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, M, N};
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// `out = lhs · rhs` with every unit owning a block of the output. The floor of both tables: it
/// asks nothing of the hardware, so it runs everywhere.
///
/// A plane's units own neighbouring blocks along `n`, taken in turns, so they read neighbouring
/// columns of the rhs at once; the launch's planes cover the output, one batch each, and the
/// launch groups them in cubes, each cube's planes side by side over a block of the output. How
/// many units a plane has is the launch's, read by the kernel: the blueprint states only how many
/// blocks a plane's units share.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnitRegistersMatmulRoutine;

impl UnitRegistersMatmulRoutine {
    /// Steps a unit holds at once: it loads a step once the last is computed.
    pub const STAGES_IN_FLIGHT: usize = 1;

    /// Blocks along `n` a plane's units take in turns under `partitioning`, as its units level
    /// states them.
    fn blocks_per_plane(partitioning: &MatmulPartitioning) -> usize {
        partitioning
            .distributed(N)
            .expect("the unit routine distributes its blocks to a plane's units")
    }

    /// The width [`prepare`](Routine::prepare) launches a plane at, beside `planes`
    /// others in a cube.
    pub fn plane_width(hw: &Hardware, planes: usize) -> u32 {
        (hw.properties.hardware.max_units_per_cube / planes as u32)
            .min(hw.dispatch_plane_size())
            .max(1)
    }

    /// `microkernel` placed by `partitioning`, its widths settled. A register block reads its
    /// operands from global memory and drains cell by cell, so the walk holds no shared memory.
    pub fn blueprint(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        microkernel: UnitRegistersMicrokernel,
        partitioning: MatmulPartitioning,
    ) -> UnitRegistersMatmulRoutineBlueprint {
        let mut vectorization = Vectorization::new();
        self.vectorize(&microkernel, &partitioning, &mut vectorization);
        MatmulBlueprint::new(hw, problem, microkernel, partitioning, vectorization, ())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitRegistersMatmulRoutineSettings {
    pub microkernel: UnitRegistersMicrokernelSettings,
    /// How the planes one cube groups are laid out over the output.
    pub planes: MatmulPlaneLayout,
    /// Blocks along `n` one plane's units take in turns, however many units the plane has.
    pub blocks_per_plane: NonZeroUsize,
    /// The order the cubes take their blocks of planes in.
    pub cube_order: CubeOrder,
}

impl UnitRegistersMatmulRoutineSettings {
    pub const DEFAULT: Self = Self {
        microkernel: UnitRegistersMicrokernelSettings::DEFAULT,
        planes: MatmulPlaneLayout::new(4, 1),
        blocks_per_plane: NonZeroUsize::new(32).unwrap(),
        cube_order: CubeOrder::RowMajor,
    };
}

impl core::fmt::Display for UnitRegistersMatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "unit-registers-matmul {} {} blocks_per_plane={}{}",
            self.microkernel,
            self.planes,
            self.blocks_per_plane,
            cube_order_name(self.cube_order)
        )
    }
}

pub type UnitRegistersMatmulRoutineBlueprint = MatmulBlueprint<UnitRegistersMicrokernel>;

impl Routine for UnitRegistersMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = UnitRegistersMicrokernel;
    type Settings = UnitRegistersMatmulRoutineSettings;
    type Blueprint = UnitRegistersMatmulRoutineBlueprint;
    type Partitioning = MatmulPartitioning;

    /// Its microkernel's blocks, each at the routine's default, then a star off the default over
    /// its own knobs: each plane layout, a column of as many planes as a full cube holds
    /// ([`Hardware::full_cube_planes`]) among them, each count of blocks a plane shares, the
    /// strips.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let lead = UnitRegistersMatmulRoutineSettings::DEFAULT;
        let blocks = UnitRegistersMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| Self::Settings {
                microkernel,
                ..lead
            });
        let planes = [(2, 1), (8, 1), (2, 2), (hw.full_cube_planes(), 1)].map(|(rows, columns)| {
            Self::Settings {
                planes: MatmulPlaneLayout::new(rows, columns),
                ..lead
            }
        });
        let shares = [16, 64].map(|blocks| Self::Settings {
            blocks_per_plane: NonZeroUsize::new(blocks).unwrap(),
            ..lead
        });
        let cube_orders = CUBE_STRIPS.map(|cube_order| Self::Settings { cube_order, ..lead });
        blocks
            .chain(planes)
            .chain(shares)
            .chain(cube_orders)
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

    /// Its units' steps: each unit reads its own block's rows of the lhs along the whole of `K`,
    /// and, beside the other units of its plane along `n`, its columns of the rhs, straight from
    /// global memory — every unit along `n` reading the same rows and every one along `m` the same
    /// columns — and writes its block the same way. Then its microkernel, for the work.
    fn estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &UnitRegistersMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let (item, partitioning) = (
            MatmulView::of::<UnitRegistersMicrokernel>(problem)
                .expect(MatmulView::PREPARED)
                .work_item(),
            &blueprint.partitioning,
        );
        let (cube, block) = (partitioning.cube_box(), partitioning.leaf());
        let plane_row = block.extent(N) * Self::blocks_per_plane(partitioning);
        let staging = MatmulStagingCount {
            depth: block.extent(K),
            stages_in_flight: Self::STAGES_IN_FLIGHT as u32,
            runs: MatmulOperands {
                lhs: cube.extent(K),
                rhs: plane_row,
                out: plane_row,
            },
            readers: MatmulOperands {
                lhs: (cube.extent(N) / block.extent(N)) as u64,
                rhs: (cube.extent(M) / block.extent(M)) as u64,
                out: 1,
            },
            drain_atomics: 0,
        };
        staging.count(&item, partitioning, estimate);
        blueprint
            .microkernel
            .estimate(&item, &(), grid.planes(), estimate);
    }

    /// The units of a plane cooperate on nothing, so the plane is launched as wide as the device
    /// dispatches one and no wider than the cube's unit budget leaves for each: any width is
    /// correct, since they take their blocks in turns, and this one is always spawnable. The
    /// width sizes the grid, never the blueprint.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        problem.check_plain()?;
        let view = MatmulView::of::<UnitRegistersMicrokernel>(problem)?;
        let microkernel =
            UnitRegistersMicrokernel::best(hw, &view.work_item(), &settings.microkernel)?;
        let planes = settings.planes;
        let partitioning = view.plane_partitioning(
            microkernel
                .tiling()
                .units_distributed(N, settings.blocks_per_plane.get()),
            &[M, N],
            PlaneOrder::new(&planes.counts(), settings.cube_order),
        );
        let cube = CubeResources {
            plane_width: Self::plane_width(hw, planes.count()),
            shared_memory: 0,
        };
        let grid = Grid::new(hw, partitioning.as_tile(), cube)?;
        let blueprint = self.blueprint(hw, problem, microkernel, partitioning);
        Ok((blueprint, grid))
    }
}

impl MatmulRoutine<UnitRegistersMicrokernel> for UnitRegistersMatmulRoutine {
    fn dispatch<O: DestinationLaunch, Out: MatmulOutput>(
        &self,
        tensors: &MatmulTensors<Out>,
        binding: MatmulBinding<O::Operand, TileArgLaunch, TileArgLaunch>,
        blueprint: UnitRegistersMatmulRoutineBlueprint,
        grid: Grid,
    ) {
        let [sum, _, _, out] = tensors.work_item().kernel_elements();
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles; it holds no shared memory.
        unsafe {
            unit_registers_matmul_kernel::launch_unchecked::<O>(
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

    /// A register block reads a col weight along `K`, contracting each line of it with a line of the
    /// lhs into a cell one value wide, so on every device the weight binds as wide as the lhs and
    /// the output is written a value at a time.
    #[test]
    fn a_col_weight_folds_with_the_lhs_into_one_value_cells() {
        use crate::kernels::tune::key::{MatmulBatching, MatmulShape, OperandKey};
        let col_weight = MatmulRoutineProblem::from(&MatmulShape {
            m: 256,
            n: 512,
            k: 64,
            batch: 1,
            elem: ElemType::Float(FloatKind::F16),
            lhs: OperandKey::float(Some(K), MatmulBatching::Batched),
            rhs: OperandKey::float(Some(K), MatmulBatching::Shared),
        });
        for hw in fixtures() {
            let (blueprint, _) = UnitRegistersMatmulRoutine
                .prepare(
                    &hw,
                    &col_weight,
                    &UnitRegistersMatmulRoutineSettings::DEFAULT,
                )
                .expect("the floor prepares everywhere");
            assert_eq!(blueprint.widths.rhs, blueprint.widths.lhs, "{hw:?}");
            assert_eq!(blueprint.widths.out, 1, "{hw:?}");
        }
    }

    fn fixtures() -> [Hardware; 7] {
        [
            devices::strix_halo(),
            devices::wgpu_no_subgroups(),
            devices::rdna_range(),
            devices::cdna_wave64(),
            devices::m2_pro(),
            devices::h100(),
            devices::cpu_12(),
        ]
    }

    /// The floor holds: every setting it lists prepares on every device described, ragged shapes
    /// included.
    #[test]
    fn every_setting_prepares_on_every_device() {
        let f16 = ElemType::Float(FloatKind::F16);
        for hw in fixtures() {
            for settings in UnitRegistersMatmulRoutine.settings(&hw) {
                for (m, n, k) in [(1, 128, 256), (37, 29, 19), (512, 4096, 4096)] {
                    let problem = MatmulRoutineProblem::dense(m, n, k, f16, MatmulLayout::Row);
                    let prepared = UnitRegistersMatmulRoutine.prepare(&hw, &problem, &settings);
                    assert!(prepared.is_ok(), "{settings} on {hw:?}: {prepared:?}");
                }
            }
        }
    }

    /// The floor of the memory-bound product too: its default prepares every decode projection,
    /// one to eight rows of `qkv` to `down` against a weight laid either way, on every device
    /// described, those whose plane width no memory-bound routine can merge across included.
    #[test]
    fn the_default_prepares_every_decode_projection_on_every_device() {
        use crate::kernels::tune::key::{MatmulBatching, MatmulShape, OperandKey};
        let f16 = ElemType::Float(FloatKind::F16);
        let settings = UnitRegistersMatmulRoutineSettings::DEFAULT;
        for hw in fixtures() {
            for m in 1..=8 {
                for (n, k) in [(4096, 2048), (2048, 2048), (6144, 2048), (2048, 6144)] {
                    for rhs in [K, N] {
                        let problem = MatmulRoutineProblem::from(&MatmulShape {
                            m,
                            n,
                            k,
                            batch: 1,
                            elem: f16,
                            lhs: OperandKey::float(Some(K), MatmulBatching::Batched),
                            rhs: OperandKey::float(Some(rhs), MatmulBatching::Shared),
                        });
                        let prepared = UnitRegistersMatmulRoutine.prepare(&hw, &problem, &settings);
                        assert!(
                            prepared.is_ok(),
                            "{m}x{k}x{n}, rhs along {rhs:?}, on {hw:?}: {prepared:?}"
                        );
                    }
                }
            }
        }
    }

    /// The plane width sizes the grid and never the blueprint: devices dispatching planes of 32
    /// and of 64 compile the same kernel, and launch it that wide.
    #[test]
    fn the_plane_width_is_the_grids_alone() {
        let problem = MatmulRoutineProblem::dense(
            512,
            4096,
            4096,
            ElemType::Float(FloatKind::F16),
            MatmulLayout::Row,
        );
        let settings = UnitRegistersMatmulRoutineSettings::DEFAULT;
        let prepare = |hw: Hardware| {
            UnitRegistersMatmulRoutine
                .prepare(&hw, &problem, &settings)
                .expect("the floor prepares everywhere")
        };
        let (wave32, wave32_grid) = prepare(devices::strix_halo());
        let (wave64, wave64_grid) = prepare(devices::cdna_wave64());
        assert_eq!(wave32, wave64);
        assert_eq!(wave32_grid.planes().width, 32);
        assert_eq!(wave64_grid.planes().width, 64);
    }

    /// A unit reads its block's `depth` of the lhs and its `columns` of the rhs and the output, so
    /// the default's four narrows every width, until the binding's mask narrows it to one.
    #[test]
    fn the_widths_are_what_the_accesses_and_the_binding_admit() {
        let hw = devices::strix_halo();
        let f16 = ElemType::Float(FloatKind::F16);
        let widths = |m, n, k| {
            let problem = MatmulRoutineProblem::dense(m, n, k, f16, MatmulLayout::Row);
            let settings = UnitRegistersMatmulRoutineSettings::DEFAULT;
            let (blueprint, _) = UnitRegistersMatmulRoutine
                .prepare(&hw, &problem, &settings)
                .expect("the floor prepares everywhere");
            let widths = blueprint.widths;
            (widths.lhs, widths.rhs, widths.out)
        };
        assert_eq!(widths(256, 512, 64), (4, 4, 4));
        // A `k` the steps overhang masks the lhs, whose contiguous axis it is.
        assert_eq!(widths(256, 512, 22), (1, 4, 4));
        // An `n` the planes overhang masks the rhs and the output.
        assert_eq!(widths(256, 516, 64), (4, 1, 1));
    }
}
