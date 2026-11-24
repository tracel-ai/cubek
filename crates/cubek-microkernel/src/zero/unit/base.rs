use crate::zero::ZeroOperand;
use cubecl::tile::Tile;
use cubecl::tile::launch::Vectorization;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube};
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// A window of an output zeroed by one unit alone.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDirectZeroMicrokernel {}

#[cube]
impl UnitDirectZeroMicrokernel {
    /// A unit's window `out`, every cell set to zero (`unit_direct_zero_microkernel`).
    pub fn zero(&self, out: &mut Tile) {
        unit_direct_zero_microkernel(out)
    }
}

/// A unit's zeroing has nothing a measurement settles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDirectZeroMicrokernelSettings;

impl core::fmt::Display for UnitDirectZeroMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("default")
    }
}

impl Microkernel for UnitDirectZeroMicrokernel {
    type WorkItem = ();
    type Settings = UnitDirectZeroMicrokernelSettings;
    type Operand = ZeroOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the unit zeroing runs on every device";

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![UnitDirectZeroMicrokernelSettings]
    }

    fn candidates(
        _hw: &Hardware,
        _item: &(),
        _settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        core::iter::once(Self {})
    }

    /// It reads nothing, and writes at whatever width its output is bound at.
    fn vectorize(&self, _: &(), _vectorization: &mut Vectorization<ZeroOperand>) {}

    /// It holds one line of zeros.
    fn estimate(&self, _item: &(), _: &(), _planes: CubePlanes, estimate: &mut KernelEstimate) {
        estimate.work = Some(UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: 1,
        });
    }
}

#[cube]
pub(crate) fn unit_direct_zero_microkernel(out: &mut Tile) {
    out.zero();
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::client::Client;
    use cubecl::ir::{ElemType, FloatKind, IntKind};
    use cubecl::prelude::*;
    use cubecl::tile::kind::Boundary;
    use cubecl::tile::launch::{BoundaryPolicy, Decline, Grid};
    use cubecl::tile::space::UnitCube;
    use cubecl::tile::{
        Dimension, Launcher, Levels, Partitioning, Space, TileSpec, TileTensor, TileTensorLaunch,
    };
    use cubecl::zspace::Shape;
    use cubek_std::devices;
    use cubek_std::microkernel::CubeFootprint;
    use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeChecker, TestInput, Written};

    #[test]
    fn it_runs_on_a_device_lacking_every_feature() {
        let hw = devices::lacking_every_feature(devices::m2_pro());
        let settings = UnitDirectZeroMicrokernel::settings(&hw)[0];
        assert!(UnitDirectZeroMicrokernel::best(&hw, &(), &settings).is_ok());
    }

    #[test]
    fn it_asks_nothing_of_its_outputs_width() {
        const WIDEST_COUNT: usize = 8;
        let mut vectorization = Vectorization::new();
        UnitDirectZeroMicrokernel {}.vectorize(&(), &mut vectorization);
        assert_eq!(
            vectorization.width(ZeroOperand::Out, |_| WIDEST_COUNT),
            WIDEST_COUNT
        );
    }

    #[test]
    fn it_holds_one_line() {
        let planes = CubePlanes { count: 1, width: 1 };
        let mut estimate = KernelEstimate {
            cubes: None,
            cube: CubeFootprint {
                units: 1,
                shared_memory: 0,
            },
            demand: None,
            staging: None,
            work: None,
        };
        UnitDirectZeroMicrokernel {}.estimate(&(), &(), planes, &mut estimate);
        assert_eq!(estimate.work.map(|work| work.registers), Some(1));
    }

    /// The rows of the output, a unit each.
    const ROWS: Dimension = Dimension(0);
    /// The cells of a row.
    const CELLS: Dimension = Dimension(1);

    /// `out`, `rows` rows of `cells` cells in `element`, row-major, each row zeroed by a unit of its
    /// own, a cube of one plane of `plane_width` units holding as many rows.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot bind the launch.
    fn launch_unit_zero(
        client: &Client,
        out: TensorArg,
        element: ElemType,
        rows: usize,
        cells: usize,
        plane_width: u32,
    ) -> Result<(), Decline> {
        let space = Space::new(&[(ROWS, rows), (CELLS, cells)]);
        let partitioning = Partitioning::new(
            space.clone(),
            Levels::leaf(&[(ROWS, 1), (CELLS, cells)])
                .launch_units(
                    &[ROWS],
                    UnitCube {
                        plane_width: plane_width as usize,
                        planes: 1,
                    },
                )
                .build(),
        );
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(plane_width),
        };
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        // A cube's units past the last row write nothing: their writes are masked.
        let out = TileTensorLaunch::new(
            out,
            element,
            1,
            TileSpec::direct(&[ROWS, CELLS]).boundary(BoundaryPolicy::Every(Boundary::Zero)),
        );
        // SAFETY: the output is bound through the launcher, which masks the writes of the units past
        // its last row, and the kernel indexes no buffer but through its tile.
        unsafe {
            unit_zero_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                out,
                launcher.partitioning_arg(),
                UnitDirectZeroMicrokernel {},
            );
        }
        Ok(())
    }

    /// Every unit's row, zeroed by the unit.
    ///
    /// Its operands' leaves:
    /// - `out`: one tensor of `rows` rows of `cells` cells, row-major.
    #[cube(launch_unchecked)]
    fn unit_zero_kernel(
        out: &mut TileTensor<'_>,
        partitioning: Partitioning,
        #[comptime] microkernel: UnitDirectZeroMicrokernel,
    ) {
        let o = out.tile_mut(&partitioning);
        for unit in &partitioning {
            let mut cells = o.at(&unit);
            microkernel.runtime().zero(&mut cells);
        }
    }

    /// An output of `rows` rows of `cells` cells in `element`, holding sevens before the launch.
    #[derive(Debug, Clone, Copy)]
    struct ZeroCase {
        pub row_count: usize,
        pub cell_count: usize,
        pub element: ElemType,
    }

    struct UnitZeroScopeChecker {
        hw: Hardware,
    }

    impl ScopeChecker for UnitZeroScopeChecker {
        type Case = ZeroCase;

        fn launch(&self, client: &Client, case: &ZeroCase) -> Result<Vec<Written>, Decline> {
            let shape = Shape::from(vec![case.row_count, case.cell_count]);
            let out = TestInput::builder(client.clone(), shape)
                .dtype(case.element)
                .custom(vec![7.0; case.row_count * case.cell_count])
                .generate_without_host_data();
            let plane_width = self.hw.plane_dim().unwrap_or(1);
            launch_unit_zero(
                client,
                out.clone().binding().into_tensor_arg(),
                case.element,
                case.row_count,
                case.cell_count,
                plane_width,
            )?;
            let host = match case.element {
                ElemType::Int(_) => HostDataType::I32,
                _ => HostDataType::F32,
            };
            Ok(vec![Written {
                name: "out",
                tensor: out,
                host,
            }])
        }

        fn reference(&self, case: &ZeroCase) -> Vec<HostData> {
            vec![HostData::row_major(
                HostDataVec::F32(vec![0.0; case.row_count * case.cell_count]),
                Shape::from(vec![case.row_count, case.cell_count]),
            )]
        }

        fn tolerance(&self, _case: &ZeroCase) -> f32 {
            0.0
        }
    }

    impl UnitZeroScopeChecker {
        /// A row length that is not a power of two, in `element`, on the test device.
        fn check_element(element: ElemType) {
            let hw = Hardware::from_client(&cubecl::test_device().client());
            Self { hw }.check([ZeroCase {
                row_count: 5,
                cell_count: 37,
                element,
            }]);
        }
    }

    #[test]
    fn a_unit_zeroes_a_row_of_floats() {
        UnitZeroScopeChecker::check_element(ElemType::Float(FloatKind::F32));
    }

    #[test]
    fn a_unit_zeroes_a_row_of_integers() {
        UnitZeroScopeChecker::check_element(ElemType::Int(IntKind::I32));
    }

    #[test]
    fn its_settings_display_as_the_default() {
        assert_eq!(UnitDirectZeroMicrokernelSettings.to_string(), "default");
    }
}
