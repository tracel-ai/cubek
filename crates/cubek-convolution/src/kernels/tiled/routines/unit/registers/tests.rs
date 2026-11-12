//! Pure tests for the unit registers routine: the decisions a correctness run cannot see.

use super::*;
use crate::components::ConvolutionOperation;
use crate::eval::benchmarks::{Conv2dElement, Conv2dProblem, problems};
use crate::kernels::tiled::routines::test_support::{dense, nvidia};
use crate::kernels::tiled::{ConvProblem, CubeStagedConvRoutine, TileConvBlueprint};
use cubek_microkernel::matmul::unit::UnitRegistersMicrokernel;
use cubek_std::devices;
use cubek_std::routine::Routine;
use cubek_std::{Hardware, UnitDispatch};

const ROUTINE: UnitRegistersConvRoutine = UnitRegistersConvRoutine;

/// A `f32` layer of one image over a 9 wide map: `channels_in` channels to 16 under a 3 wide
/// filter.
fn layer(operation: ConvolutionOperation, channels_in: usize) -> ConvProblem {
    ConvProblem::from(&Conv2dProblem {
        operation,
        input_shape: [1, channels_in, 9, 9],
        weight_shape: [16, channels_in, 3, 3],
        element: Conv2dElement::F32,
        ..dense()
    })
}

/// A CPU runtime offers no plane instruction, so the default tile strategy is the unit routine's.
#[test]
fn only_a_device_with_no_plane_instruction_leaves_the_default_to_the_unit_routine() {
    assert!(CubeStagedConvRoutine.settings(&cpu(12, 12)).is_empty());
    assert!(!CubeStagedConvRoutine.settings(&nvidia()).is_empty());
}

/// An input vector not dividing the rows is narrowed to one that does, never declined.
#[test]
fn an_input_vector_not_dividing_the_rows_is_narrowed_not_declined() {
    let hw = cpu(12, 12);
    let problem = layer(ConvolutionOperation::BackwardWeight, 10);
    for settings in ROUTINE.settings(&hw) {
        let (blueprint, _) = ROUTINE
            .prepare(&hw, &problem, &settings)
            .expect("the setting prepares");
        assert!(
            UnitRegistersConvRoutine::check_lhs_vector_axis(&problem, blueprint.widths.lhs).is_ok()
        );
    }
}

/// A CPU runtime of `threads` hardware threads over `cores` physical cores.
fn cpu(threads: u32, cores: u32) -> Hardware {
    let mut hw = devices::cpu_12();
    let topology = devices::topology(&mut hw);
    topology.vector_register_count = Some(16);
    topology.num_cpu_cores = Some(threads);
    topology.num_cpu_physical_cores = Some(cores);
    topology.max_units_per_cube = threads;
    topology.max_cube_dim = (threads, threads, threads);
    hw
}

/// The catalog problem named `name`, and the layer it states.
fn catalog_entry(name: &str) -> (ConvProblem, Conv2dProblem) {
    problems()
        .into_iter()
        .find_map(|entry| {
            (entry.id == name).then(|| (ConvProblem::from(&entry.value), entry.value))
        })
        .expect("the catalog lists the problem")
}

/// Units sweeping every cube take a core each, and the tiles go to them in turns.
#[test]
fn units_sweeping_every_cube_take_a_core_each() {
    let (problem, _) = catalog_entry("cpu_3x3_c64_56");
    for (cores, sweep) in [
        (
            32,
            UnitSweep {
                most: 25,
                least: 24,
            },
        ),
        (
            16,
            UnitSweep {
                most: 49,
                least: 49,
            },
        ),
    ] {
        let hw = cpu(cores, cores);
        assert_eq!(
            hw.unit_dispatch(),
            UnitDispatch::Sweeping {
                cores: cores as usize
            }
        );
        let units = units_of(&first_served(&hw, &problem), &problem);
        assert_eq!(units, cores as usize);
        assert_eq!(UnitSweep::of(units, 784), sweep, "{cores} cores");
    }
}

/// A core's second hardware thread adds no unit.
#[test]
fn units_number_the_physical_cores_not_the_threads() {
    let (problem, _) = catalog_entry("cpu_3x3_c64_56");
    let hw = cpu(32, 16);
    assert_eq!(hw.unit_dispatch(), UnitDispatch::Sweeping { cores: 16 });
    assert_eq!(units_of(&first_served(&hw, &problem), &problem), 16);
}

/// No unit is launched idle: fewer tiles than cores take a unit each.
#[test]
fn fewer_tiles_than_cores_take_a_unit_each() {
    let (problem, _) = catalog_entry("cpu_narrow_c64_7");
    assert_eq!(
        units_of(&first_served(&cpu(32, 16), &problem), &problem),
        14
    );
}

/// A GPU spreads its cubes rather than sweeping them, so it keeps the split covering the tiles in
/// the fewest cubes.
#[test]
fn a_gpu_keeps_the_fewest_cubes() {
    let hw = nvidia();
    assert_eq!(hw.unit_dispatch(), UnitDispatch::Spread);
    let (problem, _) = catalog_entry("cpu_3x3_c64_56");
    assert_eq!(
        planes_of(&first_served(&hw, &problem), &problem),
        AcrossCounts {
            rows: 14,
            across: 2
        }
    );
}

/// Tiles a unit sweeps over the whole launch: the most any unit does, and the least.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UnitSweep {
    most: usize,
    least: usize,
}

impl UnitSweep {
    fn of(units: usize, tiles: usize) -> Self {
        Self {
            most: tiles.div_ceil(units),
            least: tiles / units,
        }
    }
}

fn first_served(
    hw: &Hardware,
    problem: &ConvProblem,
) -> TileConvBlueprint<UnitRegistersMicrokernel> {
    ROUTINE
        .settings(hw)
        .iter()
        .find_map(|settings| ROUTINE.prepare(hw, problem, settings).ok())
        .map(|(blueprint, _)| blueprint)
        .expect("a setting serves the problem")
}

/// The planes a cube of the prepared launch lays along the rows and the across axis.
fn planes_of(
    blueprint: &TileConvBlueprint<UnitRegistersMicrokernel>,
    problem: &ConvProblem,
) -> AcrossCounts {
    let roles = problem.roles();
    let order = blueprint
        .partitioning
        .tile_partitioning()
        .levels()
        .iter()
        .find_map(|level| level.plane_order())
        .expect("the unit routine launches planes");
    AcrossCounts {
        rows: order.block_along(roles.axes().m),
        across: order.block_along(roles.free.across),
    }
}

/// The units a cube of the prepared launch holds, which take the launch's tiles in turns: one-unit
/// planes, each holding one tile of a box of one.
fn units_of(
    blueprint: &TileConvBlueprint<UnitRegistersMicrokernel>,
    problem: &ConvProblem,
) -> usize {
    let partitioning = blueprint.partitioning.tile_partitioning();
    let order = partitioning
        .levels()
        .iter()
        .find_map(|level| level.plane_order())
        .expect("the unit routine launches planes");
    for axis in problem.roles().plane_axes() {
        assert_eq!(order.block_along(axis), 1, "{axis:?}");
    }
    order.planes_per_cube()
}
