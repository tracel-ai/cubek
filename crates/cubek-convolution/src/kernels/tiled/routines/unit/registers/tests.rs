//! Pure tests for the unit registers routine: the decisions a correctness run cannot see.

use super::block::{BlockDepth, BlockForm};
use super::*;
use crate::components::ConvolutionOperation;
use crate::eval::benchmarks::{Conv2dElement, Conv2dProblem, problems};
use crate::kernels::tiled::routines::test_support::{dense, nvidia};
use crate::kernels::tiled::{ConvProblem, CubeStagedConvRoutine, TileConvBlueprint};
use cubecl::ir::{ElemType, FloatKind};
use cubek_microkernel::matmul::unit::UnitRegistersMicrokernel;
use cubek_std::devices;
use cubek_std::routine::Routine;
use cubek_std::{Hardware, UnitDispatch};

const ROUTINE: UnitRegistersConvRoutine = UnitRegistersConvRoutine;

/// One image of `f32` through `operation`, over the map and channels the tests share.
fn on(operation: ConvolutionOperation) -> ConvProblem {
    ConvProblem::from(&Conv2dProblem {
        operation,
        input_shape: [1, 64, 58, 58],
        element: Conv2dElement::F32,
        ..dense()
    })
}

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

/// Every operation prepares at each block setting, at the block its form gives that setting.
#[test]
fn every_operation_prepares_at_each_block_setting_on_a_cpu_runtime() {
    let hw = cpu(12, 12);
    for operation in [
        ConvolutionOperation::Forward,
        ConvolutionOperation::BackwardData,
        ConvolutionOperation::BackwardWeight,
    ] {
        let problem = on(operation);
        let form = BlockForm::of(&problem.roles());
        for settings in ROUTINE.settings(&hw) {
            let (blueprint, _) = ROUTINE
                .prepare(&hw, &problem, &settings)
                .unwrap_or_else(|refusal| panic!("{operation:?} at {settings}: {refusal:?}"));
            let block = (
                blueprint.microkernel.rows,
                blueprint.microkernel.settings.columns.get(),
            );
            let expected = form.size(settings.block, &hw, problem.sum);
            assert_eq!(block, expected, "{operation:?} at {settings}");
        }
    }
}

/// The block's depth is one vector register of the sums: four `f32` on 128-bit registers, eight
/// on 256, sixteen on 512, and half as many `f64`.
#[test]
fn the_block_depth_is_one_register_of_the_sums() {
    let f32_sums = on(ConvolutionOperation::Forward);
    let f64_sums = ConvProblem {
        sum: ElemType::Float(FloatKind::F64),
        ..f32_sums.clone()
    };
    for (load_width, f32_depth) in [(128, 4), (256, 8), (512, 16)] {
        let mut hw = cpu(12, 12);
        devices::topology(&mut hw).load_width = load_width;
        assert_eq!(BlockDepth::of(&hw), BlockDepth::OneRegister);
        let depth = |problem: &ConvProblem| BlockDepth::of(&hw).size(&hw, problem);
        assert_eq!(depth(&f32_sums), f32_depth);
        assert_eq!(depth(&f64_sums), f32_depth / 2);
    }
}

/// The largest block follows the registers the device keeps live: a dot block holds fewer cells
/// than an outer one, `f64` sums cost two registers each, and more registers hold more columns.
#[test]
fn the_largest_blocks_follow_the_registers_the_device_keeps_live() {
    let f32 = Conv2dElement::F32.elem_type();
    let f64 = ElemType::Float(FloatKind::F64);
    let mut hw = cpu(12, 12);
    assert_eq!(BlockForm::Dot.largest_block(&hw, f32), (4, 2));
    assert_eq!(BlockForm::Outer.largest_block(&hw, f32), (4, 8));
    assert_eq!(BlockForm::Dot.largest_block(&hw, f64), (4, 2));
    assert_eq!(BlockForm::Outer.largest_block(&hw, f64), (4, 4));
    devices::topology(&mut hw).vector_register_count = Some(32);
    assert_eq!(BlockForm::Dot.largest_block(&hw, f32), (4, 4));
    assert_eq!(BlockForm::Outer.largest_block(&hw, f32), (4, 16));
}

/// A forward reads its rhs along the channels as wide as its input, and writes its output a value
/// at a time.
#[test]
fn a_forward_reads_its_rhs_as_wide_as_its_input() {
    let hw = cpu(12, 12);
    let blueprint = first_served(&hw, &on(ConvolutionOperation::Forward));
    assert!(blueprint.widths.lhs > 1);
    assert_eq!(blueprint.widths.rhs, blueprint.widths.lhs);
    assert_eq!(blueprint.widths.out, 1);
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
    for cores in [32, 16] {
        let hw = cpu(cores, cores);
        assert_eq!(
            hw.unit_dispatch(),
            UnitDispatch::Sweeping {
                cores: cores as usize
            }
        );
        let units = units_of(&first_served(&hw, &problem), &problem);
        assert_eq!(units, cores as usize);
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
