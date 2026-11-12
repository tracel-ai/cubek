//! Tests for the unit registers convolution routine: what it prepares on a CPU runtime and the
//! widths each operation reads at.

use super::*;
use crate::components::ConvolutionOperation;
use crate::eval::benchmarks::{Conv2dElement, Conv2dProblem, problems};
use crate::kernels::tiled::routines::test_support::{F32, dense, nvidia};
use crate::kernels::tiled::{ConvProblem, CubeStagedConvRoutine, TileConvBlueprint};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::launch::Refusal;
use cubecl::tile::ops::matmul::LhsVectorAxis;
use cubek_matmul::kernels::UnitRegistersMatmulRoutineSettings;
use cubek_microkernel::matmul::MatmulOperands;
use cubek_microkernel::matmul::cube::MatmulPlaneLayout;
use cubek_microkernel::matmul::unit::UnitRegistersMicrokernel;
use cubek_std::devices;
use cubek_std::routine::Routine;
use cubek_std::{Hardware, UnitDispatch};

const ROUTINE: UnitRegistersConvRoutine = UnitRegistersConvRoutine;

/// One image of `f32` through `operation`, over the map and channels the unit tests share.
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

/// A CPU runtime offers no plane instruction, so the cube staged routine lists nothing there and
/// the default tile strategy is the unit routine's; a device with one keeps its own.
#[test]
fn only_a_device_with_no_plane_instruction_leaves_the_default_to_the_unit_routine() {
    assert!(CubeStagedConvRoutine.settings(&cpu(12, 12)).is_empty());
    assert!(!CubeStagedConvRoutine.settings(&nvidia()).is_empty());
}

/// Every operation prepares at either setting, the largest block of its own form or one cell, and
/// the block it prepares is the setting's.
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
            assert_eq!(
                block,
                form.size(settings.block, &hw, problem.sum),
                "{operation:?} at {settings}"
            );
        }
    }
}

/// The block's depth is one vector register of the sums, not the wider reads the device sizes its
/// IO to: four `f32` on 128-bit registers, eight on AVX2's 256, sixteen on 512. An `f16` input
/// sums in `f32`, so it takes the `f32` depth, and an `f64` sum half of it.
#[test]
fn the_block_depth_is_one_register_of_the_sums() {
    let device = |load_width| {
        let mut hw = cpu(12, 12);
        devices::topology(&mut hw).load_width = load_width;
        hw
    };
    let problem = |element| {
        ConvProblem::from(&Conv2dProblem {
            input_shape: [1, 64, 58, 58],
            element,
            ..dense()
        })
    };
    let depth = |load_width, element| {
        let hw = device(load_width);
        let (blueprint, _) = ROUTINE
            .prepare(&hw, &problem(element), &ROUTINE.settings(&hw)[0])
            .expect("the first setting prepares");
        blueprint.microkernel.settings.depth.get()
    };
    for element in [Conv2dElement::F16, Conv2dElement::F32] {
        assert_eq!(depth(128, element), 4, "{element:?}");
        assert_eq!(depth(256, element), 8, "{element:?}");
        assert_eq!(depth(512, element), 16, "{element:?}");
    }
    let f64_sums = ConvProblem {
        sum: ElemType::Float(FloatKind::F64),
        ..problem(Conv2dElement::F32)
    };
    for (load_width, values) in [(128, 2), (256, 4), (512, 8)] {
        let hw = device(load_width);
        assert_eq!(BlockDepth::of(&hw), BlockDepth::OneRegister);
        assert_eq!(
            BlockDepth::of(&hw).size(&hw, &f64_sums),
            values,
            "{load_width}"
        );
    }
}

/// A GPU states no vector registers and keeps its partials in scalars, so the depth is the widest
/// read of the input: 128-bit reads of `f16` hold eight, within its widest vector of eight, where
/// one register of `f32` sums would hold four.
#[test]
fn a_gpu_reads_its_depth_as_wide_as_its_input() {
    let mut hw = nvidia();
    devices::topology(&mut hw).max_vector_size = 8;
    assert_eq!(BlockDepth::of(&hw), BlockDepth::WidestRead);
    let problem = ConvProblem::from(&Conv2dProblem {
        input_shape: [1, 64, 58, 58],
        element: Conv2dElement::F16,
        ..dense()
    });
    let blueprint = first_served(&hw, &problem);
    assert_eq!(blueprint.microkernel.settings.depth.get(), 8);
}

/// An input running down the rows reads a block of the rows and the columns at the vector width
/// of the device: more registers for the columns than a dot block holds cells. Of fourteen free
/// 128-bit registers, an outer block of `f32` sums takes 4 by 8, eight cells of one register, a
/// register of rows and two of columns; of `f64` sums each costs twice that, past fourteen, so it
/// takes 4 by 4, 4·2 + 2 + 2 = 12, the most rows of the 16-cell blocks that fit.
#[test]
fn the_largest_blocks_follow_the_registers_the_device_keeps_live() {
    let f64 = ElemType::Float(FloatKind::F64);
    let mut hw = cpu(12, 12);
    assert_eq!(BlockForm::Dot.largest_block(&hw, F32), (4, 2));
    assert_eq!(BlockForm::Outer.largest_block(&hw, F32), (4, 8));
    assert_eq!(BlockForm::Dot.largest_block(&hw, f64), (4, 2));
    assert_eq!(BlockForm::Outer.largest_block(&hw, f64), (4, 4));
    devices::topology(&mut hw).vector_register_count = Some(32);
    assert_eq!(BlockForm::Dot.largest_block(&hw, F32), (4, 4));
    assert_eq!(BlockForm::Outer.largest_block(&hw, F32), (4, 16));
}

/// A forward convolution reads its col rhs along the channels with the input, a vector of each
/// merged into one value, so the output is written a value at a time.
#[test]
fn a_forward_reads_its_rhs_as_wide_as_its_input() {
    let hw = cpu(12, 12);
    let settings = ROUTINE.settings(&hw)[0];
    let (blueprint, _) = ROUTINE
        .prepare(&hw, &on(ConvolutionOperation::Forward), &settings)
        .expect("the first setting prepares");
    assert!(blueprint.widths.lhs > 1);
    assert_eq!(blueprint.widths.rhs, blueprint.widths.lhs);
    assert_eq!(blueprint.widths.out, 1);
}

/// A weight backward vectorizes its input along the accumulator's rows, which a width that does not
/// divide them cannot serve, so the leaf's refusal is returned instead of a launch that panics.
#[test]
fn an_input_vector_not_dividing_the_rows_is_refused_by_name() {
    let problem = layer(ConvolutionOperation::BackwardWeight, 10);
    assert!(matches!(
        UnitRegistersConvRoutine::check_lhs_vector_axis(&problem, 4),
        Err(Refusal::LhsVectorNotServed { .. })
    ));
    assert_eq!(
        UnitRegistersConvRoutine::check_lhs_vector_axis(&problem, 2),
        Ok(LhsVectorAxis::Rows)
    );
}

/// A weight backward whose rows the input's vectors do not divide prepares at a width its rows
/// do, rather than declining the setting.
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

/// The widths a forward prepares at divide its channels.
#[test]
fn a_forward_over_twelve_channels_reads_vectors_that_divide_them() {
    let hw = cpu(12, 12);
    let problem = layer(ConvolutionOperation::Forward, 12);
    let settings = ROUTINE.settings(&hw)[0];
    let (blueprint, _) = ROUTINE
        .prepare(&hw, &problem, &settings)
        .expect("the first setting prepares");
    assert_eq!(12 % blueprint.widths.lhs, 0);
}

fn wasted(tiles: AcrossCounts, split: AcrossCounts) -> f64 {
    let slots =
        tiles.rows.next_multiple_of(split.rows) * tiles.across.next_multiple_of(split.across);
    1.0 - (tiles.rows * tiles.across) as f64 / slots as f64
}

/// A 56 wide map in 3 wide blocks has 19 tiles along a row, so 32 planes laid along the rows
/// alone would leave most of the last group idle: the split takes the across axis too.
#[test]
fn a_56_by_56_map_splits_its_planes_over_rows_and_across() {
    let tiles = AcrossCounts {
        rows: 56_usize.div_ceil(3),
        across: 56,
    };
    let split = tiles.spread_split(32);
    assert_eq!(split, AcrossCounts { rows: 4, across: 8 });
    assert!(wasted(tiles, split) < 0.1, "{split:?}");
}

/// A 7 by 7 map is held by fewer planes than a cube offers, so the split leaves the rest out.
#[test]
fn a_7_by_7_map_takes_only_the_planes_it_fills() {
    let tiles = AcrossCounts {
        rows: 7_usize.div_ceil(3),
        across: 7,
    };
    let split = tiles.spread_split(32);
    assert_eq!(split, AcrossCounts { rows: 3, across: 7 });
    assert!(wasted(tiles, split) < 0.1, "{split:?}");
}

/// A single plane, or a single tile, splits to one.
#[test]
fn a_split_is_never_empty() {
    let one = AcrossCounts { rows: 1, across: 1 };
    let five = AcrossCounts { rows: 5, across: 5 };
    assert_eq!(five.spread_split(1), one);
    assert_eq!(one.spread_split(32), one);
}

/// A CPU runtime of `threads` hardware threads over `cores` physical cores, a cube holding a unit
/// for each thread, stating the sixteen vector registers of an x86 host, as cubecl's reports one.
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

/// The 3x3 layer over a 56 by 56 map has 14 by 56 tiles, one tile along the 64 channels out.
/// Swept by 32 or 16 cores, the cube takes a unit for every core, and the units take the 784
/// tiles in turns: on 32 the busiest sweeps 25 and the idlest 24, where 4 by 8 boxes left the
/// busiest 28; on 16 every unit sweeps 49, as 2 by 8 boxes did.
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

/// A core's second hardware thread only shares the arithmetic its first unit fills, so 32 threads
/// over 16 physical cores take the 16 units of 16 threads over 16 cores.
#[test]
fn units_number_the_physical_cores_not_the_threads() {
    let (problem, _) = catalog_entry("cpu_3x3_c64_56");
    let hw = cpu(32, 16);
    assert_eq!(hw.unit_dispatch(), UnitDispatch::Sweeping { cores: 16 });
    assert_eq!(units_of(&first_served(&hw, &problem), &problem), 16);
}

/// The units a cube takes are capped by its cores, by the planes a cube holds, and by the tiles,
/// so none is launched idle; a prime count of cores is taken whole, as no box needs to divide it.
#[test]
fn units_are_capped_by_cores_planes_and_tiles() {
    let tiles = AcrossCounts {
        rows: 14,
        across: 56,
    };
    let deal = |cores, launch_tiles, planes| {
        PlaneDeal::of(
            UnitDispatch::Sweeping { cores },
            tiles,
            launch_tiles,
            planes,
        )
    };
    assert_eq!(
        deal(16, 784, MatmulPlaneLayout::new(32, 2)),
        PlaneDeal::InTurns { units: 16 }
    );
    assert_eq!(
        deal(16, 784, MatmulPlaneLayout::new(4, 1)),
        PlaneDeal::InTurns { units: 4 }
    );
    assert_eq!(
        deal(7, 784, MatmulPlaneLayout::new(32, 1)),
        PlaneDeal::InTurns { units: 7 }
    );
    assert_eq!(
        deal(16, 14, MatmulPlaneLayout::new(32, 1)),
        PlaneDeal::InTurns { units: 14 }
    );
}

/// The 3x3 layer over a 7 by 7 map has 2 by 7 tiles and one along the channels, fewer than 16
/// cores: the cube takes a unit per tile rather than launching two units with none.
#[test]
fn fewer_tiles_than_cores_take_a_unit_each() {
    let (problem, _) = catalog_entry("cpu_narrow_c64_7");
    assert_eq!(
        units_of(&first_served(&cpu(32, 16), &problem), &problem),
        14
    );
}

/// Each catalog layer on 16 physical cores, its tiles taken in turns: the busiest unit sweeps
/// the ceiling of its share, and the edge tiles, whose input box reaches past the image and so
/// takes the checked walk, are spread as evenly.
///
/// The stride 2 layer has 7 by 28 tiles and two along its 128 channels out, 392: its busiest unit
/// sweeps 25 and its idlest 24, where 4 by 4 boxes left the busiest 28 against a share of 24.5.
/// Its edge tiles are the first tile row and the first tile column (the last row and column read
/// inside the image), 68: the first row is one run of the order, which deals one tile a unit, and
/// the first column every seventh tile, which 7, prime to 16, deals to every unit in turn, so no
/// unit holds more than 5 of them.
///
/// The stride 1 layer's 14 by 56 tiles deal 49 to every unit. Its edges are the first and last
/// tile rows and columns, 136: 14 shares 2 with 16, so the first column falls to the even units
/// and the last to the odd ones, and again none holds more than 9.
///
/// The 1x1 layer has 14 by 56 tiles and four along its 256 channels, 3136, 196 to every unit, and
/// no edge, as no box reads past the image.
#[test]
fn the_units_take_their_share_of_every_tile_and_every_edge() {
    let hw = cpu(16, 16);
    for (name, sweep, edges) in [
        (
            "cpu_3x3_s2_c128_56",
            UnitSweep {
                most: 25,
                least: 24,
            },
            68,
        ),
        (
            "cpu_3x3_c64_56",
            UnitSweep {
                most: 49,
                least: 49,
            },
            136,
        ),
        (
            "cpu_1x1_c64_256_56",
            UnitSweep {
                most: 196,
                least: 196,
            },
            0,
        ),
    ] {
        let (problem, layer) = catalog_entry(name);
        let blueprint = first_served(&hw, &problem);
        let units = units_of(&blueprint, &problem);
        assert_eq!(units, 16, "{name}");
        let checked = ForwardTiles::of(&hw, &layer, &blueprint).checked_in_order();
        assert_eq!(UnitSweep::of(units, checked.len()), sweep, "{name}");
        let total = checked.iter().filter(|&&checked| checked).count();
        assert_eq!(total, edges, "{name}");
        let busiest = (0..units)
            .map(|unit| {
                checked
                    .iter()
                    .skip(unit)
                    .step_by(units)
                    .filter(|&&c| c)
                    .count()
            })
            .max()
            .unwrap_or(0);
        assert_eq!(busiest, edges.div_ceil(units), "{name}");
    }
}

/// Tiles a unit sweeps over the whole launch: the most any unit does, which a launch waits for,
/// and the least.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UnitSweep {
    most: usize,
    least: usize,
}

impl UnitSweep {
    /// The sweep of `units` units taking `tiles` in turns.
    fn of(units: usize, tiles: usize) -> Self {
        Self {
            most: tiles.div_ceil(units),
            least: tiles / units,
        }
    }
}

/// A forward layer's output tiles as the unit routine launches them: `rows` output columns of the
/// image to a tile, one output row, and `channel_tiles` along the channels out.
struct ForwardTiles {
    rows: usize,
    channel_tiles: usize,
    batch: usize,
    out: usize,
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
}

impl ForwardTiles {
    /// The tiles of `layer`, a square forward layer, as `blueprint` launches it on `hw`.
    fn of(
        hw: &Hardware,
        layer: &Conv2dProblem,
        blueprint: &TileConvBlueprint<UnitRegistersMicrokernel>,
    ) -> Self {
        let microkernel = &blueprint.microkernel;
        let channels = layer.weight_shape[0];
        let blocks = UnitRegistersMatmulRoutineSettings::cpu_col_rhs(hw)
            .fitted_blocks_per_plane(channels, 1);
        Self {
            rows: microkernel.rows,
            channel_tiles: channels.div_ceil(microkernel.settings.columns.get() * blocks),
            batch: layer.input_shape[0],
            out: layer.out_extent()[1],
            input: layer.input_shape[3],
            kernel: layer.weight_shape[3],
            stride: layer.args.stride[1],
            padding: layer.args.padding[1],
        }
    }

    /// Whether the output positions `first..=last` read past the image along one of its axes.
    fn reads_past(&self, first: usize, last: usize) -> bool {
        first * self.stride < self.padding
            || last * self.stride + self.kernel > self.input + self.padding
    }

    /// Whether each tile reads past the image, in the order the launch's planes take them: the
    /// rows fastest, then the channels out, the across axis and the batch, as cubecl-tile numbers
    /// a launch's plane `cube · units + unit` and lays the tiles in row order. A unit `u` of
    /// `units` sweeps the tiles `u`, `u + units`, and so on.
    fn checked_in_order(&self) -> Vec<bool> {
        let row_tiles = self.out.div_ceil(self.rows);
        let mut checked = Vec::new();
        for _ in 0..self.batch {
            for across in 0..self.out {
                for _ in 0..self.channel_tiles {
                    for tile in 0..row_tiles {
                        let first = tile * self.rows;
                        checked.push(
                            self.reads_past(first, first + self.rows - 1)
                                || self.reads_past(across, across),
                        );
                    }
                }
            }
        }
        checked
    }
}

/// A GPU spreads its cubes over its multiprocessors rather than sweeping them, so the floor keeps
/// the split that covers the tiles in the fewest cubes. The 3x3 layer's 14 by 56 tiles need 28
/// cubes of at most 32 planes, 784 tiles taking at least 28 planes a cube; of the 28-plane splits
/// that reach it, 1 by 28, 2 by 14, 7 by 4 and 14 by 2, the most along the rows is 14 by 2.
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

/// The catalog's CPU problems, each as `(name, problem)`.
fn catalog() -> Vec<(String, ConvProblem)> {
    problems()
        .into_iter()
        .filter(|entry| entry.id.starts_with("cpu_"))
        .map(|entry| (entry.id, ConvProblem::from(&entry.value)))
        .collect()
}

/// The first setting the routine lists on `hw` that serves `problem`: the launch a default run
/// takes.
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

/// What a catalog problem is expected to launch at.
struct Launch {
    name: &'static str,
    rows: usize,
    columns: usize,
    units: usize,
    widths: MatmulOperands<usize>,
}

/// The register block, the units and the widths each catalog problem launches at on a 12-core CPU
/// runtime: the forward and the data backward keep the block a dot product gets, the weight
/// backward reads its input down the rows and its rhs along the columns, a block of eight of
/// them. Every problem holds more tiles than cores, so each takes a unit for every core.
#[test]
fn each_catalog_problem_launches_at_its_block_units_and_widths() {
    let expected = [
        Launch {
            name: "cpu_3x3_c64_56",
            rows: 4,
            columns: 2,
            units: 12,
            widths: MatmulOperands {
                lhs: 4,
                rhs: 4,
                out: 1,
            },
        },
        Launch {
            name: "cpu_3x3_s2_c128_56",
            rows: 4,
            columns: 2,
            units: 12,
            widths: MatmulOperands {
                lhs: 4,
                rhs: 4,
                out: 1,
            },
        },
        Launch {
            name: "cpu_narrow_c64_7",
            rows: 4,
            columns: 2,
            units: 12,
            widths: MatmulOperands {
                lhs: 4,
                rhs: 4,
                out: 1,
            },
        },
        Launch {
            name: "cpu_1x1_c64_256_56",
            rows: 4,
            columns: 2,
            units: 12,
            widths: MatmulOperands {
                lhs: 4,
                rhs: 4,
                out: 1,
            },
        },
        Launch {
            name: "cpu_bwd_data_c64_56",
            rows: 4,
            columns: 2,
            units: 12,
            widths: MatmulOperands {
                lhs: 4,
                rhs: 2,
                out: 2,
            },
        },
        Launch {
            name: "cpu_bwd_weight_c64_56",
            rows: 4,
            columns: 8,
            units: 12,
            widths: MatmulOperands {
                lhs: 4,
                rhs: 8,
                out: 1,
            },
        },
    ];
    for ((name, problem), launch) in catalog().into_iter().zip(expected) {
        assert_eq!(name, launch.name);
        let blueprint = first_served(&cpu(12, 12), &problem);
        let block = (
            blueprint.microkernel.rows,
            blueprint.microkernel.settings.columns.get(),
        );
        assert_eq!(block, (launch.rows, launch.columns), "{name}");
        assert_eq!(units_of(&blueprint, &problem), launch.units, "{name}");
        assert_eq!(blueprint.widths, launch.widths, "{name}");
    }
}
