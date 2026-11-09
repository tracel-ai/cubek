//! Tests for the unit registers convolution routine: what it prepares on a CPU runtime and the
//! widths each operation reads at.

use super::*;
use crate::kernels::tiled::routines::test_support::{F32, nvidia};
use crate::kernels::tiled::{ConvOperation, ConvProblem, CubeStagedConvRoutine};
use cubecl::tile::launch::Refusal;
use cubecl::tile::ops::matmul::LhsLine;
use cubek_std::devices;
use cubek_std::routine::Routine;

const ROUTINE: UnitRegistersConvRoutine = UnitRegistersConvRoutine::new();

fn on(operation: ConvOperation) -> ConvProblem {
    ConvProblem {
        operation,
        ..ConvProblem::dense(1, [58, 58], (64, 64), [3, 3], 1, F32)
    }
}

/// A CPU runtime offers no plane instruction, so the cube staged routine lists nothing there and
/// the default tile strategy is the unit routine's; a device with one keeps its own.
#[test]
fn only_a_device_with_no_plane_instruction_leaves_the_default_to_the_unit_routine() {
    assert!(
        CubeStagedConvRoutine
            .settings(&devices::cpu_12())
            .is_empty()
    );
    assert!(!CubeStagedConvRoutine.settings(&nvidia()).is_empty());
}

/// Every operation prepares at every setting the routine lists on a CPU runtime.
#[test]
fn every_operation_prepares_on_a_cpu_runtime() {
    let hw = devices::cpu_12();
    for operation in [
        ConvOperation::Forward,
        ConvOperation::BackwardData,
        ConvOperation::BackwardWeight,
    ] {
        for settings in ROUTINE.settings(&hw) {
            let prepared = ROUTINE.prepare(&hw, &on(operation), &settings);
            assert!(
                prepared.is_ok(),
                "{operation:?} at {settings}: {prepared:?}"
            );
        }
    }
}

/// A forward convolution reads its col rhs along the channels with the input, a line of each
/// merged into one value, so the output is written a value at a time.
#[test]
fn a_forward_reads_its_rhs_as_wide_as_its_input() {
    let hw = devices::cpu_12();
    let settings = ROUTINE.settings(&hw)[0];
    let (blueprint, _) = ROUTINE
        .prepare(&hw, &on(ConvOperation::Forward), &settings)
        .expect("the lead setting prepares");
    assert!(blueprint.widths.lhs > 1);
    assert_eq!(blueprint.widths.rhs, blueprint.widths.lhs);
    assert_eq!(blueprint.widths.out, 1);
}

/// Lines of the input that do not divide the contracted axis are refused by name: twelve channels
/// cannot be read sixteen at a time.
#[test]
fn lines_past_the_contracted_axis_are_refused_by_name() {
    let problem = ConvProblem::dense(1, [9, 9], (12, 16), [3, 3], 1, F32);
    let widths = |lhs| cubek_microkernel::matmul::MatmulOperands {
        lhs,
        rhs: lhs,
        out: 1,
    };
    assert_eq!(
        UnitRegistersConvRoutine::check_lines(&problem, &widths(16)),
        Err(CONV_UNIT_LINES_UNDIVIDED)
    );
    assert_eq!(
        UnitRegistersConvRoutine::check_lines(&problem, &widths(4)),
        Ok(())
    );
}

/// A weight backward lines its input along the accumulator's rows, which a width that does not
/// divide them cannot serve, so the leaf's refusal is returned instead of a launch that panics.
#[test]
fn an_input_line_not_dividing_the_rows_is_refused_by_name() {
    let problem = ConvProblem {
        operation: ConvOperation::BackwardWeight,
        ..ConvProblem::dense(1, [9, 9], (10, 16), [3, 3], 1, F32)
    };
    assert!(matches!(
        UnitRegistersConvRoutine::check_lhs_line(&problem, 4),
        Err(Refusal::GatherLhsLine { .. })
    ));
    assert_eq!(
        UnitRegistersConvRoutine::check_lhs_line(&problem, 2),
        Ok(LhsLine::Rows)
    );
}

/// The widths a forward prepares at divide its channels, so the check never refuses what the
/// vectorization admitted.
#[test]
fn a_forward_over_twelve_channels_reads_lines_that_divide_them() {
    let hw = devices::cpu_12();
    let problem = ConvProblem::dense(1, [9, 9], (12, 16), [3, 3], 1, F32);
    let settings = ROUTINE.settings(&hw)[0];
    let (blueprint, _) = ROUTINE
        .prepare(&hw, &problem, &settings)
        .expect("the lead setting prepares");
    assert_eq!(12 % blueprint.widths.lhs, 0);
}
