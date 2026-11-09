//! Tests for the unit registers convolution routine: what it prepares on a CPU runtime and the
//! widths each operation reads at.

use super::*;
use crate::kernels::tiled::routines::test_support::{F32, nvidia};
use crate::kernels::tiled::{ConvOperation, ConvProblem, CubeStagedConvRoutine};
use cubek_std::devices;
use cubek_std::routine::Routine;

const ROUTINE: UnitRegistersConvRoutine = UnitRegistersConvRoutine;

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
        ConvOperation::BackwardWeight,
        ConvOperation::BackwardData,
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

/// A weight backward's input and output run along the rows, which a register block reads down the
/// contraction, so every operand is read one value at a time.
#[test]
fn a_weight_backward_reads_one_value_at_a_time() {
    let hw = devices::cpu_12();
    let settings = ROUTINE.settings(&hw)[0];
    let (blueprint, _) = ROUTINE
        .prepare(&hw, &on(ConvOperation::BackwardWeight), &settings)
        .expect("the lead setting prepares");
    assert_eq!(
        (
            blueprint.widths.lhs,
            blueprint.widths.rhs,
            blueprint.widths.out
        ),
        (1, 1, 1)
    );
}
