//! The cube groups routine, the autotune's floor: a row a group of a cube's units.

use cubek_microkernel::rms_norm::GateOrder;
use cubek_rms_norm::kernels::{CubeGroupsRmsNormRoutine, RmsNormRoutineCandidate};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::rms_norm::{Declines, F16, F32, RmsNorm, Rows, Run};

const ROW_DIM: usize = 128;
const GROUPED_ROW_DIM: usize = 256;

/// The floor serves every case at every setting: a decline would be a family with nothing to fall
/// back on.
#[test]
fn cube_groups_routine_matches_the_reference() {
    let rms_norm = RmsNorm::new(CubeGroupsRmsNormRoutine, Declines::Never);
    rms_norm.check(rms_norm.every_case());
}

/// No rows is no launch: the floor hands back an empty norm, plain and grouped under a gate.
#[test]
fn no_rows_launch_nothing() {
    let client = cubecl::test_device().client();
    let candidate = RmsNormRoutineCandidate::floor(&Hardware::from_client(&client));
    let plain = Run {
        element: F32,
        output_element: F32,
        order: None,
    };
    let gated = Run {
        element: F16,
        output_element: F32,
        order: Some(GateOrder::GateThenNorm),
    };
    let cases = [
        (Rows::new(0, ROW_DIM), plain),
        (Rows::new(0, GROUPED_ROW_DIM).grouped(ROW_DIM), gated),
    ];
    for (rows, run) in cases.into_iter().filter(|(_, run)| run.runs_on(&client)) {
        let out = candidate
            .launch(&rows.call(&client, &run))
            .unwrap_or_else(|why| panic!("the floor refused {rows:?} at {run:?}: {why:?}"));
        assert_eq!(out.shape().num_elements(), 0, "{rows:?} at {run:?}");
    }
}
