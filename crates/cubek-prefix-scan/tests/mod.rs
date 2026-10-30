use cubek_prefix_scan::{
    ScanKind,
    eval::{
        correctness::{cpu_reference_result, device_result},
        problem::ScanProblem,
        strategy::strategies,
    },
    instructions::ScanInstructionKind,
};
use cubek_test_utils::{TestOutcome, assert_equals_approx};

pub fn run_prefix_scan_test(
    shape: &[usize],
    scan_kind: ScanKind,
    scan_instruction_kind: ScanInstructionKind,
) {
    let client = cubecl::test_device().client();
    let problem = ScanProblem {
        scan_kind,
        scan_instruction_kind,
        shape: shape.into(),
    };
    let expected = cpu_reference_result(client.clone(), problem.clone(), 0, None).unwrap();
    for strategy in strategies() {
        match device_result(client.clone(), strategy.value, problem.clone(), 0) {
            Ok(output) => {
                assert_equals_approx(&output, &expected, 0.0f32)
                    .as_test_outcome()
                    .enforce();
            }
            Err(err) => TestOutcome::CompileError(err).enforce(),
        }
    }
}

const TEST_DIM: &[usize] = &[1000000];

#[test]
fn test_exclusive_prefix_sum() {
    run_prefix_scan_test(TEST_DIM, ScanKind::Exclusive, ScanInstructionKind::Sum);
}

#[test]
fn test_inclusive_prefix_sum() {
    run_prefix_scan_test(TEST_DIM, ScanKind::Inclusive, ScanInstructionKind::Sum);
}

#[test]
fn test_exclusive_prefix_product() {
    run_prefix_scan_test(TEST_DIM, ScanKind::Exclusive, ScanInstructionKind::Product);
}

#[test]
fn test_inclusive_prefix_product() {
    run_prefix_scan_test(TEST_DIM, ScanKind::Inclusive, ScanInstructionKind::Product);
}
