use cubecl::frontend::Scalar;
use cubek_prefix_scan::{
    base::{PrefixScanKind, strategies},
    eval::{
        correctness::{cpu_reference_result, device_result},
        tune::PrefixScanTuneKey,
    },
    instructions::PrefixScanInstructionKind,
};
use cubek_test_utils::{TestOutcome, assert_equals_approx};

#[test]
fn test_prefix_scan() {
    let client = cubecl::test_device().client();
    for problem_size in [8, 1024, 1000000] {
        for scan_kind in PrefixScanKind::ALL {
            for scan_instruction_kind in PrefixScanInstructionKind::ALL {
                let problem = PrefixScanTuneKey {
                    dtype: f32::elem_type_native(),
                    scan_kind: *scan_kind,
                    scan_instruction_kind: *scan_instruction_kind,
                    shape: [problem_size].into(),
                };
                let expected =
                    cpu_reference_result(client.clone(), problem.clone(), 0, None).unwrap();
                for strategy in strategies() {
                    match device_result(client.clone(), strategy, problem.clone(), 0) {
                        Ok(output) => {
                            assert_equals_approx(&output, &expected, 0.01f32)
                                .as_test_outcome()
                                .enforce();
                        }
                        Err(err) => TestOutcome::CompileError(err).enforce(),
                    }
                }
            }
        }
    }
}
