pub mod backward;
pub mod forward;

use cubecl::{
    TestRuntime,
    client::ComputeClient,
    ir::{ElemType, IntKind, StorageType},
    std::tensor::TensorHandle,
};
use cubek_pool::definition::PoolError;
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, TestInput, TestOutcome, ValidationResult,
    assert_equals_approx,
};

pub fn build_output_tensor(
    client: &ComputeClient<TestRuntime>,
    output_shape: Vec<usize>,
    dtype: StorageType,
) -> TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), output_shape)
        .dtype(dtype)
        .zeros()
        .generate_without_host_data()
}

pub fn output_host_f32(
    client: &ComputeClient<TestRuntime>,
    output: TensorHandle<TestRuntime>,
) -> HostData {
    HostData::from_tensor_handle(client, output, HostDataType::F32)
}

pub fn output_host_i32(
    client: &ComputeClient<TestRuntime>,
    output: TensorHandle<TestRuntime>,
) -> HostData {
    HostData::from_tensor_handle(client, output, HostDataType::I32)
}

pub fn indices_storage_type() -> StorageType {
    StorageType::Scalar(ElemType::Int(IntKind::I32))
}

pub fn validate_test(
    result: Result<(), PoolError>,
    actual: HostData,
    expected: HostData,
    tolerance: f32,
) {
    let outcome = match ExecutionOutcome::from(result) {
        ExecutionOutcome::Executed => {
            assert_equals_approx(&actual, &expected, tolerance).as_test_outcome()
        }
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
    };
    outcome.enforce();
}

pub fn validate_indices(actual: HostData, expected: HostData) {
    if actual.shape != expected.shape {
        ValidationResult::Fail(format!(
            "Indices shape mismatch: got {:?}, expected {:?}",
            actual.shape, expected.shape,
        ))
        .as_test_outcome()
        .enforce();
        return;
    }

    let mut mismatches: Vec<(Vec<usize>, i32, i32)> = Vec::new();
    let mut total = 0usize;
    let mut mismatched = 0usize;

    for idx in actual.iter_indices() {
        total += 1;
        let got = actual.get_i32(&idx);
        let exp = expected.get_i32(&idx);
        if got != exp {
            mismatched += 1;
            if mismatches.len() < 8 {
                mismatches.push((idx, got, exp));
            }
        }
    }

    if mismatched == 0 {
        ValidationResult::Pass.as_test_outcome().enforce();
        return;
    }

    let mut message = format!(
        "Indices mismatch: {}/{} elements mismatched",
        mismatched, total
    );
    if !mismatches.is_empty() {
        message.push_str("\nFirst mismatches:");
        for (idx, got, exp) in mismatches {
            message.push_str(&format!("\n  {:?}: got {}, expected {}", idx, got, exp));
        }
    }

    ValidationResult::Fail(message).as_test_outcome().enforce();
}
