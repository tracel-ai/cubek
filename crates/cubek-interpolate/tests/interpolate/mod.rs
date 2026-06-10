mod backward;
#[cfg(feature = "benchmarks")]
mod bench_catalog;
mod forward;

use cubecl::{TestRuntime, client::ComputeClient, ir::StorageType, std::tensor::TensorHandle};
use cubek_interpolate::definition::InterpolateError;
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, TestInput, TestOutcome, assert_equals_approx,
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

pub fn validate_test(
    result: Result<(), InterpolateError>,
    actual: cubek_test_utils::HostData,
    expected: cubek_test_utils::HostData,
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
