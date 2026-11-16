pub mod backward;
mod base;
#[cfg(feature = "benchmarks")]
pub mod bench_catalog;
pub mod forward;

use cubecl::{client::Client, ir::ElemType, std::tensor::TensorHandle};
use cubek_pool::definition::PoolError;
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, TestInput, TestOutcome, assert_equals_approx,
};

pub(crate) use base::*;

pub fn build_output_tensor(
    client: &Client,
    output_shape: Vec<usize>,
    dtype: ElemType,
) -> TensorHandle {
    TestInput::builder(client.clone(), output_shape)
        .dtype(dtype)
        .zeros()
        .generate_without_host_data()
}

pub fn output_host_f32(client: &Client, output: TensorHandle) -> HostData {
    HostData::from_tensor_handle(client, output, HostDataType::F32)
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
