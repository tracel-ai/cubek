use crate::attention::assert_result;
use cubecl::TestRuntime;
use cubek_attention::{
    definition::{AttentionElems, AttentionIdent, AttentionOptions, AttentionProblem},
    launch::{Strategy, launch},
};

use cubecl::client::ComputeClient;
use cubek_test_utils::{Distribution, ExecutionOutcome, StrideSpec, TestInput, TestOutcome};

pub fn test_launch(
    client: ComputeClient<TestRuntime>,
    problem: AttentionProblem,
    strategy: Strategy,
) {
    let query_shape = problem.shape(AttentionIdent::Query);
    let key_shape = problem.shape(AttentionIdent::Key);
    let value_shape = problem.shape(AttentionIdent::Value);
    let mask_shape = problem.shape(AttentionIdent::Mask);
    let out_shape = problem.shape(AttentionIdent::Out);

    let (query_handle, query_data) = TestInput::random(
        client.clone(),
        query_shape.to_vec(),
        problem.global_dtypes.query,
        12,
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data();

    let (key_handle, key_data) = TestInput::random(
        client.clone(),
        key_shape.to_vec(),
        problem.global_dtypes.key,
        34,
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data();

    let (value_handle, value_data) = TestInput::random(
        client.clone(),
        value_shape.to_vec(),
        problem.global_dtypes.value,
        56,
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data();

    let (mask_handle, mask_data) = if problem.masked {
        let (mask_handle, mask_data) = TestInput::random(
            client.clone(),
            mask_shape.to_vec(),
            problem.global_dtypes.mask,
            78,
            Distribution::Bernoulli(0.1),
            StrideSpec::RowMajor,
        )
        .generate_with_bool_host_data();

        (Some(mask_handle), Some(mask_data))
    } else {
        (None, None)
    };

    let out_handle = TestInput::zeros(
        client.clone(),
        out_shape.to_vec(),
        problem.global_dtypes.out,
        StrideSpec::RowMajor,
    )
    .generate_without_host_data();

    match launch(
        strategy,
        &client,
        query_handle,
        key_handle,
        value_handle,
        mask_handle,
        out_handle.clone(),
        &problem.global_dtypes,
        AttentionOptions {
            causal: problem.options.causal,
            accumulator_precision: problem.options.accumulator_precision,
        },
    )
    .into()
    {
        ExecutionOutcome::Executed => assert_result(
            &query_data,
            &key_data,
            &value_data,
            mask_data.as_ref(),
            &problem,
            &client,
            out_handle,
            // TODO this is not necessarily the dtypes selected by the algorithm
            AttentionElems::from_global_types(
                &problem.global_dtypes,
                &problem.options.accumulator_precision,
            ),
        )
        .as_test_outcome(),
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
    }
    .enforce();
}
