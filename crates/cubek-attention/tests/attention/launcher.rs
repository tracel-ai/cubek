use crate::attention::assert_result;
use cubecl::TestRuntime;
use cubek_attention::launch::{
    AttentionDefinition, AttentionElems, AttentionIdent, AttentionOptions, Strategy, launch,
};

use cubecl::client::ComputeClient;
use cubek_std::test_utils::{Distribution, StrideSpec, TestInput, current_test_mode};

pub fn test_launch(
    client: ComputeClient<TestRuntime>,
    definition: AttentionDefinition,
    strategy: Strategy,
) {
    let query_shape = definition.shape(AttentionIdent::Query);
    let key_shape = definition.shape(AttentionIdent::Key);
    let value_shape = definition.shape(AttentionIdent::Value);
    let mask_shape = definition.shape(AttentionIdent::Mask);
    let out_shape = definition.shape(AttentionIdent::Out);

    let (query_handle, query_data) = TestInput::random(
        client.clone(),
        query_shape.to_vec(),
        definition.global_dtypes.query,
        12,
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data()
    .unwrap();

    let (key_handle, key_data) = TestInput::random(
        client.clone(),
        key_shape.to_vec(),
        definition.global_dtypes.key,
        34,
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data()
    .unwrap();

    let (value_handle, value_data) = TestInput::random(
        client.clone(),
        value_shape.to_vec(),
        definition.global_dtypes.value,
        56,
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data()
    .unwrap();

    let (mask_handle, mask_data) = if definition.masked {
        let (mask_handle, mask_data) = TestInput::random(
            client.clone(),
            mask_shape.to_vec(),
            definition.global_dtypes.mask,
            78,
            Distribution::Bernoulli(0.1),
            StrideSpec::RowMajor,
        )
        .generate_with_bool_host_data()
        .unwrap();

        (Some(mask_handle), Some(mask_data))
    } else {
        (None, None)
    };

    let out_handle = TestInput::zeros(
        client.clone(),
        out_shape.to_vec(),
        definition.global_dtypes.out,
    )
    .generate_without_host_data()
    .unwrap();

    match launch(
        strategy,
        &client,
        query_handle,
        key_handle,
        value_handle,
        mask_handle,
        out_handle.clone(),
        &definition.global_dtypes,
        AttentionOptions {
            causal: definition.options.causal,
            accumulator_precision: definition.options.accumulator_precision,
        },
    ) {
        Ok(_) => assert_result(
            &query_data,
            &key_data,
            &value_data,
            mask_data.as_ref(),
            &definition,
            &client,
            out_handle,
            // TODO this is not necessarily the dtypes selected by the algorithm
            AttentionElems::from_global_types(
                &definition.global_dtypes,
                &definition.options.accumulator_precision,
            ),
        ),
        Err(err) => {
            if current_test_mode().should_fail_on_test_compilation_fail() {
                panic!("Test did not run: {}", err)
            }
        }
    }
}
