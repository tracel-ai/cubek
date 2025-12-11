use crate::attention::assert_result;
use cubecl::TestRuntime;
use cubecl::std::tensor::TensorHandle;
use cubek_attention::launch::{
    AttentionDefinition, AttentionElems, AttentionIdent, AttentionOptions, Strategy, launch,
};

use cubecl::client::ComputeClient;
use cubek_std::test_utils::{
    TestMode, contiguous_strides, current_test_mode, random_bool_tensor, random_tensor,
};

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

    let (query_handle, query_data) = random_tensor(
        &client,
        definition.global_dtypes.query,
        12,
        &contiguous_strides(&query_shape, false),
        &query_shape,
    );

    let (key_handle, key_data) = random_tensor(
        &client,
        definition.global_dtypes.key,
        34,
        &contiguous_strides(&key_shape, false),
        &key_shape,
    );

    let (value_handle, value_data) = random_tensor(
        &client,
        definition.global_dtypes.value,
        56,
        &contiguous_strides(&value_shape, false),
        &value_shape,
    );

    let (mask_handle, mask_data) = if definition.masked {
        let (mask_handle, mask_data) = random_bool_tensor(
            &client,
            definition.global_dtypes.mask,
            78,
            &contiguous_strides(&mask_shape, false),
            &mask_shape,
        );
        (Some(mask_handle), Some(mask_data))
    } else {
        (None, None)
    };

    let out_handle = TensorHandle::zeros(&client, out_shape.to_vec(), definition.global_dtypes.out);

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
            mask_data.as_ref().map(|v| v.as_slice()),
            &definition,
            &client,
            out_handle,
            // TODO this is not necessarily the dtypes selected by the algorithm
            AttentionElems::from_global_types(
                &definition.global_dtypes,
                &definition.options.accumulator_precision,
            ),
        ),
        Err(err) => match current_test_mode() {
            TestMode::Skip => {}
            TestMode::Panic => panic!("Test did not run: {}", err),
            TestMode::Print => unreachable!(),
        },
    }
}
