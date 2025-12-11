use crate::attention::assert_result;
use cubecl::TestRuntime;
use cubecl::std::tensor::TensorHandle;
use cubek_attention::{Strategy, launch};

use cubecl::client::ComputeClient;
use cubek_attention::components::{AttentionElems, AttentionIdent, AttentionProblem};
use cubek_std::test_utils::{compute_strides, random_bool_tensor, random_tensor};

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

    let (query_handle, query_data) = random_tensor(
        &client,
        problem.global_dtypes.query,
        12,
        &compute_strides(&query_shape, false),
        &query_shape,
    );

    let (key_handle, key_data) = random_tensor(
        &client,
        problem.global_dtypes.key,
        34,
        &compute_strides(&key_shape, false),
        &key_shape,
    );

    let (value_handle, value_data) = random_tensor(
        &client,
        problem.global_dtypes.value,
        56,
        &compute_strides(&value_shape, false),
        &value_shape,
    );

    let (mask_handle, mask_data) = if problem.masked {
        let (mask_handle, mask_data) = random_bool_tensor(
            &client,
            problem.global_dtypes.mask,
            78,
            &compute_strides(&mask_shape, false),
            &mask_shape,
        );
        (Some(mask_handle), Some(mask_data))
    } else {
        (None, None)
    };

    let out_handle = TensorHandle::zeros(&client, out_shape.to_vec(), problem.global_dtypes.out);

    if launch(
        &strategy,
        &client,
        query_handle,
        key_handle,
        value_handle,
        mask_handle,
        out_handle.clone(),
        &problem.global_dtypes,
    )
    .is_ok()
    {
        assert_result(
            &query_data,
            &key_data,
            &value_data,
            mask_data.as_ref().map(|v| v.as_slice()),
            &problem,
            &client,
            out_handle,
            // TODO this is not necessarily the dtypes selected by the algorithm
            AttentionElems::from_problem(&problem),
        );
    }
}
