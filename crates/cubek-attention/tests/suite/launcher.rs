use crate::suite::assert_result;
use cubecl::TestRuntime;
use cubecl::std::CubeOptionArgs;
use cubecl::std::tensor::TensorHandle;

use cubecl::client::ComputeClient;
use cubek_attention::components::args::{TensorArgs, TensorInputsLaunch};
use cubek_attention::components::batch::BatchAttentionFamily;
use cubek_attention::components::{
    AttentionBlueprint, AttentionElems, AttentionIdent, AttentionProblem,
};
use cubek_attention::kernels::Algorithm;
use cubek_std::test_utils::{compute_strides, random_bool_tensor, random_tensor};

pub fn attention_test_launch<A: Algorithm>(
    client: ComputeClient<TestRuntime>,
    problem: AttentionProblem,
    settings: &A::Settings,
) {
    let blueprint = match A::blueprint(&client, &problem, settings) {
        Ok(b) => b,
        Err(_) => return,
    };
    let dtypes = match A::dtypes(&client, &problem, &blueprint) {
        Ok(d) => d,
        Err(_) => return,
    };

    test_attention_algorithm::<A>(client, problem, blueprint, dtypes);
}

pub fn test_attention_algorithm<A: Algorithm>(
    client: ComputeClient<TestRuntime>,
    problem: AttentionProblem,
    blueprint: AttentionBlueprint,
    dtypes: AttentionElems,
) {
    let query_shape = problem.shape(AttentionIdent::Query);
    let key_shape = problem.shape(AttentionIdent::Key);
    let value_shape = problem.shape(AttentionIdent::Value);
    let mask_shape = problem.shape(AttentionIdent::Mask);
    let out_shape = problem.shape(AttentionIdent::Out);

    let (query_handle, query_data) = random_tensor(
        &client,
        dtypes.query_global,
        12,
        &compute_strides(&query_shape, false),
        &query_shape,
    );

    let (key_handle, key_data) = random_tensor(
        &client,
        dtypes.key_global,
        34,
        &compute_strides(&key_shape, false),
        &key_shape,
    );

    let (value_handle, value_data) = random_tensor(
        &client,
        dtypes.value_global,
        56,
        &compute_strides(&value_shape, false),
        &value_shape,
    );

    let (mask_handle, mask_data) = if problem.masked {
        let (mask_handle, mask_data) = random_bool_tensor(
            &client,
            dtypes.mask,
            78,
            &compute_strides(&mask_shape, false),
            &mask_shape,
        );
        (Some(mask_handle), Some(mask_data))
    } else {
        (None, None)
    };
    let out = TensorHandle::zeros(&client, out_shape.to_vec(), dtypes.out_global);

    if launch_attention_algorithm::<A>(
        &client,
        &problem,
        blueprint,
        query_handle,
        key_handle,
        value_handle,
        mask_handle,
        &out,
        &dtypes,
    ) {
        assert_result(
            &query_data,
            &key_data,
            &value_data,
            mask_data.as_ref().map(|v| v.as_slice()),
            &problem,
            &client,
            out,
            dtypes,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_attention_algorithm<A: Algorithm>(
    client: &ComputeClient<TestRuntime>,
    problem: &AttentionProblem,
    blueprint: AttentionBlueprint,
    query: TensorHandle<TestRuntime>,
    key: TensorHandle<TestRuntime>,
    value: TensorHandle<TestRuntime>,
    mask: Option<TensorHandle<TestRuntime>>,
    out: &TensorHandle<TestRuntime>,
    dtypes: &AttentionElems,
) -> bool {
    let cube_count_plan = blueprint.cube_count_plan(&problem);

    unsafe {
        A::BatchAttention::launch_unchecked::<TensorArgs, TestRuntime>(
            &client,
            blueprint.cube_dim(),
            cube_count_plan.resolve(),
            TensorInputsLaunch::new(
                query.as_arg(blueprint.line_sizes.query),
                key.as_arg(blueprint.line_sizes.key),
                value.as_arg(blueprint.line_sizes.value),
                match mask.as_ref() {
                    Some(m) => CubeOptionArgs::Some(m.as_arg(blueprint.line_sizes.mask)),
                    None => CubeOptionArgs::None,
                },
            ),
            out.as_arg(blueprint.line_sizes.out),
            cube_count_plan.as_args(),
            &dtypes,
            blueprint,
        )
    }
    .is_ok()
}
