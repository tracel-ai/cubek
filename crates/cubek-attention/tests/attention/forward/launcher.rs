use crate::attention::forward::assert_result;
use cubecl::{TestRuntime, prelude::CubePrimitive as _, zspace::Shape};
use cubek_attention::{
    eval::forward::cpu_reference::assert_result_with_epsilon,
    forward::definition::{AttentionElems, AttentionIdent, AttentionOptions, AttentionProblem},
    forward::launch::{Strategy, launch_ref},
};

use cubecl::client::ComputeClient;
use cubek_test_utils::{ExecutionOutcome, TestInput, TestOutcome, launch_and_capture_outcome};

pub fn test_launch(
    client: ComputeClient<TestRuntime>,
    problem: AttentionProblem,
    strategy: Strategy,
) {
    test_launch_scaled(client, problem, strategy, 1.0, None)
}

/// [`test_launch`] with inputs drawn from `uniform(-range, range)` and an
/// optional absolute-epsilon override (the default epsilon assumes unit-range
/// inputs, so larger magnitudes must scale it along with the data).
pub fn test_launch_scaled(
    client: ComputeClient<TestRuntime>,
    problem: AttentionProblem,
    strategy: Strategy,
    range: f32,
    epsilon: Option<f32>,
) {
    test_launch_with_layouts(client, problem, strategy, range, epsilon, None)
}

/// [`test_launch_scaled`] with an explicit stride layout applied to every
/// input: attention consumers routinely pass permuted views (a projection
/// reshaped to `(b, seq, heads, hd)` then swapped to `(b, heads, seq, hd)`),
/// so the kernels must not assume packed row-major inputs.
pub fn test_launch_permuted(
    client: ComputeClient<TestRuntime>,
    problem: AttentionProblem,
    strategy: Strategy,
) {
    let heads = problem.dims.num_heads;
    let head_dim_strides = |seq: usize, dim: usize| {
        cubek_test_utils::StridedLayout::Explicit(vec![seq * heads * dim, dim, heads * dim, 1])
    };
    let layouts = InputLayouts {
        query: head_dim_strides(problem.dims.seq_q, problem.dims.head_dim),
        key: head_dim_strides(problem.dims.seq_kv, problem.dims.head_dim),
        value: head_dim_strides(problem.dims.seq_kv, problem.dims.val_dim),
    };
    test_launch_with_layouts(client, problem, strategy, 1.0, None, Some(layouts))
}

pub struct InputLayouts {
    query: cubek_test_utils::StridedLayout,
    key: cubek_test_utils::StridedLayout,
    value: cubek_test_utils::StridedLayout,
}

fn test_launch_with_layouts(
    client: ComputeClient<TestRuntime>,
    problem: AttentionProblem,
    strategy: Strategy,
    range: f32,
    epsilon: Option<f32>,
    layouts: Option<InputLayouts>,
) {
    let query_shape = problem.shape(AttentionIdent::Query);
    let key_shape = problem.shape(AttentionIdent::Key);
    let value_shape = problem.shape(AttentionIdent::Value);
    let mask_shape = problem.shape(AttentionIdent::Mask);
    let out_shape = problem.shape(AttentionIdent::Out);

    let mut query_builder = TestInput::builder(client.clone(), Shape::new(query_shape))
        .dtype(problem.global_dtypes.query);
    let mut key_builder =
        TestInput::builder(client.clone(), Shape::new(key_shape)).dtype(problem.global_dtypes.key);
    let mut value_builder = TestInput::builder(client.clone(), Shape::new(value_shape))
        .dtype(problem.global_dtypes.value);
    if let Some(layouts) = layouts {
        query_builder = query_builder.layout(layouts.query);
        key_builder = key_builder.layout(layouts.key);
        value_builder = value_builder.layout(layouts.value);
    }

    let (query_handle, query_data) = query_builder
        .uniform(12, -range, range)
        .generate_with_f32_host_data();

    let (key_handle, key_data) = key_builder
        .uniform(34, -range, range)
        .generate_with_f32_host_data();

    let (value_handle, value_data) = value_builder
        .uniform(56, -range, range)
        .generate_with_f32_host_data();

    let (mask_handle, mask_data) = if problem.masked {
        let (mask_handle, mask_data) = TestInput::builder(client.clone(), Shape::new(mask_shape))
            .dtype(problem.global_dtypes.mask)
            .bernoulli(78, 0.1)
            .generate_with_bool_host_data();

        (Some(mask_handle), Some(mask_data))
    } else {
        (None, None)
    };

    let out_handle = TestInput::builder(client.clone(), Shape::new(out_shape))
        .dtype(problem.global_dtypes.out)
        .zeros()
        .generate_without_host_data();

    let problem_for_launch = problem.clone();
    let out_binding = out_handle.clone().binding();
    let query_binding = query_handle.binding();
    let key_binding = key_handle.binding();
    let value_binding = value_handle.binding();
    let mask_binding = mask_handle.map(|m| m.binding());

    let outcome = launch_and_capture_outcome(&client, |c| {
        launch_ref(
            strategy,
            c,
            query_binding,
            key_binding,
            value_binding,
            mask_binding,
            out_binding,
            &problem_for_launch.global_dtypes,
            AttentionOptions {
                causal: problem_for_launch.options.causal,
                accumulator_precision: problem_for_launch.options.accumulator_precision,
            },
        )
        .into()
    });

    match outcome {
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e).enforce(),
        ExecutionOutcome::Executed => match epsilon {
            Some(epsilon) => assert_result_with_epsilon(
                &query_data,
                &key_data,
                &value_data,
                mask_data.as_ref(),
                &problem,
                &client,
                out_handle,
                epsilon,
            )
            .as_test_outcome()
            .enforce(),
            None => assert_result(
                &query_data,
                &key_data,
                &value_data,
                mask_data.as_ref(),
                &problem,
                &client,
                out_handle,
                AttentionElems::from_global_types(
                    &problem.global_dtypes,
                    half::f16::as_type_native_unchecked().storage_type(),
                    &problem.options.accumulator_precision,
                ),
            )
            .as_test_outcome()
            .enforce(),
        },
    }
}
