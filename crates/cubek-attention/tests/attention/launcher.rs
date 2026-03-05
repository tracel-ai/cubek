use crate::attention::assert_result;
use cubecl::{TestRuntime, prelude::CubePrimitive as _, zspace::Shape};
use cubek_attention::{
    definition::{AttentionElems, AttentionIdent, AttentionOptions, AttentionProblem},
    launch::{Strategy, launch_ref},
};

use cubecl::client::ComputeClient;
use cubek_test_utils::{
    DataKind, Distribution, ExecutionOutcome, StrideSpec, TestInput, TestOutcome,
};

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

    let (query_handle, query_data) = TestInput::new(
        client.clone(),
        Shape::new(query_shape),
        problem.global_dtypes.query,
        StrideSpec::RowMajor,
        DataKind::Random {
            seed: 12,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_with_f32_host_data();

    let (key_handle, key_data) = TestInput::new(
        client.clone(),
        Shape::new(key_shape),
        problem.global_dtypes.key,
        StrideSpec::RowMajor,
        DataKind::Random {
            seed: 34,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_with_f32_host_data();

    let (value_handle, value_data) = TestInput::new(
        client.clone(),
        Shape::new(value_shape),
        problem.global_dtypes.value,
        StrideSpec::RowMajor,
        DataKind::Random {
            seed: 56,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_with_f32_host_data();

    let (mask_handle, mask_data) = if problem.masked {
        let (mask_handle, mask_data) = TestInput::new(
            client.clone(),
            Shape::new(mask_shape),
            problem.global_dtypes.mask,
            StrideSpec::RowMajor,
            DataKind::Random {
                seed: 78,
                distribution: Distribution::Bernoulli(0.1),
            },
        )
        .generate_with_bool_host_data();

        (Some(mask_handle), Some(mask_data))
    } else {
        (None, None)
    };

    let out_handle = TestInput::new(
        client.clone(),
        Shape::new(out_shape),
        problem.global_dtypes.out,
        StrideSpec::RowMajor,
        DataKind::Zeros,
    )
    .generate_without_host_data();

    let client = client.clone();
    let client_cloned = client.clone();
    let problem_for_launch = problem.clone();
    let out_handle_for_launch = out_handle.clone();

    let launch = client_cloned.exclusive(move || {
        let result = launch_ref(
            strategy,
            &client,
            query_handle.binding(),
            key_handle.binding(),
            value_handle.binding(),
            mask_handle.map(|m| m.binding()),
            out_handle_for_launch.binding(),
            &problem_for_launch.global_dtypes,
            AttentionOptions {
                causal: problem_for_launch.options.causal,
                accumulator_precision: problem_for_launch.options.accumulator_precision,
            },
        )
        .into();

        let errors = client.flush_errors();
        (result, errors)
    });

    match launch {
        Ok((ExecutionOutcome::Executed, errors)) => {
            for error in errors.iter() {
                match error {
                    cubecl::server::ServerError::Launch(_) => {
                        return TestOutcome::CompileError(format!("{errors:?}")).enforce();
                    }
                    _ => panic!("{errors:?}"),
                }
            }

            assert_result(
                &query_data,
                &key_data,
                &value_data,
                mask_data.as_ref(),
                &problem,
                &client_cloned,
                out_handle,
                AttentionElems::from_global_types(
                    &problem.global_dtypes,
                    half::f16::as_type_native_unchecked(),
                    &problem.options.accumulator_precision,
                ),
            )
            .as_test_outcome()
            .enforce();
        }

        Ok((ExecutionOutcome::CompileError(e), _)) => {
            TestOutcome::CompileError(e).enforce();
        }

        Err(err) => {
            TestOutcome::CompileError(err.to_string()).enforce();
        }
    }
}
