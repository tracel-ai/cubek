//! Matmul with bias (accumulator) test.
//!
//! The standard `test_matmul_strategy` / `test_matmul_routine` helpers always
//! pass `ComptimeOptionArgs::None` for `acc`, so regressions in the accumulator
//! load path slip through undetected. This test launches the kernel with a real
//! bias tensor and validates `out = lhs @ rhs + bias`.

use cubecl::{
    TestRuntime,
    frontend::CubePrimitive,
    ir::AddressType,
    prelude::*,
    std::tensor::{TensorHandle, launch::ViewArg, layout::VirtualLayoutLaunch},
};
use cubek_matmul::{
    components::{
        batch::BatchMatmulFamily,
        global::memory::{BatchLayout, BatchLayoutLaunch, GlobalLayout, GlobalLayoutLaunch},
    },
    definition::{
        AvailableVectorSizes, Blueprint as _, MatmulElems, MatmulProblem, cube_mapping_launch,
    },
    launch::{TensorArgs, TensorInputsLaunch, TensorOutputLaunch},
    routines::{BlueprintStrategy, Routine, simple_unit::SimpleUnitAlgorithm},
};
use cubek_std::MatrixLayout;
use cubek_test_utils::{
    DataKind, Distribution, HostData, HostDataType, HostDataVec, StrideSpec, TestInput,
    TestOutcome, assert_equals_approx,
};

use crate::suite::{layout_to_stride_spec, reference::matmul_cpu_reference};

#[test]
pub fn test_matmul_with_bias_simple_unit_f32() {
    type Algorithm = SimpleUnitAlgorithm;

    let client = TestRuntime::client(&Default::default());

    let elems = MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems();

    let mut problem = MatmulProblem::from_parameters(
        64,
        64,
        64,
        [1].into(),
        [1].into(),
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        elems,
        AddressType::default(),
    );

    let (lhs, lhs_data) = TestInput::new(
        client.clone(),
        problem.lhs_shape.clone(),
        problem.global_dtypes.lhs,
        layout_to_stride_spec(problem.lhs_layout),
        DataKind::Random {
            seed: 1234,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_with_f32_host_data();

    let (rhs, rhs_data) = TestInput::new(
        client.clone(),
        problem.rhs_shape.clone(),
        problem.global_dtypes.rhs,
        layout_to_stride_spec(problem.rhs_layout),
        DataKind::Random {
            seed: 5678,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_with_f32_host_data();

    let (bias, bias_data) = TestInput::new(
        client.clone(),
        problem.out_shape.clone(),
        problem.global_dtypes.out,
        StrideSpec::RowMajor,
        DataKind::Random {
            seed: 9999,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_with_f32_host_data();

    let out = TestInput::new(
        client.clone(),
        problem.out_shape.clone(),
        problem.global_dtypes.out,
        StrideSpec::RowMajor,
        DataKind::Zeros,
    )
    .generate_without_host_data();

    problem.lhs_strides = lhs.strides().clone();
    problem.rhs_strides = rhs.strides().clone();

    let dtypes = MatmulElems::from_globals(&problem.global_dtypes.clone());

    let outcome =
        launch_with_bias::<Algorithm>(&client, &problem, &lhs, &rhs, &bias, &out, &dtypes);

    match outcome {
        Ok(()) => {
            let no_bias_reference = matmul_cpu_reference(&lhs_data, &rhs_data, &problem);
            let mut expected = no_bias_reference.clone();
            add_bias_to_reference(&mut expected, &bias_data);

            let actual = HostData::from_tensor_handle(&client, out, HostDataType::F32);

            // Guard: if we accidentally built a reference where the bias was a
            // no-op (or the kernel is silently dropping it), the main assertion
            // below would pass against a matmul-without-bias result. Make that
            // scenario loud.
            let (HostDataVec::F32(actual_vec), HostDataVec::F32(no_bias_vec)) =
                (&actual.data, &no_bias_reference.data)
            else {
                unreachable!("F32 data by construction")
            };
            let bias_contribution: f32 = actual_vec
                .iter()
                .zip(no_bias_vec.iter())
                .map(|(x, y)| (x - y).abs())
                .sum();
            assert!(
                bias_contribution > 1e-3,
                "kernel output matches the no-bias reference; bias was not applied",
            );

            let epsilon = dtypes.acc_global.epsilon() as f32 * 500.0;
            assert_equals_approx(&actual, &expected, epsilon)
                .as_test_outcome()
                .enforce();
        }
        Err(e) => TestOutcome::CompileError(e).enforce(),
    }
}

/// Adds the bias tensor into the CPU matmul reference, in-place.
fn add_bias_to_reference(reference: &mut HostData, bias: &HostData) {
    assert_eq!(reference.shape, bias.shape);
    let HostDataVec::F32(ref mut out) = reference.data else {
        panic!("reference must be F32 for this helper");
    };
    let HostDataVec::F32(ref bias_vec) = bias.data else {
        panic!("bias must be F32 for this helper");
    };
    assert_eq!(out.len(), bias_vec.len());
    for (o, b) in out.iter_mut().zip(bias_vec.iter()) {
        *o += *b;
    }
}

fn launch_with_bias<A: Routine<()>>(
    client: &ComputeClient<TestRuntime>,
    problem: &MatmulProblem,
    lhs: &TensorHandle<TestRuntime>,
    rhs: &TensorHandle<TestRuntime>,
    bias: &TensorHandle<TestRuntime>,
    out: &TensorHandle<TestRuntime>,
    dtypes: &MatmulElems,
) -> Result<(), String> {
    let vector_sizes = AvailableVectorSizes::from_type_sizes(
        client,
        dtypes.lhs_global.size(),
        dtypes.rhs_global.size(),
        dtypes.acc_global.size(),
    )
    .filter_lhs_with_tensor(lhs.strides(), lhs.shape(), problem.lhs_layout)
    .filter_rhs_with_tensor(rhs.strides(), rhs.shape(), problem.rhs_layout)
    .filter_out_with_tensor(out.strides(), out.shape())
    .pick_max()
    .map_err(|e| format!("vector size: {e}"))?;

    let device_settings = A::device_settings(client, vector_sizes);
    let expand_info = A::expand_blueprint(problem, &device_settings, &BlueprintStrategy::default())
        .map_err(|e| format!("expand: {e}"))?;
    let launch_info =
        A::prepare(problem, &device_settings, expand_info).map_err(|e| format!("prepare: {e}"))?;

    let blueprint = launch_info.blueprint;
    let dtypes = launch_info.dtypes.clone();
    let vector_sizes = launch_info.vector_sizes;

    let lhs_binding = lhs.clone().binding();
    let rhs_binding = rhs.clone().binding();
    let bias_binding = bias.clone().binding();
    let out_binding = out.clone().binding();

    let lhs_view = ViewArg::new_tensor::<GlobalLayout>(
        lhs_binding.clone().into_tensor_arg(),
        GlobalLayoutLaunch::from_handle(
            &lhs_binding,
            vector_sizes.lhs,
            blueprint.lhs_global_layout_config(),
        ),
    );
    let rhs_view = ViewArg::new_tensor::<GlobalLayout>(
        rhs_binding.clone().into_tensor_arg(),
        GlobalLayoutLaunch::from_handle(
            &rhs_binding,
            vector_sizes.rhs,
            blueprint.rhs_global_layout_config(),
        ),
    );
    let acc_view = ViewArg::new_tensor::<GlobalLayout>(
        bias_binding.clone().into_tensor_arg(),
        GlobalLayoutLaunch::from_handle(
            &bias_binding,
            vector_sizes.out,
            blueprint.out_global_layout_config(),
        ),
    );
    let out_view = ViewArg::new_tensor::<GlobalLayout>(
        out_binding.clone().into_tensor_arg(),
        GlobalLayoutLaunch::from_handle(
            &out_binding,
            vector_sizes.out,
            blueprint.out_global_layout_config(),
        ),
    );

    let lhs_batch = VirtualLayoutLaunch::new::<BatchLayout>(BatchLayoutLaunch::from_handle(
        &lhs_binding,
        problem,
    ));
    let rhs_batch = VirtualLayoutLaunch::new::<BatchLayout>(BatchLayoutLaunch::from_handle(
        &rhs_binding,
        problem,
    ));
    let acc_batch = VirtualLayoutLaunch::new::<BatchLayout>(BatchLayoutLaunch::from_handle(
        &bias_binding,
        problem,
    ));
    let out_batch = VirtualLayoutLaunch::new::<BatchLayout>(BatchLayoutLaunch::from_handle(
        &out_binding,
        problem,
    ));

    let inputs = TensorInputsLaunch::new(
        lhs_batch,
        lhs_view,
        rhs_batch,
        rhs_view,
        ComptimeOptionArgs::Some(acc_batch),
        ComptimeOptionArgs::Some(acc_view),
    );
    let output = TensorOutputLaunch::new(out_view, out_batch);

    unsafe {
        A::BatchMatmul::launch_unchecked::<TensorArgs, TestRuntime>(
            client,
            launch_info.cube_dim,
            launch_info.cube_count_plan.resolve(),
            AddressType::U32,
            inputs,
            output,
            (),
            cube_mapping_launch(&launch_info.cube_count_plan),
            blueprint,
            &dtypes,
            &vector_sizes,
        )
        .map_err(|e| format!("launch: {e:?}"))?;
    }

    client.flush().map_err(|e| format!("flush: {e:?}"))?;
    Ok(())
}
