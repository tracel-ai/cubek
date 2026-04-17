use cubecl::{TestRuntime, ir::ElemType, ir::FloatKind, ir::StorageType, prelude::*};
use cubek_matmul::{
    definition::{MatmulElems, MatmulGlobalElems, MatmulProblem},
    launch::Strategy,
    launch::launch_ref,
};
use cubek_quant::scheme::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{
    DataKind, ExecutionOutcome, HostData, HostDataType, InputDataType, TestInput, TestOutcome,
    TestTensor, assert_equals_approx,
};

use crate::suite::layout_to_stride_spec;

/// Test for matrix multiplication with a quantized LHS.
/// Note: This test might require a runtime that supports the chosen quantization scheme.
#[test]
pub fn test_matmul_quantized_lhs() {
    let client = TestRuntime::client(&Default::default());
    let m = 64;
    let n = 64;
    let k = 64;

    // Quantization scheme: Symmetric, Tensor-wise, Q8S (i8), PackedU32 storage
    let scheme = QuantScheme::default()
        .with_mode(QuantMode::Symmetric)
        .with_level(QuantLevel::Tensor)
        .with_value(QuantValue::Q8S)
        .with_store(QuantStore::PackedU32(0))
        .with_param(QuantParam::F32);

    let lhs_dtype = InputDataType::Quantized(scheme);
    let rhs_dtype = InputDataType::from(ElemType::Float(FloatKind::F32));
    let out_dtype = InputDataType::from(ElemType::Float(FloatKind::F32));

    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        vec![1].into(),
        vec![1].into(),
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        lhs_dtype.scheme().as_ref(),
        None,
        MatmulGlobalElems {
            // The quantized view dequantizes to float, so the kernel sees the
            // output float type, not the packed storage type.
            lhs: out_dtype.storage_type(),
            rhs: rhs_dtype.storage_type(),
            out: out_dtype.storage_type(),
        },
        cubecl::ir::AddressType::U32,
    );

    // Generate LHS
    // Handle is quantized but
    let lhs = TestInput::new(
        client.clone(),
        problem.lhs_shape.clone(),
        lhs_dtype,
        layout_to_stride_spec(problem.lhs_layout),
        DataKind::Random {
            seed: 1234,
            distribution: cubek_test_utils::Distribution::Uniform(-1., 1.),
        },
    )
    .generate_test_tensor();

    // Generate RHS (f32)
    let rhs = TestInput::new(
        client.clone(),
        problem.rhs_shape.clone(),
        rhs_dtype,
        layout_to_stride_spec(problem.rhs_layout),
        DataKind::Random {
            seed: 5678,
            distribution: cubek_test_utils::Distribution::Uniform(-1., 1.),
        },
    )
    .generate_test_tensor();

    let out = TestInput::new(
        client.clone(),
        problem.out_shape.clone(),
        out_dtype,
        layout_to_stride_spec(MatrixLayout::RowMajor),
        DataKind::Zeros,
    )
    .generate_without_host_data();

    let mut problem = problem;
    // Don't override lhs_strides: for quantized tensors, the handle carries packed
    // strides (e.g. for [1,64,16]) but the problem needs the original float strides
    // which from_parameters already computed correctly.
    problem.rhs_strides = rhs.handle.strides().clone();

    let lhs_binding = test_tensor_to_binding(lhs.clone());
    let rhs_binding = test_tensor_to_binding(rhs.clone());
    let out_binding = out.clone().binding();

    let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes.clone());

    let strategy = Strategy::Naive;
    let outcome: ExecutionOutcome = launch_ref(
        &strategy,
        &client,
        lhs_binding,
        rhs_binding,
        out_binding,
        &mut dtypes,
    )
    .into();

    match outcome {
        ExecutionOutcome::Executed => {
            let expected =
                crate::suite::reference::matmul_cpu_reference(&lhs.host, &rhs.host, &problem);
            let actual = HostData::from_tensor_handle(&client, out, HostDataType::F32);
            // Use relaxed tolerance for quantized matmul: Q8S quantization introduces
            // ~1/127 error per element, accumulated over k dot-product terms.
            let quant_epsilon = 0.05;
            assert_equals_approx(&actual, &expected, quant_epsilon).as_test_outcome()
        }
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
    }
    .enforce()
}

/// Helper to convert TestTensor (which may be marked as quantized) to InputBinding.
fn test_tensor_to_binding(tensor: TestTensor) -> InputBinding<TestRuntime> {
    match tensor.quantization {
        Some(q) => InputBinding::Quantized {
            data: tensor.handle.clone().binding(),
            data_dtype: tensor.handle.dtype,
            scale: q.scale.clone().binding(),
            scale_dtype: q.scale.dtype,
            shape: q.shape,
            scheme: q.scheme,
        },
        None => InputBinding::Normal(tensor.handle.clone().binding(), tensor.handle.dtype),
    }
}
