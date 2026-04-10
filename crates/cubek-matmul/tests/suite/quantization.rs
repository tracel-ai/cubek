use cubecl::{TestRuntime, prelude::*, ir::ElemType, ir::StorageType, ir::FloatKind, ir::IntKind, ir::UIntKind};
use cubek_matmul::{
    definition::{MatmulProblem, MatmulElems, MatmulGlobalElems},
    launch::Strategy,
    launch::launch_ref,
};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{
    DataKind, TestInput, TestTensor, ExecutionOutcome, TestOutcome,
};
use cubek_quant::{scheme::{QuantScheme, QuantMode, QuantLevel, QuantValue, QuantStore, QuantParam}, quantize};

use crate::{suite::assert_result, suite::layout_to_stride_spec};

/// Test for matrix multiplication with a quantized LHS.
/// This test demonstrates how to use `TestTensor` to mark a handle as quantized
/// and pass it to the matmul kernel via `InputBinding::Quantized`.
/// Note: This test might require a runtime that supports the chosen quantization scheme.
#[test]
pub fn test_matmul_quantized_lhs() {
    let client = TestRuntime::client(&Default::default());
    // Small problem to ensure vector_size=1
    let m = 16;
    let n = 16;
    let k = 16;
    
    let problem = MatmulProblem::from_parameters(
        m, n, k,
        vec![1].into(), vec![1].into(),
        MatrixLayout::RowMajor, MatrixLayout::RowMajor, MatrixLayout::RowMajor,
        None, None,
        MatmulGlobalElems {
            lhs: f32::as_type_native_unchecked().storage_type(),
            rhs: f32::as_type_native_unchecked().storage_type(),
            out: f32::as_type_native_unchecked().storage_type(),
        },
        cubecl::ir::AddressType::U32,
    );

    // Generate LHS (f32)
    let lhs = TestInput::new(
        client.clone(),
        problem.lhs_shape.clone(),
        problem.global_dtypes.lhs,
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
        problem.global_dtypes.rhs,
        layout_to_stride_spec(problem.rhs_layout),
        DataKind::Random {
            seed: 5678,
            distribution: cubek_test_utils::Distribution::Uniform(-1., 1.),
        },
    )
    .generate_test_tensor();

    // Quantization scheme: Symmetric, Tensor-wise, Q8S (i8), PackedU32 storage
    let scheme = QuantScheme::default()
        .with_mode(QuantMode::Symmetric)
        .with_level(QuantLevel::Tensor)
        .with_value(QuantValue::Q8S)
        .with_store(QuantStore::PackedU32(0))
        .with_param(QuantParam::F32);
    
    // Actually quantize LHS on device for a realistic test.
    // Host data remains f32 for easy reference comparison.
    let lhs = quantize_lhs(&client, lhs, scheme);

    let out = TestInput::new(
        client.clone(),
        problem.out_shape.clone(),
        problem.global_dtypes.out,
        layout_to_stride_spec(MatrixLayout::RowMajor),
        DataKind::Zeros,
    )
    .generate_without_host_data();

    let mut problem = problem;
    problem.lhs_strides = lhs.handle.strides().clone();
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
            assert_result(&lhs.host, &rhs.host, &problem, &client, out, dtypes).as_test_outcome()
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

/// Helper to quantize a TestTensor on device while keeping f32 host data.
fn quantize_lhs(
    client: &ComputeClient<TestRuntime>,
    lhs: TestTensor,
    scheme: QuantScheme,
) -> TestTensor {
    let shape = lhs.handle.shape().clone();
    let dtype_input = lhs.handle.dtype;
    
    // Scale for symmetric tensor-wise is 1x1.
    // We use a dummy scale for this integration test.
    let scale_handle = TestInput::new(
        client.clone(),
        [1, 1],
        f32::as_type_native_unchecked().storage_type(),
        cubek_test_utils::StrideSpec::RowMajor,
        DataKind::Custom { data: vec![1.0 / 127.0] },
    ).generate();
    
    let quant_dtype = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
    let mut quant_shape = shape.clone();
    let last_dim = quant_shape.len() - 1;
    quant_shape[last_dim] /= scheme.num_quants();

    let output_handle = TestInput::new(
        client.clone(),
        quant_shape,
        quant_dtype,
        cubek_test_utils::StrideSpec::RowMajor,
        DataKind::Zeros,
    ).generate();
    
    let out_scale_handle = TestInput::new(
        client.clone(),
        [1, 1],
        f32::as_type_native_unchecked().storage_type(),
        cubek_test_utils::StrideSpec::RowMajor,
        DataKind::Zeros,
    ).generate();

    // Use cubek-quant to actually quantize the data on device.
    quantize::launch_ref(
        client,
        lhs.handle.clone().binding(),
        output_handle.clone().binding(),
        scale_handle.clone().binding(),
        out_scale_handle.clone().binding(),
        &scheme,
        storage_to_elem(dtype_input),
    ).expect("Quantization failed");
    
    // Create new TestTensor with quantized handle and original f32 host data.
    let mut quantized_tensor = TestTensor {
        handle: output_handle,
        host: lhs.host,
        quantization: None,
    };
    
    // Mark the handle as quantized for the matmul dispatcher.
    quantized_tensor.mark_quantized(scheme, out_scale_handle)
}

fn storage_to_elem(st: StorageType) -> ElemType {
    match st {
        StorageType::Scalar(elem) => elem,
        _ => panic!("Unsupported storage type {:?}", st),
    }
}
