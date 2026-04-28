use cubecl::{Runtime, TestRuntime, features::TypeUsage, ir::ElemType, ir::FloatKind, prelude::*};
use cubek_matmul::{
    definition::{MatmulElems, MatmulGlobalElems, MatmulProblem},
    launch::Strategy,
    launch::launch_ref,
};
use cubek_quant::scheme::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, InputDataType, TestInput, TestOutcome, TestTensor,
    assert_equals_approx,
};

use crate::suite::layout_to_stride_spec;

/// Configuration for a parameterized quantized matmul test.
struct QuantizedMatmulCase {
    m: usize,
    n: usize,
    k: usize,
    batches: Vec<usize>,
    lhs_scheme: Option<QuantScheme>,
    rhs_scheme: Option<QuantScheme>,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    strategy: Strategy,
    /// Extra tolerance multiplier applied on top of the per-quant-value baseline.
    epsilon_scale: f32,
}

impl Default for QuantizedMatmulCase {
    fn default() -> Self {
        Self {
            m: 64,
            n: 64,
            k: 64,
            batches: vec![1],
            lhs_scheme: None,
            rhs_scheme: None,
            lhs_layout: MatrixLayout::RowMajor,
            rhs_layout: MatrixLayout::RowMajor,
            strategy: Strategy::Naive,
            epsilon_scale: 1.0,
        }
    }
}

/// Baseline tolerance for tensor-wise symmetric quantization, scaled by sqrt(k)
/// to model accumulated dot-product error. Derived from the quant value's
/// effective range so it extends automatically to new QuantValues.
fn tolerance_for(scheme: &QuantScheme, k: usize, scale: f32) -> f32 {
    let (q_min, q_max) = scheme.value.range();
    let max_abs_q = q_max.abs().max(q_min.abs());
    // Per-element dequantization error in input space (inputs scaled to [-1, 1]
    // map to [-q_max, q_max] quant codes, so per-code step in input space is
    // 1/max_abs_q).
    let per_elem = 1.0 / max_abs_q;
    per_elem * (k as f32).sqrt() * scale
}

fn tensor_scheme(value: QuantValue) -> QuantScheme {
    QuantScheme::default()
        .with_mode(QuantMode::Symmetric)
        .with_level(QuantLevel::Tensor)
        .with_value(value)
        .with_store(QuantStore::PackedU32(0))
        .with_param(QuantParam::F32)
}

fn tensor_scheme_store(value: QuantValue, store: QuantStore) -> QuantScheme {
    tensor_scheme(value).with_store(store)
}

fn block_scheme(value: QuantValue, block_size: impl AsRef<[u8]>) -> QuantScheme {
    tensor_scheme(value).with_level(QuantLevel::block(block_size))
}

/// Skips a test when the runtime lacks `i8` conversion support, which
/// `cubek_quant::quantize_native` currently requires for `QuantStore::Native`.
/// Returns `true` if the caller should proceed.
fn native_quant_supported() -> bool {
    let client = TestRuntime::client(&Default::default());
    i8::supported_uses(&client).contains(TypeUsage::Conversion)
}

fn run_quantized_matmul(case: QuantizedMatmulCase) {
    let client = TestRuntime::client(&Default::default());

    let lhs_dtype = match case.lhs_scheme {
        Some(scheme) => InputDataType::Quantized(scheme),
        None => InputDataType::from(ElemType::Float(FloatKind::F32)),
    };
    let rhs_dtype = match case.rhs_scheme {
        Some(scheme) => InputDataType::Quantized(scheme),
        None => InputDataType::from(ElemType::Float(FloatKind::F32)),
    };
    let out_dtype = InputDataType::from(ElemType::Float(FloatKind::F32));

    let problem = MatmulProblem::from_parameters(
        case.m,
        case.n,
        case.k,
        case.batches.clone().into(),
        case.batches.clone().into(),
        case.lhs_layout,
        case.rhs_layout,
        MatrixLayout::RowMajor,
        lhs_dtype.scheme().as_ref(),
        rhs_dtype.scheme().as_ref(),
        MatmulGlobalElems {
            lhs: out_dtype.storage_type(),
            rhs: out_dtype.storage_type(),
            out: out_dtype.storage_type(),
        },
        cubecl::ir::AddressType::U32,
    );

    let lhs = TestInput::builder(client.clone(), problem.lhs_shape.clone())
        .dtype(lhs_dtype)
        .stride(layout_to_stride_spec(problem.lhs_layout))
        .uniform(1234, -1., 1.)
        .generate_test_tensor();

    let rhs = TestInput::builder(client.clone(), problem.rhs_shape.clone())
        .dtype(rhs_dtype)
        .stride(layout_to_stride_spec(problem.rhs_layout))
        .uniform(5678, -1., 1.)
        .generate_test_tensor();

    let out = TestInput::builder(client.clone(), problem.out_shape.clone())
        .dtype(out_dtype)
        .stride(layout_to_stride_spec(MatrixLayout::RowMajor))
        .zeros()
        .generate_without_host_data();

    // The handle strides for quantized tensors are packed-view strides and must NOT
    // replace the problem's float-level strides computed by `from_parameters`. For
    // non-quantized sides, re-reading the handle is harmless and keeps the problem in
    // sync with any layout quirks of the generator.
    let mut problem = problem;
    if case.lhs_scheme.is_none() {
        problem.lhs_strides = lhs.handle.strides().clone();
    }
    if case.rhs_scheme.is_none() {
        problem.rhs_strides = rhs.handle.strides().clone();
    }

    let lhs_binding = test_tensor_to_binding(lhs.clone());
    let rhs_binding = test_tensor_to_binding(rhs.clone());
    let out_binding = out.clone().binding();

    let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes.clone());

    let outcome: ExecutionOutcome = launch_ref(
        &case.strategy,
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
            let tolerance = [case.lhs_scheme.as_ref(), case.rhs_scheme.as_ref()]
                .into_iter()
                .flatten()
                .map(|s| tolerance_for(s, case.k, case.epsilon_scale))
                .fold(0.0_f32, f32::max)
                .max(1e-3);
            assert_equals_approx(&actual, &expected, tolerance).as_test_outcome()
        }
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
    }
    .enforce()
}

#[test]
pub fn test_matmul_quantized_lhs() {
    run_quantized_matmul(QuantizedMatmulCase {
        lhs_scheme: Some(tensor_scheme(QuantValue::Q8S)),
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_lhs_q8f() {
    run_quantized_matmul(QuantizedMatmulCase {
        lhs_scheme: Some(tensor_scheme(QuantValue::Q8F)),
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_rhs() {
    // Transformer-style: float activations against quantized weights.
    run_quantized_matmul(QuantizedMatmulCase {
        rhs_scheme: Some(tensor_scheme(QuantValue::Q8S)),
        strategy: Strategy::Auto,
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_both() {
    run_quantized_matmul(QuantizedMatmulCase {
        lhs_scheme: Some(tensor_scheme(QuantValue::Q8S)),
        rhs_scheme: Some(tensor_scheme(QuantValue::Q8S)),
        strategy: Strategy::Auto,
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_lhs_rect() {
    // Non-square, non-power-of-two-k to exercise stride/vectorization edge cases.
    run_quantized_matmul(QuantizedMatmulCase {
        m: 48,
        n: 80,
        k: 128,
        lhs_scheme: Some(tensor_scheme(QuantValue::Q8S)),
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_lhs_batched() {
    run_quantized_matmul(QuantizedMatmulCase {
        batches: vec![2, 3],
        lhs_scheme: Some(tensor_scheme(QuantValue::Q8S)),
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_rhs_col_major() {
    // Col-major quantized RHS forces a non-contiguous path through launch_naive's
    // rhs re-layout logic.
    run_quantized_matmul(QuantizedMatmulCase {
        rhs_layout: MatrixLayout::ColMajor,
        rhs_scheme: Some(tensor_scheme(QuantValue::Q8S)),
        strategy: Strategy::Auto,
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_lhs_auto() {
    run_quantized_matmul(QuantizedMatmulCase {
        lhs_scheme: Some(tensor_scheme(QuantValue::Q8S)),
        strategy: Strategy::Auto,
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_lhs_q4s() {
    // Q4S packs 8 values per u32; inner dim (k=64) must divide by 8.
    run_quantized_matmul(QuantizedMatmulCase {
        lhs_scheme: Some(tensor_scheme(QuantValue::Q4S)),
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_lhs_q4f() {
    run_quantized_matmul(QuantizedMatmulCase {
        lhs_scheme: Some(tensor_scheme(QuantValue::Q4F)),
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_rhs_q4s() {
    run_quantized_matmul(QuantizedMatmulCase {
        rhs_scheme: Some(tensor_scheme(QuantValue::Q4S)),
        strategy: Strategy::Auto,
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_mixed_q8s_q4s() {
    // Q8S on LHS, Q4S on RHS. Different packing factors (4 vs 8) on each side
    // stress the quantized view's stride/packing handling independently.
    run_quantized_matmul(QuantizedMatmulCase {
        lhs_scheme: Some(tensor_scheme(QuantValue::Q8S)),
        rhs_scheme: Some(tensor_scheme(QuantValue::Q4S)),
        strategy: Strategy::Auto,
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_lhs_q4s_large_k() {
    run_quantized_matmul(QuantizedMatmulCase {
        m: 32,
        n: 32,
        k: 256,
        lhs_scheme: Some(tensor_scheme(QuantValue::Q4S)),
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_lhs_q4s_batched() {
    run_quantized_matmul(QuantizedMatmulCase {
        batches: vec![2, 3],
        lhs_scheme: Some(tensor_scheme(QuantValue::Q4S)),
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_both_q4s() {
    run_quantized_matmul(QuantizedMatmulCase {
        lhs_scheme: Some(tensor_scheme(QuantValue::Q4S)),
        rhs_scheme: Some(tensor_scheme(QuantValue::Q4S)),
        strategy: Strategy::Auto,
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_lhs_q2s() {
    // Q2 has only 3 levels (±1); per-element error is ~0.5 in input space.
    // Use a larger epsilon_scale to absorb the big accumulated dot-product error.
    run_quantized_matmul(QuantizedMatmulCase {
        lhs_scheme: Some(tensor_scheme(QuantValue::Q2S)),
        epsilon_scale: 4.0,
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_lhs_q2f() {
    run_quantized_matmul(QuantizedMatmulCase {
        lhs_scheme: Some(tensor_scheme(QuantValue::Q2F)),
        epsilon_scale: 4.0,
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_both_q2s() {
    // Both sides quantized with Q2S: the output should still be in the
    // right ballpark, but error compounds across both sides.
    run_quantized_matmul(QuantizedMatmulCase {
        m: 32,
        n: 32,
        k: 64,
        lhs_scheme: Some(tensor_scheme(QuantValue::Q2S)),
        rhs_scheme: Some(tensor_scheme(QuantValue::Q2S)),
        strategy: Strategy::Auto,
        epsilon_scale: 8.0,
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_lhs_q8s_native() {
    // Native store = 1:1 packing. `cubek_quant::quantize` routes this through
    // `quantize_native` which asserts `i8` conversion support; skip the test
    // on runtimes that lack it.
    if !native_quant_supported() {
        eprintln!("skipping: runtime lacks i8 conversion for Native quantization");
        return;
    }
    run_quantized_matmul(QuantizedMatmulCase {
        lhs_scheme: Some(tensor_scheme_store(QuantValue::Q8S, QuantStore::Native)),
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_rhs_q8s_native() {
    if !native_quant_supported() {
        eprintln!("skipping: runtime lacks i8 conversion for Native quantization");
        return;
    }
    run_quantized_matmul(QuantizedMatmulCase {
        rhs_scheme: Some(tensor_scheme_store(QuantValue::Q8S, QuantStore::Native)),
        strategy: Strategy::Auto,
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_lhs_q8f_native() {
    if !native_quant_supported() {
        eprintln!("skipping: runtime lacks i8 conversion for Native quantization");
        return;
    }
    run_quantized_matmul(QuantizedMatmulCase {
        lhs_scheme: Some(tensor_scheme_store(QuantValue::Q8F, QuantStore::Native)),
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_lhs_q8s_block16() {
    // Block-level isn't supported by `Strategy::Naive`. Use
    // `Strategy::Auto` here so the tiling path, which handles block-scaled
    // inputs natively, is exercised instead.
    run_quantized_matmul(QuantizedMatmulCase {
        lhs_scheme: Some(block_scheme(QuantValue::Q8S, [16u8])),
        strategy: Strategy::Auto,
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_lhs_q8s_block32() {
    run_quantized_matmul(QuantizedMatmulCase {
        lhs_scheme: Some(block_scheme(QuantValue::Q8S, [32u8])),
        strategy: Strategy::Auto,
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_rhs_q8s_block16() {
    run_quantized_matmul(QuantizedMatmulCase {
        rhs_scheme: Some(block_scheme(QuantValue::Q8S, [16u8])),
        strategy: Strategy::Auto,
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_both_q8s_block16() {
    run_quantized_matmul(QuantizedMatmulCase {
        lhs_scheme: Some(block_scheme(QuantValue::Q8S, [16u8])),
        rhs_scheme: Some(block_scheme(QuantValue::Q8S, [16u8])),
        strategy: Strategy::Auto,
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_lhs_q4s_block32() {
    run_quantized_matmul(QuantizedMatmulCase {
        lhs_scheme: Some(block_scheme(QuantValue::Q4S, [32u8])),
        strategy: Strategy::Auto,
        ..Default::default()
    });
}

#[test]
pub fn test_matmul_quantized_lhs_q4s_rect() {
    // Non-square with Q4S (packing factor 8 on inner dim). k=128 keeps
    // divisibility by 8; n=80 exercises a non-power-of-two output.
    run_quantized_matmul(QuantizedMatmulCase {
        m: 48,
        n: 80,
        k: 128,
        lhs_scheme: Some(tensor_scheme(QuantValue::Q4S)),
        ..Default::default()
    });
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
