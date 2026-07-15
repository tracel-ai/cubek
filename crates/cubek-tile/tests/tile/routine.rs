//! [`routine::matmul`] end to end: the host entry a backend's selector binds to, driven the
//! way `metabolic`'s tiled gemv will drive it — a partitioner plan in, values out. The
//! quantized cases run the production scheme family (packed-u32, block scales along the
//! weight's `d_out`).

use cubecl::{
    TestRuntime,
    bytes::Bytes,
    prelude::*,
    quant::scheme::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue},
};
use cubek_test_utils::{
    HostData, HostDataType, HostDataVec, StridedLayout, TestInput, TestOutcome, ValidationResult,
    assert_equals_approx, pack_q_values,
};
use cubek_tile::{
    ByAxis, CubeAxis, Cut, Distribution, Leaf, Partitioner, Schedule, Tiling, WalkOrder,
    routine::{self, K, M, N, Operand},
};

use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::Shape;

/// One staged level of `tm×tn×tk` register-leaf tiles, walked by one cube.
fn staged_plan(tm: usize, tn: usize, tk: usize) -> Partitioner {
    Partitioner::row_major(
        ByAxis::new(&[(M, tm), (N, tn), (K, tk)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .staged()
}

fn f32_binding(
    client: &ComputeClient<TestRuntime>,
    rows: usize,
    cols: usize,
    values: Vec<f32>,
) -> TensorBinding<TestRuntime> {
    assert_eq!(values.len(), rows * cols);
    TensorBinding {
        handle: client.create(Bytes::from_elems(values)).binding(),
        strides: vec![cols, 1].into(),
        shape: vec![rows, cols].into(),
        runtime: core::marker::PhantomData,
    }
}

/// Plain f32 through the routine: the unquantized baseline the quant cases diff against.
#[test]
fn routine_matmul_plain() {
    run_routine_matmul((8, 8, 8), staged_plan(4, 4, 4), None);
}

/// The production scheme: packed-u32 Q8S weight, one scale per `(k, N-group)` block.
#[test]
fn routine_matmul_rhs_packed_q8() {
    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block([1, 4]))
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q8S)
        .with_param(QuantParam::F32);
    run_routine_matmul((8, 8, 8), staged_plan(4, 4, 4), Some(scheme));
}

/// The `q4s` twin — 8 values per word, so it needs a device whose vectors reach the packing
/// factor (the same gate a selector applies before picking this plan).
#[test]
fn routine_matmul_rhs_packed_q4() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if client.properties().hardware.max_vector_size < 8 {
        TestOutcome::Validated(ValidationResult::Skipped(
            "device vectors cap below q4's packing factor".to_string(),
        ))
        .enforce();
        return;
    }
    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block([1, 8]))
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q4S)
        .with_param(QuantParam::F32);
    run_routine_matmul((8, 16, 8), staged_plan(4, 8, 4), Some(scheme));
}

/// The decode shape: a single activation row against the packed weight.
#[test]
fn routine_matmul_gemv_row() {
    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block([1, 4]))
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q8S)
        .with_param(QuantParam::F32);
    run_routine_matmul((1, 8, 8), staged_plan(1, 4, 4), Some(scheme));
}

/// A multi-cube plan: `N` spread across cubes, the shape a gemv selector actually emits
/// (`M = 1` → `N` across the device).
#[test]
fn routine_matmul_gemv_row_n_across_cubes() {
    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block([1, 4]))
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q8S)
        .with_param(QuantParam::F32);
    let plan = Tiling::new()
        .extents(&[(M, 1), (N, 16), (K, 8)])
        .level(WalkOrder::RowMajor, Schedule::Staged, |l| {
            l.axis(M, Cut::sequential(1))
                .axis(N, Cut::cube(CubeAxis::X, 4))
                .axis(K, Cut::sequential(4))
        })
        .leaf(Leaf::Register)
        .partitioner()
        .clone();
    run_routine_matmul((1, 16, 8), plan, Some(scheme));
}

/// The quant-needs-staging gate fires on the caller's thread, not as swallowed zeros.
#[test]
#[should_panic(expected = "requires a staged level")]
fn routine_matmul_quant_direct_panics() {
    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block([1, 4]))
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q8S)
        .with_param(QuantParam::F32);
    let plan = Partitioner::row_major(
        ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .direct();
    run_routine_matmul((8, 8, 8), plan, Some(scheme));
}

/// The dtype gate fires on the caller's thread: a plain operand bound at anything but the
/// out type would be a silent bit-reinterpretation.
#[test]
#[should_panic(expected = "served unconverted")]
fn routine_matmul_plain_dtype_mismatch_panics() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let a = f32_binding(&client, 4, 4, vec![0.0; 16]);
    let b = f32_binding(&client, 4, 4, vec![0.0; 16]);
    let out = f32_binding(&client, 4, 4, vec![0.0; 16]);
    let f32_ty = f32::as_type_native_unchecked().storage_type();
    let u32_ty = u32::as_type_native_unchecked().storage_type();
    routine::matmul::<TestRuntime>(
        &client,
        (4, 4, 4),
        staged_plan(4, 4, 4),
        Operand::plain(a),
        Operand::plain(b),
        out,
        [f32_ty, u32_ty, f32_ty],
    );
}

/// Build the operands, run [`routine::matmul`], and check
/// `C[i,j] = Σ_p A[i,p] · B_deq[p,j]` against a host reference.
fn run_routine_matmul(
    (m, n, k): (usize, usize, usize),
    plan: Partitioner,
    rhs_scheme: Option<QuantScheme>,
) {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // A: deterministic smallish floats.
    let a_vals: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
    let a = f32_binding(&client, m, k, a_vals.clone());
    let out_alloc = client.create(Bytes::from_elems(vec![0.0f32; m * n]));
    let out = TensorBinding::<TestRuntime> {
        handle: out_alloc.clone().binding(),
        strides: vec![n, 1].into(),
        shape: vec![m, n].into(),
        runtime: core::marker::PhantomData,
    };

    // B and its dequantized host twin.
    let (b_op, b_deq, b_dtype) = match rhs_scheme {
        None => {
            let vals: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) * 0.5).collect();
            let b = f32_binding(&client, k, n, vals.clone());
            (
                Operand::plain(b),
                vals,
                f32::as_type_native_unchecked().storage_type(),
            )
        }
        Some(scheme) => {
            let (lo, hi) = scheme.value.range();
            let (lo, hi) = (lo as i32, hi as i32);
            let span = hi - lo + 1;
            let q: Vec<i32> = (0..k * n).map(|i| lo + (i as i32 % span)).collect();

            let words = pack_q_values(&q, &scheme);
            let b = TensorBinding::<TestRuntime> {
                handle: client.create(Bytes::from_elems(words)).binding(),
                strides: vec![n, 1].into(),
                shape: vec![k, n].into(),
                runtime: core::marker::PhantomData,
            };

            let bn = match scheme.level {
                QuantLevel::Block(bs) => bs.to_dim_vec(2)[1] as usize,
                _ => unreachable!(),
            };
            let sn = n / bn;
            let scale_vals: Vec<f32> = (0..k * sn).map(|g| 0.05 * (g + 1) as f32).collect();
            let scales = TestInput::builder(client.clone(), Shape::from(vec![k, sn]))
                .custom(scale_vals.clone())
                .generate_without_host_data();

            let deq: Vec<f32> = q
                .iter()
                .enumerate()
                .map(|(idx, &v)| {
                    let (p, j) = (idx / n, idx % n);
                    v as f32 * scale_vals[p * sn + j / bn]
                })
                .collect();
            (
                Operand::quantized(b, scales.binding(), scheme),
                deq,
                u32::as_type_native_unchecked().storage_type(),
            )
        }
    };

    let f32_ty = f32::as_type_native_unchecked().storage_type();
    let out_handle = TensorHandle::<TestRuntime>::new(
        out_alloc,
        vec![m, n],
        vec![n, 1],
        f32::as_type_native_unchecked().storage_type(),
    );
    routine::matmul::<TestRuntime>(
        &client,
        (m, n, k),
        plan,
        Operand::plain(a),
        b_op,
        out,
        [f32_ty, b_dtype, f32_ty],
    );

    let shape = Shape::from(vec![m, n]);
    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    let expected = HostData {
        data: HostDataVec::F32(
            (0..m * n)
                .map(|idx| {
                    let (i, j) = (idx / n, idx % n);
                    (0..k).map(|p| a_vals[i * k + p] * b_deq[p * n + j]).sum()
                })
                .collect(),
        ),
        strides: StridedLayout::RowMajor.compute_strides(&shape),
        shape,
    };
    assert_equals_approx(&got, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}
