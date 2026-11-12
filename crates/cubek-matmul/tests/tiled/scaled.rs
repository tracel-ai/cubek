//! A float rhs multiplied by a scale tensor over its batches and `K` as the cube staged routine
//! fills its stage, against an `f64` reference; and the routines that decline it.

use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::{PlainTensor, ScaledTensor, StoredTensor};
use cubek_matmul::kernels::{
    CubeStagedMatmulRoutine, MATMUL_SCALED_LHS_UNREAD, MATMUL_SCALED_UNREAD, MatmulRoutine,
    PlaneShuffleMatmulRoutine, UnitRegistersMatmulRoutine,
};
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult, client};
use half::f16;

const F16: ElemType = ElemType::Float(FloatKind::F16);
const F32: ElemType = ElemType::Float(FloatKind::F32);

const BATCHES: [usize; 2] = [3, 2];
const M: usize = 64;
const N: usize = 48;
const K: usize = 96;

/// Loose enough for an `f16` output and a sum over `K`; a scale applied to the wrong position is
/// off by far more.
const TOLERANCE: f32 = 5e-3;

/// Each operand's seed, so no two hold the same values.
const LHS_SEED: usize = 1;
const RHS_SEED: usize = 2;
const SCALE_SEED: usize = 3;

/// Values a half holds exactly, so the host and the device start from the same numbers.
fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let h = (i.wrapping_mul(2654435761).wrapping_add(seed * 40503)) % 1999;
            f16::from_f32(h as f32 / 1999.0 - 0.5).to_f32()
        })
        .collect()
}

/// Skips the test where the cube staged routine prepares no scaled input on this device.
fn skip_unprepared(why: Refusal) {
    TestOutcome::Validated(ValidationResult::Skipped(format!(
        "the cube staged routine does not prepare a scaled input here: {why:?}"
    )))
    .enforce();
}

fn upload(client: &Client, data: &[f32], dims: &[usize], element: ElemType) -> PlainTensor {
    PlainTensor::of_handle(
        TestInput::builder(client.clone(), Shape::from(dims.to_vec()))
            .dtype(element)
            .custom(data.to_vec())
            .generate_without_host_data(),
    )
}

/// What the fill leaves in the stage for `value`: the product rounded to the element.
fn staged(element: ElemType, value: f64) -> f64 {
    match element {
        F16 => f16::from_f64(value).to_f64(),
        _ => value as f32 as f64,
    }
}

/// How the scale `[3, 2, 96]` varies along the batch axes.
#[derive(Debug, Clone, Copy)]
enum Scale {
    /// Along both.
    PerBatch,
    /// Along the second only: stride zero over the first, a view of `[2, 96]`.
    SharedOverFirstBatch,
}

/// `[3, 2, 64, 96] · [3, 2, 96, 48]` with the rhs, at `element`, multiplied by a scale at
/// `scale_element`.
fn answers_the_reference(element: ElemType, scale_element: ElemType, scale: Scale) {
    let client = client();
    let hw = Hardware::from_client(&client);
    let [b0, b1] = BATCHES;
    let batches = b0 * b1;
    let lhs_host = values(batches * M * K, LHS_SEED);
    let rhs_host = values(batches * K * N, RHS_SEED);
    let scale_rows = match scale {
        Scale::PerBatch => batches,
        Scale::SharedOverFirstBatch => b1,
    };
    let scale_host: Vec<f32> = values(scale_rows * K, SCALE_SEED)
        .into_iter()
        .map(|v| v + 0.6)
        .collect();
    let scale_row = |b: usize| match scale {
        Scale::PerBatch => b,
        Scale::SharedOverFirstBatch => b % b1,
    };
    let lhs = upload(&client, &lhs_host, &[b0, b1, M, K], element);
    let rhs = upload(&client, &rhs_host, &[b0, b1, K, N], element);
    let scale = match scale {
        Scale::PerBatch => upload(&client, &scale_host, &[b0, b1, K], scale_element),
        Scale::SharedOverFirstBatch => upload(&client, &scale_host, &[b1, K], scale_element)
            .viewed(Metadata::new(Shape::from(vec![b0, b1, K]), vec![0, K, 1])),
    };
    let lhs = StoredTensor::Plain(lhs);
    let rhs = StoredTensor::Scaled(ScaledTensor::new(rhs, scale).expect("a scale over K"));
    let out = match CubeStagedMatmulRoutine.launch_first_served_on(&hw, &client, lhs, rhs) {
        Ok(out) => out,
        Err(why) => return skip_unprepared(why),
    };
    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    for b in 0..batches {
        for i in 0..M {
            for j in 0..N {
                let want: f64 = (0..K)
                    .map(|k| {
                        let weight = scale_host[scale_row(b) * K + k] as f64;
                        let l = lhs_host[(b * M + i) * K + k] as f64;
                        let r = rhs_host[(b * K + k) * N + j] as f64;
                        l * staged(element, weight * r)
                    })
                    .sum();
                let have = got.get_f32(&[b / b1, b % b1, i, j]);
                assert!(
                    (have as f64 - want).abs() <= TOLERANCE as f64 * (1.0 + want.abs()),
                    "batch {b}, [{i}, {j}] is {have}, want {want}"
                );
            }
        }
    }
}

#[test]
fn a_scaled_rhs_answers_the_reference_at_f16() {
    answers_the_reference(F16, F16, Scale::PerBatch);
}

#[test]
fn a_scaled_rhs_answers_the_reference_at_f32() {
    answers_the_reference(F32, F32, Scale::PerBatch);
}

#[test]
fn an_f32_scale_over_an_f16_rhs_answers_the_reference() {
    answers_the_reference(F16, F32, Scale::PerBatch);
}

#[test]
fn a_scale_broadcast_over_a_batch_axis_answers_the_reference() {
    answers_the_reference(F16, F16, Scale::SharedOverFirstBatch);
}

/// The routines whose fill applies no scale decline a scaled rhs by name, whether they tile their
/// rows or hold every one whole, and every routine declines a scaled lhs.
#[test]
fn the_routines_applying_no_scale_decline_it() {
    let client = client();
    let hw = Hardware::from_client(&client);
    let scaled_rhs = || {
        let lhs = upload(&client, &values(M * K, LHS_SEED), &[M, K], F16);
        let rhs = upload(&client, &values(K * N, RHS_SEED), &[K, N], F16);
        let scale = upload(&client, &values(K, SCALE_SEED), &[K], F16);
        let rhs = ScaledTensor::new(rhs, scale).expect("a scale over K");
        (StoredTensor::Plain(lhs), StoredTensor::Scaled(rhs))
    };
    let (lhs, rhs) = scaled_rhs();
    assert_eq!(
        UnitRegistersMatmulRoutine
            .launch_first_served_on(&hw, &client, lhs, rhs)
            .err(),
        Some(MATMUL_SCALED_UNREAD)
    );
    let (lhs, rhs) = scaled_rhs();
    assert_eq!(
        PlaneShuffleMatmulRoutine
            .launch_first_served_on(&hw, &client, lhs, rhs)
            .err(),
        Some(MATMUL_SCALED_UNREAD)
    );
    // A scaled lhs is built as a `[m, k]` matrix under a scale over its first dim.
    let lhs = upload(&client, &values(M * K, LHS_SEED), &[M, K], F16);
    let scale = upload(&client, &values(M, SCALE_SEED), &[M], F16);
    let lhs = StoredTensor::Scaled(ScaledTensor::new(lhs, scale).expect("a scale over M"));
    let rhs = StoredTensor::Plain(upload(&client, &values(K * N, RHS_SEED), &[K, N], F16));
    assert_eq!(
        CubeStagedMatmulRoutine
            .launch_first_served_on(&hw, &client, lhs, rhs)
            .err(),
        Some(MATMUL_SCALED_LHS_UNREAD)
    );
}
