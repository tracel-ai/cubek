//! A float rhs multiplied by a scale tensor over its batches and `K` as the cube staged routine
//! fills its stage, against an `f64` reference; and the routines that decline it.

use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind, IntKind};
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_matmul::kernels::{
    CubeStagedMatmulRoutine, MATMUL_SCALE_NOT_OVER_RHS, MATMUL_SCALED_RHS_ALONG_K,
    MATMUL_SCALED_UNREAD, MatmulRoutine, MatmulTensors, PlaneShuffleMatmulRoutine,
    UnitRegistersMatmulRoutine,
};
use cubek_std::Hardware;
use cubek_std::routine::Routine;
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

/// A scaled product `[3, 2, 64, k] · [.., k, n]` the cube staged routine answers.
#[derive(Debug, Clone, Copy)]
enum Case {
    /// An f16 rhs under an f16 scale, a row of it a batch.
    F16,
    /// An f32 rhs under an f32 scale: skipped where the planes take no f32 fragments, as a scaled
    /// rhs is read as stored, never rounded to a narrower instruction's element.
    F32,
    F32ScaleOverF16,
    /// The scale stored once over the first batch axis, viewed across it by a stride of zero.
    ScaleSharedOverFirstBatch,
    /// The rhs and its scale one along the first batch axis, broadcast against the lhs's.
    RhsSharedOverFirstBatch,
    /// A `k` and an `n` that overhang a stage.
    Overhanging,
}

impl Case {
    /// The rhs's element, then the scale's.
    fn elements(self) -> [ElemType; 2] {
        match self {
            Self::F32 => [F32, F32],
            Self::F32ScaleOverF16 => [F16, F32],
            _ => [F16, F16],
        }
    }

    /// `k`, then `n`.
    fn extents(self) -> [usize; 2] {
        match self {
            Self::Overhanging => [100, 50],
            _ => [K, N],
        }
    }

    /// The first batch dim the rhs, then the scale, is stored at.
    fn first_batches(self) -> [usize; 2] {
        match self {
            Self::ScaleSharedOverFirstBatch => [BATCHES[0], 1],
            Self::RhsSharedOverFirstBatch => [1, 1],
            _ => [BATCHES[0], BATCHES[0]],
        }
    }

    /// Skips the case where the device runs no plane at its element, which a refusal of the scale
    /// itself is not.
    fn skip(self, why: Refusal) {
        assert_ne!(why, MATMUL_SCALED_UNREAD, "{self:?}");
        let skipped = format!("{self:?}: no plane at this element here: {why:?}");
        TestOutcome::Validated(ValidationResult::Skipped(skipped)).enforce();
    }

    /// The product on the device against the reference.
    fn answers_the_reference(self) {
        let client = client();
        let hw = Hardware::from_client(&client);
        let ([element, scale_element], [k, n]) = (self.elements(), self.extents());
        let ([b0, b1], [rhs_b0, scale_b0]) = (BATCHES, self.first_batches());
        let lhs_host = values(b0 * b1 * M * k, LHS_SEED);
        let rhs_host = values(rhs_b0 * b1 * k * n, RHS_SEED);
        let scale_host: Vec<f32> = values(scale_b0 * b1 * k, SCALE_SEED)
            .into_iter()
            .map(|v| v + 0.6)
            .collect();
        let lhs = upload(&client, &lhs_host, &[b0, b1, M, k], element);
        let rhs = upload(&client, &rhs_host, &[rhs_b0, b1, k, n], element);
        let scale = upload(&client, &scale_host, &[scale_b0, b1, k], scale_element);
        let strides = vec![(scale_b0 / rhs_b0) * b1 * k, k, 1];
        let scale = scale.viewed(Metadata::new(Shape::from(vec![rhs_b0, b1, k]), strides));
        let out = match CubeStagedMatmulRoutine.launch_scaled_on(&hw, &client, lhs, rhs, scale) {
            Ok(out) => out,
            Err(why) => return self.skip(why),
        };
        let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
        for (b, [first, second]) in (0..b0 * b1).map(|b| (b, [b / b1, b % b1])) {
            let rhs_b = (first % rhs_b0) * b1 + second;
            let scale_b = (first % scale_b0) * b1 + second;
            for (i, j) in (0..M).flat_map(|i| (0..n).map(move |j| (i, j))) {
                let want: f64 = (0..k)
                    .map(|c| {
                        let weight = scale_host[scale_b * k + c] as f64;
                        let r = rhs_host[(rhs_b * k + c) * n + j] as f64;
                        lhs_host[(b * M + i) * k + c] as f64 * staged(element, weight * r)
                    })
                    .sum();
                let have = got.get_f32(&[first, second, i, j]) as f64;
                assert!(
                    (have - want).abs() <= TOLERANCE as f64 * (1.0 + want.abs()),
                    "{self:?}: batch {b}, [{i}, {j}] is {have}, want {want}"
                );
            }
        }
    }
}

#[test]
fn a_scaled_rhs_answers_the_reference_at_f16() {
    Case::F16.answers_the_reference();
}

#[test]
fn a_scaled_rhs_answers_the_reference_at_f32() {
    Case::F32.answers_the_reference();
}

#[test]
fn an_f32_scale_over_an_f16_rhs_answers_the_reference() {
    Case::F32ScaleOverF16.answers_the_reference();
}

#[test]
fn a_scale_broadcast_over_a_batch_axis_answers_the_reference() {
    Case::ScaleSharedOverFirstBatch.answers_the_reference();
}

#[test]
fn a_scaled_rhs_broadcast_over_the_lhs_batches_answers_the_reference() {
    Case::RhsSharedOverFirstBatch.answers_the_reference();
}

#[test]
fn a_scaled_rhs_whose_k_and_n_overhang_a_stage_answers_the_reference() {
    Case::Overhanging.answers_the_reference();
}

/// The routines whose fill applies no scale decline a scaled rhs by name, whether they tile their
/// rows or hold every one whole, as the cube staged routine declines one laid along `K`; a scale
/// not laid over the rhs's batches and `K`, or over an integer rhs, is declined.
#[test]
fn a_scaled_rhs_is_declined_where_it_is_not_read() {
    let client = client();
    let hw = Hardware::from_client(&client);
    let lhs = |element| upload(&client, &values(M * K, LHS_SEED), &[M, K], element);
    let rhs = |element| upload(&client, &values(K * N, RHS_SEED), &[K, N], element);
    let scale = |extent| upload(&client, &values(extent, SCALE_SEED), &[extent], F16);
    let scaled = |rows_kind, element, extent| {
        MatmulTensors::scaled(
            &client,
            lhs(element),
            rhs(element),
            scale(extent),
            rows_kind,
        )
    };
    let unit = scaled(UnitRegistersMatmulRoutine::ROWS_KIND, F16, K).and_then(|tensors| {
        UnitRegistersMatmulRoutine.prepare_first_served(&hw, &tensors.problem())
    });
    assert_eq!(unit.map(|_| ()), Err(MATMUL_SCALED_UNREAD));
    let plane = scaled(PlaneShuffleMatmulRoutine::ROWS_KIND, F16, K);
    assert_eq!(plane.map(|_| ()), Err(MATMUL_SCALED_UNREAD));
    let tiled = CubeStagedMatmulRoutine::ROWS_KIND;
    let over_n = scaled(tiled, F16, N);
    assert_eq!(over_n.map(|_| ()), Err(MATMUL_SCALE_NOT_OVER_RHS));
    let integer = scaled(tiled, ElemType::Int(IntKind::I32), K);
    assert_eq!(integer.map(|_| ()), Err(MATMUL_SCALE_NOT_OVER_RHS));
    let columns = upload(&client, &values(K * N, RHS_SEED), &[N, K], F16)
        .viewed(Metadata::new(Shape::from(vec![K, N]), vec![1, K]));
    let along_k =
        CubeStagedMatmulRoutine.launch_scaled_on(&hw, &client, lhs(F16), columns, scale(K));
    assert_eq!(along_k.map(|_| ()), Err(MATMUL_SCALED_RHS_ALONG_K));
}
