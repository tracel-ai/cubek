//! A packed activation against a float weight through the staged routine, as W4A4 runs a
//! projection whose weight was left in float: every setting that prepares on this device binds
//! both inputs and answers the dequantized product.

use super::indexed::upload;
use super::packed::{PackedFormat, PackedSlabs, PackedSlabsSpec};
use cubecl::prelude::*;
use cubecl::zspace::Shape;
use cubek_matmul::kernels::tensor::{PlainTensor, StoredTensor};
use cubek_matmul::kernels::{CubeStagedMatmulRoutine, MatmulRoutine};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, TestOutcome, ValidationResult, client};
use half::bf16;

/// Loose enough for a `bf16` output.
const BF16_TOLERANCE: f32 = 1e-2;

/// Qwen3.8-27B's 48-wide gated-delta projection at a prompt's rows, and the same 44 columns wide:
/// NVFP4 activations against a `bf16` weight. A box 32 columns wide reaches past the weight's
/// columns, which binds it at a width dividing them; on an RTX PRO 6000 a box bound eight wide
/// past 48, which no mask stated, refused to bind and the model did not load. Skipped on a device
/// where no setting prepares, one without the matrix instructions the staged routine contracts
/// on.
#[test]
fn a_packed_activation_against_a_float_weight_answers_at_every_setting() {
    let (m, k) = (512, 5120);
    let client = client();
    let hw = Hardware::from_client(&client);
    let activation = PackedSlabs::new(
        &client,
        PackedSlabsSpec {
            format: PackedFormat::Nvfp4,
            operand: MatmulOperand::Lhs,
            slabs: 1,
            rows: m,
            k,
            seed: 1,
        },
    );
    let routine = CubeStagedMatmulRoutine;
    let mut ran = 0;
    for n in [48, 44] {
        let weight: Vec<bf16> = (0..k * n)
            .map(|i| bf16::from_f32(((i * 7919 % 2003) as f32 / 1001.5) - 1.))
            .collect();
        let expected: Vec<f32> = (0..m * n)
            .map(|at| {
                let (row, column) = (at / n, at % n);
                (0..k)
                    .map(|kk| activation.cell(0, kk, row) * weight[kk * n + column].to_f32())
                    .sum()
            })
            .collect();
        for settings in routine
            .packed_settings(&hw)
            .into_iter()
            .chain(routine.packed_one_step_settings())
        {
            let (out, bound) = PlainTensor::empty_contiguous(
                &client,
                Shape::from(vec![m, n]),
                bf16::elem_type_native(),
            );
            let launched = routine.launch_into(
                &hw,
                &client,
                StoredTensor::Packed(activation.matrix()),
                StoredTensor::Plain(upload(&client, &[k, n], &weight)),
                bound,
                &settings,
            );
            if launched.is_err() {
                continue;
            }
            ran += 1;
            let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
            for (at, &want) in expected.iter().enumerate() {
                let have = got.get_f32(&[at / n, at % n]);
                assert!(
                    (have - want).abs() <= BF16_TOLERANCE * (1. + want.abs()),
                    "{settings}, {n} columns: element {at} is {have}, want {want}"
                );
            }
        }
    }
    if ran == 0 {
        TestOutcome::Validated(ValidationResult::Skipped(
            "no staged setting prepares a packed activation on this device".to_string(),
        ))
        .enforce();
    }
}

/// NVFP4 activations against an NVFP4 weight, Qwen3.8-27B's projections at a prompt's rows: every
/// setting that prepares on this device answers the dequantized product. On an RTX PRO 6000 every
/// setting whose planes hold a grid of block-scaled fragments, rather than one, answered zero: the
/// grid's fragments were loaded into copies and contracted unwritten, and the model's prompt was
/// lost. Skipped on a device where none prepares.
#[test]
fn a_packed_activation_against_a_packed_weight_answers_at_every_setting() {
    let (m, n, k) = (512, 128, 5120);
    let client = client();
    let hw = Hardware::from_client(&client);
    let slabs = |operand, rows, seed| {
        PackedSlabs::new(
            &client,
            PackedSlabsSpec {
                format: PackedFormat::Nvfp4,
                operand,
                slabs: 1,
                rows,
                k,
                seed,
            },
        )
    };
    let (activation, weight) = (
        slabs(MatmulOperand::Lhs, m, 1),
        slabs(MatmulOperand::Rhs, n, 2),
    );
    let expected: Vec<f32> = (0..m * n)
        .map(|at| {
            let (row, column) = (at / n, at % n);
            (0..k)
                .map(|kk| activation.cell(0, kk, row) * weight.cell(0, kk, column))
                .sum()
        })
        .collect();
    let routine = CubeStagedMatmulRoutine;
    let mut ran = 0;
    let mut wrong = Vec::new();
    for settings in routine
        .packed_settings(&hw)
        .into_iter()
        .chain(routine.packed_one_step_settings())
        .chain(routine.block_scaled_settings(&hw))
    {
        let (out, bound) = PlainTensor::empty_contiguous(
            &client,
            Shape::from(vec![m, n]),
            bf16::elem_type_native(),
        );
        let launched = routine.launch_into(
            &hw,
            &client,
            StoredTensor::Packed(activation.matrix()),
            StoredTensor::Packed(weight.matrix()),
            bound,
            &settings,
        );
        if launched.is_err() {
            continue;
        }
        ran += 1;
        let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
        let off = expected
            .iter()
            .enumerate()
            .filter(|&(at, &want)| {
                (got.get_f32(&[at / n, at % n]) - want).abs() > BF16_TOLERANCE * (1. + want.abs())
            })
            .count();
        if off > 0 {
            wrong.push(format!("{settings}: {off} wrong"));
        }
    }
    if ran == 0 {
        TestOutcome::Validated(ValidationResult::Skipped(
            "no staged setting prepares two packed inputs on this device".to_string(),
        ))
        .enforce();
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
