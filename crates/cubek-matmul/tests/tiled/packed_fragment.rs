//! An NVFP4 weight through the cube shuffle's tensor-core planes, as a decode runs a projection:
//! every fragment setting that prepares on this device, its rows along `m` or along `n`, answers
//! the dequantized product from one row to sixteen.

use super::indexed::upload;
use super::packed::{PackedFormat, PackedSlabs, PackedSlabsSpec};
use cubecl::prelude::*;
use cubecl::zspace::Shape;
use cubek_matmul::kernels::tensor::{PlainTensor, StoredTensor};
use cubek_matmul::kernels::{CubeShuffleMatmulRoutine, MatmulRoutine};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_microkernel::matmul::cube::CubeShufflePlaneSettings;
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, TestOutcome, ValidationResult, client};
use half::f16;

/// Loose enough for an `f16` output summed over a few hundred products.
const F16_TOLERANCE: f32 = 1e-2;

/// Every tensor-core setting of the cube shuffle that prepares an NVFP4 weight answers its
/// product, at rows a fragment holds whole and rows it pads. Skipped on a device offering none
/// of the instructions its planes decode into.
#[test]
fn every_fragment_setting_answers_an_nvfp4_weight() {
    let (n, k) = (128, 1024);
    let client = client();
    let hw = Hardware::from_client(&client);
    let weight = PackedSlabs::new(
        &client,
        PackedSlabsSpec {
            format: PackedFormat::Nvfp4,
            operand: MatmulOperand::Rhs,
            slabs: 1,
            rows: n,
            k,
            seed: 2,
        },
    );
    let routine = CubeShuffleMatmulRoutine;
    let settings: Vec<_> = routine
        .packed_settings(&hw)
        .into_iter()
        .filter(|settings| {
            matches!(
                settings.microkernel.plane,
                CubeShufflePlaneSettings::Fragment(_)
            )
        })
        .collect();
    let mut ran = 0;
    let mut wrong = Vec::new();
    for m in [1, 5, 8, 13, 16] {
        let activation: Vec<f16> = (0..m * k)
            .map(|i| f16::from_f32(((i * 7919 % 2003) as f32 / 1001.5) - 1.))
            .collect();
        let expected: Vec<f32> = (0..m * n)
            .map(|at| {
                let (row, column) = (at / n, at % n);
                (0..k)
                    .map(|kk| activation[row * k + kk].to_f32() * weight.cell(0, kk, column))
                    .sum()
            })
            .collect();
        for setting in &settings {
            let (out, bound) = PlainTensor::empty_contiguous(
                &client,
                Shape::from(vec![m, n]),
                f16::elem_type_native(),
            );
            let launched = routine.launch_into(
                &hw,
                &client,
                StoredTensor::Plain(upload(&client, &[m, k], &activation)),
                StoredTensor::Packed(weight.matrix()),
                bound,
                setting,
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
                    let have = got.get_f32(&[at / n, at % n]);
                    (have - want).abs() > F16_TOLERANCE * (1. + want.abs())
                })
                .count();
            if off > 0 {
                wrong.push(format!("{setting} at {m} rows: {off} of {} cells", m * n));
            }
        }
    }
    assert!(wrong.is_empty(), "wrong products:\n{}", wrong.join("\n"));
    if ran == 0 {
        TestOutcome::Validated(ValidationResult::Skipped(
            "no tensor-core setting of the cube shuffle prepares an NVFP4 weight here".to_string(),
        ))
        .enforce();
    }
    println!("{ran} fragment launches answered");
}
