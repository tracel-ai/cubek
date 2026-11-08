//! A packed weight read with each owner of a run of columns walking `K` rotated: the plane and the
//! cube shuffle at their rotated packed leads answer the product they answer in order.

use super::indexed::upload;
use super::packed::{PackedFormat, PackedSlabs, PackedSlabsSpec};
use cubecl::prelude::*;
use cubecl::zspace::Shape;
use cubek_matmul::kernels::tensor::{PlainTensor, StoredTensor};
use cubek_matmul::kernels::{
    CubeShuffleMatmulRoutine, CubeShuffleMatmulRoutineSettings, MatmulRoutine,
    PlaneShuffleMatmulRoutine, PlaneShuffleMatmulRoutineSettings,
};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_microkernel::matmul::cube::CubeShuffleMicrokernelSettings;
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, TestOutcome, ValidationResult, client};
use half::f16;

/// Loose enough for an `f16` output.
const HALF_TOLERANCE: f32 = 1e-2;

/// One row against weights of many runs of columns, so the owners start at different steps, and
/// a `K` of many steps, so a rotation wraps: each shuffle's packed lead, in order and rotated,
/// answers the dequantized product wherever it prepares. Skipped on a device where none prepares.
#[test]
fn a_rotated_walk_down_k_answers_the_product_in_order() {
    let client = client();
    let hw = Hardware::from_client(&client);
    let mut ran = 0;
    for format in [PackedFormat::Q8, PackedFormat::Q4, PackedFormat::Nvfp4] {
        for (m, n, k) in [(1, 256, 2048), (3, 96, 1024)] {
            let weight = PackedSlabs::new(
                &client,
                PackedSlabsSpec {
                    format,
                    operand: MatmulOperand::Rhs,
                    slabs: 1,
                    rows: n,
                    k,
                    seed: 3,
                },
            );
            let lhs: Vec<f16> = (0..m * k)
                .map(|i| f16::from_f32(((i * 7919 % 2003) as f32 / 1001.5) - 1.))
                .collect();
            let expected: Vec<f32> = (0..m * n)
                .map(|at| {
                    let (row, column) = (at / n, at % n);
                    (0..k)
                        .map(|kk| lhs[row * k + kk].to_f32() * weight.cell(0, kk, column))
                        .sum()
                })
                .collect();
            let check = |name: String, launch: &dyn Fn(StoredTensor, PlainTensor) -> bool| {
                let (out, bound) = PlainTensor::empty_contiguous(
                    &client,
                    Shape::from(vec![m, n]),
                    f16::elem_type_native(),
                );
                let lhs = StoredTensor::Plain(upload(&client, &[m, k], &lhs));
                if !launch(lhs, bound) {
                    return false;
                }
                let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
                for (at, &want) in expected.iter().enumerate() {
                    let have = got.get_f32(&[at / n, at % n]);
                    assert!(
                        (have - want).abs() <= HALF_TOLERANCE * (1. + want.abs()),
                        "{name}, {format:?} {m}x{n}x{k}: element {at} is {have}, want {want}"
                    );
                }
                true
            };
            let planes = [
                PlaneShuffleMatmulRoutineSettings::PACKED,
                PlaneShuffleMatmulRoutineSettings::PACKED_ROTATED,
            ];
            for plane in planes {
                ran += usize::from(check(plane.to_string(), &|lhs, out| {
                    PlaneShuffleMatmulRoutine
                        .launch_into(
                            &hw,
                            &client,
                            lhs,
                            StoredTensor::Packed(weight.matrix()),
                            out,
                            &plane,
                        )
                        .is_ok()
                }));
            }
            let cubes = [
                CubeShuffleMicrokernelSettings::PACKED,
                CubeShuffleMicrokernelSettings::PACKED_ROTATED,
            ];
            for microkernel in cubes {
                let cube = CubeShuffleMatmulRoutineSettings { microkernel };
                ran += usize::from(check(cube.to_string(), &|lhs, out| {
                    CubeShuffleMatmulRoutine
                        .launch_into(
                            &hw,
                            &client,
                            lhs,
                            StoredTensor::Packed(weight.matrix()),
                            out,
                            &cube,
                        )
                        .is_ok()
                }));
            }
        }
    }
    if ran == 0 {
        TestOutcome::Validated(ValidationResult::Skipped(
            "no rotated shuffle setting prepares a packed weight on this device".to_string(),
        ))
        .enforce();
    }
}
