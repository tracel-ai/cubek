//! The cube shuffle's split of `K` against an NVFP4 weight, as a decode step runs a projection:
//! the weight read a scale block at a time, its scales a byte each, every plane of a cube one turn
//! of `K`, the cubes walking runs of the boxes of columns.

use super::indexed::upload;
use super::packed::{PackedFormat, PackedSlabs, PackedSlabsSpec};
use cubecl::prelude::*;
use cubecl::zspace::Shape;
use cubek_matmul::kernels::tensor::{PlainTensor, StoredTensor};
use cubek_matmul::kernels::{
    CubeShuffleMatmulRoutine, CubeShuffleMatmulRoutineSettings, MatmulRoutine,
};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, TestOutcome, ValidationResult, client};
use half::bf16;

/// Loose enough for a `bf16` output.
const BF16_TOLERANCE: f32 = 1e-2;

/// One, three and eight rows against NVFP4 weights whose `K` takes three to twenty-four planes,
/// their columns a whole number of eight-column boxes or not: every product answers the
/// dequantized one, its planes taking `K` in runs or in turns
/// ([`PACKED_SPLIT_K_ONE_TRANSACTION`](CubeShuffleMatmulRoutineSettings::PACKED_SPLIT_K_ONE_TRANSACTION)),
/// where a turn the planes do not fill is the last. Before a read could take two words, the words
/// were bound a read a stored word, so a read of a block stepped its rows twice as far and read
/// past the weight. Skipped on a device where the settings do not prepare, one committing to no
/// plane width.
#[test]
fn the_split_k_answers_a_decode_step() {
    for settings in [
        CubeShuffleMatmulRoutineSettings::PACKED_SPLIT_K,
        CubeShuffleMatmulRoutineSettings::PACKED_SPLIT_K_ONE_TRANSACTION,
    ] {
        split_k_answers_a_decode_step(&settings);
    }
}

fn split_k_answers_a_decode_step(settings: &CubeShuffleMatmulRoutineSettings) {
    let client = client();
    let hw = Hardware::from_client(&client);
    let (mut ran, mut declined) = (0, Vec::new());
    for (m, n, k) in [
        (1, 64, 2048),
        (1, 44, 4096),
        (3, 40, 1536),
        (8, 72, 4096),
        (1, 48, 12288),
    ] {
        let weight = PackedSlabs::new(
            &client,
            PackedSlabsSpec {
                format: PackedFormat::Nvfp4,
                operand: MatmulOperand::Rhs,
                slabs: 1,
                rows: n,
                k,
                seed: 3,
            },
        );
        let x: Vec<bf16> = (0..m * k)
            .map(|i| bf16::from_f32(((i * 7919 % 2003) as f32 / 1001.5) - 1.))
            .collect();
        let expected: Vec<f32> = (0..m * n)
            .map(|at| {
                let (row, column) = (at / n, at % n);
                (0..k)
                    .map(|kk| x[row * k + kk].to_f32() * weight.cell(0, kk, column))
                    .sum()
            })
            .collect();
        let (out, bound) = PlainTensor::empty_contiguous(
            &client,
            Shape::from(vec![m, n]),
            bf16::elem_type_native(),
        );
        let launched = CubeShuffleMatmulRoutine.launch_into(
            &hw,
            &client,
            StoredTensor::Plain(upload(&client, &[m, k], &x)),
            StoredTensor::Packed(weight.matrix()),
            bound,
            settings,
        );
        if let Err(why) = launched {
            declined.push(format!("m {m}, n {n}, k {k}: {why:?}"));
            continue;
        }
        ran += 1;
        let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
        for (at, &want) in expected.iter().enumerate() {
            let have = got.get_f32(&[at / n, at % n]);
            assert!(
                (have - want).abs() <= BF16_TOLERANCE * (1. + want.abs()),
                "{settings}, m {m}, n {n}, k {k}: element {at} is {have}, want {want}"
            );
        }
    }
    // A device it prepares on prepares every shape here.
    assert!(
        ran == 0 || declined.is_empty(),
        "{settings} declined: {declined:?}"
    );
    if ran == 0 {
        TestOutcome::Validated(ValidationResult::Skipped(
            "the split of K does not prepare on this device".to_string(),
        ))
        .enforce();
    }
}
