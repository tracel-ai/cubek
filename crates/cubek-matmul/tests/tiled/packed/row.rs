//! FP8-row activations against an MXFP4 weight with row-factored scales, W4A8 with its scales a
//! row: the activation's
//! `e4m3` values contract as they are stored, the weight's codes decode into `e4m3` under their
//! block's offset as the fill stages them, and both rows' scales come out of the sum, applied
//! once to its output.

use cubecl::ir::{ElemType, FloatKind};
use cubecl::quant::scheme::QuantScheme;
use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::Shape;
use cubek_matmul::kernels::tensor::{
    PlainTensor, ROW_SCALED_UNREAD, ROW_SCALED_UNSERVED, RowScaledTensor, StoredTensor,
};
use cubek_matmul::kernels::{
    CubeStagedAsyncCopyMatmulRoutine, CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings,
    HypercubeLastArrivalMatmulRoutine, MATMUL_PACKED_FLOAT_ELEMENT_UNVERIFIED, MatmulTensors,
};
use cubek_microkernel::matmul::{MatmulOperand, MatmulRowsKind};
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::Declines;
use crate::tiled::packed::{PackedFormat, PackedProduct, PackedSlabs, PackedSlabsSpec, Stored};
use crate::tiled::{ByRoutine, upload};

/// The products every routine is held to: a model's projection at a prompt's rows, and rows,
/// columns and a depth no box divides, a prompt's ragged last chunk.
fn products() -> Vec<PackedProduct> {
    let w4a8 = PackedProduct::of(
        Stored::Fp8Row { seed: 1 },
        Stored::Packed {
            format: PackedFormat::Mxfp4RowFactored,
            seed: 2,
        },
    );
    vec![
        w4a8.clone().m(512).n(128).k(2048),
        w4a8.clone().m(33).n(96).k(1024),
        w4a8.m(300).n(44).k(1408),
    ]
}

/// Every setting of the units' staged routine against a packed weight, each launched apart, so a
/// device whose instructions take no `e4m3` product declines them all, which skips.
#[test]
fn an_fp8_row_activation_against_a_row_factored_mxfp4_weight_answers_on_the_units() {
    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    let settings = [
        CubeStagedMatmulRoutine.packed_settings(staged.hw()),
        vec![CubeStagedMatmulRoutineSettings::DEFAULT],
    ]
    .concat();
    staged.check(staged.at(settings, &products()));
}

/// The same on the copy engine's staged routine, on a device that has one.
#[test]
fn an_fp8_row_activation_against_a_row_factored_mxfp4_weight_answers_on_the_copy_engine() {
    let async_copy = ByRoutine::new(CubeStagedAsyncCopyMatmulRoutine, Declines::AsItStates);
    async_copy.check(async_copy.at(
        CubeStagedAsyncCopyMatmulRoutine.packed_settings(async_copy.hw()),
        &products(),
    ));
}

/// The same on the hypercube, its cubes sharing each box along `K`: each run's part is parked
/// under both rows' scales, which the sum took out of the decode, and the last run adds them up.
#[test]
fn an_fp8_row_activation_against_a_row_factored_mxfp4_weight_answers_on_the_hypercube() {
    let hypercube = ByRoutine::new(HypercubeLastArrivalMatmulRoutine, Declines::AsItStates);
    hypercube.check(hypercube.at(
        HypercubeLastArrivalMatmulRoutine.packed_settings(hypercube.hw()),
        &products(),
    ));
}

/// The weight the products above read spreads its block scales further below its rows' largest
/// than the decode into `e4m3` takes exactly, and the load step counts those blocks.
#[test]
fn factoring_mxfp4_scales_by_row_counts_the_blocks_the_decode_rounds() {
    let client = cubecl::test_device().client();
    let weight = PackedSlabs::new(
        &client,
        PackedSlabsSpec {
            format: PackedFormat::Mxfp4RowFactored,
            operand: MatmulOperand::Rhs,
            slabs: 1,
            rows: 96,
            k: 1024,
            seed: 2,
        },
    );
    let lossy = weight
        .lossy_blocks(&client)
        .expect("an MXFP4 weight with row-factored scales");
    assert!(lossy > 0, "no block of the spread weight was counted");
    assert!(lossy < 96 * 1024 / 32, "every block was counted");
}

/// `rows × k` `e4m3` values under a scale a row, every one zero.
fn row_scaled(rows: usize, k: usize) -> StoredTensor {
    let client = cubecl::test_device().client();
    let values = e4m3_zeros(rows, k);
    let scales = upload(&client, &[rows, 1], &vec![1f32; rows]);
    StoredTensor::RowScaled(
        RowScaledTensor::new(values, scales, QuantScheme::fp8_per_row()).expect("FP8-row"),
    )
}

fn e4m3_zeros(rows: usize, k: usize) -> PlainTensor {
    let client = cubecl::test_device().client();
    PlainTensor::of_handle(TensorHandle::new_contiguous(
        Shape::from(vec![rows, k]),
        client.create_from_slice(&vec![0u8; rows * k]),
        ElemType::Float(FloatKind::E4M3),
    ))
}

fn row_factored_weight(n: usize, k: usize) -> StoredTensor {
    let client = cubecl::test_device().client();
    let spec = PackedSlabsSpec {
        format: PackedFormat::Mxfp4RowFactored,
        operand: MatmulOperand::Rhs,
        slabs: 1,
        rows: n,
        k,
        seed: 3,
    };
    StoredTensor::Packed(PackedSlabs::new(&client, spec).matrix())
}

/// `e4m3` values beside a packed input are read only under their scales a row: plain, they are
/// declined as a float operand of an element the packed routines do not take, and a row-scaled
/// tensor is declined anywhere but as the lhs against a packed rhs, the one place its scales come
/// out of the sum.
#[test]
fn e4m3_values_beside_a_packed_input_are_read_only_as_a_row_scaled_lhs() {
    let client = cubecl::test_device().client();
    let (m, n, k) = (8, 32, 256);
    let declined = |lhs, rhs| {
        MatmulTensors::new(&client, lhs, rhs, MatmulRowsKind::Tiled)
            .err()
            .map(|why| format!("{why:?}"))
    };
    let named = |decline: cubecl::tile::launch::Decline| Some(format!("{decline:?}"));

    assert_eq!(
        declined(
            StoredTensor::Plain(e4m3_zeros(m, k)),
            row_factored_weight(n, k)
        ),
        named(MATMUL_PACKED_FLOAT_ELEMENT_UNVERIFIED)
    );
    assert_eq!(declined(row_scaled(m, k), row_factored_weight(n, k)), None);
    let bf16_weight = StoredTensor::Plain(upload(&client, &[k, n], &vec![half::bf16::ZERO; k * n]));
    assert_eq!(
        declined(row_scaled(m, k), bf16_weight),
        named(ROW_SCALED_UNSERVED)
    );
    assert_eq!(
        declined(row_factored_weight(m, k), row_scaled(k, n)),
        named(ROW_SCALED_UNSERVED)
    );
}

/// A row-scaled tensor holds 8-bit float values in dense rows under one `f32` a row, its scheme's
/// only level, and refuses anything else.
#[test]
fn a_row_scaled_tensor_refuses_what_it_does_not_hold() {
    let client = cubecl::test_device().client();
    let (rows, k) = (4, 64);
    let scales = || upload(&client, &[rows, 1], &vec![1f32; rows]);
    let new = |values, scales, scheme| RowScaledTensor::new(values, scales, scheme).err();

    assert_eq!(
        new(e4m3_zeros(rows, k), scales(), QuantScheme::fp8_per_row()),
        None
    );
    // A per-tensor level in place of the row's.
    let per_tensor = QuantScheme::fp8_per_row().per_tensor(cubecl::quant::scheme::ScaleDtype::F32);
    assert_eq!(
        new(e4m3_zeros(rows, k), scales(), per_tensor),
        Some(ROW_SCALED_UNREAD)
    );
    // One scale short.
    let short = upload(&client, &[rows - 1, 1], &vec![1f32; rows - 1]);
    assert_eq!(
        new(e4m3_zeros(rows, k), short, QuantScheme::fp8_per_row()),
        Some(ROW_SCALED_UNREAD)
    );
    // Values of another element than the scheme's.
    let halves = upload(&client, &[rows, k], &vec![half::bf16::ZERO; rows * k]);
    assert_eq!(
        new(halves, scales(), QuantScheme::fp8_per_row()),
        Some(ROW_SCALED_UNREAD)
    );
}
