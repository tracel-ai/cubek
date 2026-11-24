//! FP8-row activations against an MXFP4 weight with row-factored scales, W4A8 with its scales a
//! row: the activation's `e4m3` values contract as they are stored, the weight's codes decode
//! into `e4m3` under their block's offset as the fill stages them, and both rows' scales come out
//! of the sum, applied once to its output.

use cubecl::ir::{ElemType, FloatKind};
use cubecl::quant::scheme::QuantValue;
use cubek_matmul::kernels::{
    CubeStagedAsyncCopyMatmulRoutine, CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings,
    HypercubeLastArrivalMatmulRoutine,
};
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::Declines;
use crate::tiled::ByRoutine;
use crate::tiled::packed::{PackedFormat, PackedProduct, Stored};

/// The products every routine is held to, FP8-row activations of `value` against `format`: a
/// model's projection at a prompt's rows, and rows, columns and a depth no box divides, a
/// prompt's ragged last chunk, `k` whole words of the format's block scales.
fn products_of(value: QuantValue, format: PackedFormat) -> Vec<PackedProduct> {
    let w4a8 = PackedProduct::of(
        Stored::Fp8Row { value, seed: 1 },
        Stored::Packed { format, seed: 2 },
    );
    let ragged_k = match format {
        PackedFormat::E2m1Blocks64RowFactored => 1280,
        _ => 1408,
    };
    vec![
        w4a8.clone().m(512).n(128).k(2048),
        w4a8.clone().m(33).n(96).k(1024),
        w4a8.m(300).n(44).k(ragged_k),
    ]
}

/// The products of the routine tests: `e4m3` activations against MXFP4 with row-factored scales,
/// plain MXFP8, and `e2m1` under blocks of 64, and `e5m2` ones against MXFP4 with row-factored
/// scales.
fn products() -> Vec<PackedProduct> {
    [
        (QuantValue::E4M3, PackedFormat::Mxfp4RowFactored),
        (QuantValue::E4M3, PackedFormat::Mxfp8),
        (QuantValue::E4M3, PackedFormat::E2m1Blocks64RowFactored),
        (QuantValue::E5M2, PackedFormat::Mxfp4RowFactored),
    ]
    .into_iter()
    .flat_map(|(value, format)| products_of(value, format))
    .collect()
}

/// Every setting of the units' staged routine against each product, each launched apart. A device
/// whose manual MMA multiplies `e4m3` answers at some of them; one whose does not declines them
/// all, which skips, as one without `e5m2`'s declines those products.
#[test]
fn an_fp8_row_activation_against_a_row_factored_mxfp4_weight_answers_on_the_units() {
    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    let settings = [
        CubeStagedMatmulRoutine.packed_settings(staged.hw()),
        vec![CubeStagedMatmulRoutineSettings::DEFAULT],
    ]
    .concat();
    let cases = staged.at(settings, &products());
    let e4m3 = ElemType::Float(FloatKind::E4M3);
    match staged
        .hw()
        .manual_mma(e4m3, e4m3, ElemType::Float(FloatKind::F32))
    {
        true => staged.check_some(cases),
        false => staged.check(cases),
    }
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
