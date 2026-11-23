//! FP8-row activations against an MXFP4 weight with row-factored scales, W4A8 with its scales a
//! row: the activation's `e4m3` values contract as they are stored, the weight's codes decode
//! into `e4m3` under their block's offset as the fill stages them, and both rows' scales come out
//! of the sum, applied once to its output.

use cubek_matmul::kernels::{
    CubeStagedAsyncCopyMatmulRoutine, CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings,
    HypercubeLastArrivalMatmulRoutine,
};
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::Declines;
use crate::tiled::ByRoutine;
use crate::tiled::packed::{PackedFormat, PackedProduct, Stored};

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

/// Every setting of the units' staged routine against a packed weight, each launched apart. A
/// device whose manual MMA multiplies `e4m3` answers at some of them; one whose does not
/// declines them all, which skips.
#[test]
fn an_fp8_row_activation_against_a_row_factored_mxfp4_weight_answers_on_the_units() {
    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    let settings = [
        CubeStagedMatmulRoutine.packed_settings(staged.hw()),
        vec![CubeStagedMatmulRoutineSettings::DEFAULT],
    ]
    .concat();
    let cases = staged.at(settings, &products());
    match staged.hw().e4m3_mma() {
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
