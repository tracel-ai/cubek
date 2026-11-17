//! A packed activation, as W4A4 runs a projection: against a float weight, and against a packed
//! one, through the cube staged routines that stage a packed input for a matrix instruction.

use cubek_matmul::kernels::{
    CubeStagedAsyncCopyMatmulRoutine, CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings,
    CubeStagedTmaFillerMatmulRoutine, CubeStagedTmaMatmulRoutine, CubeStagedTmaStoredMatmulRoutine,
};
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::Declines;
use crate::tiled::ByRoutine;
use crate::tiled::packed::{PackedProduct, Stored};

/// NVFP4 activations against a `bf16` weight: a gated-delta projection 48 columns wide at a
/// prompt's rows, and the same 44 wide. A box 32 columns wide reaches past the weight's columns,
/// which binds it at a width dividing them; on an RTX PRO 6000 a box bound eight wide past 48,
/// which no mask stated, refused to bind and the model did not load.
#[test]
fn a_packed_activation_against_a_float_weight_answers_at_every_setting() {
    let projection = PackedProduct::of(Stored::Nvfp4 { seed: 1 }, Stored::Bf16)
        .m(512)
        .k(5120);
    let products = [projection.clone().n(48), projection.n(44)];
    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    let settings = [
        CubeStagedMatmulRoutine.packed_settings(staged.hw()),
        CubeStagedMatmulRoutine.packed_one_step_settings(),
    ]
    .concat();
    staged.check(staged.at(settings, &products));
}

/// NVFP4 activations against an NVFP4 weight, a model's projections at a prompt's rows, and at
/// rows and columns no box divides, a prompt's ragged last chunk: every box past the matrix's edge
/// reads zeros and writes nothing. On an RTX PRO 6000 every setting whose planes hold a grid of
/// block-scaled fragments, rather than one, answered zero: the grid's fragments were loaded into
/// copies and contracted unwritten, and the model's prompt was lost.
#[test]
fn a_packed_activation_against_a_packed_weight_answers_at_every_setting() {
    let both = PackedProduct::of(Stored::Nvfp4 { seed: 1 }, Stored::Nvfp4 { seed: 2 }).k(5120);
    let products = [
        both.clone().m(512).n(128),
        both.clone().m(33).n(96),
        both.m(300).n(96),
    ];

    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    let settings = [
        CubeStagedMatmulRoutine.packed_settings(staged.hw()),
        CubeStagedMatmulRoutine.packed_one_step_settings(),
        vec![CubeStagedMatmulRoutineSettings::DEFAULT],
    ]
    .concat();
    staged.check(staged.at(settings, &products));

    let async_copy = ByRoutine::new(CubeStagedAsyncCopyMatmulRoutine, Declines::AsItStates);
    let settings = [
        CubeStagedAsyncCopyMatmulRoutine.packed_settings(async_copy.hw()),
        CubeStagedAsyncCopyMatmulRoutine.block_scaled_settings(async_copy.hw()),
    ]
    .concat();
    async_copy.check(async_copy.at(settings, &products));

    let filler = ByRoutine::new(CubeStagedTmaFillerMatmulRoutine, Declines::AsItStates);
    filler.check(filler.at(
        CubeStagedTmaFillerMatmulRoutine.packed_settings(filler.hw()),
        &products,
    ));

    let tma = ByRoutine::new(CubeStagedTmaMatmulRoutine, Declines::AsItStates);
    tma.check(tma.at(
        CubeStagedTmaMatmulRoutine.block_scaled_settings(tma.hw()),
        &products,
    ));

    let stored = ByRoutine::new(CubeStagedTmaStoredMatmulRoutine, Declines::AsItStates);
    stored.check(stored.at(
        CubeStagedTmaStoredMatmulRoutine.block_scaled_settings(stored.hw()),
        &products,
    ));
}
