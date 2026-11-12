//! The cube staged routines: a cube's planes contract the stages the cube fills, on the matrix
//! instruction the device offers. They differ in who fills the stages (the units, the copy engine,
//! the tensor memory accelerator, a plane set aside to drive it) and in who stores the box, and
//! each declines what its engine does not run.

use cubek_matmul::kernels::{
    CubeStagedAsyncCopyMatmulRoutine, CubeStagedMatmulRoutine, CubeStagedTmaFillerMatmulRoutine,
    CubeStagedTmaMatmulRoutine, CubeStagedTmaStoredMatmulRoutine, CubeStagedWgmmaMatmulRoutine,
};
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{Declines, F16, F32, Product, RhsTiles};
use crate::tiled::ByRoutine;

#[test]
fn every_cube_staged_setting_matches_the_host() {
    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    staged.check(staged.at_every_setting(&Product::many_rows()));
}

#[test]
fn every_cube_staged_setting_sums_i32_exactly() {
    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    staged.check(staged.at_every_setting(&Product::integers(32)));
}

/// The weight stored in `32 × 32` tiles beside a row-major activation: the same product as the
/// row-major one, each tile read through the binding's tiling.
#[test]
fn the_cube_staged_lead_reads_a_weight_stored_in_tiles() {
    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    let products = [F16, F32].map(|element| {
        Product::of(element)
            .m(128)
            .n(256)
            .k(384)
            .rhs_in(RhsTiles { n: 32, k: 32 })
    });
    staged.check(staged.at_its_lead(&products));
}

/// `f16` inputs into an `f32` output: the sum is cast to the output's element as it drains.
#[test]
fn the_cube_staged_lead_writes_a_wider_output() {
    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    let product = Product::of(F16).m(64).n(48).k(96).writing(F32);
    staged.check(staged.at_its_lead(&[product]));
}

/// The stages filled by the copy engine, where the device has one.
#[test]
fn every_async_copy_setting_matches_the_host() {
    let staged = ByRoutine::new(CubeStagedAsyncCopyMatmulRoutine, Declines::AsItStates);
    staged.check(staged.at_every_setting(&Product::many_rows()));
}

/// The stages filled by the tensor memory accelerator, where the device has one.
#[test]
fn every_tma_setting_matches_the_host() {
    let staged = ByRoutine::new(CubeStagedTmaMatmulRoutine, Declines::AsItStates);
    staged.check(staged.at_every_setting(&Product::many_rows()));
}

/// The stages filled by a plane set aside to drive the tensor memory accelerator.
#[test]
fn every_tma_filler_setting_matches_the_host() {
    let staged = ByRoutine::new(CubeStagedTmaFillerMatmulRoutine, Declines::AsItStates);
    staged.check(staged.at_every_setting(&Product::many_rows()));
}

/// The box stored by the tensor memory accelerator too.
#[test]
fn every_tma_stored_setting_matches_the_host() {
    let staged = ByRoutine::new(CubeStagedTmaStoredMatmulRoutine, Declines::AsItStates);
    staged.check(staged.at_every_setting(&Product::many_rows()));
}

/// The stages contracted by warpgroup instructions, where the device has them.
#[test]
fn every_wgmma_setting_matches_the_host() {
    let staged = ByRoutine::new(CubeStagedWgmmaMatmulRoutine, Declines::AsItStates);
    staged.check(staged.at_every_setting(&Product::many_rows()));
}
