//! The cube shuffle routine: every cube owns every row of a run of the output's columns, its planes
//! splitting the contraction between them. It holds at most a decode step's rows.

use cubek_matmul::kernels::CubeShuffleMatmulRoutine;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{Declines, Product};
use crate::tiled::ByRoutine;

#[test]
fn every_cube_shuffle_setting_matches_the_host() {
    let shuffle = ByRoutine::new(CubeShuffleMatmulRoutine, Declines::AsItStates);
    shuffle.check(shuffle.at_every_setting(&Product::few_rows()));
}

#[test]
fn every_cube_shuffle_setting_sums_i32_exactly() {
    let shuffle = ByRoutine::new(CubeShuffleMatmulRoutine, Declines::AsItStates);
    shuffle.check(shuffle.at_every_setting(&Product::integers(4)));
}
