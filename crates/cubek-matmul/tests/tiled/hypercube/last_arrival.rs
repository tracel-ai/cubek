//! The hypercube last-arrival routine: every tile's steps of `K` laid end to end and shared between
//! a fixed grid of cubes, the last cube to arrive at a tile writing it.

use cubek_matmul::kernels::HypercubeLastArrivalMatmulRoutine;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{Declines, Product};
use crate::tiled::ByRoutine;

#[test]
fn every_last_arrival_setting_matches_the_host() {
    let last_arrival = ByRoutine::new(HypercubeLastArrivalMatmulRoutine, Declines::AsItStates);
    last_arrival.check(last_arrival.at_every_setting(&Product::many_rows()));
}
