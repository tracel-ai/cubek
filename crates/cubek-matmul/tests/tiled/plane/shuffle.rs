//! The plane shuffle routine: every plane owns every row of a run of the output's columns, its
//! units shuffling their partial sums together. It holds at most a decode step's rows.

use cubek_matmul::kernels::PlaneShuffleMatmulRoutine;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{Declines, I32, Product};
use crate::tiled::ByRoutine;

#[test]
fn every_plane_shuffle_setting_matches_the_host() {
    let shuffle = ByRoutine::new(PlaneShuffleMatmulRoutine, Declines::AsItStates);
    shuffle.check(shuffle.at_every_setting(&Product::few_rows()));
}

/// The second product's depth no line wider than one divides, so a col rhs binds one element at a
/// time.
#[test]
fn every_plane_shuffle_setting_sums_i32_exactly() {
    let shuffle = ByRoutine::new(PlaneShuffleMatmulRoutine, Declines::AsItStates);
    let prime_depth = Product::of(I32).m(4).n(24).k(41);
    let products = [
        Product::integers(4),
        vec![prime_depth.clone(), prime_depth.rhs_along_k()],
    ]
    .concat();
    shuffle.check(shuffle.at_every_setting(&products));
}
