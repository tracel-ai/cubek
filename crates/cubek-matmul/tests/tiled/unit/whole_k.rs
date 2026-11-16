//! The unit whole-`K` routine: every unit owns every row of a run of the output's columns and walks
//! all of `K` alone. It holds at most a decode step's rows.

use cubek_matmul::kernels::UnitWholeKMatmulRoutine;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{Declines, Product};
use crate::tiled::ByRoutine;

#[test]
fn every_unit_whole_k_setting_matches_the_host() {
    let whole_k = ByRoutine::new(UnitWholeKMatmulRoutine, Declines::AsItStates);
    whole_k.check(whole_k.at_every_setting(&Product::few_rows()));
}

#[test]
fn every_unit_whole_k_setting_sums_i32_exactly() {
    let whole_k = ByRoutine::new(UnitWholeKMatmulRoutine, Declines::AsItStates);
    whole_k.check(whole_k.at_every_setting(&Product::integers(4)));
}
