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

/// A projection to fewer columns than a cube of planes reads, a gated delta net's per-head gates
/// (`5120` to `48`): its rhs laid along `k`, and stored in the device's tiles, which a plane past
/// the last run of columns would read inside one tile, where nothing masks it. Skipped in tiles
/// where the device stores no float input in them.
#[test]
fn every_plane_shuffle_setting_matches_the_host_on_a_projection_narrower_than_a_cube() {
    use cubek_matmul::tiled::storage::{StoredValues, stored_tile};
    use cubek_microkernel::matmul::MatmulOperand;
    use cubek_std::matmul_axes::{K, N};
    use cubek_test_utils::{TestOutcome, ValidationResult};

    use crate::matmul::{F16, RhsTiles};

    let shuffle = ByRoutine::new(PlaneShuffleMatmulRoutine, Declines::AsItStates);
    let heads = Product::of(F16).m(1).n(48).k(5120);
    shuffle.check(shuffle.at_every_setting(&[heads.clone().rhs_along_k()]));
    match stored_tile(
        shuffle.hw(),
        MatmulOperand::Rhs,
        StoredValues::Elements(F16),
        F16,
    ) {
        Some(tile) => shuffle.check(shuffle.at_every_setting(&[heads.rhs_in(RhsTiles {
            n: tile.extent(N),
            k: tile.extent(K),
        })])),
        None => TestOutcome::Validated(ValidationResult::Skipped(
            "the device stores no float input in tiles".to_string(),
        ))
        .enforce(),
    }
}
