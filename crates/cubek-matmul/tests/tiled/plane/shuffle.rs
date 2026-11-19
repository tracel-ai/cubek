//! The plane shuffle routine: every plane owns every row of a run of the output's columns, its
//! units shuffling their partial sums together. It holds at most a decode step's rows.

use cubek_matmul::kernels::PlaneShuffleMatmulRoutine;
use cubek_matmul::tiled::storage::{StoredValues, stored_tile};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_std::matmul_axes::{K, N};
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{BF16, Declines, I32, Product, RhsTiles, TileRows};
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

/// A bf16 decode projection against a weight stored as a load stores it on the test device, each
/// tile row split at the widest read: every setting either declines it or matches the host.
/// Skipped where the device stores no bf16 weight in tiles.
#[test]
fn every_plane_shuffle_setting_reads_a_weight_stored_in_tiles() {
    let shuffle = ByRoutine::new(PlaneShuffleMatmulRoutine, Declines::AsItStates);
    let values = StoredValues::Elements(BF16);
    let Some(tile) = stored_tile(shuffle.hw(), MatmulOperand::Rhs, values, BF16) else {
        return;
    };
    let tiles = RhsTiles {
        n: tile.extent(N),
        k: tile.extent(K),
        rows: TileRows::InReads,
    };
    let products = [48, 96].map(|n| Product::of(BF16).m(1).n(n).k(5120).rhs_in(tiles));
    shuffle.check(shuffle.at_every_setting(&products));
}
