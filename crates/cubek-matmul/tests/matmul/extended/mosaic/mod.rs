mod layouts;

use crate::matmul::test_matmul_strategy;
use cubecl::{Runtime, frontend::CubePrimitive, ir::AddressType, zspace::shape};
use cubek_matmul::{
    definition::{MatmulElems, MatmulGlobalElems, MatmulProblem},
    launch::Strategy,
    routines::mosaic::MosaicStrategy,
};
use cubek_std::MatrixLayout;

type TestRuntime = cubecl::TestRuntime;

fn elems_f32() -> MatmulGlobalElems {
    MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems()
}

/// Run `batch × (m, k) @ (k, n)` through Mosaic with the given square sub-tile edge.
fn check(batch: usize, m: usize, n: usize, k: usize, tile_size: usize) {
    let client = TestRuntime::client(&Default::default());
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![batch],
        shape![batch],
        MatrixLayout::RowMajor,
        MatrixLayout::ColMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        elems_f32(),
        AddressType::U32,
    );
    test_matmul_strategy(
        client,
        problem,
        Strategy::Mosaic(MosaicStrategy { tile_size }),
    );
}

#[test]
fn very_small_square() {
    check(1, 8, 8, 8, 4);
}

#[test]
fn small_square() {
    check(1, 32, 32, 64, 8);
}

#[test]
fn rectangular() {
    check(1, 48, 32, 64, 16);
}

#[test]
fn single_tile() {
    check(1, 8, 8, 8, 8);
}

#[test]
fn many_tiles_default_size() {
    check(1, 64, 64, 128, MosaicStrategy::default().tile_size);
}

#[test]
fn batched_small() {
    check(4, 16, 16, 32, 8);
}

#[test]
fn batched_rectangular() {
    check(3, 32, 48, 64, 16);
}
