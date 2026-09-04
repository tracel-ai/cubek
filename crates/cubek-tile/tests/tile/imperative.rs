//! The walk written by the kernel: the levels are loops, the stages are rings the kernel
//! allocates, and the leaf runs under the register block the kernel states. The reference every
//! routine's rewrite is measured against.
#![allow(non_snake_case)]

use cubecl::{Runtime, TestRuntime, prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput, TileInput, assert_equals_approx};
use cubek_tile::*;

use super::references;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// The software instruction the leaf runs: a 16-cell budget, no edge split, no lane fan-out.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// `c = a · b` over a K walk whose blocks of `a` and `b` are double-buffered in shared memory,
/// the leaf running the software instruction on the block's final tiles.
#[cube(launch)]
fn ring_matmul<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();

    // The cube's walk: one block of K per region, both operands staged for it.
    let walk = Walk::over(c.op_space(&a, &b));
    let mut ring = Ring::smem(&walk, &a, &b, StageStorage::Strided, depth);
    pipelined(walk, &mut ring, |slot, region| {
        let c_block = c.at(region);
        slot.consume(|a_s, b_s| {
            // The block's own grid of final tiles, each contracted by the leaf.
            for cell in Walk::over(c_block.op_space(a_s, b_s)) {
                let mut c_cell = c_block.at(&cell);
                c_cell.mma_with(
                    &a_s.at(&cell),
                    &b_s.at(&cell),
                    REGISTER_BLOCK,
                    Semiring::SUM_PROD,
                );
            }
        });
    });
}

fn check_ring_matmul(m: usize, n: usize, k: usize, block_k: usize, depth: usize) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tile = 4usize;
    let dtype = f32::elem_type_native();
    let space = Tiling::over(&[(M, m), (N, n), (K, k)])
        .level(|l| {
            l.walk(&[(M, m), (N, n), (K, block_k)]);
        })
        .level(|l| {
            l.walk(&[(M, tile), (N, tile), (K, tile)]);
        })
        .build();

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .tile(&[tile, tile])
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .tile(&[tile, tile])
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[tile, tile])
        .uniform(7, -100.0, 100.0);

    ring_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        a.arg(),
        b.arg(),
        c.arg(),
        space,
        depth,
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let expected = references::tiled_matmul(m, n, k, tile);
    let (_, expected) = TestInput::builder(client, shape![m / tile, n / tile, tile, tile])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

#[test]
fn a_hand_written_ring_walk_single_buffered() {
    check_ring_matmul(8, 8, 16, 4, 1);
}

#[test]
fn a_hand_written_ring_walk_double_buffered() {
    check_ring_matmul(8, 8, 16, 4, 2);
}

#[test]
fn a_hand_written_ring_walk_triple_buffered_over_an_odd_walk() {
    check_ring_matmul(8, 8, 20, 4, 3);
}
