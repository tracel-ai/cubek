//! The walk written by the kernel: the levels are loops, the stages are rings the kernel
//! allocates, and the leaf runs under the register block the kernel states. The reference every
//! routine's rewrite is measured against.
#![allow(non_snake_case)]

use cubecl::{prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput, TileInput, assert_equals_approx};
use cubek_tile::*;

use super::references;
use super::{Form, implied};

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
    space: Partitioning,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));
    c.zero();

    // The cube's walk: one block of K per region, both operands staged for it.
    let walk = space.walk();
    let mut ring = Ring::smem(&walk, &a, &b, StageStorage::Strided, depth);
    pipelined(walk, &mut ring, |slot, region| {
        let c_block = c.at(region);
        slot.consume(|a_s, b_s| {
            // The block's own grid of final tiles, each contracted by the leaf.
            for cell in region {
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

/// [`ring_matmul`] with the fill the kernel's own: same schedule, and the operands are written
/// into the slot by this kernel rather than by the ring.
///
/// What it proves is that the two halves are separable: the prologue, the lap that prefetches one
/// region ahead of the one it computes, and the publish on the draining consume are the part worth
/// sharing; where the bytes come from and what happens to them on the way in is the kernel's.
///
/// Here the fill is the ring's own sources and the answer is unchanged, which is the honest
/// control: a fill that read somewhere else would be testing the caller, not the seam.
#[cube(launch)]
fn ring_matmul_filled_by_the_kernel<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));
    c.zero();

    let walk = space.walk();
    let mut ring = Ring::smem(&walk, &a, &b, StageStorage::Strided, depth);
    pipelined_with(
        walk,
        &mut ring,
        |slot, region| {
            slot.fill(|staged, pipe| {
                pipe.fill(&mut staged.0, &a.at(region));
                pipe.fill(&mut staged.1, &b.at(region));
            });
        },
        |slot, region| {
            let c_block = c.at(region);
            slot.consume(|a_s, b_s| {
                for cell in region {
                    let mut c_cell = c_block.at(&cell);
                    c_cell.mma_with(
                        &a_s.at(&cell),
                        &b_s.at(&cell),
                        REGISTER_BLOCK,
                        Semiring::SUM_PROD,
                    );
                }
            });
        },
    );
}

/// The same contraction with the ring's two steps written out and the plane's role read off it:
/// the shape a specialized kernel takes. With no plane set aside to fill, every plane computes,
/// so the fill arm is emitted and never entered and the compute arm does both halves itself.
#[cube(launch)]
fn role_split_matmul<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));
    c.zero();

    let walk = space.walk();
    let mut ring = Ring::smem(&walk, &a, &b, StageStorage::Strided, 1usize);

    match ring.role() {
        Role::Fill => {
            for region in space.walk() {
                ring.fill(0usize, &region);
            }
        }
        Role::Compute => {
            for region in space.walk() {
                ring.fill(0usize, &region);
                ring.slot_mut(0usize).publish();
                let c_block = c.at(&region);
                ring.consume(0usize, |a_s, b_s| {
                    for cell in &region {
                        let mut c_cell = c_block.at(&cell);
                        c_cell.mma_with(
                            &a_s.at(&cell),
                            &b_s.at(&cell),
                            REGISTER_BLOCK,
                            Semiring::SUM_PROD,
                        );
                    }
                });
            }
        }
    }
}

fn check_ring_matmul(m: usize, n: usize, k: usize, block_k: usize, depth: usize) {
    check_ring_matmul_with(m, n, k, block_k, depth, false)
}

/// [`check_ring_matmul`], filling through the ring or through the kernel's own closure.
fn check_ring_matmul_with(
    m: usize,
    n: usize,
    k: usize,
    block_k: usize,
    depth: usize,
    kernel_fills: bool,
) {
    let client = cubecl::test_device().client();
    let tile = 4usize;
    let dtype = f32::elem_type_native();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Tiling::leaf(&[(M, tile), (N, tile), (K, tile)])
                .walk(&[(M, m / tile), (N, n / tile), (K, block_k / tile)])
                .walk_every(&[M, N, K])
                .levels(),
        ),
        Form::Static,
    );

    let a = TileInput::builder(&client, launcher.space().project(&[M, K]))
        .tile(&[tile, tile])
        .arange();
    let b = TileInput::builder(&client, launcher.space().project(&[K, N]))
        .tile(&[tile, tile])
        .arange();
    let c = TileInput::builder(&client, launcher.space().project(&[M, N]))
        .tile(&[tile, tile])
        .uniform(7, -100.0, 100.0);

    match kernel_fills {
        false => ring_matmul::launch(
            &client,
            launcher.cube_count(),
            CubeDim::new_single(),
            a.arg(),
            b.arg(),
            c.arg(),
            launcher.partitioning_arg(),
            depth,
            dtype,
        ),
        true => ring_matmul_filled_by_the_kernel::launch(
            &client,
            launcher.cube_count(),
            CubeDim::new_single(),
            a.arg(),
            b.arg(),
            c.arg(),
            launcher.partitioning_arg(),
            depth,
            dtype,
        ),
    }

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let expected = references::tiled_matmul(m, n, k, tile);
    let (_, expected) = TestInput::builder(client, shape![m / tile, n / tile, tile, tile])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// The two roles meet on a barrier and nowhere else, so a device without one is told so by name
/// rather than handed two loops with no rendezvous between them.
#[test]
#[should_panic(expected = "carries no barrier type")]
fn a_device_without_barriers_refuses_a_walk_filled_by_planes_of_its_own() {
    let (m, n, k, tile) = (8usize, 8usize, 16usize, 4usize);
    let client = cubecl::test_device().client();
    implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Tiling::leaf(&[(M, tile), (N, tile), (K, tile)])
                .walk(&[(M, m / tile), (N, n / tile), (K, 1)])
                .walk_every(&[M, N, K])
                .filled_by(1)
                .levels(),
        ),
        Form::Static,
    );
}

/// The specialized shape runs and is right where no plane is set aside: the compute arm fills
/// and reads its own slot, and the fill arm compiles beside it, entered by nobody.
#[test]
fn a_role_split_walk_with_no_filling_plane_is_the_walk_it_always_was() {
    let (m, n, k, tile) = (8usize, 8usize, 16usize, 4usize);
    let client = cubecl::test_device().client();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Tiling::leaf(&[(M, tile), (N, tile), (K, tile)])
                .walk(&[(M, m / tile), (N, n / tile), (K, 1)])
                .walk_every(&[M, N, K])
                .levels(),
        ),
        Form::Static,
    );
    let a = TileInput::builder(&client, launcher.space().project(&[M, K]))
        .tile(&[tile, tile])
        .arange();
    let b = TileInput::builder(&client, launcher.space().project(&[K, N]))
        .tile(&[tile, tile])
        .arange();
    let c = TileInput::builder(&client, launcher.space().project(&[M, N]))
        .tile(&[tile, tile])
        .uniform(7, -100.0, 100.0);

    role_split_matmul::launch(
        &client,
        launcher.cube_count(),
        CubeDim::new_single(),
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        f32::elem_type_native(),
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

/// The kernel's own fill through the ring's schedule, at each buffering depth.
///
/// Same walk, same answer, and the only difference is who writes the slot. A kernel that wants a
/// third operand or a transform on the way in keeps the protocol it would otherwise have had to
/// re-derive.
#[test]
fn a_kernel_may_bring_its_own_fill_single_buffered() {
    check_ring_matmul_with(8, 8, 16, 8, 1, true);
}

#[test]
fn a_kernel_may_bring_its_own_fill_double_buffered() {
    check_ring_matmul_with(8, 8, 16, 8, 2, true);
}

#[test]
fn a_kernel_may_bring_its_own_fill_over_an_odd_walk() {
    check_ring_matmul_with(8, 8, 24, 8, 3, true);
}
