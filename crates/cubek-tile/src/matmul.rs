//! Where "matmul" is *lowered*. There is no `Matmul` type — a matmul is what
//! `c.mma(a, b)` does when `c` is a whole tensor: reconstruct the operation's
//! [`Space`] from the operands, partition it, locate each operand
//! [`at`](Tile::at) every region, and recurse (`acc.mma(…)` falls through to the leaf).
//! The *move* — stage or not, pipeline or not — is read from the partitioner's
//! [`Schedule`], so the kernel body stays `c.mma(a, b)` and the choice rides in
//! the partitioner. The MNK labels are the client's; the DSL sees a contraction.

use cubecl::prelude::*;

use super::*;

/// Pick the lowering move from the accumulator's [`Schedule`] and run it. Called
/// by [`Tile::mma`] for a whole-tensor accumulator.
#[cube]
pub fn mma_lower<E: Numeric>(lhs: &Tile<E>, rhs: &Tile<E>, out: &mut Tile<E>) {
    match comptime!(out.space.partitioner().schedule()) {
        Schedule::Direct => mma_direct::<E>(lhs, rhs, out),
        Schedule::Staged => mma_staged::<E>(lhs, rhs, out),
        Schedule::DoubleBuffered => mma_double::<E>(lhs, rhs, out),
    }
}

/// `Direct`: no staging — each operand sub-tile feeds the leaf straight from its
/// tiled layout. Partition the space, locate each tile, recurse.
#[cube]
fn mma_direct<E: Numeric>(lhs: &Tile<E>, rhs: &Tile<E>, out: &mut Tile<E>) {
    let walk = Walk::over(comptime!(Space::merge(&[
        &lhs.space, &rhs.space, &out.space
    ])));
    for i in 0..walk.total() {
        out.mma_at(&lhs, &rhs, &walk.region(i));
    }
}

/// `Staged`: stage each operand sub-tile into shared memory, then recurse.
#[cube]
fn mma_staged<E: Numeric>(lhs: &Tile<E>, rhs: &Tile<E>, out: &mut Tile<E>) {
    // Stage one sub-tile of the head level — its space is this level's divide (the
    // located sub-tile's space), so the buffer mirrors what `at` produces and
    // carries any remaining finer levels for the recursion below.
    let a_sub = comptime!(lhs.space.divide());
    let b_sub = comptime!(rhs.space.divide());
    let a_smem = Shared::<[Vector<E, Const<1>>]>::new_slice(a_sub.tile_size());
    let b_smem = Shared::<[Vector<E, Const<1>>]>::new_slice(b_sub.tile_size());
    let mut a_tile = Tile::smem(&a_smem, a_sub);
    let mut b_tile = Tile::smem(&b_smem, b_sub);

    let walk = Walk::over(comptime!(Space::merge(&[
        &lhs.space, &rhs.space, &out.space
    ])));
    for i in 0..walk.total() {
        let region = walk.region(i);
        a_tile.stage(&lhs.at(&region));
        b_tile.stage(&rhs.at(&region));
        out.at(&region).mma(&a_tile, &b_tile);
    }
}

/// `DoubleBuffered`: two staged buffers per operand, prefetching the next region
/// into the idle slot while computing the current one. The loop stays runtime; the
/// slot choice stays comptime by writing the two ping-pong phases explicitly.
/// `sync_cube()` between phases keeps a slot from being reused while another unit
/// still reads it (several planes = several cores). (Demo: an even region count.)
#[cube]
fn mma_double<E: Numeric>(lhs: &Tile<E>, rhs: &Tile<E>, out: &mut Tile<E>) {
    // Two shared-memory buffers per operand, wrapped as a `Ring`. Allocated here
    // (caller scope) because a view-backed buffer must outlive the ring. Each buffer
    // mirrors a located sub-tile of the head level (this level's divide).
    let a_sub = comptime!(lhs.space.divide());
    let b_sub = comptime!(rhs.space.divide());
    let a0 = Shared::<[Vector<E, Const<1>>]>::new_slice(a_sub.tile_size());
    let a1 = Shared::<[Vector<E, Const<1>>]>::new_slice(a_sub.tile_size());
    let b0 = Shared::<[Vector<E, Const<1>>]>::new_slice(b_sub.tile_size());
    let b1 = Shared::<[Vector<E, Const<1>>]>::new_slice(b_sub.tile_size());
    let mut a_buf = Sequence::new();
    a_buf.push(Tile::smem(&a0, comptime!(a_sub.clone())));
    a_buf.push(Tile::smem(&a1, comptime!(a_sub.clone())));
    let mut b_buf = Sequence::new();
    b_buf.push(Tile::smem(&b0, comptime!(b_sub.clone())));
    b_buf.push(Tile::smem(&b1, comptime!(b_sub.clone())));
    let mut a = Ring::new(a_buf);
    let mut b = Ring::new(b_buf);

    let walk = Walk::over(comptime!(Space::merge(&[
        &lhs.space, &rhs.space, &out.space
    ])));

    // prologue: prime slot 0 with region 0.
    let r0 = walk.region(0);
    a.stage(0usize, &lhs.at(&r0));
    b.stage(0usize, &rhs.at(&r0));
    sync_cube();

    let n = walk.total();
    for p in 0..n / 2 {
        let even = p * 2;
        let odd = even + 1;

        // phase 0: prefetch the odd region into slot 1, compute the even region.
        a.stage(1usize, &lhs.at(&walk.region(even + 1)));
        b.stage(1usize, &rhs.at(&walk.region(even + 1)));
        out.at(&walk.region(even)).mma(a.get(0usize), b.get(0usize));
        sync_cube();

        // phase 1: prefetch the next even region into slot 0, compute the odd region.
        if odd + 1 < n {
            a.stage(0usize, &lhs.at(&walk.region(odd + 1)));
            b.stage(0usize, &rhs.at(&walk.region(odd + 1)));
        }
        out.at(&walk.region(odd)).mma(a.get(1usize), b.get(1usize));
        sync_cube();
    }
}
