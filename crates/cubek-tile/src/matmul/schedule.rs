//! The three lowering schedules behind [`Tile::mma`](super::Tile): [`Direct`](Schedule::Direct)
//! (no staging), [`Staged`](Schedule::Staged), and [`DoubleBuffered`](Schedule::DoubleBuffered).
//! Each receives the level's [`Walk`] from `Tile::mma`, so the schedules themselves carry no
//! extent or merge logic.

use cubecl::prelude::*;

use crate::*;

/// `Direct`: no staging
#[cube]
pub(crate) fn mma_direct<Lhs: Numeric, Rhs: Numeric, Acc>(
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    out: &mut Tile<Acc>,
    space: Space,
) where
    Acc: Numeric,
{
    for region in Walk::over(space) {
        out.at(&region).mma(&lhs.at(&region), &rhs.at(&region));
    }
}

/// `Staged`: stage each operand sub-tile into shared memory, then recurse. Each buffer keeps
/// its own served type.
#[cube]
pub(crate) fn mma_staged<Lhs: Numeric, Rhs: Numeric, Acc>(
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    out: &mut Tile<Acc>,
    space: Space,
) where
    Acc: Numeric,
{
    // The buffer's space is this level's divide, so it mirrors what `at` produces and
    // carries any remaining finer levels. Each smem buffer is staged at its source operand's
    // physical width, so the scalar slice holds `tile_size * width` entries.
    let a_sub = comptime!(lhs.space.divide());
    let b_sub = comptime!(rhs.space.divide());
    let a_width = comptime!(lhs.vector_size);
    let b_width = comptime!(rhs.vector_size);
    let a_smem = Shared::<[Lhs]>::new_slice(a_sub.tile_size() * a_width);
    let b_smem = Shared::<[Rhs]>::new_slice(b_sub.tile_size() * b_width);
    let mut a_tile = Tile::smem(&a_smem, a_sub, a_width);
    let mut b_tile = Tile::smem(&b_smem, b_sub, b_width);

    for region in Walk::over(space) {
        a_tile.stage(&lhs.at(&region));
        b_tile.stage(&rhs.at(&region));
        out.at(&region).mma(&a_tile, &b_tile);
    }
}

/// `DoubleBuffered`: two staged buffers per operand, prefetching the next region into the idle
/// slot while computing the current one.
#[cube]
pub(crate) fn mma_double<Lhs: Numeric, Rhs: Numeric, Acc>(
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    out: &mut Tile<Acc>,
    space: Space,
) where
    Acc: Numeric,
{
    // Allocated here in caller scope because a view-backed buffer must outlive the ring. Each smem
    // buffer is staged at its source operand's physical width (`tile_size * width` scalar entries).
    let a_sub = comptime!(lhs.space.divide());
    let b_sub = comptime!(rhs.space.divide());
    let a_width = comptime!(lhs.vector_size);
    let b_width = comptime!(rhs.vector_size);
    let a0 = Shared::<[Lhs]>::new_slice(a_sub.tile_size() * a_width);
    let a1 = Shared::<[Lhs]>::new_slice(a_sub.tile_size() * a_width);
    let b0 = Shared::<[Rhs]>::new_slice(b_sub.tile_size() * b_width);
    let b1 = Shared::<[Rhs]>::new_slice(b_sub.tile_size() * b_width);
    let mut a_buf = Sequence::new();
    a_buf.push(Tile::smem(&a0, comptime!(a_sub.clone()), a_width));
    a_buf.push(Tile::smem(&a1, comptime!(a_sub.clone()), a_width));
    let mut b_buf = Sequence::new();
    b_buf.push(Tile::smem(&b0, comptime!(b_sub.clone()), b_width));
    b_buf.push(Tile::smem(&b1, comptime!(b_sub.clone()), b_width));
    let mut a = Ring::new(a_buf);
    let mut b = Ring::new(b_buf);

    // Double-buffering needs random access (prefetch the next region), so it indexes the `walk`
    // by hand rather than iterating.
    let walk = Walk::over(space);

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
