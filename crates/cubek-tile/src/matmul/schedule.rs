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

/// `DoubleBuffered`: two [`Stream`] lanes, each holding both operands' staged tiles. The schedule
/// drives them `load`/`mma`, `load`/`mma`, alternating lanes so one lane's fill overlaps the other's
/// compute. The rendezvous is not a call the schedule makes — it lives inside `load`/`mma` (a
/// cube-wide `sync_cube` for a strided fill, the lane's mbarrier for a TMA fill).
///
/// The fill kind is decided from the operands and carried as each lane's barriers: a TMA source
/// gets a per-lane `full`/`empty` mbarrier pair, a strided source none. TMA is dormant — no launch
/// path builds a tma source yet — so in practice both lanes run strided.
#[cube]
pub(crate) fn mma_double<Lhs: Numeric, Rhs: Numeric, Acc>(
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    out: &mut Tile<Acc>,
    space: Space,
) where
    Acc: Numeric,
{
    // Allocated here in caller scope because a view-backed buffer must outlive the lanes. Each smem
    // buffer is staged at its source operand's physical width (`tile_size * width` scalar entries).
    let a_sub = comptime!(lhs.space.divide());
    let b_sub = comptime!(rhs.space.divide());
    let a_width = comptime!(lhs.vector_size);
    let b_width = comptime!(rhs.vector_size);
    let a0 = Shared::<[Lhs]>::new_slice(a_sub.tile_size() * a_width);
    let a1 = Shared::<[Lhs]>::new_slice(a_sub.tile_size() * a_width);
    let b0 = Shared::<[Rhs]>::new_slice(b_sub.tile_size() * b_width);
    let b1 = Shared::<[Rhs]>::new_slice(b_sub.tile_size() * b_width);

    // The operands hand back each lane's barriers for their delivery; the schedule stays out of the
    // fill kind, the mbarrier counts, and the fences.
    let (bar0, bar1) = lhs.double_buffer_barriers(rhs);

    let mut s0 = Stream::new(
        Tile::smem(&a0, comptime!(a_sub.clone()), a_width),
        Tile::smem(&b0, comptime!(b_sub.clone()), b_width),
        bar0,
    );
    let mut s1 = Stream::new(
        Tile::smem(&a1, comptime!(a_sub.clone()), a_width),
        Tile::smem(&b1, comptime!(b_sub.clone()), b_width),
        bar1,
    );

    // Double-buffering needs random access (prefetch the next region), so it indexes the `walk`
    // by hand rather than iterating.
    let walk = Walk::over(space);
    let n = walk.total();

    // prologue: prime lane 0 with region 0.
    s0.load(lhs, rhs, walk.region(0));

    for p in 0..n / 2 {
        let even = p * 2;
        let odd = even + 1;

        // prefetch the odd region into lane 1 (its fill overlaps the compute below), then compute
        // the even region on lane 0.
        s1.load(lhs, rhs, walk.region(even + 1));
        s0.mma(out, walk.region(even));

        // prefetch the next even region back into lane 0 (if it exists), then compute the odd
        // region on lane 1.
        if odd + 1 < n {
            s0.load(lhs, rhs, walk.region(odd + 1));
        }
        s1.mma(out, walk.region(odd));
    }
}
