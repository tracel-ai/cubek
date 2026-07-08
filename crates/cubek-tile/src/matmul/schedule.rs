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
pub(crate) fn mma_staged<Lhs: Numeric, Rhs: Numeric, Acc: Numeric>(
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    out: &mut Tile<Acc>,
    space: Space,
) {
    // Each smem buffer mirrors what `at` produces one level down (its `divide()`) and carries any
    // remaining finer levels, at the source operand's physical width.
    let mut a_tile = lhs.smem_like();
    let mut b_tile = rhs.smem_like();

    for region in Walk::over(space) {
        a_tile.copy_from(&lhs.at(&region));
        b_tile.copy_from(&rhs.at(&region));
        out.at(&region).mma(&a_tile, &b_tile);
    }
}

/// `DoubleBuffered`: two [`Staging`] slots, each a `(Tile<Lhs>, Tile<Rhs>)` payload, driven
/// `fill`/`consume` on alternating slots so one slot's fill overlaps the other's compute. Each slot's
/// synchronization is wrapped inside `fill`/`consume`, not here.
#[cube]
pub(crate) fn mma_double<Lhs: Numeric, Rhs: Numeric, Acc: Numeric>(
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    out: &mut Tile<Acc>,
    space: Space,
) {
    // Each slot stages `lhs`/`rhs` into its own smem; the sync strategy and the allocation both live
    // in `Staging::new` (deduced from the operands' delivery), so the schedule stays out of it.
    let mut s0 = Staging::new(lhs, rhs);
    let mut s1 = Staging::new(lhs, rhs);

    // Double-buffering needs random access (prefetch the next region), so it indexes the `walk`
    // by hand rather than iterating.
    let walk = Walk::over(space);
    let n = walk.total();

    // prologue: prime slot 0 with region 0.
    let first = walk.region(0);
    s0.fill(|s, pipe| {
        pipe.fill(&mut s.0, &lhs.at(&first));
        pipe.fill(&mut s.1, &rhs.at(&first));
    });

    for p in 0..n / 2 {
        let even = p * 2;
        let odd = even + 1;

        // prefetch the odd region into slot 1 (its fill overlaps the compute below), then compute
        // the even region on slot 0.
        let odd_region = walk.region(odd);
        s1.fill(|s, pipe| {
            pipe.fill(&mut s.0, &lhs.at(&odd_region));
            pipe.fill(&mut s.1, &rhs.at(&odd_region));
        });
        let even_region = walk.region(even);
        s0.consume(|a, b| out.at(&even_region).mma(a, b));

        // prefetch the next even region back into slot 0 (if it exists), then compute the odd
        // region on slot 1.
        if odd + 1 < n {
            let next_even = walk.region(odd + 1);
            s0.fill(|s, pipe| {
                pipe.fill(&mut s.0, &lhs.at(&next_even));
                pipe.fill(&mut s.1, &rhs.at(&next_even));
            });
        }
        let odd_region = walk.region(odd);
        s1.consume(|a, b| out.at(&odd_region).mma(a, b));
    }
}
