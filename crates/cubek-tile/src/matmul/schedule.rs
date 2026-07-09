//! The walks behind [`Tile::mma`](super::Tile), one per [`Schedule`]. A schedule's body
//! is pure structure; kind decisions (slot store, rendezvous, fill dispatch) are
//! delegated, chiefly to [`Staging::new`].

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `Direct`: no staging, every read goes to where the operand lives. The static twin
    /// steps with indexed `comptime!` because each unrolled copy stamps different host
    /// data, which the runtime iterator sugar cannot carry.
    pub(crate) fn mma_direct<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        space: Space,
    ) {
        if self.tile_kind.static_level(comptime!(self.space.clone())) {
            let walk = comptime!(StaticWalk::over_fastest(
                &Space::merge(&[&lhs.space, &rhs.space]),
                self.space.axis_at(self.space.rank() - 2),
            ));
            #[unroll]
            for i in 0..comptime!(walk.total()) {
                let region = comptime!(walk.region(i));
                self.at_static(&region)
                    .mma(&lhs.at_static(&region), &rhs.at_static(&region));
            }
        } else {
            for region in Walk::over(space) {
                self.at(&region).mma(&lhs.at(&region), &rhs.at(&region));
            }
        }
    }

    /// `Staged`: per region, fill a [`Staging`] slot with the operands and consume it
    /// into the recursion. `consume_final` every region, since no later fill publishes
    /// within an iteration.
    pub(crate) fn mma_staged<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        space: Space,
    ) {
        let mut slot = Staging::new(lhs, rhs, comptime!(self.space.clone()));
        for region in Walk::over(space) {
            slot.fill(|s, pipe| {
                pipe.fill(&mut s.0, &lhs.at(&region));
                pipe.fill(&mut s.1, &rhs.at(&region));
            });
            slot.consume_final(|a, b| self.at(&region).mma(a, b));
        }
    }

    /// `DoubleBuffered`: two [`Staging`] slots driven `fill`/`consume` on alternating
    /// regions so one slot's fill overlaps the other's compute.
    pub(crate) fn mma_double<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        space: Space,
    ) {
        let mut s0 = Staging::new(lhs, rhs, comptime!(self.space.clone()));
        let mut s1 = Staging::new(lhs, rhs, comptime!(self.space.clone()));

        // Double-buffering needs random access (prefetch the next region), so it indexes the
        // `walk` by hand rather than iterating.
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

            // prefetch the odd region into slot 1 (its fill overlaps the compute below), then
            // compute the even region on slot 0.
            let odd_region = walk.region(odd);
            s1.fill(|s, pipe| {
                pipe.fill(&mut s.0, &lhs.at(&odd_region));
                pipe.fill(&mut s.1, &rhs.at(&odd_region));
            });
            let even_region = walk.region(even);
            s0.consume(|a, b| self.at(&even_region).mma(a, b));

            // prefetch the next even region back into slot 0 (if it exists), then compute
            // the odd region on slot 1; on the walk's final region no fill follows, so
            // `consume_final` publishes slot 1 itself.
            let odd_region = walk.region(odd);
            if odd + 1 < n {
                let next_even = walk.region(odd + 1);
                s0.fill(|s, pipe| {
                    pipe.fill(&mut s.0, &lhs.at(&next_even));
                    pipe.fill(&mut s.1, &rhs.at(&next_even));
                });
                s1.consume(|a, b| self.at(&odd_region).mma(a, b));
            } else {
                s1.consume_final(|a, b| self.at(&odd_region).mma(a, b));
            }
        }

        // An odd total leaves the last region primed in slot 0 with no consumer in the
        // loop; no fill follows, so `consume_final` publishes it.
        if n % 2 == 1 {
            let last = walk.region(n - 1);
            s0.consume_final(|a, b| self.at(&last).mma(a, b));
        }
    }
}
