//! The walks behind [`Tile::mma`](super::Tile), one per [`Schedule`]. A schedule's body
//! is pure structure; kind decisions (slot store, rendezvous, fill dispatch) are
//! delegated, chiefly to [`Staging::new`].

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `Direct`: no staging, every read goes to where the operand lives. A fragment
    /// output demands the unrolled walk (its coordinates fold to constants, which
    /// select fragments); a memory output keeps the compact runtime loop.
    pub(crate) fn mma_direct<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        space: Space,
    ) {
        if self.tile_kind.static_level(comptime!(self.space.clone())) {
            let walk = Walk::over_fastest(
                comptime!(Space::merge(&[&lhs.space, &rhs.space])),
                comptime!(self.space.axis_at(self.space.rank() - 2)),
            );
            for region in walk.unrolled() {
                self.at(&region).mma(&lhs.at(&region), &rhs.at(&region));
            }
        } else {
            for region in Walk::over(space) {
                self.at(&region).mma(&lhs.at(&region), &rhs.at(&region));
            }
        }
    }

    /// `Staged`: per region, fill a [`Staging`] slot with the operands and consume it
    /// into the recursion. An operand the walk leaves unchanged (its space lacks every
    /// walked axis — the same structural fact as broadcast omission) fills its slot once,
    /// above the loop: re-filling it per region would move the same window again. E.g. an
    /// N-walk refills one B fragment per step while the whole A partition stays put — the
    /// legacy single-buffered register budget. `consume_final` every region, since no
    /// later fill publishes within an iteration.
    ///
    /// The walk unrolls when the level *cuts* a fragment-partition output (each region
    /// selects its block, which takes comptime coordinates) and on a static
    /// register-staged level (comptime regions land window offsets as immediates — the
    /// fill-side win at the thin selector point). An smem-staged level stays rolled:
    /// unrolling would re-stage the recursion's shared memory per copy.
    pub(crate) fn mma_staged<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        space: Space,
    ) {
        // A barrier pipeline arrives `full` once per fill, so a TMA pair keeps the joint
        // per-region fill; splitting an invariant fill out would corrupt its phase. A
        // dynamic level can't decide invariance at comptime, so it keeps it too.
        let lhs_tma = lhs.is_tma();
        let rhs_tma = rhs.is_tma();
        let spec = comptime!(space.clone());
        let split = comptime!(spec.is_static() && !lhs_tma && !rhs_tma);
        let lhs_once = comptime!(split && spec.walk_invariant(&lhs.space));
        let rhs_once = comptime!(split && spec.walk_invariant(&rhs.space));

        let cuts = self.tile_kind.cuts_partition(comptime!(self.space.clone()));
        let register = comptime!(
            self.space.partitioner().leaf().is_cmma()
                && partition_level(&self.space.divide()).is_some()
                && Space::merge(&[&lhs.space, &rhs.space]).static_walkable()
        );
        let unroll = comptime!(cuts || register);

        let mut slot = Staging::new(lhs, rhs, comptime!(self.space.clone()));
        let walk = Walk::over(space);
        if comptime!(lhs_once || rhs_once) {
            // An invariant operand's window ignores the walked axes, so the first
            // region's window is every region's window.
            let first = walk.region(0);
            slot.fill(|s, pipe| {
                if comptime!(lhs_once) {
                    pipe.fill(&mut s.0, &lhs.at(&first));
                }
                if comptime!(rhs_once) {
                    pipe.fill(&mut s.1, &rhs.at(&first));
                }
            });
        }
        let walk = if comptime!(unroll) { walk.unrolled() } else { walk };
        for region in walk {
            slot.fill(|s, pipe| {
                if comptime!(!lhs_once) {
                    pipe.fill(&mut s.0, &lhs.at(&region));
                }
                if comptime!(!rhs_once) {
                    pipe.fill(&mut s.1, &rhs.at(&region));
                }
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
