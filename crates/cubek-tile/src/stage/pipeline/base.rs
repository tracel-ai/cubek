//! The buffered walk: `depth` [`Slot`]s driven as a circular software pipeline, `depth - 1`
//! regions in flight while one computes.

use cubecl::frontend::branch::{if_else_expand, if_expand};
use cubecl::ir::Scope;
use cubecl::prelude::*;
use cubecl::unexpanded;

use crate::*;

/// What a plane does with the stages' slots.
#[derive(CubeType, CubeTypeMut, IntoRuntime)]
#[cube(runtime_variants)]
pub enum Role {
    /// Fills the slots and takes no tile.
    Fill,
    /// Reads the slots and computes.
    Compute,
}

/// The `depth` slots of one buffered walk and the operands they are filled from.
#[derive(CubeType)]
pub struct Stages<T: CubeType> {
    pub(crate) slots: Sequence<Slot<T>>,
    pub(crate) sources: T,
    #[allow(dead_code)]
    #[cube(comptime)]
    pub(crate) depth: usize,
    /// Planes that only fill these slots.
    #[cube(comptime)]
    pub(crate) fillers: usize,
}

#[cube]
impl<T: CubeType> Stages<T> {
    /// Wrap already-built slots over the operands they stage.
    pub(crate) fn wrap(
        slots: Sequence<Slot<T>>,
        sources: T,
        #[comptime] depth: usize,
        #[comptime] fillers: usize,
    ) -> Stages<T> {
        comptime!(assert!(
            depth > 0,
            "Stages: a pipeline needs at least one slot"
        ));
        Stages::<T> {
            slots,
            sources,
            depth,
            fillers,
        }
    }

    /// What this plane does with the stages' slots; filler planes sit at the end of the cube.
    pub fn role(&self) -> Role {
        if comptime!(self.fillers == 0) {
            Role::new_Compute()
        } else if UNIT_POS >= Meeting::consumers(comptime!(self.fillers)) {
            Role::new_Fill()
        } else {
            Role::new_Compute()
        }
    }

    /// Slot `index`, known at comptime.
    pub fn slot_mut(&mut self, #[comptime] index: usize) -> &mut Slot<T> {
        self.slots.index_mut(index)
    }
}

/// How stages fill their slots from their own sources, at expand level.
pub trait StagesFill {
    /// Whether any operand's window is fixed across the walk.
    fn has_fixed(&self, scope: &Scope) -> bool;
    /// Fill slot `slot`'s fixed operands from `region`'s window.
    fn fill_fixed(&mut self, scope: &Scope, slot: usize, region: &RegionExpand);
    /// Fill slot `slot`'s streamed operands from `region`'s window.
    fn fill_streamed(&mut self, scope: &Scope, slot: usize, region: &RegionExpand);
    /// Publish slot `slot`'s last fill where no later fill will (see [`Slot::publish`]).
    fn publish(&mut self, scope: &Scope, slot: usize);
}

impl<T: CubeType> Stages<T> {
    /// Walk `walk`'s regions, filling slots from the stages' sources and `compute` consuming each
    /// through [`consume`](Slot::consume).
    pub fn pipelined<F>(&mut self, _walk: Walk, _compute: F)
    where
        F: FnMut(&mut Slot<T>, &Region),
    {
        unexpanded!()
    }
}

impl<T: CubeType> StagesExpand<T>
where
    StagesExpand<T>: StagesFill,
{
    pub fn __expand_pipelined_method<F>(&mut self, scope: &Scope, walk: WalkExpand, compute: F)
    where
        F: FnMut(&Scope, &mut SlotExpand<T>, &RegionExpand),
    {
        schedule::run(
            scope,
            walk,
            self,
            |scope, stages, slot, region| stages.fill_streamed(scope, slot, region),
            compute,
        )
    }
}

/// Where a walk's next stage waits while the previous one is contracted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Prefetch {
    /// In its own slot, filled before the previous contraction.
    InSlots,
    /// In registers, loaded before the contraction and stored after; bounded by
    /// [`fits`](Prefetch::fits).
    InRegisters,
}

impl Prefetch {
    /// Scalars one unit holds when `units` prefetch `elements` values read `width` at a time.
    pub fn scalars(elements: usize, units: usize, width: usize) -> usize {
        UnitLines::new(elements.div_ceil(width), units).scalars(width)
    }

    /// Whether a register prefetch of `scalars` a unit, over both operands, is accepted.
    pub fn fits(scalars: usize) -> bool {
        scalars <= MOST_FETCHED_SCALARS
    }
}

impl<Lhs: Numeric, Rhs: Numeric> Stages<OperandPair<Lhs, Rhs>> {
    /// Walk `walk`'s stages with the schedule `prefetch` selects.
    pub fn walk<F>(&mut self, _walk: Walk, _prefetch: Prefetch, _compute: F)
    where
        F: FnMut(&mut Slot<OperandPair<Lhs, Rhs>>, &Region),
    {
        unexpanded!()
    }
}

impl<Lhs: Numeric, Rhs: Numeric> StagesExpand<OperandPair<Lhs, Rhs>> {
    pub fn __expand_walk_method<F>(
        &mut self,
        scope: &Scope,
        walk: WalkExpand,
        prefetch: Prefetch,
        compute: F,
    ) where
        F: FnMut(&Scope, &mut SlotExpand<OperandPair<Lhs, Rhs>>, &RegionExpand),
    {
        match prefetch {
            Prefetch::InSlots => self.__expand_pipelined_method(scope, walk, compute),
            Prefetch::InRegisters => self.__expand_prefetched_method(scope, walk, compute),
        }
    }
}

impl<Lhs: Numeric, Rhs: Numeric> Stages<OperandPair<Lhs, Rhs>> {
    /// [`pipelined`](Stages::pipelined) with the next region's loads held in registers across the
    /// current contraction.
    ///
    /// Requires depth 1 or 2, every unit copying straight from a plain operand, and loads within
    /// [`Prefetch::fits`]; otherwise refused at expansion. Fixed operands fall back to
    /// [`pipelined`](Stages::pipelined).
    pub fn prefetched<F>(&mut self, _walk: Walk, _compute: F)
    where
        F: FnMut(&mut Slot<OperandPair<Lhs, Rhs>>, &Region),
    {
        unexpanded!()
    }
}

impl<Lhs: Numeric, Rhs: Numeric> StagesExpand<OperandPair<Lhs, Rhs>> {
    pub fn __expand_prefetched_method<F>(&mut self, scope: &Scope, walk: WalkExpand, compute: F)
    where
        F: FnMut(&Scope, &mut SlotExpand<OperandPair<Lhs, Rhs>>, &RegionExpand),
    {
        if self.has_fixed(scope) {
            return self.__expand_pipelined_method(scope, walk, compute);
        }
        self.__expand_assert_copied_by_every_unit_method(scope);
        assert!(
            self.depth <= 2,
            "Stages::prefetched: {} slots hold one region ahead, so the slots past a second \
             would sit idle; build them with a depth of 1 or 2",
            self.depth
        );
        let total = walk.__expand_total_method(scope);
        self.prime(scope, &walk, &total);
        self.prefetching_laps(scope, walk, total, compute);
    }

    /// Fill and publish the first region before any contraction.
    fn prime(&mut self, scope: &Scope, walk: &WalkExpand, total: &NativeExpand<usize>) {
        let any = 0usize.into_expand(scope).__expand_lt_method(scope, total);
        if_expand(scope, any, |scope| {
            let first = walk.__expand_region_method(scope, FIRST_SLOT.into_expand(scope));
            self.fill_streamed(scope, FIRST_SLOT, &first);
            self.publish(scope, FIRST_SLOT);
        });
    }

    /// The laps whose every region has a next one to fetch, run with no branch around the fetch,
    /// so what it computes that no lap changes (each unit's place in the stage) leaves the loop.
    /// How many ran.
    fn steady_prefetching_laps<F>(
        &mut self,
        scope: &Scope,
        walk: &WalkExpand,
        total: NativeExpand<usize>,
        lhs: &mut <Array<Vector<Lhs, super::fill::FL>> as CubeType>::ExpandType,
        rhs: &mut <Array<Vector<Rhs, super::fill::FR>> as CubeType>::ExpandType,
        compute: &mut F,
    ) -> NativeExpand<usize>
    where
        F: FnMut(&Scope, &mut SlotExpand<OperandPair<Lhs, Rhs>>, &RegionExpand),
    {
        // The laps whose every region has a next one to fetch: no branch around the fetch, so what
        // it computes that no lap changes (each unit's place in the stage) leaves the loop.
        let depth = self.depth;
        let steady = steady_laps(scope, total, 1, depth);
        let mut steady_body = |scope: &Scope, lap: NativeExpand<usize>| {
            for j in 0..depth {
                let target = (j + 1) % depth;
                let region_idx = lap
                    .__expand_times_method(scope, depth.into_expand(scope))
                    .__expand_plus_method(scope, j.into_expand(scope));
                let next = region_idx.__expand_plus_method(scope, 1usize.into_expand(scope));
                let upcoming = walk.__expand_region_method(scope, next);
                self.__expand_fetch_method(scope, target, &upcoming, lhs, rhs);
                let region = walk.__expand_region_method(scope, region_idx);
                let slot = self.__expand_slot_mut_method(scope, j);
                compute(scope, slot, &region);
                // One slot: every unit has read it before any overwrites it.
                if depth == 1 {
                    self.publish(scope, FIRST_SLOT);
                }
                self.__expand_store_method(scope, target, lhs, rhs);
                self.publish(scope, target);
            }
        };
        run_laps(
            scope,
            0usize.into_expand(scope),
            steady,
            walk.unroll,
            &mut steady_body,
        );
        steady
    }

    /// Each lap fetches the next region into registers, contracts, then stores into the freed slot.
    fn prefetching_laps<F>(
        &mut self,
        scope: &Scope,
        walk: WalkExpand,
        total: NativeExpand<usize>,
        mut compute: F,
    ) where
        F: FnMut(&Scope, &mut SlotExpand<OperandPair<Lhs, Rhs>>, &RegionExpand),
    {
        let depth = self.depth;
        let (mut lhs, mut rhs) = self.__expand_fetch_buffers_method(scope);
        let steady = self.steady_prefetching_laps(
            scope,
            &walk,
            total,
            &mut lhs,
            &mut rhs,
            &mut compute,
        );
        // The last laps, where the walk runs out under them.
        let mut body = |scope: &Scope, lap: NativeExpand<usize>| {
            for j in 0..depth {
                let target = (j + 1) % depth;
                let region_idx = lap
                    .__expand_times_method(scope, depth.into_expand(scope))
                    .__expand_plus_method(scope, j.into_expand(scope));
                let next = region_idx.__expand_plus_method(scope, 1usize.into_expand(scope));
                let in_walk = region_idx.__expand_lt_method(scope, &total);
                if_expand(scope, in_walk, |scope| {
                    let prefetching = next.__expand_lt_method(scope, &total);
                    if_expand(scope, prefetching, |scope| {
                        let upcoming = walk.__expand_region_method(scope, next);
                        self.__expand_fetch_method(scope, target, &upcoming, &mut lhs, &mut rhs);
                    });
                    let region = walk.__expand_region_method(scope, region_idx);
                    let slot = self.__expand_slot_mut_method(scope, j);
                    compute(scope, slot, &region);
                    if_expand(scope, prefetching, |scope| {
                        // One slot: every unit must finish reading before any overwrites it.
                        if depth == 1 {
                            self.publish(scope, FIRST_SLOT);
                        }
                        self.__expand_store_method(scope, target, &lhs, &rhs);
                        self.publish(scope, target);
                    });
                });
            }
        };
        // One lap is left whenever the walk holds a region (`ceil(t / d) - floor((t - 1) / d)`),
        // none when it holds nothing, and the body guards every region it names: it runs once,
        // straight, rather than as a loop around a copy of the contraction.
        body(scope, steady);
    }
}

/// Laps of `depth` regions each, out of a walk of `total`, in which every region has the one
/// `ahead` of it to fill: `(total - ahead) / depth`, none where the walk is shorter than `ahead`.
/// What a schedule runs with no branch around its fills before the laps where the walk runs out.
fn steady_laps(
    scope: &Scope,
    total: NativeExpand<usize>,
    ahead: usize,
    depth: usize,
) -> NativeExpand<usize> {
    total
        .__expand_max_with_method(scope, ahead.into_expand(scope))
        .__expand_minus_method(scope, ahead.into_expand(scope))
        .__expand_divided_by_method(scope, depth.into_expand(scope))
}

/// `body` on every lap in `from..to`, unrolled where the walk says.
fn run_laps(
    scope: &Scope,
    from: NativeExpand<usize>,
    to: NativeExpand<usize>,
    unroll: bool,
    body: &mut dyn FnMut(&Scope, NativeExpand<usize>),
) {
    let range = RangeExpand::new(from, to);
    if unroll {
        range.expand_unroll(scope, body);
    } else {
        range.expand(scope, body);
    }
}

/// The shared schedule: prologue, laps prefetching one region ahead, drain publishing.
mod schedule {
    use super::*;

    pub(super) fn run<T: CubeType, Fill, F>(
        scope: &Scope,
        walk: WalkExpand,
        stages: &mut StagesExpand<T>,
        mut fill: Fill,
        mut compute: F,
    ) where
        StagesExpand<T>: StagesFill,
        Fill: FnMut(&Scope, &mut StagesExpand<T>, usize, &RegionExpand),
        F: FnMut(&Scope, &mut SlotExpand<T>, &RegionExpand),
    {
        let depth = stages.depth;
        let unroll = walk.unroll;
        let total = walk.__expand_total_method(scope);

        if stages.has_fixed(scope) {
            let first = walk.__expand_region_method(scope, FIRST_SLOT.into_expand(scope));
            for slot in 0..depth {
                stages.fill_fixed(scope, slot, &first);
            }
        }

        for slot in 0..depth - 1 {
            let index = slot.into_expand(scope);
            let cond = index.__expand_lt_method(scope, &total);
            if_expand(scope, cond, |scope| {
                let region = walk.__expand_region_method(scope, slot.into_expand(scope));
                fill(scope, stages, slot, &region);
            });
        }

        let laps = total
            .__expand_plus_method(scope, (depth - 1).into_expand(scope))
            .__expand_divided_by_method(scope, depth.into_expand(scope));
        let steady = steady_ahead_laps(scope, &walk, stages, total, &mut fill, &mut compute);
        let mut body = |scope: &Scope, lap: NativeExpand<usize>| {
            for j in 0..depth {
                let region_idx = lap
                    .__expand_times_method(scope, depth.into_expand(scope))
                    .__expand_plus_method(scope, j.into_expand(scope));
                if depth == 1 {
                    let region = walk.__expand_region_method(scope, region_idx);
                    fill(scope, stages, FIRST_SLOT, &region);
                    stages.publish(scope, FIRST_SLOT);
                    let slot = stages.__expand_slot_mut_method(scope, FIRST_SLOT);
                    compute(scope, slot, &region);
                } else {
                    let ahead =
                        region_idx.__expand_plus_method(scope, (depth - 1).into_expand(scope));
                    let prefetching = ahead.__expand_lt_method(scope, &total);
                    let in_walk = region_idx.__expand_lt_method(scope, &total);
                    // Compute is emitted once, after the fill branch, keeping its accumulator live.
                    if_else_expand(scope, prefetching, |scope| {
                        let prefetch = walk.__expand_region_method(scope, ahead);
                        fill(scope, stages, (j + depth - 1) % depth, &prefetch);
                    })
                    .or_else(scope, |scope| {
                        // Draining: no fill follows, so this consume publishes.
                        if_expand(scope, in_walk, |scope| {
                            stages.publish(scope, j);
                        });
                    });
                    if_expand(scope, in_walk, |scope| {
                        let region = walk.__expand_region_method(scope, region_idx);
                        let slot = stages.__expand_slot_mut_method(scope, j);
                        compute(scope, slot, &region);
                    });
                }
            }
        };
        run_laps(scope, steady, laps, unroll, &mut body);
    }

    /// Deeper than one slot, the laps whose every region has the one `depth - 1` ahead to fill,
    /// run with no branch around the fill, so what it computes that no lap changes (each unit's
    /// place in the stage) leaves the loop; the last laps keep the branches. How many ran.
    fn steady_ahead_laps<T: CubeType, Fill, F>(
        scope: &Scope,
        walk: &WalkExpand,
        stages: &mut StagesExpand<T>,
        total: NativeExpand<usize>,
        fill: &mut Fill,
        compute: &mut F,
    ) -> NativeExpand<usize>
    where
        Fill: FnMut(&Scope, &mut StagesExpand<T>, usize, &RegionExpand),
        F: FnMut(&Scope, &mut SlotExpand<T>, &RegionExpand),
    {
        let depth = stages.depth;
        if depth == 1 {
            return 0usize.into_expand(scope);
        }
        let steady = steady_laps(scope, total, depth - 1, depth);
        let mut steady_body = |scope: &Scope, lap: NativeExpand<usize>| {
            for j in 0..depth {
                let region_idx = lap
                    .__expand_times_method(scope, depth.into_expand(scope))
                    .__expand_plus_method(scope, j.into_expand(scope));
                let ahead = region_idx.__expand_plus_method(scope, (depth - 1).into_expand(scope));
                let prefetch = walk.__expand_region_method(scope, ahead);
                fill(scope, stages, (j + depth - 1) % depth, &prefetch);
                let region = walk.__expand_region_method(scope, region_idx);
                let slot = stages.__expand_slot_mut_method(scope, j);
                compute(scope, slot, &region);
            }
        };
        run_laps(
            scope,
            0usize.into_expand(scope),
            steady,
            walk.unroll,
            &mut steady_body,
        );
        steady
    }
}
