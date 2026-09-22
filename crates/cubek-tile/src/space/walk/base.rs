//! The [`Walk`]: the regions one [`Level`] of a [`Space`] hands the instance running the code.
//! Kernel loops rarely name one: `for cube in space` and `for plane in cube` each build a walk
//! of the next level ([`Region::walk`]); [`over`](Region::over) walks a level stated beside them.
//!
//! A walk is a sequence with random access and nothing else: how many regions this instance owns
//! at the level ([`total`](Walk::total)) and which one an index names ([`region`](Walk::region)).
//! It holds no current region: `for region in walk` is `for i in 0..total { region(i) }`.
//!
//! Its knobs: order ([`reversed`](Walk::reversed)), unrolling ([`unrolled`](Walk::unrolled)), a run
//! of the flat grid ([`range`](Walk::range)), one axis's coordinate ([`routed`](Walk::routed)).
//! Holding several at once is a schedule's job: [`pipelined`](crate::pipelined) indexes it by hand.
//!
//! Each region is a [`Region`], the path of levels from the space to that box; a [`Tile`]
//! windows itself to it with `at`, applying the steps below its own depth.

use cubecl::prelude::*;
use cubecl::unexpanded;

use crate::{Axis, Coords, Known, KnownExpand, Level, Region, RegionExpand, Space};

use super::deal::AxisDeal;
use crate::space::partition::{GridCount, in_plane_axes, swizzled_positions};
use crate::{Spread, StepOrder};

/// The runtime odometer over a [`Space`]'s tiles under one [`Level`].
#[derive(CubeType)]
pub struct Walk {
    /// Per-axis walk counts: this instance's share on `Spatial` axes, the whole grid on
    /// `Sequential` ones.
    counts: Coords<usize>,
    /// Per-axis hardware-instance coordinate, folded through any shared hardware dim;
    /// `0` for `Sequential`. Loop-invariant, so decoded once at construction rather
    /// than per region.
    positions: Coords<usize>,
    /// Per-axis spread factor combining a step with its position: the instance's tile
    /// share (`Contiguous`) or the instance count (`Interleaved`); `1` for `Sequential`.
    scales: Coords<usize>,
    /// Per-axis coordinate the kernel routed rather than the odometer counted, `0` on every axis
    /// it did not ([`routed`](Walk::routed)).
    route: Coords<u32>,
    /// Which axes [`route`](Self::route) speaks for, so the rest fold their digit as ever and pay
    /// nothing. A routed axis takes that coordinate whole: folding in an instance position would
    /// hand each lane of a distributed axis a different one, which is not what naming one means.
    #[cube(comptime)]
    routed_at: Vec<usize>,
    /// Where in this level's flat step space the walk starts. `0` for a whole walk, so it
    /// folds away; a portion of the flat grid ([`range`](Walk::range)) starts at its own.
    base: usize,
    steps: usize,
    /// The path above this walk: what every region it hands out is one level below.
    parent: Region,
    /// The space the regions are cut from, which is also what a ring sizes its slots to
    /// ([`Ring::smem`](crate::Ring::smem)).
    #[cube(comptime)]
    pub(crate) space: Space,
    /// The level this walk steps: the statement the loop made.
    #[cube(comptime)]
    pub(crate) level: Level,
    /// How the level deals each axis of the space, in axis order.
    #[cube(comptime)]
    deals: Vec<AxisDeal>,
    /// Whether iterating this walk unrolls (the one codegen choice folding cannot
    /// make): fragment outputs demand it, memory outputs prefer the compact loop.
    #[cube(comptime)]
    pub(crate) unroll: bool,
    /// The order the steps visit the odometer in ([`reversed`](Walk::reversed)).
    #[cube(comptime)]
    order: StepOrder,
}

#[cube]
impl Walk {
    // The loops are `#[unroll]`ed cube loops over a comptime `Vec`, not host iteration.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn of(space: &Space, #[comptime] level: Level, parent: Region) -> Walk {
        let host = comptime!(space.clone());
        let rank = comptime!(host.rank());
        // A cube level dealing its boxes in an order other than the grid's own decodes its two
        // in-plane axes *together*, from the flat dispatch index, because a swizzle is a joint
        // permutation and the loop below reads one hardware dimension per axis.
        let in_plane = comptime!(
            level
                .order()
                .swizzles()
                .then(|| in_plane_axes(&host, &level))
        );
        let deals = comptime!(
            (0..rank)
                .map(|p| AxisDeal::new(&level, &host, p, in_plane))
                .collect::<Vec<_>>()
        );

        // The grid per axis: a stated count is the constant it states; an every-level's is the
        // extent handed down in its tile, the one division of a launch (folded where the extent
        // is static).
        let mut grid = Coords::<usize>::new();
        #[unroll]
        for p in 0..rank {
            match comptime!(level.grid(space.axis_at(p))) {
                GridCount::Const(n) => grid.push(n.runtime()),
                GridCount::Extent(tile) => grid.push(space.count(p, tile)),
            }
        }

        // Per-axis instance counts, `1` where the axis is walked: the grid itself where every
        // worker takes one tile, the stated count where the grid is dealt across workers in runs.
        // Folded, so a constant grid's decode below folds too (`/1`, `%1` vanish; `%` gets a
        // constant divisor).
        let mut instances = Coords::<usize>::new();
        #[unroll]
        for p in 0..rank {
            match comptime!(deals[p].clone()) {
                AxisDeal::Dealt {
                    across: Some(workers),
                    ..
                } => instances.push(workers.runtime()),
                AxisDeal::Dealt { .. } => instances.push(grid.at(p)),
                AxisDeal::Walked => instances.push(1usize),
            }
        }
        let swizzled = swizzled_positions(
            comptime!(host.clone()),
            comptime!(level.clone()),
            &instances,
        );

        let mut counts = Coords::<usize>::new();
        let mut positions = Coords::<usize>::new();
        let mut scales = Coords::<usize>::new();
        #[unroll]
        for p in 0..rank {
            match comptime!(deals[p].clone()) {
                AxisDeal::Walked => {
                    counts.push(grid.at(p));
                    positions.push(0usize);
                    scales.push(1usize);
                }
                AxisDeal::Dealt {
                    takers,
                    dim,
                    spread,
                    across,
                    divides,
                    inner,
                    unspanned,
                    swizzled: joint,
                } => {
                    // Mixed-radix stride for axes sharing one hardware dim: the product of the
                    // later same-dimension axes' instance counts (the earlier axis is the more
                    // significant digit); `1` when this axis owns its dimension. Both halves: the
                    // odometer is the level's and this space may be a subspace of it, so axes here
                    // carry a possibly-runtime count and unspanned ones a comptime one.
                    let inner_weight = instances.product(inner) * comptime!(unspanned).runtime();
                    let position = match comptime!(joint) {
                        Some(i) => swizzled.at(i),
                        None => AxisDeal::hardware(takers, dim)
                            .divided_by(inner_weight)
                            .remainder(instances.at(p)),
                    };
                    // One tile a worker, or this worker's run of a grid dealt across them, cut
                    // short where the grid does not divide.
                    let run = match comptime!(across) {
                        Some(workers) => grid
                            .at(p)
                            .plus(comptime!(workers - 1).runtime())
                            .divided_by(workers.runtime()),
                        None => 1usize.runtime(),
                    };
                    counts.push(AxisDeal::tiles(
                        grid.at(p),
                        position,
                        instances.at(p),
                        run,
                        spread,
                        divides,
                    ));
                    positions.push(position);
                    match comptime!(spread) {
                        Spread::Contiguous => scales.push(run),
                        Spread::Interleaved => scales.push(instances.at(p)),
                    }
                }
            }
        }

        // Folded, not accumulated: a static walk's total stays a constant, so
        // `#[unroll] for region in walk` can unroll it.
        let steps = counts.product(comptime!((0..rank).collect::<Vec<_>>()));

        Walk {
            counts,
            positions,
            scales,
            route: Coords::constant(comptime!(vec![0; rank])),
            routed_at: comptime!(Vec::new()),
            base: 0usize,
            steps,
            parent,
            order: comptime!(StepOrder::Forward),
            deals,
            space: host,
            level,
            unroll: comptime!(false),
        }
    }

    /// This walk taking one step along `axis`, at the coordinate `coord` states rather than the
    /// one the odometer would have counted.
    ///
    /// What routing is: an operand's window on that axis is placed by a value the kernel read (an
    /// expert per token, a physical page per logical one), not by a loop; the axis keeps its true
    /// extent while the walk visits one coordinate, ordinary to everything below (`at` unchanged).
    ///
    /// The caller owns the coordinate, as it owns [`range`](Walk::range)'s bounds: a value past
    /// the axis's extent windows past the buffer, and nothing here can check it.
    pub fn routed(self, #[comptime] axis: Axis, coord: usize) -> Walk {
        let rank = comptime!(self.space.rank());
        let at = comptime!(self.space.position(axis));
        comptime!(assert!(
            self.space.contains(axis),
            "Walk::routed: {axis:?} is not an axis of this walk's space, so it has no \
             coordinate to state"
        ));

        let mut counts = Coords::<usize>::new();
        let mut route = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            // One of it, so its digit folds to the constant `0` and the routed coordinate is
            // the whole of what `resolve` pushes for this axis.
            //
            // Clamped to the tiles the axis has: the coordinate came from data, so it can name one
            // it does not. Reading the wrong tile beats reading past the buffer, and a refusal is
            // unavailable here (a panic in a cube verb dies on a kernel-expansion thread, unseen).
            if comptime!(p == at) {
                // The axis's own tiles, not `counts`, which on a distributed axis is this
                // instance's share of them and would clamp every lane to its first.
                let last = comptime!(self.level.tiles(&self.space, axis) - 1).runtime();
                counts.push(1usize);
                route.push(coord.min_with(last).retyped::<u32>());
            } else {
                counts.push(self.counts.at(p));
                route.push(self.route.at(p));
            }
        }
        let steps = counts.product(comptime!((0..rank).collect::<Vec<_>>()));

        Walk {
            counts,
            positions: self.positions,
            scales: self.scales,
            route,
            routed_at: comptime!({
                let mut routed_at = self.routed_at.clone();
                routed_at.push(at);
                routed_at
            }),
            base: self.base,
            steps,
            parent: self.parent,
            deals: comptime!(self.deals.clone()),
            space: comptime!(self.space.clone()),
            level: comptime!(self.level.clone()),
            unroll: comptime!(self.unroll),
            order: comptime!(self.order),
        }
    }

    /// Returns the regions count
    pub fn total(&self) -> usize {
        self.steps
    }

    /// Returns the ith region of the walk
    pub fn region(&self, i: usize) -> Region {
        let idx = self
            .base
            .plus(StepOrder::step(i, self.steps, comptime!(self.order)));
        self.parent
            .below(self.resolve(idx), comptime!(self.level.clone()))
    }

    /// Unravel a runtime step `idx` to its per-axis coordinates: each axis's odometer
    /// [`digit`](Walk::digit), [`fold`](Walk::fold)ed with its instance position. A constant `idx`
    /// (an unrolled walk's) folds through, so a static walk's regions can select fragments.
    fn resolve(&self, idx: usize) -> Coords<u32> {
        let mut coords = Coords::<u32>::new();

        #[unroll]
        for p in 0..comptime!(self.space.rank()) {
            if comptime!(self.routed_at.contains(&p)) {
                coords.push(self.route.at(p));
            } else {
                coords.push(self.fold(self.digit(idx, p), p).retyped::<u32>());
            }
        }
        coords
    }

    /// The odometer digit of step `idx` along axis `p` (last axis fastest): divide off
    /// the later axes' counts, keep the remainder of this one. Constant counts fold.
    fn digit(&self, idx: usize, #[comptime] p: usize) -> usize {
        let rank = comptime!(self.space.rank());
        let quot = idx.divided_by(
            self.counts
                .product(comptime!(((p + 1)..rank).collect::<Vec<_>>())),
        );
        // `% count` is a no-op when `idx` has no more significant digit: a range fact that
        // folding (which only sees values) cannot know. Those digits are absent when every earlier
        // count is the constant one (stated, one tile a worker, or unnamed), as their product says.
        //
        // A count of one is the exception: there `% 1` folds to the constant `0`, which `quot`
        // alone would not.
        let count = self.counts.at(p);
        let one = count.constant();
        let earlier = self
            .counts
            .product(comptime!((0..p).collect::<Vec<_>>()))
            .constant();
        if comptime!(one != Some(1) && earlier == Some(1)) {
            quot
        } else {
            quot.remainder(count)
        }
    }

    /// Fold axis `p`'s instance position into its `digit` per the [`Spread`]: an
    /// instance owns a contiguous run (`digit + pos·share`) or the instances take
    /// turns (`digit·instances + pos`); a walked digit passes through.
    fn fold(&self, digit: usize, #[comptime] p: usize) -> usize {
        match comptime!(self.deals[p].clone()) {
            AxisDeal::Walked => digit,
            AxisDeal::Dealt {
                spread: Spread::Contiguous,
                ..
            } => digit.plus(self.positions.at(p).times(self.scales.at(p))),
            AxisDeal::Dealt {
                spread: Spread::Interleaved,
                ..
            } => digit.times(self.scales.at(p)).plus(self.positions.at(p)),
        }
    }
}

/// Iterating a `Walk` visits its regions in order, so `for region in walk` is equivalent to
/// `for i in 0..walk.total() {let region = walk.region(i); ...}`
/// Schedules that need random access (prefetch, double-buffering) still index by hand.
impl IntoIterator for Walk {
    type Item = Region;
    type IntoIter = std::vec::IntoIter<Region>;

    fn into_iter(self) -> Self::IntoIter {
        let mut regions = Vec::new();
        for i in 0..self.total() {
            regions.push(self.region(i));
        }
        regions.into_iter()
    }
}

impl Iterable for WalkExpand {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        let start = 0usize.into_expand(scope);
        let total = self.__expand_total_method(scope);
        let range = RangeExpand::new(start, total);
        if self.unroll {
            range.expand_unroll(scope, &mut |scope, i| {
                body(scope, self.__expand_region_method(scope, i));
            });
        } else {
            range.expand(scope, &mut |scope, i| {
                body(scope, self.__expand_region_method(scope, i));
            });
        }
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        let start = 0usize.into_expand(scope);
        let total = self.__expand_total_method(scope);
        RangeExpand::new(start, total).expand_unroll(scope, &mut |scope, i| {
            body(scope, self.__expand_region_method(scope, i));
        });
    }

    /// The region count when it folds to a constant, which is what lets `for region in walk`
    /// drop the loop around a single region: a level that cuts nothing is one region, walked
    /// straight through rather than under a one-trip loop.
    fn const_len(&self) -> Option<usize> {
        crate::algebra::constant(&self.steps).map(|n| n as usize)
    }
}

/// The settings a walk is told after it is built. They change comptime fields, or swap in the
/// handles a runtime window states, and emit no instruction of their own, so they are written
/// once on the expand type rather than as a `#[cube]` rebuild of every field.
impl Walk {
    /// This walk with its steps visited last to first.
    pub fn reversed(self) -> Walk {
        unexpanded!()
    }

    /// This walk, unrolled when iterated: each region's coordinates fold to comptime
    /// constants (static spaces only; the trip count must be constant).
    pub fn unrolled(self) -> Walk {
        unexpanded!()
    }

    /// This walk over the `steps` regions starting at flat step `base`, rather than all of its
    /// own from zero.
    ///
    /// How a level deals its grid out as contiguous runs, not a rectangular block per axis: every
    /// axis stays `Sequential`, so the flat index carries every coordinate and an instance's share
    /// is a range of it. `base` and `steps` are runtime, so launch-sized runs walk the same loop.
    ///
    /// The caller owns the range: `base + steps` past this walk's own [`total`](Walk::total)
    /// reads coordinates that are not in the grid, and nothing here can check it.
    pub fn range(self, _base: usize, _steps: usize) -> Walk {
        unexpanded!()
    }

    /// The depth of the regions this walk hands out: one below its path.
    pub(crate) fn depth(&self) -> usize {
        self.parent.path.depth() + 1
    }
}

impl WalkExpand {
    pub fn __expand_reversed_method(mut self, _scope: &Scope) -> Self {
        self.order = StepOrder::Reversed;
        self
    }

    pub fn __expand_unrolled_method(mut self, _scope: &Scope) -> Self {
        self.unroll = true;
        self
    }

    pub fn __expand_range_method(
        mut self,
        _scope: &Scope,
        base: NativeExpand<usize>,
        steps: NativeExpand<usize>,
    ) -> Self {
        self.base = base;
        self.steps = steps;
        self
    }

    pub(crate) fn depth(&self) -> usize {
        self.parent.path.depth() + 1
    }
}
