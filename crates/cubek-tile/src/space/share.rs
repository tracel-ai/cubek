//! What the hardware instances are to a tile's cells, once a level has been distributed out.
//!
//! Two questions, at two scopes. A plane's units share registers and combine there, so a folding
//! drain asks what each unit holds of a cell ([`UnitShare`]), and folds the partials with
//! [`UnitShare::fold`]. Planes and cubes share none; the one question is whether an instance holds
//! a whole cell ([`SplitShare`]).
//!
//! Both are read off a [`Level`] against the space an operand spans, level by level on the way
//! down a partitioning ([`under`](UnitShare::under)), since the level that spreads an axis is only
//! known then.

use cubecl::prelude::*;

use crate::{Axis, Carrier, ComputeScope, Count, Coverage, Level, Monoid, Space};

/// What the plane's units each hold of a tile's cells, once a units level is distributed out. An axis
/// the tile does not span is *folded* (units cover disjoint slices, each holds a partial); one it
/// does span is *carried* (each unit gets a different cell). The case says how a partial drains.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UnitShare {
    /// Nothing rides the units, so every unit repeats the same work over the same whole cells: a
    /// store lands the same value however many make it, a fold must land once.
    Repeated,
    /// The units carry cells of their own and fold nothing: each reads and writes as it is.
    Whole,
    /// Nothing carried, everything folded: every unit of the plane holds a partial of the *same*
    /// cell, and the plane's own reduction is the drain.
    Plane,
    /// Both, so the plane splits into groups: one cell each, several cells in flight at once.
    /// `fold_mask` is the set of unit-index bits the folded axes occupy, so a cell's partials
    /// live on exactly the units that agree outside it and differ inside.
    Group { fold_mask: usize },
}

impl CubeDebug for UnitShare {}
impl UnitShare {
    /// What `level` makes the plane's units to the cells of an operand spanning `spanned`: an
    /// axis the operand does not span is folded (units hold partials), one it spans is carried.
    /// A level that is not the units' leaves the units as they were ([`Repeated`](Self::Repeated)).
    pub fn new(level: &Level, spanned: &Space) -> UnitShare {
        if level.coverage() != Coverage::Distribute(ComputeScope::Unit) {
            return UnitShare::Repeated;
        }
        // Innermost first, so `weight` is the axis's stride in the unit index as it is reached,
        // the same least-significant-last ordering the walk decodes with.
        let (mut weight, mut fold_mask) = (1usize, 0usize);
        for axis in level.axes().into_iter().rev() {
            // Distributed to however many units the launch runs, and the level's only unit axis:
            // carried where the operand spans it, and where it does not, every unit holds a
            // partial of the same cell.
            if let Count::Distributed(_) = level.cut(axis).count {
                match spanned.contains(axis) {
                    true => continue,
                    false => return UnitShare::Plane,
                }
            }
            let plane_units = level
                .cut(axis)
                .count
                .stated()
                .expect("Level::new: a units level carries stated counts");
            if plane_units == 1 {
                continue;
            }
            assert!(
                plane_units.is_power_of_two(),
                "UnitShare: {axis:?} rides {plane_units} units, which is not a power of two, so its \
                 partials are not a bit range"
            );
            if !spanned.contains(axis) {
                fold_mask |= (plane_units - 1) * weight;
            }
            weight *= plane_units;
        }
        match fold_mask {
            // Nothing rides the units at all when every count is one.
            0 if weight == 1 => UnitShare::Repeated,
            0 => UnitShare::Whole,
            // Every unit's bit folded: nothing is carried, so the plane shares the one cell.
            mask if mask == weight - 1 => UnitShare::Plane,
            fold_mask => UnitShare::Group { fold_mask },
        }
    }

    /// This share under `parent`'s: the folds compose, since each level takes its own bits of the
    /// unit index, and units that once carried cells of their own keep doing so.
    /// [`Plane`](Self::Plane) spans every unit, so nothing folds under it.
    pub fn under(self, parent: UnitShare) -> UnitShare {
        match (parent, self) {
            (UnitShare::Repeated, share) | (share, UnitShare::Repeated) => share,
            (UnitShare::Whole, share) | (share, UnitShare::Whole) => share,
            (UnitShare::Group { fold_mask: a }, UnitShare::Group { fold_mask: b }) => {
                UnitShare::Group { fold_mask: a | b }
            }
            _ => panic!("UnitShare::under: {self:?} under {parent:?}: nothing folds under a plane"),
        }
    }

    /// Whether a cell's partials sit on several units, so a drain has to fold before it writes.
    pub fn folds(self) -> bool {
        matches!(self, UnitShare::Plane | UnitShare::Group { .. })
    }

    /// The units over which `axis` of a units level is distributed, as a share: the whole plane where
    /// they are all of it, a group otherwise. What a row-owning verb asks of the units level it
    /// runs under.
    pub fn of_units(plane_units: usize, plane: usize) -> UnitShare {
        match plane_units {
            1 => UnitShare::Repeated,
            n if n == plane => UnitShare::Plane,
            n => {
                assert!(
                    n.is_power_of_two() && n < plane,
                    "UnitShare::of_units: a group of {n} units in a plane of {plane} is not a bit \
                     range of the unit index"
                );
                UnitShare::Group { fold_mask: n - 1 }
            }
        }
    }
}

#[cube]
impl UnitShare {
    /// Combine the partials of one cell under `monoid`, leaving every unit that holds one holding
    /// the total: the plane instruction where the whole plane shares the cell, a butterfly over the
    /// group's unit bits where a group does, `value` itself where nothing is folded.
    ///
    /// One butterfly step per bit of the group's mask. A cell's partials sit on the units that
    /// agree outside the mask and differ inside it, so an xor by a single mask bit stays within the
    /// group: every group folds at once, each over its own cell, with no guard and no branch.
    pub fn fold_of<T: Carrier + CubePrimitive<Scalar: PlaneNumeric>>(
        value: T,
        #[comptime] share: UnitShare,
        #[comptime] monoid: Monoid,
    ) -> T {
        match comptime!(share) {
            UnitShare::Repeated | UnitShare::Whole => value,
            UnitShare::Plane => match comptime!(monoid) {
                Monoid::Sum => plane_sum(value),
                Monoid::Prod => plane_prod(value),
                Monoid::Max => plane_max(value),
                Monoid::Min => plane_min(value),
            },
            UnitShare::Group { fold_mask } => {
                let mut total = value;
                #[unroll]
                for bit in 0..comptime!(usize::BITS - fold_mask.leading_zeros()) {
                    if comptime!(fold_mask & (1 << bit) != 0) {
                        total = monoid
                            .combine::<T>(total, plane_shuffle_xor(total, comptime!(1u32 << bit)));
                    }
                }
                total
            }
        }
    }
}

impl UnitShare {
    /// [`fold_of`](Self::fold_of) as a method on the share.
    pub fn fold<T: Carrier + CubePrimitive<Scalar: PlaneNumeric>>(
        self,
        value: T,
        monoid: Monoid,
    ) -> T {
        UnitShare::fold_of::<T>(value, self, monoid)
    }

    pub fn __expand_fold_method<T: Carrier + CubePrimitive<Scalar: PlaneNumeric>>(
        self,
        scope: &Scope,
        value: T::ExpandType,
        monoid: Monoid,
    ) -> T::ExpandType {
        UnitShare::__expand_fold_of::<T>(scope, value, self, monoid)
    }
}

/// What one instance holds of a tile's cells, across the scopes whose instances can only meet in
/// the destination: `Plane` and `Cube`. Coarser than [`UnitShare`], whose units share registers
/// and elect a writer (a mask); planes and cubes share none, so each folds its own contribution.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SplitShare {
    /// Every cell this instance writes is its own outright, so the drain is a store.
    Whole,
    /// Several instances hold partials of the same cell, so the drain has to fold rather than
    /// store. A contraction cut at plane or cube scope is the way to get here.
    Partial,
}

impl SplitShare {
    /// What one instance of an operand spanning `spanned` holds of its cells after `level` is
    /// distributed out over `space`: [`Partial`](SplitShare::Partial) where a plane or cube axis the
    /// operand does not span is distributed across several instances, so each contracts a slice.
    ///
    /// Asked with the level's whole space, not the operand's projection: a projection has dropped
    /// the contracted axis and so cannot tell a split from a cut whose edge is the whole axis.
    /// Conservative where the count is not comptime: whole would lose every partial but one.
    pub fn new(level: &Level, space: &Space, spanned: &Space) -> SplitShare {
        match level.coverage() {
            Coverage::Walk | Coverage::Distribute(ComputeScope::Unit) => return SplitShare::Whole,
            Coverage::Distribute(ComputeScope::Plane)
            | Coverage::Distribute(ComputeScope::Cube) => {}
        }
        // A grid shared as one index is not distributed by axis: a share of it covers part of a cell
        // whenever the index runs over an axis the operand does not span, and which part is not
        // something the per-axis cuts record.
        if level.shared_by().is_some() {
            let unspanned = level.axes().iter().any(|axis| !spanned.contains(*axis));
            return match unspanned {
                true => SplitShare::Partial,
                false => SplitShare::Whole,
            };
        }
        // An axis the operand spans is carried, not split: it gives each instance a cell of its
        // own rather than a slice of one.
        let split = level.axes().into_iter().any(|axis: Axis| {
            !spanned.contains(axis) && level.instances_along(space, axis) != Some(1)
        });
        match split {
            true => SplitShare::Partial,
            false => SplitShare::Whole,
        }
    }

    /// This share under `parent`'s: partial stays partial.
    pub fn under(self, parent: SplitShare) -> SplitShare {
        match (parent, self) {
            (SplitShare::Whole, SplitShare::Whole) => SplitShare::Whole,
            (SplitShare::Partial, _) | (_, SplitShare::Partial) => SplitShare::Partial,
        }
    }
}

/// A count a units level states along an axis, as the units it takes.
impl From<Count> for usize {
    fn from(count: Count) -> usize {
        count
            .stated()
            .expect("a units level's count is stated; every tile is the launch's")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Levels, Space};

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// A contraction distributed out across cubes leaves each of them a slice of every output cell.
    /// Read off the *output's* subspace, which does not span `K`, against the level's whole
    /// space, which still names it.
    #[test]
    fn a_cube_cut_contraction_is_partial_to_the_output() {
        let space = Space::new(&[(M, 4), (N, 4), (K, 8)]);
        let level = Levels::leaf(&[(K, 4)]).cubes(&[K]).level();
        assert_eq!(
            SplitShare::new(&level, &space, &space.subspace(&[M, N])),
            SplitShare::Partial
        );
        // The operands span `K`, so their own cells are whole: nothing about a split is
        // visible from a space that covers the axis being split.
        assert_eq!(
            SplitShare::new(&level, &space, &space.subspace(&[M, K])),
            SplitShare::Whole
        );
    }

    /// The same at plane scope: planes of one cube share no registers either, so a contraction
    /// distributed out across them leaves each holding a slice, exactly as cubes do.
    #[test]
    fn a_plane_cut_contraction_is_partial_to_the_output() {
        let space = Space::new(&[(M, 4), (N, 4), (K, 8)]);
        let level = Levels::leaf(&[(K, 4)]).planes(&[(K, 2)]).level();
        assert_eq!(
            SplitShare::new(&level, &space, &space.subspace(&[M, N])),
            SplitShare::Partial
        );
    }

    /// A grid shared as one index is not an axis, and the index runs over the contraction: a share
    /// of it can start and end inside a cell's contraction, so the output is partial even though
    /// no axis of the level rides the cubes.
    #[test]
    fn a_shared_grid_is_partial_to_the_output() {
        let space = Space::new(&[(M, 8), (N, 8), (K, 8)]);
        let level = Levels::leaf(&[(M, 4), (N, 4), (K, 8)])
            .cubes(&[M, N, K])
            .shared_by(3)
            .level();
        assert_eq!(
            SplitShare::new(&level, &space, &space.subspace(&[M, N])),
            SplitShare::Partial
        );
        // An operand spanning every axis of the grid holds whole cells of its own, the same way
        // it does under a cut.
        assert_eq!(SplitShare::new(&level, &space, &space), SplitShare::Whole);
    }

    /// A cube cut whose edge is the whole axis distributes out one tile, so it is not a split at all.
    /// The level's whole space is asked because a mapping parameterised by its split count writes
    /// the same cut at `splits` one, and refusing it refuses the control it is compared against.
    #[test]
    fn a_cube_cut_of_the_whole_axis_is_not_a_split() {
        let space = Space::new(&[(M, 4), (N, 4), (K, 8)]);
        let level = Levels::leaf(&[(N, 1), (K, 8)]).cubes(&[N, K]).level();
        assert_eq!(
            SplitShare::new(&level, &space, &space.subspace(&[M, N])),
            SplitShare::Whole
        );
    }

    /// The same cut on an axis the output *does* span is a plain output split: each cube owns
    /// its columns outright and there is nothing to combine.
    #[test]
    fn a_cube_cut_output_axis_stays_whole() {
        let space = Space::new(&[(M, 4), (N, 8), (K, 4)]);
        let level = Levels::leaf(&[(N, 4)]).cubes(&[N]).level();
        assert_eq!(
            SplitShare::new(&level, &space, &space.subspace(&[M, N])),
            SplitShare::Whole
        );
    }

    /// A units level over the contraction alone folds every unit's bit: the plane shares the cell.
    /// One over an axis the operand spans carries; a units level naming both is a group per cell.
    #[test]
    fn a_units_level_folds_what_the_operand_does_not_span() {
        let space = Space::new(&[(M, 4), (N, 8), (K, 32)]);
        let out = space.subspace(&[M, N]);
        let plane = Levels::leaf(&[(K, 1)]).units(&[(K, 32)]).level();
        assert_eq!(UnitShare::new(&plane, &out), UnitShare::Plane);
        let whole = Levels::leaf(&[(N, 1)]).units(&[(N, 8)]).level();
        assert_eq!(UnitShare::new(&whole, &out), UnitShare::Whole);
        let team = Levels::leaf(&[(N, 1), (K, 1)])
            .units(&[(N, 8), (K, 4)])
            .level();
        assert_eq!(
            UnitShare::new(&team, &out),
            UnitShare::Group { fold_mask: 3 }
        );
        let walk = Levels::leaf(&[(K, 8)]).walk_every(&[K]).level();
        assert_eq!(UnitShare::new(&walk, &out), UnitShare::Repeated);
    }

    /// Descending composes the folds and keeps the carry.
    #[test]
    fn shares_compose_level_by_level() {
        let group = UnitShare::Group { fold_mask: 3 };
        assert_eq!(group.under(UnitShare::Repeated), group);
        assert_eq!(UnitShare::Whole.under(group), group);
        assert_eq!(
            UnitShare::Group { fold_mask: 12 }.under(group),
            UnitShare::Group { fold_mask: 15 }
        );
        assert_eq!(
            UnitShare::Repeated.under(UnitShare::Whole),
            UnitShare::Whole
        );
        assert_eq!(
            SplitShare::Whole.under(SplitShare::Partial),
            SplitShare::Partial
        );
    }
}
