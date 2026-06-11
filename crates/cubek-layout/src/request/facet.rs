//! The constraint facets and the any-of [`AxisSet`] they range over.

use cubecl::zspace::SmallVec;

use crate::{Axis, MAX_AXES};

use super::ConcreteLayout;

/// An any-of set of axes. A [`Facet`] over an `AxisSet` is satisfied when *any* member
/// satisfies it, so `Innermost({M, K})` accepts both row- and col-major.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct AxisSet(SmallVec<[Axis; MAX_AXES]>);

impl AxisSet {
    pub fn new(axes: &[Axis]) -> Self {
        AxisSet(SmallVec::from_slice(axes))
    }

    pub fn one(axis: Axis) -> Self {
        AxisSet::new(&[axis])
    }

    pub fn contains(&self, axis: Axis) -> bool {
        self.0.contains(&axis)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = Axis> + '_ {
        self.0.iter().copied()
    }
}

/// One facet of a desired physical layout. Every variant is extent-independent: tile edges
/// and divisors are hardware constants, clamped or padded only at realization.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Facet {
    /// One of these axes is the innermost (contiguous) physical axis, i.e. where
    /// vectorization lands.
    Innermost(AxisSet),
    /// These axes occupy the innermost physical slots (row/col-major, batch-trailing), in
    /// any order among themselves.
    Minor(AxisSet),
    /// Split `axis` into storage tiles of `edge`, raising the physical rank. A finer tiling
    /// (edge a multiple of the request's) also satisfies it.
    Tiled { axis: Axis, edge: usize },
    /// `axis`'s logical extent is a multiple of `by` (pad on relayout).
    Divisible { axis: Axis, by: usize },
}

impl Facet {
    /// Whether this facet is satisfied by a concrete physical layout.
    pub(super) fn holds(&self, layout: &ConcreteLayout) -> bool {
        match self {
            Facet::Innermost(set) => layout.innermost().map(|a| set.contains(a)).unwrap_or(false),
            // The set occupies the innermost physical slots: the largest in-set suffix covers
            // every member, so no foreign axis is more inner than any of them (order-free).
            Facet::Minor(set) => {
                let block = layout.inner_block(|a| set.contains(a));
                set.iter().all(|a| block.contains(&a))
            }
            Facet::Tiled { axis, edge } => match layout.leaf_edge(*axis) {
                Some(leaf) => leaf % edge == 0,
                None => false,
            },
            Facet::Divisible { axis, by } => layout.extent(*axis).is_multiple_of(*by),
        }
    }
}
