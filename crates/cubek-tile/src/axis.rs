//! The opaque axis label every layout and space concept is keyed on, and the per-axis
//! map built on it. A client gives a label meaning (matmul's `M`/`N`/`K`, reduce's reduce
//! axis); the vocabulary stays agnostic.

use cubecl::zspace::SmallVec;

/// Inline capacity for per-axis allocations in small vectors (spills to heap if exceeded).
pub(crate) const MAX_AXES: usize = 6;

/// Inline capacity for per-level allocations in small vectors (spills to heap if exceeded).
pub(crate) const MAX_LEVELS: usize = 6;

/// A labeled axis. The `u8` is a client-assigned index, not a position.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Axis(pub u8);

/// A comptime map from [`Axis`] to a value, in declared order. This is the
/// canonical axis order and the order a [`Region`](crate::Region)'s coordinates come in.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ByAxis<T: Copy> {
    entries: SmallVec<[(Axis, T); MAX_AXES]>,
}

impl<T: Copy> ByAxis<T> {
    /// Order is significant.
    pub fn new(entries: &[(Axis, T)]) -> Self {
        ByAxis {
            entries: SmallVec::from_slice(entries),
        }
    }

    pub fn get(&self, axis: Axis) -> T {
        self.entries
            .iter()
            .find(|(a, _)| *a == axis)
            .expect("ByAxis::get: axis not present")
            .1
    }

    /// A new map with `f` applied to every value, axis order preserved.
    pub fn map<U: Copy>(&self, mut f: impl FnMut(Axis, T) -> U) -> ByAxis<U> {
        ByAxis {
            entries: self.entries.iter().map(|&(a, v)| (a, f(a, v))).collect(),
        }
    }

    pub fn axis_at(&self, i: usize) -> Axis {
        self.entries[i].0
    }

    /// The values in axis order.
    pub fn values(&self) -> impl Iterator<Item = T> + '_ {
        self.entries.iter().map(|&(_, v)| v)
    }

    pub fn position(&self, axis: Axis) -> usize {
        self.entries
            .iter()
            .position(|(a, _)| *a == axis)
            .expect("ByAxis::position: axis not present")
    }

    pub fn contains(&self, axis: Axis) -> bool {
        self.entries.iter().any(|(a, _)| *a == axis)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
