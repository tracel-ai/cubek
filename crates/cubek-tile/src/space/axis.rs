//! The per-axis comptime map keyed on [`Axis`].

use cubecl::zspace::SmallVec;

use crate::{Axis, MAX_AXES};

/// A comptime map from [`Axis`] to a value, in declared order. This is the
/// canonical axis order and the order a [`Region`](super::Region)'s coordinates come in.
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
