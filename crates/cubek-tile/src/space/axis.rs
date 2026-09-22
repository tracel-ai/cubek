//! The opaque axis label every layout and space concept is keyed on. A client gives a label
//! meaning (matmul's `M`/`N`/`K`, reduce's reduce axis); the vocabulary stays agnostic.

use cubecl::zspace::SmallVec;
use serde::{Deserialize, Serialize};

use crate::Space;

/// A labeled axis. The `u8` is a client-assigned index, not a position.
///
/// Serialized as that index, so a client's persisted record (an autotune key naming the axis an
/// operand is contiguous along) can carry one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct Axis(pub u8);

/// A comptime map from [`Axis`] to a value, in declared order: the canonical axis order, and the
/// order a [`Region`](crate::Region)'s coordinates come in.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct AxisMap<T: Copy> {
    entries: SmallVec<[(Axis, T); Space::MAX_RANK]>,
}

impl<T: Copy> AxisMap<T> {
    /// Order is significant.
    pub(crate) fn new(entries: &[(Axis, T)]) -> Self {
        AxisMap {
            entries: SmallVec::from_slice(entries),
        }
    }

    pub(crate) fn get(&self, axis: Axis) -> T {
        self.entries
            .iter()
            .find(|(a, _)| *a == axis)
            .expect("AxisMap::get: axis not present")
            .1
    }

    /// A new map with `f` applied to every value, axis order preserved.
    pub(crate) fn map<U: Copy>(&self, mut f: impl FnMut(Axis, T) -> U) -> AxisMap<U> {
        AxisMap {
            entries: self.entries.iter().map(|&(a, v)| (a, f(a, v))).collect(),
        }
    }

    pub(crate) fn axis_at(&self, i: usize) -> Axis {
        self.entries[i].0
    }

    /// The axes in declared order.
    pub(crate) fn axes(&self) -> impl Iterator<Item = Axis> + '_ {
        self.entries.iter().map(|&(a, _)| a)
    }

    pub(crate) fn position(&self, axis: Axis) -> usize {
        self.entries
            .iter()
            .position(|(a, _)| *a == axis)
            .expect("AxisMap::position: axis not present")
    }

    pub(crate) fn contains(&self, axis: Axis) -> bool {
        self.entries.iter().any(|(a, _)| *a == axis)
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}
