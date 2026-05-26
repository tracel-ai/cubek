//! Labeled axes and the per-axis comptime map that keys everything else.

use cubecl::zspace::SmallVec;

/// Maximum number of axes the engine carries in one space. Bump if a problem
/// declares more.
pub const MAX_AXES: usize = 6;

/// An opaque axis label. The engine never matches on a specific one — it only
/// compares, hashes, and looks them up. A client declares its own set (matmul:
/// `M`, `N`, `K`); the count and meaning are the client's, not the engine's.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Axis(pub u8);

/// A comptime map from [`Axis`] to a value, kept in declared order (the space's
/// canonical axis order, and the order a `Point`'s coordinates come in). Backed
/// by a [`SmallVec`] inline up to [`MAX_AXES`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ByAxis<T: Copy> {
    entries: SmallVec<[(Axis, T); MAX_AXES]>,
}

impl<T: Copy> ByAxis<T> {
    /// Build from an ordered `(axis, value)` list. The order is significant: it
    /// defines the axis order the engine iterates and the layout of a `Point`.
    pub fn new(entries: &[(Axis, T)]) -> Self {
        ByAxis {
            entries: SmallVec::from_slice(entries),
        }
    }

    /// The value carried for `axis` (panics if absent).
    pub fn get(&self, axis: Axis) -> T {
        self.entries
            .iter()
            .find(|(a, _)| *a == axis)
            .expect("ByAxis::get: axis not present")
            .1
    }

    /// The axis at position `i` in declared order.
    pub fn axis_at(&self, i: usize) -> Axis {
        self.entries[i].0
    }

    /// The declared-order position of `axis` (panics if absent).
    pub fn position(&self, axis: Axis) -> usize {
        self.entries
            .iter()
            .position(|(a, _)| *a == axis)
            .expect("ByAxis::position: axis not present")
    }

    /// Whether `axis` is carried.
    pub fn contains(&self, axis: Axis) -> bool {
        self.entries.iter().any(|(a, _)| *a == axis)
    }

    /// Number of axes carried.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}
