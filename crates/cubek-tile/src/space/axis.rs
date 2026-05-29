//! Labeled axes and the per-axis comptime map that keys everything else.

use cubecl::zspace::SmallVec;

/// Maximum number of axes the engine carries in one space. Bump if a problem
/// declares more.
pub const MAX_AXES: usize = 6;

/// An opaque axis label. The engine only compares/hashes/looks up axes, never
/// matching a specific one; a client declares its own set (matmul: `M`, `N`, `K`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Axis(pub u8);

/// A comptime map from [`Axis`] to a value, in declared order (the canonical axis
/// order, and the order a `Point`'s coordinates come in).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ByAxis<T: Copy> {
    entries: SmallVec<[(Axis, T); MAX_AXES]>,
}

impl<T: Copy> ByAxis<T> {
    /// Build from an ordered `(axis, value)` list; the order is significant.
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

    /// Whether no axes are carried.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
