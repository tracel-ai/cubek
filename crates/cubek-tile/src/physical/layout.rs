//! The concrete physical layout of a stored buffer: its axes in major-to-minor order, with a
//! storage-tiled axis contributing several fragments. Built from a real tensor and its
//! [`Projection`](crate::Projection) at realization; constructed directly in tests.

use cubecl::zspace::SmallVec;

use crate::{Axis, MAX_AXES};

/// One physical axis (dimension) of a stored buffer: the logical [`Axis`] it belongs to and
/// its extent. Storage tiling is *not* an annotation here; a tiled logical axis contributes
/// several `PhysicalAxis` entries (one per nesting level, outer grid to inner leaf), so tiling
/// is just higher physical rank with the label repeated, mirroring the `[grid…, tile…]` buffer
/// the tile engine ([`Projection`](crate::Projection)) actually launches.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PhysicalAxis {
    axis: Axis,
    extent: usize,
}

impl PhysicalAxis {
    pub fn new(axis: Axis, extent: usize) -> Self {
        PhysicalAxis { axis, extent }
    }

    pub fn axis(&self) -> Axis {
        self.axis
    }

    pub fn extent(&self) -> usize {
        self.extent
    }
}

/// A concrete physical layout: its axes in major (outer) to minor (inner) order, the last
/// being innermost/contiguous. A storage-tiled axis appears as several entries, level-major
/// (coarse grid outer, leaf inner), so the rank can exceed the number of logical axes. Built
/// from a real tensor and its [`Projection`](crate::Projection) at realization; constructed directly
/// in tests.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ConcreteLayout {
    axes: SmallVec<[PhysicalAxis; MAX_AXES]>,
}

impl ConcreteLayout {
    /// `axes` listed major-to-minor; the last is innermost.
    pub fn new(axes: &[PhysicalAxis]) -> Self {
        ConcreteLayout {
            axes: SmallVec::from_slice(axes),
        }
    }

    /// The physical axes, major-to-minor (a storage-tiled logical axis repeats, one per level).
    pub fn axes(&self) -> &[PhysicalAxis] {
        &self.axes
    }

    /// The distinct logical axes in first-occurrence order, the axes the operand spans, with each
    /// storage-tiled axis (which contributes several physical fragments) collapsed to one entry.
    pub fn distinct_axes(&self) -> SmallVec<[Axis; MAX_AXES]> {
        let mut out = SmallVec::new();
        for a in &self.axes {
            if !out.contains(&a.axis) {
                out.push(a.axis);
            }
        }
        out
    }
}
