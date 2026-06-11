//! The concrete physical layout a [`LayoutRequest`](super::LayoutRequest) is tested against.

use cubecl::zspace::SmallVec;

use crate::{Axis, MAX_AXES};

/// One physical axis (dimension) of a stored buffer: the logical [`Axis`] it belongs to and
/// its extent. Storage tiling is *not* an annotation here — a tiled logical axis contributes
/// several `PhysicalAxis` entries (one per nesting level, outer grid to inner leaf), so tiling
/// is just higher physical rank with the label repeated, mirroring the `[grid…, tile…]` buffer
/// the tile engine ([`Storage`](crate::Storage)) actually launches.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PhysicalAxis {
    axis: Axis,
    extent: usize,
}

impl PhysicalAxis {
    pub fn new(axis: Axis, extent: usize) -> Self {
        PhysicalAxis { axis, extent }
    }
}

/// A concrete physical layout: its axes in major (outer) to minor (inner) order, the last
/// being innermost/contiguous. A storage-tiled axis appears as several entries, level-major
/// (coarse grid outer, leaf inner), so the rank can exceed the number of logical axes. Built
/// from a real tensor and its [`Storage`](crate::Storage) at realization; constructed directly
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

    pub(super) fn innermost(&self) -> Option<Axis> {
        self.axes.last().map(|a| a.axis)
    }

    /// Labels of the largest physical suffix all of whose axes pass `keep`, innermost first.
    /// The innermost contiguous block, used to test the [`Minor`](super::Facet::Minor) facet.
    pub(super) fn inner_block(&self, keep: impl Fn(Axis) -> bool) -> SmallVec<[Axis; MAX_AXES]> {
        let mut block = SmallVec::new();
        for a in self.axes.iter().rev() {
            if !keep(a.axis) {
                break;
            }
            block.push(a.axis);
        }
        block
    }

    /// The leaf (innermost-fragment) extent of `axis` when it is storage-tiled, i.e. it
    /// contributes more than one physical axis; `None` for a plain single-fragment axis.
    pub(super) fn leaf_edge(&self, axis: Axis) -> Option<usize> {
        let frags: SmallVec<[usize; MAX_AXES]> = self
            .axes
            .iter()
            .filter(|a| a.axis == axis)
            .map(|a| a.extent)
            .collect();
        // Untiled (one fragment) is not storage-tiled; the leaf is the innermost (last) one.
        (frags.len() > 1).then(|| *frags.last().unwrap())
    }

    /// Logical extent of `axis`: the product of its physical fragments.
    pub(super) fn extent(&self, axis: Axis) -> usize {
        let mut found = false;
        let mut extent = 1;
        for a in self.axes.iter().filter(|a| a.axis == axis) {
            found = true;
            extent *= a.extent;
        }
        assert!(found, "ConcreteLayout::extent: axis not present");
        extent
    }
}
