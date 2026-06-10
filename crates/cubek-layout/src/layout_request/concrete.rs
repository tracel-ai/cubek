//! The concrete physical layout a [`LayoutRequest`](super::LayoutRequest) is tested against.

use cubecl::zspace::SmallVec;

use crate::{Axis, MAX_AXES};

/// One axis in a physical layout: its label, logical extent, and storage tile edge (`None`
/// when untiled, i.e. a plain strided axis).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PhysicalAxis {
    axis: Axis,
    extent: usize,
    tile_edge: Option<usize>,
}

impl PhysicalAxis {
    /// A plain strided axis (not storage-tiled).
    pub fn untiled(axis: Axis, extent: usize) -> Self {
        PhysicalAxis {
            axis,
            extent,
            tile_edge: None,
        }
    }

    /// A storage-tiled axis whose leaf block holds `edge` elements.
    pub fn tiled(axis: Axis, extent: usize, edge: usize) -> Self {
        PhysicalAxis {
            axis,
            extent,
            tile_edge: Some(edge),
        }
    }
}

/// A concrete physical layout: its axes in major (outer) to minor (inner) order, the last
/// being innermost/contiguous. Built from a real tensor and its [`Storage`](crate::Storage)
/// at realization; constructed directly in tests.
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

    /// The innermost `n` axes, in any order. Fewer than `n` when the layout has lower rank.
    pub(super) fn minor(&self, n: usize) -> SmallVec<[Axis; MAX_AXES]> {
        let start = self.axes.len().saturating_sub(n);
        self.axes[start..].iter().map(|a| a.axis).collect()
    }

    pub(super) fn tile_edge(&self, axis: Axis) -> Option<usize> {
        self.axes.iter().find(|a| a.axis == axis)?.tile_edge
    }

    pub(super) fn extent(&self, axis: Axis) -> usize {
        self.axes
            .iter()
            .find(|a| a.axis == axis)
            .expect("ConcreteLayout::extent: axis not present")
            .extent
    }
}
