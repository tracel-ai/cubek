//! One axis's size ([`Extent`]), and every axis's at once ([`Shape`]): the comptime half of a
//! [`Space`](crate::Space), the same value on the host and in a kernel.

use crate::{Axis, AxisMap};

/// One axis's size.
/// `Static` is a comptime constant (a tile edge);
/// `Dynamic` is a runtime scalar resolved in-kernel from the tensor shape.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Extent {
    Static(usize),
    Dynamic,
}

impl Extent {
    /// The comptime size; panics on `Dynamic` (a runtime extent has no comptime value;
    /// resolve it from the tensor shape).
    pub fn get(self) -> usize {
        match self {
            Extent::Static(n) => n,
            Extent::Dynamic => {
                panic!("Extent::get: this axis is Dynamic; its size is only known at runtime")
            }
        }
    }

    pub fn is_dynamic(self) -> bool {
        matches!(self, Extent::Dynamic)
    }
}

/// Every axis with its comptime extent, in canonical order: what a space is at compile time,
/// and what its identity is. A `Dynamic` axis has its size beside it at runtime, on the
/// [`Space`](crate::Space).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Shape {
    extents: AxisMap<Extent>,
}

impl Shape {
    pub(crate) fn new(extents: &[(Axis, Extent)]) -> Self {
        Shape {
            extents: AxisMap::new(extents),
        }
    }

    pub fn rank(&self) -> usize {
        self.extents.len()
    }

    pub fn axis_at(&self, i: usize) -> Axis {
        self.extents.axis_at(i)
    }

    pub fn position(&self, axis: Axis) -> usize {
        self.extents.position(axis)
    }

    pub fn contains(&self, axis: Axis) -> bool {
        self.extents.contains(axis)
    }

    pub fn axes(&self) -> impl Iterator<Item = Axis> + '_ {
        self.extents.axes()
    }

    pub(crate) fn extent_raw(&self, axis: Axis) -> Extent {
        self.extents.get(axis)
    }

    /// The axis's comptime size; panics on a [`Dynamic`](Extent::Dynamic) axis. The leaf and
    /// smem consumers all run on fully-divided (`Static`) spaces, so this is what they call.
    pub fn extent(&self, axis: Axis) -> usize {
        self.extent_raw(axis).get()
    }

    pub fn extent_at(&self, i: usize) -> usize {
        self.extent(self.axis_at(i))
    }

    pub fn is_dynamic(&self, axis: Axis) -> bool {
        self.extent_raw(axis).is_dynamic()
    }

    /// Every axis is [`Static`](Extent::Static), so a walk over it is fully comptime.
    pub(crate) fn is_static(&self) -> bool {
        self.axes().all(|axis| !self.is_dynamic(axis))
    }

    /// The listed axes with their extents, in the order listed.
    pub(crate) fn subspace(&self, axes: &[Axis]) -> Shape {
        let entries: Vec<_> = axes.iter().map(|&a| (a, self.extent_raw(a))).collect();
        Shape::new(&entries)
    }
}
