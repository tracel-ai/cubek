//! The coordinate space a tile lives in. An operation's space is the union of
//! its operands' spaces; the axes the output drops are the ones it contracts.

use cubecl::zspace::SmallVec;

use super::{Axis, ByAxis, MAX_AXES};

/// A coordinate space: every axis with its extent, ordered (the leaf and a
/// `Point` read them positionally). A tile lives in its own space (matmul's
/// `lhs ∈ {M,K}`, `rhs ∈ {K,N}`, `out ∈ {M,N}`); an operation ranges over their
/// [`union`](Space::union).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Space {
    extents: ByAxis<usize>,
}

impl Space {
    /// Build from an ordered `(axis, extent)` list — the canonical axis order.
    pub fn new(extents: &[(Axis, usize)]) -> Self {
        Space {
            extents: ByAxis::new(extents),
        }
    }

    /// Extent (size) of the space along an axis.
    pub fn extent(&self, axis: Axis) -> usize {
        self.extents.get(axis)
    }

    /// The axis at position `i` in canonical order.
    pub fn axis_at(&self, i: usize) -> Axis {
        self.extents.axis_at(i)
    }

    /// The canonical-order position of `axis`.
    pub fn position(&self, axis: Axis) -> usize {
        self.extents.position(axis)
    }

    /// The space's dimension (number of axes).
    pub fn rank(&self) -> usize {
        self.extents.len()
    }

    /// Whether `axis` is one of this space's axes.
    pub fn contains(&self, axis: Axis) -> bool {
        self.extents.contains(axis)
    }

    /// The smallest space containing every `part`: each axis carried by any part,
    /// in first-appearance order. A shared axis must agree on extent across parts
    /// (panics otherwise). E.g. `{M,K} ∪ {K,N} ∪ {M,N} = {M,N,K}`.
    pub fn union(parts: &[&Space]) -> Space {
        let mut entries: SmallVec<[(Axis, usize); MAX_AXES]> = SmallVec::new();
        for part in parts {
            let mut i = 0;
            while i < part.rank() {
                let axis = part.axis_at(i);
                let extent = part.extent(axis);
                match entries.iter().find(|(a, _)| *a == axis) {
                    Some((_, seen)) => assert!(
                        *seen == extent,
                        "Space::union: axis appears with conflicting extents"
                    ),
                    None => entries.push((axis, extent)),
                }
                i += 1;
            }
        }
        Space {
            extents: ByAxis::new(&entries),
        }
    }

    /// The subspace over `axes` (in the given order), extents copied from self.
    pub fn select(&self, axes: &[Axis]) -> Space {
        Space::new(
            &axes
                .iter()
                .map(|&a| (a, self.extent(a)))
                .collect::<Vec<_>>(),
        )
    }

    /// The axes in this space but not in `output` — those an operation contracts.
    /// Matmul over `{M,N,K}` with output `{M,N}` contracts `{K}`.
    pub fn contracting(&self, output: &Space) -> SmallVec<[Axis; MAX_AXES]> {
        let mut contracted = SmallVec::new();
        let mut i = 0;
        while i < self.rank() {
            let axis = self.axis_at(i);
            if !output.contains(axis) {
                contracted.push(axis);
            }
            i += 1;
        }
        contracted
    }
}
