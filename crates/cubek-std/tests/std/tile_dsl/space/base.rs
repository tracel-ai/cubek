//! The coordinate space a tile lives in. An operation's space is the union of
//! its operands' spaces; the axes the output drops are the ones it contracts.

use cubecl::zspace::SmallVec;

use super::{Axis, ByAxis, MAX_AXES};

/// A coordinate space: every axis with its extent. A tile lives in its own space
/// (matmul's `lhs ∈ {M, K}`, `rhs ∈ {K, N}`, `out ∈ {M, N}`); an operation ranges
/// over their [`union`](Space::union), `{M,K} ∪ {K,N} ∪ {M,N} = {M,N,K}`. The
/// axes are ordered: the leaf reads them positionally and a `Point`'s coordinates
/// come in this order.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Space {
    extents: ByAxis<u32>,
}

impl Space {
    /// Build from an ordered `(axis, extent)` list — the canonical axis order.
    pub fn new(extents: &[(Axis, u32)]) -> Self {
        Space {
            extents: ByAxis::new(extents),
        }
    }

    /// Extent (size) of the space along an axis.
    pub fn extent(&self, axis: Axis) -> u32 {
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

    /// The smallest space containing every `part`: each axis any part carries,
    /// with its extent, in first-appearance order. A shared axis must agree on
    /// extent across the parts that carry it — matmul's `K` is the same size in
    /// both operands — and panics otherwise. This is how an operation recovers the
    /// space it ranges over from its operands (`{M,K} ∪ {K,N} ∪ {M,N} = {M,N,K}`).
    pub fn union(parts: &[&Space]) -> Space {
        let mut entries: SmallVec<[(Axis, u32); MAX_AXES]> = SmallVec::new();
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

    /// The axes this space contracts when an operation projects it onto `output`:
    /// those in the space but not in `output`. A matmul over `{M,N,K}` with output
    /// `{M,N}` contracts `{K}`.
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
