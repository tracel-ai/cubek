//! The coordinate space a tile lives in. An operation's space is the merge of
//! its operands' spaces; the axes the output drops are contracted.

use cubecl::zspace::SmallVec;

use crate::Partitioner;

use super::{Axis, ByAxis, MAX_AXES};

/// Every axis with its extent, in canonical order. A tile lives in its own space
/// (matmul's `lhs ∈ {M,K}`, `rhs ∈ {K,N}`, `out ∈ {M,N}`); an operation ranges over
/// their [`merge`](Space::merge).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Space {
    extents: ByAxis<usize>,
    partitioner: Partitioner,
}

impl Space {
    pub fn new(extents: &[(Axis, usize)]) -> Self {
        Space {
            extents: ByAxis::new(extents),
            partitioner: Partitioner::Final,
        }
    }

    /// Chain coarse-to-fine for multi-level tiling; each call appends to the end of
    /// the chain (see [`Partitioner::append`]).
    pub fn with_partitioner(mut self, partitioner: Partitioner) -> Self {
        self.partitioner = self.partitioner.append(partitioner);
        self
    }

    pub fn partitioner(&self) -> &Partitioner {
        &self.partitioner
    }

    pub fn is_final(&self) -> bool {
        self.partitioner.is_final()
    }

    pub fn extent(&self, axis: Axis) -> usize {
        self.extents.get(axis)
    }

    pub fn extent_at(&self, i: usize) -> usize {
        self.extent(self.axis_at(i))
    }

    pub fn axis_at(&self, i: usize) -> Axis {
        self.extents.axis_at(i)
    }

    pub fn position(&self, axis: Axis) -> usize {
        self.extents.position(axis)
    }

    pub fn rank(&self) -> usize {
        self.extents.len()
    }

    pub fn contains(&self, axis: Axis) -> bool {
        self.extents.contains(axis)
    }

    /// The smallest space containing every `part`, axes in first-appearance order. A
    /// shared axis is broadcast-merged via [`merge_level`] (`n ∪ n = n`, `1 ∪ n = n`, else
    /// conflict); an omitted axis broadcasts along all of it. E.g.
    /// `{M,K} ∪ {K,N} ∪ {M,N} = {M,N,K}`.
    pub fn merge(parts: &[&Space]) -> Space {
        let mut entries: SmallVec<[(Axis, usize); MAX_AXES]> = SmallVec::new();

        for part in parts {
            for axis in part.axes() {
                let extent = part.extent(axis);
                match entries.iter_mut().find(|(a, _)| *a == axis) {
                    Some(slot) => slot.1 = merge_level(slot.1, extent),
                    None => entries.push((axis, extent)),
                }
            }
        }
        // Operands of one operation share its partitioner, so the merge carries
        // the first part that has one.
        let partitioner = parts
            .iter()
            .map(|p| &p.partitioner)
            .find(|p| !p.is_final())
            .cloned()
            .unwrap_or(Partitioner::Final);

        Space {
            extents: ByAxis::new(&entries),
            partitioner,
        }
    }

    pub fn project(&self, axes: &[Axis]) -> Space {
        let entries = axes
            .iter()
            .map(|&a| (a, self.extent(a)))
            .collect::<Vec<_>>();
        Space {
            extents: ByAxis::new(&entries),
            partitioner: self.partitioner.clone(),
        }
    }

    /// Tiles along `axis`: `extent / sub-tile edge`.
    pub fn count(&self, axis: Axis) -> usize {
        self.extent(axis) / self.partitioner().edge(axis)
    }

    /// The axes in this space but not in `output`, i.e. those contracted.
    pub fn contracting(&self, output: &Space) -> SmallVec<[Axis; MAX_AXES]> {
        self.axes().filter(|&axis| !output.contains(axis)).collect()
    }

    pub fn axes(&self) -> Axes<'_> {
        Axes { space: self, i: 0 }
    }

    /// The child space one level down: every axis shrunk to its partitioner's sub-tile
    /// edge, that level consumed. Position-free shape; the positions are the [`Walk`].
    pub fn divide(&self) -> Space {
        let entries = self
            .axes()
            .map(|axis| (axis, self.partitioner.edge(axis)))
            .collect::<Vec<_>>();
        Space {
            extents: ByAxis::new(&entries),
            partitioner: self.partitioner.next().clone(),
        }
    }

    /// Divide until no partitioner level is left. Its extents are the finest tile
    /// shape, used to size the staging buffers and to read the final tile's `mr`/`nr`/`kc`.
    pub fn final_space(&self) -> Space {
        let mut space = self.clone();
        while !space.is_final() {
            space = space.divide();
        }
        space
    }

    pub fn tile_size(&self) -> usize {
        self.axes().map(|axis| self.extent(axis)).product()
    }
}

/// Broadcast rule for one axis when [`merge`](Space::merge)ing spaces: equal
/// sizes agree, a `1` yields to the other, anything else conflicts.
fn merge_level(a: usize, b: usize) -> usize {
    match (a, b) {
        (1, b) => b,
        (a, 1) => a,
        (a, b) if a == b => a,
        _ => panic!("Space::merge: axis appears with conflicting extents"),
    }
}

pub struct Axes<'a> {
    space: &'a Space,
    i: usize,
}

impl Iterator for Axes<'_> {
    type Item = Axis;

    fn next(&mut self) -> Option<Axis> {
        if self.i < self.space.rank() {
            let axis = self.space.axis_at(self.i);
            self.i += 1;
            Some(axis)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.space.rank() - self.i;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Axes<'_> {}

impl<'a> IntoIterator for &'a Space {
    type Item = Axis;
    type IntoIter = Axes<'a>;

    fn into_iter(self) -> Axes<'a> {
        self.axes()
    }
}
