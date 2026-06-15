//! The coordinate space a tile lives in. An operation's space is the merge of
//! its operands' spaces; the axes the output drops are contracted.

use cubecl::zspace::SmallVec;

use crate::{Axis, MAX_AXES, Partitioner};

use super::ByAxis;

/// One axis's size. `Static` is a comptime constant (a tile edge, or a problem dim we
/// deliberately specialize on); `Dynamic` is a runtime scalar resolved in-kernel from the
/// tensor shape. A `Dynamic` axis carries no value, so two problem shapes that differ only
/// in their dynamic dims produce the *same* `Space` — hence one compiled kernel rather than
/// one per shape. [`divide`](Space::divide) always yields `Static` children (a sub-tile edge
/// is comptime), so dynamism only ever lives at the top level.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Extent {
    Static(usize),
    Dynamic,
}

impl Extent {
    /// The comptime size; panics on `Dynamic` (a runtime extent has no comptime value —
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

/// Every axis with its extent, in canonical order. A tile lives in its own space
/// (matmul's `lhs ∈ {M,K}`, `rhs ∈ {K,N}`, `out ∈ {M,N}`); an operation ranges over
/// their [`merge`](Space::merge).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Space {
    extents: ByAxis<Extent>,
    partitioner: Partitioner,
}

impl Space {
    pub fn new(extents: &[(Axis, usize)]) -> Self {
        let extents: Vec<_> = extents
            .iter()
            .map(|&(a, n)| (a, Extent::Static(n)))
            .collect();
        Space::from_extents(&extents)
    }

    /// Construct directly from [`Extent`]s (the form `merge`/`project`/`divide` round-trip).
    pub fn from_extents(extents: &[(Axis, Extent)]) -> Self {
        Space {
            extents: ByAxis::new(extents),
            partitioner: Partitioner::Final,
        }
    }

    /// Flip the listed axes to [`Dynamic`](Extent::Dynamic), keeping the partitioner. The
    /// launch side computes geometry from the concrete (real-extent) space, then derives the
    /// kernel's space with this so distinct input shapes hit one compiled kernel.
    pub fn with_dynamic(mut self, axes: &[Axis]) -> Self {
        let entries: Vec<_> = self
            .axes()
            .map(|a| {
                let extent = if axes.contains(&a) {
                    Extent::Dynamic
                } else {
                    self.extents.get(a)
                };
                (a, extent)
            })
            .collect();
        self.extents = ByAxis::new(&entries);
        self
    }

    /// Every axis [`Dynamic`]: the kernel form for an operation whose problem dims are all
    /// runtime (the common case — see [`with_dynamic`](Space::with_dynamic)).
    pub fn all_dynamic(self) -> Self {
        let axes: Vec<_> = self.axes().collect();
        self.with_dynamic(&axes)
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

    /// The axis's comptime size; panics on a [`Dynamic`](Extent::Dynamic) axis. The leaf and
    /// smem consumers all run on fully-divided (`Static`) spaces, so this is what they call.
    pub fn extent(&self, axis: Axis) -> usize {
        self.extents.get(axis).get()
    }

    pub fn extent_raw(&self, axis: Axis) -> Extent {
        self.extents.get(axis)
    }

    pub fn is_dynamic(&self, axis: Axis) -> bool {
        self.extents.get(axis).is_dynamic()
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
        let mut entries: SmallVec<[(Axis, Extent); MAX_AXES]> = SmallVec::new();

        for part in parts {
            for axis in part.axes() {
                let extent = part.extent_raw(axis);
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
            .map(|&a| (a, self.extent_raw(a)))
            .collect::<Vec<_>>();
        Space {
            extents: ByAxis::new(&entries),
            partitioner: self.partitioner.clone(),
        }
    }

    /// Tiles along `axis`: `ceil(extent / sub-tile edge)`, so an indivisible axis gets a
    /// trailing partial tile (its overhang is masked at read/write).
    pub fn count(&self, axis: Axis) -> usize {
        self.extent(axis).div_ceil(self.partitioner().edge(axis))
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
        // A sub-tile edge is always comptime, so a child is fully `Static` whatever the
        // parent was: dynamism lives only at the top level.
        let entries = self
            .axes()
            .map(|axis| (axis, Extent::Static(self.partitioner.edge(axis))))
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

/// Broadcast rule for one axis when [`merge`](Space::merge)ing spaces: equal sizes agree, a
/// static `1` yields to the other, anything else conflicts. A `Dynamic` axis subsumes any
/// non-broadcast operand — its runtime size is the merged one — so the merge stays dynamic.
fn merge_level(a: Extent, b: Extent) -> Extent {
    match (a, b) {
        (Extent::Static(1), b) => b,
        (a, Extent::Static(1)) => a,
        (Extent::Dynamic, _) | (_, Extent::Dynamic) => Extent::Dynamic,
        (Extent::Static(a), Extent::Static(b)) if a == b => Extent::Static(a),
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
