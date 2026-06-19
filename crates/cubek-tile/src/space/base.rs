//! The coordinate space a tile lives in. An operation's space is the merge of
//! its operands' spaces; the axes the output drops are contracted.

use cubecl::prelude::*;
use cubecl::zspace::SmallVec;

use crate::{Axis, MAX_AXES, Partitioner};

use super::ByAxis;

/// One axis's size.
/// `Static` is a comptime constant (a tile edge);
/// `Dynamic` is a runtime scalar resolved in-kernel from the tensor shape.
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

/// Every axis's extent: the comptime `kinds` (`Static(n)` | `Dynamic`) plus, for the `Dynamic` ones,
/// their runtime `sizes`. The kinds stay comptime so static tile counts fold and the walk unrolls;
/// the sizes are the runtime half a `Dynamic` axis needs, which a comptime `Extent` can't hold. Only
/// the top operation space carries any sizes (filled from the operands); `divide` yields `Static`
/// children, so the whole interior has none.
#[derive(CubeType, Clone, Debug)]
pub struct Extents {
    #[cube(comptime)]
    kinds: ByAxis<Extent>,
    sizes: Sequence<usize>,
}

impl Extents {
    /// A fully-`Static` (or yet-unresolved) extents — no runtime sizes.
    fn fixed(kinds: ByAxis<Extent>) -> Self {
        Extents {
            kinds,
            sizes: Sequence::new(),
        }
    }

    fn get(&self, axis: Axis) -> Extent {
        self.kinds.get(axis)
    }
    fn axis_at(&self, i: usize) -> Axis {
        self.kinds.axis_at(i)
    }
    fn position(&self, axis: Axis) -> usize {
        self.kinds.position(axis)
    }
    fn contains(&self, axis: Axis) -> bool {
        self.kinds.contains(axis)
    }
    fn len(&self) -> usize {
        self.kinds.len()
    }
}

#[cube]
impl Extents {
    /// Axis `p`'s tile count for a sub-tile `edge`: a `Static` axis folds to a comptime constant (so
    /// the walk loop unrolls), a `Dynamic` axis ceil-divides its runtime size. The `Static`/`Dynamic`
    /// match is comptime, so an all-`Static` extents never touches `sizes`.
    pub fn count(&self, #[comptime] p: usize, #[comptime] edge: usize) -> usize {
        match comptime!(self.kinds.get(self.kinds.axis_at(p))) {
            Extent::Static(n) => comptime!(n.div_ceil(edge)).runtime(),
            Extent::Dynamic => (*self.sizes.index(p)).div_ceil(edge),
        }
    }
}

/// Every axis with its extent, in canonical order. A tile lives in its own space
/// (matmul's `lhs ∈ {M,K}`, `rhs ∈ {K,N}`, `out ∈ {M,N}`); an operation ranges over
/// their [`merge`](Space::merge).
#[derive(CubeType, Clone, Debug)]
pub struct Space {
    pub(crate) extents: Extents,
    #[cube(comptime)]
    partitioner: Partitioner,
}

// Identity is the comptime tiling spec only — the `Extents` sizes are runtime, never a key.
impl PartialEq for Space {
    fn eq(&self, other: &Self) -> bool {
        self.extents.kinds == other.extents.kinds && self.partitioner == other.partitioner
    }
}
impl Eq for Space {}
impl std::hash::Hash for Space {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.extents.kinds.hash(state);
        self.partitioner.hash(state);
    }
}

/// Comptime tiling spec read off a runtime `Space`'s `#[cube(comptime)]` data. Tiles carry a comptime
/// `Space`, so only [`Walk::over`](crate::Walk) — which takes the runtime operation space built by
/// `merged_space` — needs these; everything else calls the host methods directly.
impl SpaceExpand {
    fn comptime(&self) -> Space {
        Space {
            extents: Extents::fixed(self.extents.kinds.clone()),
            partitioner: self.partitioner.clone(),
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn clone(&self) -> Space {
        self.comptime()
    }

    pub fn rank(&self) -> usize {
        self.extents.kinds.len()
    }

    pub fn axis_at(&self, i: usize) -> Axis {
        self.extents.kinds.axis_at(i)
    }

    pub fn partitioner(&self) -> Partitioner {
        self.partitioner.clone()
    }
}

#[cube]
impl Space {
    /// The runtime operation space for a tiling level: the comptime tiling spec plus the runtime
    /// `sizes` of its `Dynamic` axes (per-axis, aligned to axis order; empty when fully `Static`).
    /// [`Walk::over`](crate::Walk) reads them through [`Extents::count`].
    pub fn with_sizes(#[comptime] space: Space, sizes: Sequence<usize>) -> Space {
        Space {
            extents: Extents {
                kinds: comptime!(space.extents.kinds.clone()),
                sizes,
            },
            partitioner: comptime!(space.partitioner.clone()),
        }
    }

    /// Merge two already-sized runtime spaces into the operation space: the comptime structure is
    /// the host [`merge`](Space::merge) of their specs, and each merged axis takes its runtime size
    /// from whichever input spans it. A fully-`Static` merge carries no runtime sizes.
    pub fn merge_with(&self, other: &Space) -> Space {
        let merged = comptime!(Space::merge(&[&self.clone(), &other.clone()]));
        let mut sizes = Sequence::<usize>::new();
        if comptime!(!merged.is_static()) {
            #[unroll]
            for p in 0..comptime!(merged.rank()) {
                let axis = comptime!(merged.axis_at(p));
                if comptime!(self.clone().contains(axis)) {
                    sizes.push(self.size(axis));
                } else {
                    sizes.push(other.size(axis));
                }
            }
        }
        Space::with_sizes(merged, sizes)
    }

    /// This (sized) space's runtime size along `axis`, read from the per-axis `sizes`. Only valid
    /// once filled (a `Dynamic` axis on an unsized space has none).
    fn size(&self, #[comptime] axis: Axis) -> usize {
        *self
            .extents
            .sizes
            .index(comptime!(self.clone().position(axis)))
    }
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
            extents: Extents::fixed(ByAxis::new(extents)),
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
        self.extents = Extents::fixed(ByAxis::new(&entries));
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

    /// Every axis is [`Static`](Extent::Static), so the walk is fully comptime. True at every
    /// interior tiling level, since [`divide`](Space::divide) yields `Static` children; only the top
    /// merge can be dynamic.
    pub fn is_static(&self) -> bool {
        self.axes().all(|axis| !self.is_dynamic(axis))
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
            extents: Extents::fixed(ByAxis::new(&entries)),
            partitioner,
        }
    }

    pub fn project(&self, axes: &[Axis]) -> Space {
        let entries = axes
            .iter()
            .map(|&a| (a, self.extent_raw(a)))
            .collect::<Vec<_>>();
        Space {
            extents: Extents::fixed(ByAxis::new(&entries)),
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
            extents: Extents::fixed(ByAxis::new(&entries)),
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
