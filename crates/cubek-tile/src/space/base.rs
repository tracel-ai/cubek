//! The coordinate space a tile lives in. An operation's space is the merge of
//! its operands' spaces; the axes the output drops are contracted.

use cubecl::prelude::*;
use cubecl::zspace::SmallVec;

use crate::{Axis, ComputeScope, Distribution, LaneShare, Leaf, MAX_AXES, Partitioner};

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

/// What backs a staged matmul operand, the [`Space::operand_stage`] classification. `Plane` stages
/// straight into plane-private tile partitions; `Smem` into a shared buffer the leaf reads windows
/// from. Read by the staging store ([`Staging::new`]) and the schedule's unroll (a plane stage
/// selects tiles by comptime coordinate, so its walk must be unrolled).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OperandStage {
    Plane,
    Smem,
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

    /// This space's runtime size along `axis`: a `Static` axis folds to its comptime extent
    /// (so a fully-static operand needs no `sizes` at all), a `Dynamic` one reads the
    /// per-axis `sizes` — only valid once filled.
    fn size(&self, #[comptime] axis: Axis) -> usize {
        match comptime!(self.clone().extent_raw(axis)) {
            Extent::Static(n) => comptime!(n).runtime(),
            Extent::Dynamic => *self
                .extents
                .sizes
                .index(comptime!(self.clone().position(axis))),
        }
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
            partitioner: Partitioner::Final(Leaf::Register),
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

    /// Resolve every `Unit` axis's deferred lane count to `Instances(plane_size)`. The
    /// launch's stamping pass ([`Space::launcher`] applies it), so a partitioner declares a
    /// `Unit` split without knowing the hardware warp width and geometry/walk only ever see
    /// a concrete count.
    pub fn resolve_lanes(mut self, plane_size: usize) -> Self {
        self.partitioner = self.partitioner.resolve_lanes(plane_size);
        self
    }

    /// Chain coarse-to-fine for multi-level tiling; each call appends to the end of
    /// the chain (see [`Partitioner::append`]).
    pub fn with_partitioner(mut self, partitioner: Partitioner) -> Self {
        self.partitioner = self.partitioner.append(partitioner);
        self
    }

    /// Set the chain-end [`Leaf`] after all levels are stacked (appending a level resets
    /// it); the public surface is the order-safe [`Tiling::leaf`](crate::LeveledTiling::leaf).
    pub(crate) fn with_leaf(mut self, leaf: Leaf) -> Self {
        self.partitioner = self.partitioner.with_leaf(leaf);
        self
    }

    pub fn partitioner(&self) -> &Partitioner {
        &self.partitioner
    }

    pub fn is_final(&self) -> bool {
        self.partitioner.is_final()
    }

    /// How this output plan's operands stage: [`Plane`](OperandStage::Plane) when the leaf
    /// contracts plane tiles and the level below is their grid (operands stage straight into
    /// plane-private tiles), else [`Smem`](OperandStage::Smem). The plan's own fact, so no consumer
    /// reassembles it from the leaf and the partition level.
    pub(crate) fn operand_stage(&self) -> OperandStage {
        match self.partitioner().leaf().is_plane()
            && crate::partition_level(&self.divide()).is_some()
        {
            true => OperandStage::Plane,
            false => OperandStage::Smem,
        }
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

    /// Whether this level's walk is host data: every extent `Static` and every axis
    /// `Sequential` (no hardware digit to decode), so an unrolled walk's regions fold
    /// to comptime coordinates.
    pub(crate) fn static_walkable(&self) -> bool {
        self.is_static()
            && self.axes().all(|axis| {
                matches!(
                    self.partitioner().distribution(axis),
                    Distribution::Sequential
                )
            })
    }

    pub fn extent_at(&self, i: usize) -> usize {
        self.extent(self.axis_at(i))
    }

    pub fn axis_at(&self, i: usize) -> Axis {
        self.extents.axis_at(i)
    }

    /// Whether axis position `p` is `Spatial` `TilesEach(1)`: its walk count is
    /// comptime `1`, so a step decode can skip it.
    pub(crate) fn single_tile_at(&self, p: usize) -> bool {
        self.partitioner()
            .distribution(self.axis_at(p))
            .single_tile()
    }

    /// Whether this level cuts `axis` into a single, statically-known tile — so its walk
    /// coordinate is a constant `0`, even on a rolled walk. A `Dynamic` axis (only the top
    /// level) has no comptime count and is never statically single; the `&&` short-circuits
    /// before [`count`](Space::count), which panics on `Dynamic`.
    pub(crate) fn single_static_tile(&self, axis: Axis) -> bool {
        !self.is_dynamic(axis) && self.count(axis) == 1
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
            .unwrap_or(Partitioner::Final(Leaf::Register));

        Space {
            extents: Extents::fixed(ByAxis::new(&entries)),
            partitioner,
        }
    }

    /// Reorder so `fastest` walks innermost (last axis fastest): each coarser-axis
    /// window then feeds a consecutive burst of steps — the unrolled fragment walk's
    /// emission order.
    pub fn with_fastest(&self, fastest: Axis) -> Space {
        let mut axes: Vec<Axis> = self.axes().filter(|&a| a != fastest).collect();
        axes.push(fastest);
        self.project(&axes)
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

    /// Whether `axis` overhangs its tiling: some level's sub-tile edge fails to divide the
    /// extent handed to it (the top extent at the first level, the parent edge below), leaving
    /// a partial tile that needs masking. Host-side, on the concrete (real-extent) space —
    /// a [`Dynamic`](Extent::Dynamic) axis panics.
    pub fn overhangs(&self, axis: Axis) -> bool {
        assert!(
            !self.is_dynamic(axis),
            "Space::overhangs: axis {axis:?} is Dynamic; call on the concrete space, not the kernel-form one"
        );
        let mut extent = self.extent(axis);
        let mut partitioner = &self.partitioner;
        while !partitioner.is_final() {
            let edge = partitioner.edge(axis);
            if !extent.is_multiple_of(edge) {
                return true;
            }
            extent = edge;
            partitioner = partitioner.next();
        }
        false
    }

    /// Whether a walk over this level leaves `operand`'s window unchanged: every axis the
    /// walk actually steps (more than one tile) is absent from the operand — the same
    /// structural fact as broadcast omission. A [`Staged`](crate::Schedule::Staged) walk
    /// fills such an operand once, above the loop. Host-side, static extents.
    pub fn walk_invariant(&self, operand: &Space) -> bool {
        self.axes()
            .all(|axis| self.count(axis) == 1 || !operand.contains(axis))
    }

    /// What the plane's lanes hold of this space's cells: `Partial` once a level spreads an axis
    /// the space doesn't span, since each lane then covers a disjoint slice of it.
    pub(crate) fn lane_share(&self) -> LaneShare {
        if self.partitioner.is_final() {
            return LaneShare::Whole;
        }
        for axis in self.partitioner.axes() {
            if self.contains(axis) {
                continue;
            }
            if let Distribution::Spatial {
                scope: ComputeScope::Unit,
                coverage,
                ..
            } = self.partitioner.distribution(axis)
                && coverage.instances_const().is_some_and(|lanes| lanes > 1)
            {
                return LaneShare::Partial;
            }
        }
        LaneShare::Whole
    }

    /// The axes in this space but not in `output`, i.e. those contracted.
    pub fn contracting(&self, output: &Space) -> SmallVec<[Axis; MAX_AXES]> {
        self.axes().filter(|&axis| !output.contains(axis)).collect()
    }

    /// The single axis this operand contracts against `output`:
    /// [`contracting`](Space::contracting) with the one-axis contract asserted.
    pub fn contraction(&self, output: &Space) -> Axis {
        let contracted = self.contracting(output);
        assert!(
            contracted.len() == 1,
            "Space::contraction: exactly one contracted axis expected"
        );
        contracted[0]
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
