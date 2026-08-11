//! The coordinate space a tile lives in. An operation's space is the merge of
//! its operands' spaces; the axes the output drops are contracted.

use cubecl::prelude::*;
use cubecl::zspace::SmallVec;

use crate::{Axis, ComputeScope, Distribution, LaneShare, Leaf, LevelRole, MAX_AXES, Partitioner};

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
    /// A fully-`Static` (or yet-unresolved) extents, with no runtime sizes.
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

// Identity is the comptime tiling spec only; the `Extents` sizes are runtime, never a key.
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
/// `Space`, so only [`Walk::over`](crate::Walk), which takes the runtime operation space
/// [`witnessed_space`](crate::witnessed_space) builds from an op's operands, needs these;
/// everything else calls the host methods directly.
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
    /// runtime (the common case; see [`with_dynamic`](Space::with_dynamic)).
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

    pub fn partitioner(&self) -> &Partitioner {
        &self.partitioner
    }

    pub fn is_final(&self) -> bool {
        self.partitioner.is_final()
    }

    /// How an operand that becomes `leaf` stages under this plan: [`Plane`](OperandStage::Plane)
    /// when a plane fragment is fed by a partition grid just below, else [`Smem`](OperandStage::Smem).
    pub(crate) fn operand_stage(&self, leaf: Leaf) -> OperandStage {
        match self.partitioner() {
            Partitioner::Level(_) => match (leaf, self.partitioner().next()) {
                (Leaf::Cmma | Leaf::Mma { .. }, Partitioner::Level(sub)) => match sub.role() {
                    LevelRole::Partition => OperandStage::Plane,
                    LevelRole::Instance => OperandStage::Smem,
                },
                _ => OperandStage::Smem,
            },
            Partitioner::Final => OperandStage::Smem,
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

    /// Whether this level cuts `axis` into a single, statically-known tile, so its walk
    /// coordinate is a constant `0`, even on a rolled walk. A `Dynamic` axis (only the top
    /// level) has no comptime count and is never statically single; the `&&` short-circuits
    /// before [`count`](Space::count), which panics on `Dynamic`.
    pub(crate) fn single_static_tile(&self, axis: Axis) -> bool {
        !self.is_dynamic(axis) && self.count(axis) == 1
    }

    /// Whether this level cuts its tiles into an m×n grid larger than 1×1, so each region must be
    /// selected by a comptime coordinate. A final tile, an instance level, and a degenerate 1×1
    /// partition (a k-step walk) all cut nothing.
    pub(crate) fn cuts_tiles(&self) -> bool {
        match self.partitioner() {
            Partitioner::Final => false,
            Partitioner::Level(level) => match level.role() {
                LevelRole::Instance => false,
                LevelRole::Partition => crate::partition_grid(self) != (1, 1),
            },
        }
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

    /// Reorder so `fastest` walks innermost (last axis fastest): each coarser-axis
    /// window then feeds a consecutive burst of steps: the unrolled fragment walk's
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
    /// a partial tile that needs masking. Host-side, on the concrete (real-extent) space;
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
    /// walk actually steps (more than one tile) is absent from the operand: the same
    /// structural fact as broadcast omission. A [`Staged`](crate::Schedule::Staged) walk
    /// fills such an operand once, above the loop. Host-side, static extents.
    pub fn walk_invariant(&self, operand: &Space) -> bool {
        self.axes()
            .all(|axis| self.count(axis) == 1 || !operand.contains(axis))
    }

    /// What the plane's lanes hold of this space's cells: a `Unit` axis the space doesn't span is
    /// *folded* across the lanes, so each holds only a partial; one it does span is *carried*,
    /// giving each lane a cell of its own.
    ///
    /// Which lanes hold partials of one cell is a question about the lane index's digits, so the
    /// answer is a bit mask. `Walk::from_counts` decodes a `Unit` axis as
    /// `UNIT_POS_X / inner_weight % instances`, which for power-of-two counts is a contiguous run
    /// of bits; the folded axes' runs are exactly the bits a cell's partials differ in. Fold
    /// everything and that mask is the whole plane ([`LaneShare::Plane`]); fold under a carry and
    /// it is a [`LaneShare::Group`], whatever order the axes sit in.
    pub(crate) fn lane_share(&self) -> LaneShare {
        if self.partitioner.is_final() {
            return LaneShare::Whole;
        }
        // Innermost first, so `weight` is the axis's stride in the lane index as it is reached —
        // the same least-significant-last ordering `Walk::from_counts` decodes with.
        let (mut weight, mut fold_mask) = (1usize, 0usize);
        for axis in self.partitioner.axes().into_iter().rev() {
            let Distribution::Spatial {
                scope: ComputeScope::Unit,
                coverage,
                ..
            } = self.partitioner.distribution(axis)
            else {
                continue;
            };
            // Asserted, not skipped: a `Unit` axis always resolves to `Instances`
            // (`Distribution::unit` defers through `PlaneLanes`), and passing over one whose
            // count we could not read would shift every inner axis's bits by its width.
            let lanes = coverage
                .instances_const()
                .expect("Space::lane_share: a Unit axis must carry a const instance count");
            if lanes == 1 {
                continue;
            }
            assert!(
                lanes.is_power_of_two(),
                "Space::lane_share: {axis:?} rides {lanes} lanes, which is not a power of two, so its partials are not a bit range"
            );
            if !self.contains(axis) {
                fold_mask |= (lanes - 1) * weight;
            }
            weight *= lanes;
        }
        match fold_mask {
            0 => LaneShare::Whole,
            // Every lane's bit folded: nothing is carried, so the plane shares the one cell.
            mask if mask == weight - 1 => LaneShare::Plane,
            fold_mask => LaneShare::Group { fold_mask },
        }
    }

    /// The axes in this space but not in `output`, i.e. those contracted.
    pub fn contracting(&self, output: &Space) -> SmallVec<[Axis; MAX_AXES]> {
        self.axes().filter(|&axis| !output.contains(axis)).collect()
    }

    /// The axes `operands` jointly contract against `output`: [`contracting`](Space::contracting)
    /// over their [`merge`](Space::merge), so an axis only one operand spans still counts. How many
    /// there are is what picks a leaf's microkernel, so every site that deduces a 2-D single-`K`
    /// shape asks here rather than reading an operand's rank.
    pub fn contracted(operands: &[&Space], output: &Space) -> SmallVec<[Axis; MAX_AXES]> {
        Space::merge(operands).contracting(output)
    }

    /// The `k` edge this operand contracts over against `output`: the product of every
    /// [`contracting`](Space::contracting) axis's extent. An instruction sees one contraction
    /// depth, not a list of axes.
    ///
    /// Reads the extents off this space as it stands, like every other consumer of a tile's edges
    /// ([`matrix_split`](crate::matrix_split)); call it on the [`final_space`](Space::final_space)
    /// when the caller holds a level above the leaf.
    pub fn contracted_extent(&self, output: &Space) -> usize {
        self.contracting(output)
            .iter()
            .map(|&axis| self.extent(axis))
            .product()
    }

    /// Whether `lhs` and `rhs` enumerate their contracted axes in the same order.
    ///
    /// A fragment groups its `k` edge by extent alone ([`matrix_split`](crate::matrix_split)), so
    /// two operands listing the same axes in different orders contract mismatched positions with
    /// no shape mismatch to catch it. Each operand's order is its own [`TileSpec`](crate::TileSpec)
    /// axis list, which is stated per operand, so nothing upstream forces them to agree.
    pub fn contraction_agrees(lhs: &Space, rhs: &Space, output: &Space) -> bool {
        lhs.contracting(output) == rhs.contracting(output)
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

    /// The space of one *sub-tile* this level hands out, rather than of the whole region it
    /// covers: every extent becomes the partitioner's comptime sub-tile edge, and the level
    /// itself is kept. Sits between [`divide`](Space::divide), which takes those same edges
    /// but also consumes the level to answer "the child one level down", and the untouched
    /// space, which describes the region.
    ///
    /// This is the shape a register-resident form actually has. A
    /// [`mirror`](crate::PlanePartition::mirror)ed accumulator sizes its fragments from the
    /// partitioner alone and never reads the extents, so it must not inherit their dynamism:
    /// the kernel-form space is [`all_dynamic`](Space::all_dynamic), and a plane tile has no
    /// buffer bound to resolve a `Dynamic` axis back from.
    ///
    /// A [`Final`](Partitioner::Final) space has no level left to read edges from, and is
    /// already the tile, so it is returned unchanged.
    pub fn sub_tile_space(&self) -> Space {
        if self.is_final() {
            return self.clone();
        }
        let entries = self
            .axes()
            .map(|axis| (axis, Extent::Static(self.partitioner.edge(axis))))
            .collect::<Vec<_>>();
        Space {
            extents: Extents::fixed(ByAxis::new(&entries)),
            partitioner: self.partitioner.clone(),
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
/// non-broadcast operand (its runtime size is the merged one), so the merge stays dynamic.
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

/// A single-level space whose every axis is one sequential cut of its extent: what the shape
/// helpers are exercised against, since they read extents and axis order rather than the walk.
#[cfg(test)]
pub(crate) fn flat_space(extents: &[(Axis, usize)]) -> Space {
    use crate::{Cut, Schedule, Tiling, WalkOrder};
    Tiling::new()
        .extents(extents)
        .level(WalkOrder::RowMajor, Schedule::Direct, |mut l| {
            for &(axis, e) in extents {
                l = l.axis(axis, Cut::sequential(e));
            }
            l
        })
        .build()
}

#[cfg(test)]
mod contraction_tests {
    use crate::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);
    const R: Axis = Axis(3);

    /// A matmul contracts one axis, so the `k` edge is that axis's extent, as it always was.
    #[test]
    fn a_matmul_contracts_its_one_axis() {
        let lhs = flat_space(&[(M, 8), (K, 4)]);
        let out = flat_space(&[(M, 8), (N, 8)]);
        assert_eq!(lhs.contracted_extent(&out), 4);
    }

    /// A convolution contracts its taps and its channels at once; the instruction sees one `k`,
    /// which is their product.
    #[test]
    fn a_convolution_contracts_taps_times_channels() {
        let lhs = flat_space(&[(M, 8), (R, 3), (K, 4)]);
        let out = flat_space(&[(M, 8), (N, 8)]);
        assert_eq!(lhs.contracted_extent(&out), 12);
    }

    /// An operand spanning only output axes contracts nothing, and an empty product is `1`.
    #[test]
    fn contracting_nothing_is_a_unit_depth() {
        let lhs = flat_space(&[(M, 8), (N, 8)]);
        let out = flat_space(&[(M, 8), (N, 8)]);
        assert_eq!(lhs.contracted_extent(&out), 1);
    }

    /// The `A` and `B` roles of a convolution, listing taps then channels in the order each
    /// operand's own spec states them.
    #[test]
    fn operands_listing_one_contraction_order_agree() {
        let lhs = flat_space(&[(M, 8), (R, 3), (K, 4)]);
        let rhs = flat_space(&[(R, 3), (K, 4), (N, 8)]);
        let out = flat_space(&[(M, 8), (N, 8)]);
        assert!(Space::contraction_agrees(&lhs, &rhs, &out));
    }

    /// The same axes and the same `k`, listed the other way round on `rhs`: nothing about the
    /// shapes distinguishes this from the case above, so the order has to be compared.
    #[test]
    fn a_permuted_contraction_order_disagrees() {
        let lhs = flat_space(&[(M, 8), (R, 3), (K, 4)]);
        let rhs = flat_space(&[(K, 4), (R, 3), (N, 8)]);
        let out = flat_space(&[(M, 8), (N, 8)]);
        assert!(!Space::contraction_agrees(&lhs, &rhs, &out));
    }
}
