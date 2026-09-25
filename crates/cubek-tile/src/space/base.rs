//! The coordinate space a tile lives in. An operation's space is the merge of
//! its operands' spaces; the axes the output drops are contracted.

use cubecl::prelude::*;
use cubecl::zspace::SmallVec;

use crate::{Axis, Extent, Level, Shape};

/// Every axis with its extent, in canonical order. A tile lives in its own space
/// (matmul's `lhs ∈ {M,K}`, `rhs ∈ {K,N}`, `out ∈ {M,N}`); an operation ranges over
/// their [`merge`](Space::merge).
///
/// The [`Shape`] is comptime, so static tile counts fold and the walk unrolls; the `sizes` are
/// the runtime half, one per axis (positional) where any axis is dynamic and empty otherwise.
/// Only the top operation space carries sizes; a level's child is `Static`.
#[derive(CubeType, CubeLaunch, Clone, Debug)]
pub struct Space {
    #[cube(comptime)]
    pub(crate) shape: Shape,
    pub(crate) sizes: Sequence<usize>,
}

// Identity is the comptime shape only; the sizes are runtime, never a key.
impl PartialEq for Space {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
    }
}
impl Eq for Space {}
impl std::hash::Hash for Space {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.shape.hash(state);
    }
}

/// The comptime reads of a space, as the host reads them: what `comptime!(space.rank())` and the
/// like resolve to on the space a kernel is handed.
impl SpaceExpand {
    /// The comptime shape of a runtime `Space`, as a host `Space` with no sizes.
    pub(crate) fn comptime(&self) -> Space {
        Space::from_shape(self.shape.clone())
    }

    #[allow(clippy::should_implement_trait)]
    pub fn clone(&self) -> Space {
        self.comptime()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn axis_at(&self, i: usize) -> Axis {
        self.shape.axis_at(i)
    }

    pub fn position(&self, axis: Axis) -> usize {
        self.shape.position(axis)
    }

    pub fn contains(&self, axis: Axis) -> bool {
        self.shape.contains(axis)
    }

    pub fn axes(&self) -> impl Iterator<Item = Axis> + '_ {
        self.shape.axes()
    }

    pub fn extent(&self, axis: Axis) -> usize {
        self.shape.extent(axis)
    }

    pub fn extent_at(&self, i: usize) -> usize {
        self.shape.extent_at(i)
    }

    pub fn is_dynamic(&self, axis: Axis) -> bool {
        self.shape.is_dynamic(axis)
    }

    /// The listed axes with their extents, in the order listed.
    pub fn subspace(&self, axes: &[Axis]) -> Space {
        Space::from_shape(self.shape.subspace(axes))
    }
}

impl Space {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn axis_at(&self, i: usize) -> Axis {
        self.shape.axis_at(i)
    }

    pub fn position(&self, axis: Axis) -> usize {
        self.shape.position(axis)
    }

    pub fn contains(&self, axis: Axis) -> bool {
        self.shape.contains(axis)
    }

    pub fn axes(&self) -> impl Iterator<Item = Axis> + '_ {
        self.shape.axes()
    }

    pub(crate) fn extent_raw(&self, axis: Axis) -> Extent {
        self.shape.extent_raw(axis)
    }

    /// The axis's comptime size; panics on a [`Dynamic`](Extent::Dynamic) axis. The leaf
    /// and smem consumers all run on fully-divided (`Static`) spaces, so this is what they
    /// call.
    pub fn extent(&self, axis: Axis) -> usize {
        self.shape.extent(axis)
    }

    pub fn extent_at(&self, i: usize) -> usize {
        self.shape.extent_at(i)
    }

    pub fn is_dynamic(&self, axis: Axis) -> bool {
        self.shape.is_dynamic(axis)
    }

    /// The listed axes with their extents, in the order listed.
    pub fn subspace(&self, axes: &[Axis]) -> Space {
        Space::from_shape(self.shape.subspace(axes))
    }
}

#[cube]
impl Space {
    /// The runtime operation space: the comptime space plus the runtime `sizes` of its
    /// `Dynamic` axes (per-axis, aligned to axis order; empty when fully `Static`).
    /// A walk reads them through [`count`](Space::count).
    pub(crate) fn with_sizes(#[comptime] space: Space, sizes: Sequence<usize>) -> Space {
        Space {
            shape: comptime!(space.shape.clone()),
            sizes,
        }
    }

    /// Axis `p`'s tile count for a sub-tile `edge`: a `Static` axis folds to a comptime constant
    /// (so the walk loop unrolls), a `Dynamic` axis ceil-divides its runtime size. The match is
    /// comptime, so an all-`Static` space never touches `sizes`.
    pub fn count(&self, #[comptime] p: usize, #[comptime] edge: usize) -> usize {
        match comptime!(self.shape.extent_raw(self.shape.axis_at(p))) {
            Extent::Static(n) => comptime!(n.div_ceil(edge)).runtime(),
            Extent::Dynamic => (*self.sizes.index(p)).div_ceil(edge),
        }
    }
}

impl Space {
    /// The most axes a space holds inline; a per-axis small vector spills to the heap past it.
    pub const MAX_RANK: usize = 6;

    pub fn new(extents: &[(Axis, usize)]) -> Self {
        let extents: Vec<_> = extents
            .iter()
            .map(|&(a, n)| (a, Extent::Static(n)))
            .collect();
        Space::from_extents(&extents)
    }

    /// Every axis dynamic: the kernel form, its extents resolved in-kernel
    /// from the tensors, so one compiled kernel serves every shape; the launch keeps the concrete
    /// space beside it.
    pub fn dynamic(axes: &[Axis]) -> Self {
        let extents: Vec<_> = axes.iter().map(|&a| (a, Extent::Dynamic)).collect();
        Space::from_extents(&extents)
    }

    /// Construct directly from [`Extent`]s (the form `merge`/`project`/`divide` round-trip).
    pub(crate) fn from_extents(extents: &[(Axis, Extent)]) -> Self {
        Space::from_shape(Shape::new(extents))
    }

    /// A host space of `shape`, with no runtime sizes.
    pub(crate) fn from_shape(shape: Shape) -> Self {
        Space {
            shape,
            sizes: Sequence::new(),
        }
    }

    /// Flip the listed axes to [`Dynamic`](Extent::Dynamic), The
    /// launch side computes geometry from the concrete (real-extent) space, then derives the
    /// kernel's space with this so distinct input shapes hit one compiled kernel.
    pub fn with_dynamic(mut self, axes: &[Axis]) -> Self {
        for axis in axes {
            assert!(
                self.contains(*axis),
                "Space::with_dynamic: {axis:?} is not an axis of this space"
            );
        }
        let entries: Vec<_> = self
            .axes()
            .map(|a| {
                let extent = if axes.contains(&a) {
                    Extent::Dynamic
                } else {
                    self.extent_raw(a)
                };
                (a, extent)
            })
            .collect();
        self.shape = Shape::new(&entries);
        self
    }

    /// Every axis dynamic: the kernel form for an operation whose problem dims are all
    /// runtime (the common case; see [`with_dynamic`](Space::with_dynamic)).
    pub fn all_dynamic(self) -> Self {
        let axes: Vec<_> = self.axes().collect();
        self.with_dynamic(&axes)
    }

    /// Every axis is [`Static`](Extent::Static), so the walk is fully comptime. True at every
    /// interior level, since a level's child is `Static`; only the top merge can be dynamic.
    pub(crate) fn is_static(&self) -> bool {
        self.shape.is_static()
    }

    /// The static extents, in axis order.
    pub fn extents(&self) -> Vec<(Axis, usize)> {
        self.axes().map(|axis| (axis, self.extent(axis))).collect()
    }

    /// The leaf `levels` reach below this space: each level's child of the last.
    pub(crate) fn leaf(&self, levels: &[Level]) -> Space {
        levels
            .iter()
            .fold(self.clone(), |space, level| level.child(&space))
    }

    /// The smallest space containing every `part`, axes in first-appearance order. A shared axis
    /// is broadcast-merged via [`merge_level`] (`n ∪ n = n`, `1 ∪ n = n`, else conflict); an
    /// omitted axis broadcasts along all of it. E.g. `{M,K} ∪ {K,N} ∪ {M,N} = {M,N,K}`.
    pub fn merge(parts: &[&Space]) -> Space {
        let mut entries: SmallVec<[(Axis, Extent); Space::MAX_RANK]> = SmallVec::new();

        for part in parts {
            for axis in part.axes() {
                let extent = part.extent_raw(axis);
                match entries.iter_mut().find(|(a, _)| *a == axis) {
                    Some(slot) => slot.1 = merge_level(slot.1, extent),
                    None => entries.push((axis, extent)),
                }
            }
        }
        Space::from_shape(Shape::new(&entries))
    }

    /// The axes in this space but not in `output`, i.e. those contracted.
    pub fn difference(&self, output: &Space) -> SmallVec<[Axis; Space::MAX_RANK]> {
        self.axes().filter(|&axis| !output.contains(axis)).collect()
    }

    /// How many contracted values one step consumes off a `width`-wide line of this operand.
    ///
    /// A line folds into one accumulator cell only where it runs along the fastest of
    /// `contracted`, absent from the accumulator, so its units are partials of one cell; skipping
    /// the test silently merges distinct cells. The width must divide the axis: no masked tail.
    pub(crate) fn contracted_per_step(&self, contracted: &[Axis], width: usize) -> usize {
        let lined = self.axis_at(self.rank() - 1);
        let folds = width > 1
            && contracted.last() == Some(&lined)
            && self.extent(lined).is_multiple_of(width);
        if folds { width } else { 1 }
    }

    /// The axes `operands` jointly contract against `output`: [`difference`](Space::difference)
    /// over their [`merge`](Space::merge), so an axis only one operand spans still counts. Their
    /// number picks a leaf's instruction, so a site deducing a 2-D single-`K` shape asks here.
    ///
    /// An axis is kept if it varies or every operand shares it. One that does neither, as a routed
    /// axis ([`Walk::routed`](crate::Walk::routed)) leaves, is a fixed coordinate, not a sum. A
    /// shared axis stays even at one value: separable factors are named by contracted position.
    pub fn contracted(operands: &[&Space], output: &Space) -> SmallVec<[Axis; Space::MAX_RANK]> {
        let merged = Space::merge(operands);
        // Read raw: a `Dynamic` extent is not known to be one, and asking its comptime size panics.
        let varies = |axis: Axis| merged.extent_raw(axis) != Extent::Static(1);
        let shared = |axis: Axis| operands.iter().all(|operand| operand.contains(axis));
        merged
            .difference(output)
            .into_iter()
            .filter(|&axis| varies(axis) || shared(axis))
            .collect()
    }

    /// The `k` edge this operand contracts over against `output`: the product of every
    /// [`difference`](Space::difference) axis's extent. An instruction sees one contraction
    /// depth, not a list of axes.
    ///
    /// Reads the extents off this space as it stands, like every other consumer of a tile's edges
    /// ([`MatrixAxes::whole`](crate::MatrixAxes::whole)).
    pub(crate) fn contracted_extent(&self, output: &Space) -> usize {
        self.difference(output)
            .iter()
            .map(|&axis| self.extent(axis))
            .product()
    }

    /// Whether `lhs` and `rhs` enumerate their contracted axes in the same order.
    ///
    /// Fragments group `k` by extent alone ([`MatrixAxes::whole`](crate::MatrixAxes::whole)), so
    /// operands listing the same axes in different orders contract mismatched positions, unseen by
    /// shape checks; each operand's [`TileSpec`](crate::TileSpec) axis order is its own.
    ///
    /// A routed axis sits in one operand's list only, so each list is narrowed to the joint
    /// [`contracted`](Space::contracted) axes first. Compared raw, every routed contraction would
    /// read as a disagreement.
    pub(crate) fn contraction_agrees(lhs: &Space, rhs: &Space, output: &Space) -> bool {
        let joint = Space::contracted(&[lhs, rhs], output);
        let listed = |operand: &Space| -> SmallVec<[Axis; Space::MAX_RANK]> {
            operand
                .difference(output)
                .into_iter()
                .filter(|axis| joint.contains(axis))
                .collect()
        };
        listed(lhs) == listed(rhs)
    }

    /// How many cells the space holds: the product of its extents.
    pub fn cells(&self) -> usize {
        self.axes().map(|axis| self.extent(axis)).product()
    }
}

#[cube]
impl Space {
    /// Axis `i`'s extent, preserving a dynamic extent as its runtime size. This is the form a
    /// coordinate-backed operand carries as its real-data bound across [`Space::divide`] calls.
    pub(crate) fn runtime_extent_at(&self, #[comptime] i: usize) -> usize {
        match comptime!(self.shape.extent_raw(self.shape.axis_at(i))) {
            Extent::Static(n) => comptime!(n).runtime(),
            Extent::Dynamic => *self.sizes.index(i),
        }
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
        let lhs = Space::new(&[(M, 8), (K, 4)]);
        let out = Space::new(&[(M, 8), (N, 8)]);
        assert_eq!(lhs.contracted_extent(&out), 4);
    }

    /// A convolution contracts its taps and its channels at once; the instruction sees one `k`,
    /// which is their product.
    #[test]
    fn a_convolution_contracts_taps_times_channels() {
        let lhs = Space::new(&[(M, 8), (R, 3), (K, 4)]);
        let out = Space::new(&[(M, 8), (N, 8)]);
        assert_eq!(lhs.contracted_extent(&out), 12);
    }

    /// An operand spanning only output axes contracts nothing, and an empty product is `1`.
    #[test]
    fn contracting_nothing_is_a_unit_depth() {
        let lhs = Space::new(&[(M, 8), (N, 8)]);
        let out = Space::new(&[(M, 8), (N, 8)]);
        assert_eq!(lhs.contracted_extent(&out), 1);
    }

    /// The `A` and `B` roles of a convolution, listing taps then channels in the order each
    /// operand's own spec states them.
    #[test]
    fn operands_listing_one_contraction_order_agree() {
        let lhs = Space::new(&[(M, 8), (R, 3), (K, 4)]);
        let rhs = Space::new(&[(R, 3), (K, 4), (N, 8)]);
        let out = Space::new(&[(M, 8), (N, 8)]);
        assert!(Space::contraction_agrees(&lhs, &rhs, &out));
    }

    /// The same axes and the same `k`, listed the other way round on `rhs`: nothing about the
    /// shapes distinguishes this from the case above, so the order has to be compared.
    #[test]
    fn a_permuted_contraction_order_disagrees() {
        let lhs = Space::new(&[(M, 8), (R, 3), (K, 4)]);
        let rhs = Space::new(&[(K, 4), (R, 3), (N, 8)]);
        let out = Space::new(&[(M, 8), (N, 8)]);
        assert!(!Space::contraction_agrees(&lhs, &rhs, &out));
    }
}
