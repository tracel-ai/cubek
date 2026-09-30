//! The coordinate space a tile lives in.

use cubecl::prelude::*;
use cubecl::zspace::SmallVec;

use crate::{Axis, Extent, Level, Shape};

/// Every axis with its extent, in canonical order, plus the runtime sizes of dynamic axes.
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

/// The comptime reads of a runtime space.
impl SpaceExpand {
    /// The comptime shape of a runtime `Space`.
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

    /// Whether `other` is this space with some one-wide axes dropped, order and extents kept.
    pub(crate) fn narrows_to(&self, other: &Space) -> bool {
        self.axes()
            .filter(|&axis| other.contains(axis))
            .eq(other.axes())
            && self.axes().last().is_some_and(|axis| other.contains(axis))
            && self.axes().all(|axis| match other.contains(axis) {
                true => self.extent(axis) == other.extent(axis),
                false => self.extent(axis) == 1,
            })
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

    /// The axis's comptime size; panics on a `Dynamic` axis.
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
    /// `Dynamic` axes (aligned to axis order; empty when fully `Static`).
    pub(crate) fn with_sizes(#[comptime] space: Space, sizes: Sequence<usize>) -> Space {
        Space {
            shape: comptime!(space.shape.clone()),
            sizes,
        }
    }

    /// Axis `p`'s tile count for a sub-tile `edge`; comptime for a `Static` axis.
    pub fn count(&self, #[comptime] p: usize, #[comptime] edge: usize) -> usize {
        match comptime!(self.shape.extent_raw(self.shape.axis_at(p))) {
            Extent::Static(n) => comptime!(n.div_ceil(edge)).runtime(),
            Extent::Dynamic => (*self.sizes.index(p)).div_ceil(edge),
        }
    }
}

impl Space {
    /// The most axes a space holds inline.
    pub const MAX_RANK: usize = 6;

    pub fn new(extents: &[(Axis, usize)]) -> Self {
        let extents: Vec<_> = extents
            .iter()
            .map(|&(a, n)| (a, Extent::Static(n)))
            .collect();
        Space::from_extents(&extents)
    }

    /// Every axis dynamic, extents resolved in-kernel from the tensors.
    pub fn dynamic(axes: &[Axis]) -> Self {
        let extents: Vec<_> = axes.iter().map(|&a| (a, Extent::Dynamic)).collect();
        Space::from_extents(&extents)
    }

    /// Construct directly from [`Extent`]s.
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

    /// Flip the listed axes to `Dynamic`.
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

    /// Every axis dynamic (see [`with_dynamic`](Space::with_dynamic)).
    pub fn all_dynamic(self) -> Self {
        let axes: Vec<_> = self.axes().collect();
        self.with_dynamic(&axes)
    }

    /// Every axis is [`Static`](Extent::Static), so the walk is fully comptime.
    pub(crate) fn is_static(&self) -> bool {
        self.shape.is_static()
    }

    /// The static extents, in axis order.
    pub fn extents(&self) -> Vec<(Axis, usize)> {
        self.axes().map(|axis| (axis, self.extent(axis))).collect()
    }

    /// The leaf `levels` reach below this space.
    pub(crate) fn leaf(&self, levels: &[Level]) -> Space {
        levels
            .iter()
            .fold(self.clone(), |space, level| level.child(&space))
    }

    /// The smallest space containing every `part`, axes in first-appearance order, shared axes
    /// broadcast-merged. E.g. `{M,K} ∪ {K,N} ∪ {M,N} = {M,N,K}`.
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

    /// The axes in this space but not in `output`: those contracted.
    pub fn difference(&self, output: &Space) -> SmallVec<[Axis; Space::MAX_RANK]> {
        self.axes().filter(|&axis| !output.contains(axis)).collect()
    }

    /// How many contracted values one step consumes off a `width`-wide line of this operand.
    pub(crate) fn contracted_per_step(&self, contracted: &[Axis], width: usize) -> usize {
        let lined = self.axis_at(self.rank() - 1);
        let whole_lines = match self.extent_raw(lined) {
            Extent::Static(extent) => extent.is_multiple_of(width),
            Extent::Dynamic => true,
        };
        let folds = width > 1 && contracted.last() == Some(&lined) && whole_lines;
        if folds { width } else { 1 }
    }

    /// The axes `operands` jointly contract against `output`.
    pub fn contracted(operands: &[&Space], output: &Space) -> SmallVec<[Axis; Space::MAX_RANK]> {
        let merged = Space::merge(operands);
        // Read raw: asking a `Dynamic` extent's comptime size panics.
        let varies = |axis: Axis| merged.extent_raw(axis) != Extent::Static(1);
        let shared = |axis: Axis| operands.iter().all(|operand| operand.contains(axis));
        merged
            .difference(output)
            .into_iter()
            .filter(|&axis| varies(axis) || shared(axis))
            .collect()
    }

    /// The `k` edge this operand contracts over against `output`.
    pub(crate) fn contracted_extent(&self, output: &Space) -> usize {
        self.difference(output)
            .iter()
            .map(|&axis| self.extent(axis))
            .product()
    }

    /// Whether `lhs` and `rhs` enumerate their jointly contracted axes in the same order.
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

    /// How many cells the space holds.
    pub fn cells(&self) -> usize {
        self.axes().map(|axis| self.extent(axis)).product()
    }

    /// How many rows the space holds: its cells along every axis but the innermost, which a row
    /// runs along. The innermost may be one a walk has yet to cut, whose extent only the launch
    /// knows.
    pub fn rows(&self) -> usize {
        (0..self.rank() - 1).map(|p| self.extent_at(p)).product()
    }

    /// How many cells one row holds: the innermost axis's extent.
    pub fn columns(&self) -> usize {
        self.extent_at(self.rank() - 1)
    }
}

#[cube]
impl Space {
    /// Axis `i`'s extent, a dynamic extent as its runtime size.
    pub(crate) fn runtime_extent_at(&self, #[comptime] i: usize) -> usize {
        match comptime!(self.shape.extent_raw(self.shape.axis_at(i))) {
            Extent::Static(n) => comptime!(n).runtime(),
            Extent::Dynamic => *self.sizes.index(i),
        }
    }
}

/// Broadcast rule for one axis when [`merge`](Space::merge)ing spaces.
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

    /// A matmul's `k` is its one contracted axis.
    #[test]
    fn a_matmul_contracts_its_one_axis() {
        let lhs = Space::new(&[(M, 8), (K, 4)]);
        let out = Space::new(&[(M, 8), (N, 8)]);
        assert_eq!(lhs.contracted_extent(&out), 4);
    }

    /// A convolution's `k` is taps times channels.
    #[test]
    fn a_convolution_contracts_taps_times_channels() {
        let lhs = Space::new(&[(M, 8), (R, 3), (K, 4)]);
        let out = Space::new(&[(M, 8), (N, 8)]);
        assert_eq!(lhs.contracted_extent(&out), 12);
    }

    /// Contracting nothing is `k = 1`.
    #[test]
    fn contracting_nothing_is_a_unit_depth() {
        let lhs = Space::new(&[(M, 8), (N, 8)]);
        let out = Space::new(&[(M, 8), (N, 8)]);
        assert_eq!(lhs.contracted_extent(&out), 1);
    }

    /// Same contraction order on both operands.
    #[test]
    fn operands_listing_one_contraction_order_agree() {
        let lhs = Space::new(&[(M, 8), (R, 3), (K, 4)]);
        let rhs = Space::new(&[(R, 3), (K, 4), (N, 8)]);
        let out = Space::new(&[(M, 8), (N, 8)]);
        assert!(Space::contraction_agrees(&lhs, &rhs, &out));
    }

    /// A permuted contraction order disagrees.
    #[test]
    fn a_permuted_contraction_order_disagrees() {
        let lhs = Space::new(&[(M, 8), (R, 3), (K, 4)]);
        let rhs = Space::new(&[(K, 4), (R, 3), (N, 8)]);
        let out = Space::new(&[(M, 8), (N, 8)]);
        assert!(!Space::contraction_agrees(&lhs, &rhs, &out));
    }
}
