//! How an operand's logical [`Space`](crate::Space) axes map onto its buffer's
//! [`PhysicalAxis`](crate::PhysicalAxis) positions.
//!
//! The default is one logical axis per physical axis with coefficient `1`: logical rank equals
//! physical rank, one tile coordinate moves one physical axis by one sub-tile, and sibling windows
//! never overlap. Matmul fits because `K` genuinely is a physical axis of `lhs` and `rhs`.
//!
//! A gather-reduce over an *abstract* dimension does not fit. A stencil's reduce axes address the
//! same input physical axis its output axes address, each with its own coefficient:
//!
//! ```text
//! space {Oh, Ow, Cout, Rh, Rw, Cin}
//!   input physical axis Ih <- Oh*stride_h + Rh*dilation_h
//!         physical axis Iw <- Ow*stride_w + Rw*dilation_w
//!         physical axis Cin
//! ```
//!
//! So a physical axis is an affine combination of logical axes, and consecutive windows along `Oh`
//! overlap by the receptive field. [`Projection`] is that combination;
//! [`direct`](Projection::direct) is the degenerate one every current operand uses.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;
use cubecl::zspace::SmallVec;

use crate::{
    Axis, ConcreteLayout, Coords, Fold, FoldExpand, FoldSeq, FoldSeqExpand, MAX_AXES, Storage,
};

/// How far one unit of a logical axis's coordinate moves along one physical axis. Mirrors
/// [`Extent`](crate::Extent): `Static` is a comptime constant so the advance folds the way
/// [`window_start`](crate::MemData) needs, `Dynamic` is reserved for a runtime stride/dilation
/// and is rejected by [`Projection::validate`] until the runtime half exists.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Scale {
    Static(usize),
    Dynamic,
}

impl Scale {
    /// The comptime coefficient; panics on `Dynamic`.
    pub fn get(self) -> usize {
        match self {
            Scale::Static(n) => n,
            Scale::Dynamic => {
                panic!(
                    "Scale::get: this coefficient is Dynamic; its value is only known at runtime"
                )
            }
        }
    }

    pub fn is_dynamic(self) -> bool {
        matches!(self, Scale::Dynamic)
    }
}

/// One logical axis's contribution to one physical axis: `digit * scale`, where the digit is the
/// whole coordinate unless the axis is spread over several physical axes, which
/// [`Projection::digit`] reads off the map's own shape rather than off a stored constant.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AxisTerm {
    pub axis: Axis,
    pub scale: Scale,
}

/// One [`PhysicalAxis`](crate::PhysicalAxis) as an affine combination of logical axes' digits:
/// `physical = Σ digit(axis) * scale`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PhysicalAxisMap {
    terms: SmallVec<[AxisTerm; MAX_AXES]>,
}

impl PhysicalAxisMap {
    /// The identity map: this physical axis *is* `axis`, coefficient `1`. What every operand of a
    /// non-gather operation uses on every physical axis.
    pub fn of(axis: Axis) -> Self {
        PhysicalAxisMap {
            terms: SmallVec::from_slice(&[AxisTerm {
                axis,
                scale: Scale::Static(1),
            }]),
        }
    }

    /// An affine combination, e.g. `affine(&[(Oh, stride), (Rh, dilation)])`.
    pub fn affine(terms: &[(Axis, usize)]) -> Self {
        PhysicalAxisMap {
            terms: terms
                .iter()
                .map(|&(axis, scale)| AxisTerm {
                    axis,
                    scale: Scale::Static(scale),
                })
                .collect(),
        }
    }

    pub fn terms(&self) -> &[AxisTerm] {
        &self.terms
    }

    /// `axis`'s coefficient, `0` when it does not address this physical axis.
    pub fn scale(&self, axis: Axis) -> usize {
        self.terms
            .iter()
            .find(|t| t.axis == axis)
            .map_or(0, |t| t.scale.get())
    }

    /// Whether this physical axis is exactly `axis` at coefficient `1`. Says nothing about digit
    /// extraction, which is a property of the whole [`Projection`] (how many physical axes carry
    /// `axis`), not of one map.
    pub fn is_identity(&self, axis: Axis) -> bool {
        matches!(
            self.terms.as_slice(),
            [AxisTerm {
                axis: a,
                scale: Scale::Static(1)
            }] if *a == axis
        )
    }
}

/// An operand's logical axes mapped onto its buffer's physical axes.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Projection {
    /// One entry per physical axis, in buffer order.
    physical: SmallVec<[PhysicalAxisMap; MAX_AXES]>,
    /// The logical axes this operand spans, in its own order. This becomes the tile's
    /// [`Space`](crate::Space) axis order, so the last entry is the vectorized axis.
    axes: SmallVec<[Axis; MAX_AXES]>,
}

impl Projection {
    /// One logical axis per physical axis at coefficient `1`, logical order equal to buffer order:
    /// the mapping [`TileSpec::new`](crate::TileSpec::new) builds and every non-gather operand uses.
    pub fn direct(axes: &[Axis]) -> Self {
        Projection {
            physical: axes.iter().map(|&a| PhysicalAxisMap::of(a)).collect(),
            axes: SmallVec::from_slice(axes),
        }
    }

    /// [`direct`](Projection::direct) over a space's own axes: what a materialized tile (an smem
    /// stage, a fragment) maps through, whatever its source mapped through.
    pub fn direct_over(space: &crate::Space) -> Self {
        Projection::direct(&space.axes().collect::<Vec<_>>())
    }

    /// Build the projection off a realized [`ConcreteLayout`]: one identity term per physical
    /// axis, in buffer order, with logical order [`distinct_axes`](ConcreteLayout::distinct_axes).
    /// A storage-tiled logical axis labels several physical axes, which is the whole encoding of
    /// its tiling: the digit each fragment carries follows from that repetition
    /// ([`digit`](Projection::digit)), so no extent is read here and none is baked in. What
    /// [`TileSpec::from_concrete`](crate::TileSpec::from_concrete) builds every operand's mapping
    /// from.
    pub fn of_layout(layout: &ConcreteLayout) -> Projection {
        Projection {
            physical: layout
                .axes()
                .iter()
                .map(|pa| PhysicalAxisMap::of(pa.axis()))
                .collect(),
            axes: layout.distinct_axes(),
        }
    }

    /// [`GmemLayout`](crate::GmemLayout)'s own physical-position map: position `p`
    /// (`0..start_axis + num_tiled`) labeled by the synthetic axis `Axis(p)`. A `GmemLayout`
    /// addresses its buffer by physical position, already resolved past any gather one layer up,
    /// so it never needs the operand's real axis labels. Positions from `start_axis` on are the
    /// tiled block, each labeling `levels + 1` physical axes in the buffer's `[pre…, grid…, tile…]`
    /// order, at any depth.
    pub fn of_tiling(start_axis: usize, num_tiled: usize, levels: usize) -> Projection {
        let axes: Vec<Axis> = (0..start_axis + num_tiled).map(|p| Axis(p as u8)).collect();
        let mut physical: Vec<PhysicalAxisMap> = axes[..start_axis]
            .iter()
            .map(|&a| PhysicalAxisMap::of(a))
            .collect();
        for _ in 0..=levels {
            physical.extend(axes[start_axis..].iter().map(|&a| PhysicalAxisMap::of(a)));
        }
        Projection::new(&axes, &physical)
    }

    /// Whether this projection's physical rank exceeds its logical rank: some axis is split
    /// across several physical fragments (storage-tiled). Only [`of_tiling`](Projection::of_tiling)
    /// and a tiled [`of_layout`](Projection::of_layout) ever produce this; every other constructor
    /// keeps physical and logical rank equal.
    pub fn is_tiled(&self) -> bool {
        self.physical.len() != self.axes.len()
    }

    /// Where `axis`'s digit at physical axis `pa` sits in the buffer's mixed radix: the physical
    /// positions of that axis's *finer* fragments (whose extents are the block this digit sits
    /// above), and the position whose extent is this digit's radix, `None` for the outermost
    /// fragment, which has no enclosing block and keeps the full quotient.
    ///
    /// Positional, not numeric: the radix is looked up in the buffer's own `physical_shape` at use
    /// time rather than baked in here. That is what lets one representation serve an smem stage,
    /// whose extents are comptime and fold the arithmetic away, and a gmem tensor, whose tile
    /// extents are genuine runtime values. An axis carried by a single physical axis yields
    /// `(&[], None)`: the whole coordinate, no arithmetic at all.
    pub fn digit(&self, pa: usize, axis: Axis) -> (SmallVec<[usize; MAX_AXES]>, Option<usize>) {
        let carriers = self.carriers(axis);
        assert!(
            carriers.contains(&pa),
            "Projection::digit: physical axis {pa} does not carry {axis:?} (carried by {carriers:?})"
        );
        let finer = carriers.iter().copied().filter(|&q| q > pa).collect();
        (finer, (carriers[0] != pa).then_some(pa))
    }

    /// The physical axes carrying `axis`, in buffer order: one entry unless the axis is
    /// storage-tiled, in which case its extents multiply back to the logical one
    /// ([`logical_extent`]). Never empty: every caller decomposes a coordinate along `axis`, and an
    /// axis addressing no physical axis has no decomposition, so that is a malformed projection
    /// rather than an empty answer ([`validate`](Projection::validate) rules it out up front).
    pub fn carriers(&self, axis: Axis) -> SmallVec<[usize; MAX_AXES]> {
        let carriers: SmallVec<[usize; MAX_AXES]> = (0..self.physical.len())
            .filter(|&q| self.physical[q].terms().iter().any(|t| t.axis == axis))
            .collect();
        assert!(
            !carriers.is_empty(),
            "Projection::carriers: {axis:?} addresses no physical axis of this operand"
        );
        carriers
    }

    /// `axes` in the tile's logical order, `physical` one per physical axis in buffer order.
    pub fn new(axes: &[Axis], physical: &[PhysicalAxisMap]) -> Self {
        Projection {
            physical: physical.iter().cloned().collect(),
            axes: SmallVec::from_slice(axes),
        }
    }

    /// Whether this is the [`direct`](Projection::direct) mapping. Every generalized path is
    /// gated on this being `false`, so a direct operand keeps its exact previous codegen.
    pub fn is_direct(&self) -> bool {
        self.physical.len() == self.axes.len()
            && self
                .physical
                .iter()
                .zip(self.axes.iter())
                .all(|(map, &axis)| map.is_identity(axis))
    }

    pub fn physical_rank(&self) -> usize {
        self.physical.len()
    }

    pub fn logical_rank(&self) -> usize {
        self.axes.len()
    }

    pub fn logical_axes(&self) -> &[Axis] {
        &self.axes
    }

    /// `axis`'s index in the logical order, which is the order a coordinate comes in.
    pub fn position(&self, axis: Axis) -> usize {
        self.axes
            .iter()
            .position(|&a| a == axis)
            .expect("Projection::position: axis not spanned by this operand")
    }

    pub fn physical_axis(&self, pa: usize) -> &PhysicalAxisMap {
        &self.physical[pa]
    }

    /// `axis`'s coefficient along physical axis `pa`, `0` when it does not address it.
    pub fn scale(&self, pa: usize, axis: Axis) -> usize {
        self.physical[pa].scale(axis)
    }

    /// How far one *tile* step along `axis` moves physical axis `pa`, given that level's sub-tile
    /// `edge`. The direct case is `edge`, which is what [`MemData::at`](crate::MemData) has always
    /// added.
    pub fn origin_step(&self, pa: usize, axis: Axis, edge: usize) -> usize {
        edge * self.scale(pa, axis)
    }

    /// How many elements of physical axis `pa` a region covers, given each logical axis's extent:
    /// the receptive field `1 + Σ (extent - 1) * scale`. A single coefficient-`1` term collapses to
    /// `extent`, so a direct operand's window is its sub-tile edge as before; two terms give the
    /// overlapping stencil window.
    pub fn span(&self, pa: usize, extent_of: impl Fn(Axis) -> usize) -> usize {
        1 + self.physical[pa]
            .terms()
            .iter()
            .map(|t| (extent_of(t.axis) - 1) * t.scale.get())
            .sum::<usize>()
    }

    /// Whether every physical axis carries exactly one logical axis at coefficient `1`, so the
    /// physical coordinates uniquely determine the logical ones and [`fold_physical`] can invert
    /// [`GmemLayout::to_source_pos`](crate::GmemLayout::to_source_pos). True for
    /// [`of_layout`](Projection::of_layout) and [`of_tiling`](Projection::of_tiling); false for an
    /// affine (gather/stencil) map, which mixes several logical coordinates into one physical cell.
    pub fn is_invertible(&self) -> bool {
        self.physical
            .iter()
            .all(|m| matches!(m.terms(), [t] if matches!(t.scale, Scale::Static(1))))
    }

    /// The phase-1 contract for a non-[`direct`](Projection::direct) projection. A direct one is
    /// unconstrained: it is what the engine has always done.
    pub fn validate(&self, storage: &Storage) {
        assert!(
            !self
                .physical
                .iter()
                .any(|m| m.terms().iter().any(|t| t.scale.is_dynamic())),
            "Projection: Dynamic scales are reserved for a runtime stride/dilation and are not \
             implemented yet"
        );
        if self.is_direct() {
            return;
        }
        assert!(
            !self.physical.is_empty() && !self.axes.is_empty(),
            "Projection: an operand must span at least one logical and one physical axis"
        );
        // The innermost physical axis is addressed in *lines*, not elements: `MemData::at` divides
        // its edge by the vector size and `of_impl` counts its physical shape in lines. That
        // arithmetic is only sound when it is one logical axis at coefficient 1.
        let innermost = self.axes[self.axes.len() - 1];
        assert!(
            self.physical[self.physical.len() - 1].is_identity(innermost),
            "Projection: the innermost physical axis must be the operand's last logical axis at \
             coefficient 1 (it is addressed in vector lines)"
        );
        // A tiled `[grid…, tile…]` buffer splits an advance into two digits, which is not linear
        // in the edge, so it cannot absorb a scaled one. Projected operands are bare gmem.
        assert!(
            storage.start_axis == 0 && storage.levels == 0,
            "Projection: a projected operand must be untiled gmem (start_axis 0, levels 0); \
             got {storage:?}"
        );
        for &axis in self.axes.iter() {
            assert!(
                self.physical.iter().any(|m| m.scale(axis) != 0),
                "Projection: logical axis {axis:?} addresses no physical axis"
            );
        }
    }
}

/// The logical extent per axis, folded from `projection`'s physical shape: a single-carrier axis
/// passes its physical extent through, a storage-tiled one multiplies its fragments' extents back
/// together. Reduces to `physical_shape` itself for an untiled projection. A free function, not a
/// method: `Projection` stays comptime-only (never a [`CubeType`]), so the runtime folding it
/// drives lives in an ordinary `#[cube]` function, like every other comptime/runtime mix in this
/// crate ([`top_window`](crate::GmemLayout)).
#[cube]
pub fn logical_extent(
    #[comptime] projection: Projection,
    physical_shape: &Coords<u32>,
) -> Coords<u32> {
    let mut bound = Coords::<u32>::new();
    #[unroll]
    for i in 0..comptime!(projection.logical_rank()) {
        let picks = comptime!(projection.carriers(projection.axes[i]).to_vec());
        bound.push(physical_shape.fproduct(picks));
    }
    bound
}

/// The line offset one `edge`-sized tile step along `axis` moves under `projection`: `axis`'s
/// digits taken of `edge` itself rather than of a coordinate, dotted with `strides`. Exact because
/// one edge-step's offset is linear in the tile index: every edge divides its enclosing block, so
/// decomposing the step size the same way as a coordinate reconstructs the same advance. The
/// radices come from `physical_shape`, so a static store folds the whole thing to a constant and a
/// runtime-shaped one stays exact.
#[cube]
pub fn step_offset(
    #[comptime] projection: Projection,
    #[comptime] axis: Axis,
    #[comptime] edge: usize,
    physical_shape: &Coords<u32>,
    strides: &Coords<u32>,
) -> u32 {
    let carriers = comptime!(projection.carriers(axis));
    let picks = comptime!((0..carriers.len()).collect::<Vec<_>>());
    let mut parts = Sequence::<u32>::new();
    #[unroll]
    for k in 0..comptime!(carriers.len()) {
        let pa = comptime!(carriers[k]);
        let (finer, modulo) = comptime!(projection.digit(pa, axis));
        let scale = comptime!(projection.scale(pa, axis) as u32);
        let quot = comptime!(edge as u32)
            .runtime()
            .fdiv(physical_shape.fproduct(comptime!(finer.to_vec())));
        let digit = match comptime!(modulo) {
            Some(m) => quot.frem(physical_shape.at(m)),
            None => quot,
        };
        parts.push(digit.fmul(comptime!(scale).runtime()).fmul(strides.at(pa)));
    }
    parts.fsum(picks)
}

/// The inverse of [`GmemLayout::to_source_pos`](crate::GmemLayout::to_source_pos): the logical
/// coordinate under `projection` that produced physical digits `digits` (one entry per physical
/// axis, already decoded off the flat physical index). Requires
/// [`Projection::is_invertible`]; a gathered (affine, scale != 1) projection never reaches this,
/// since a stage is always materialized dense through [`Projection::direct_over`] first.
#[cube]
pub fn fold_physical(
    #[comptime] projection: Projection,
    digits: &Coords<u32>,
    physical_shape: &Coords<u32>,
) -> CoordsDyn {
    comptime!(assert!(
        projection.is_invertible(),
        "Projection::fold_physical: not invertible (an affine term mixes several logical \
         coordinates into one physical cell)"
    ));
    let mut out = CoordsDyn::new();
    #[unroll]
    for i in 0..comptime!(projection.logical_rank()) {
        let axis = comptime!(projection.axes[i]);
        let carriers = comptime!(projection.carriers(axis));
        let picks = comptime!((0..carriers.len()).collect::<Vec<_>>());
        let mut parts = Sequence::<u32>::new();
        #[unroll]
        for k in 0..comptime!(carriers.len()) {
            let pa = comptime!(carriers[k]);
            // Each digit weighs the block it sits above, the same finer extents that stripped it.
            let (finer, _) = comptime!(projection.digit(pa, axis));
            parts.push(
                digits
                    .at(pa)
                    .fmul(physical_shape.fproduct(comptime!(finer.to_vec()))),
            );
        }
        out.push(parts.fsum(picks));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Axis = Axis(0);
    const B: Axis = Axis(1);
    const R: Axis = Axis(2);

    #[test]
    fn direct_is_direct() {
        let p = Projection::direct(&[A, B]);
        assert!(p.is_direct());
        assert_eq!(p.physical_rank(), 2);
        assert_eq!(p.logical_axes(), &[A, B]);
    }

    #[test]
    fn direct_span_and_step_are_the_edge() {
        let p = Projection::direct(&[A, B]);
        assert_eq!(p.origin_step(0, A, 8), 8);
        assert_eq!(p.origin_step(0, B, 8), 0);
        assert_eq!(p.span(0, |_| 8), 8);
    }

    /// `I <- A*stride + R*dilation`: one tile step along `A` moves `edge*stride`, and a region
    /// covering `ea` outputs and `er` taps spans the receptive field.
    #[test]
    fn affine_span_is_the_receptive_field() {
        let p = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 3)]),
                PhysicalAxisMap::of(B),
            ],
        );
        assert!(!p.is_direct());
        assert_eq!(p.origin_step(0, A, 4), 8);
        assert_eq!(p.origin_step(0, R, 4), 12);
        assert_eq!(p.scale(0, B), 0);
        // 1 + (4-1)*2 + (3-1)*3 = 13
        assert_eq!(p.span(0, |a| if a == A { 4 } else { 3 }), 13);
        // A single tap at stride 1 is a plain contiguous window.
        let q = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 1), (R, 1)]),
                PhysicalAxisMap::of(B),
            ],
        );
        assert_eq!(q.span(0, |a| if a == A { 4 } else { 1 }), 4);
    }

    #[test]
    #[should_panic(expected = "innermost physical axis")]
    fn innermost_must_be_identity() {
        Projection::new(
            &[A, R],
            &[PhysicalAxisMap::of(A), PhysicalAxisMap::affine(&[(R, 2)])],
        )
        .validate(&Storage::passthrough(0, 0));
    }

    #[test]
    #[should_panic(expected = "untiled gmem")]
    fn projected_rejects_tiled_storage() {
        Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 3)]),
                PhysicalAxisMap::of(B),
            ],
        )
        .validate(&Storage::passthrough(0, 1));
    }

    #[test]
    #[should_panic(expected = "addresses no physical axis")]
    fn every_axis_must_address_a_physical_axis() {
        Projection::new(
            &[A, R, B],
            &[PhysicalAxisMap::affine(&[(A, 2)]), PhysicalAxisMap::of(B)],
        )
        .validate(&Storage::passthrough(0, 0));
    }

    /// A direct projection is never constrained: tiled storage is exactly what it is for.
    #[test]
    fn direct_accepts_tiled_storage() {
        Projection::direct(&[A, B]).validate(&Storage::passthrough(0, 1));
    }

    /// `[batch, m_grid, n_grid, m_tile, n_tile]`: a passthrough axis carries one physical axis and
    /// no digit arithmetic, while each tiled axis's grid digit strips its tile fragment and keeps
    /// the full quotient, and its tile digit strips nothing and takes its own extent as the radix.
    #[test]
    fn of_layout_batch_and_grid_tile() {
        use crate::PhysicalAxis;

        const BATCH: Axis = Axis(3);
        let layout = ConcreteLayout::new(&[
            PhysicalAxis::new(BATCH, 2),
            PhysicalAxis::new(A, 4),
            PhysicalAxis::new(B, 4),
            PhysicalAxis::new(A, 8),
            PhysicalAxis::new(B, 8),
        ]);
        let p = Projection::of_layout(&layout);

        assert_eq!(p.logical_axes(), &[BATCH, A, B]);
        assert_eq!(p.physical_rank(), 5);
        assert!(p.is_tiled());
        assert!(p.is_invertible());

        assert_eq!(p.digit(0, BATCH), (SmallVec::new(), None));
        assert_eq!(p.digit(1, A), (SmallVec::from_slice(&[3]), None));
        assert_eq!(p.digit(2, B), (SmallVec::from_slice(&[4]), None));
        assert_eq!(p.digit(3, A), (SmallVec::new(), Some(3)));
        assert_eq!(p.digit(4, B), (SmallVec::new(), Some(4)));
    }

    /// An untiled layout is [`direct`](Projection::direct): one physical axis per logical one, so
    /// every digit is the whole coordinate and the addressing is the plain strided dot.
    #[test]
    fn of_layout_untiled_is_identity() {
        use crate::PhysicalAxis;

        let layout = ConcreteLayout::new(&[PhysicalAxis::new(A, 4), PhysicalAxis::new(B, 8)]);
        let p = Projection::of_layout(&layout);
        assert_eq!(p, Projection::direct(&[A, B]));
        assert!(p.is_invertible());
        assert_eq!(p.digit(0, A), (SmallVec::new(), None));
    }

    /// The synthetic per-position map addresses the same `[pre…, grid…, tile…]` buffer as a
    /// realized layout of the same shape, at any depth: two levels give three fragments per tiled
    /// position, each stripping the ones below it.
    #[test]
    fn of_tiling_matches_a_realized_layout() {
        use crate::PhysicalAxis;

        let p = Projection::of_tiling(1, 2, 2);
        assert_eq!(p.physical_rank(), 7);
        assert_eq!(p.logical_rank(), 3);
        // `[batch, m_grid, n_grid, m_mid, n_mid, m_tile, n_tile]`, `M` at positions 1, 3, 5.
        assert_eq!(p.digit(1, Axis(1)), (SmallVec::from_slice(&[3, 5]), None));
        assert_eq!(p.digit(3, Axis(1)), (SmallVec::from_slice(&[5]), Some(3)));
        assert_eq!(p.digit(5, Axis(1)), (SmallVec::new(), Some(5)));
        assert_eq!(p.digit(0, Axis(0)), (SmallVec::new(), None));

        let realized = ConcreteLayout::new(&[
            PhysicalAxis::new(Axis(0), 2),
            PhysicalAxis::new(Axis(1), 4),
            PhysicalAxis::new(Axis(2), 4),
            PhysicalAxis::new(Axis(1), 2),
            PhysicalAxis::new(Axis(2), 2),
            PhysicalAxis::new(Axis(1), 8),
            PhysicalAxis::new(Axis(2), 8),
        ]);
        assert_eq!(p, Projection::of_layout(&realized));
    }

    /// Folding `Σ digit * (extents it stripped)` back reconstructs the coordinate the digits were
    /// decomposed from, which is what `fold_physical` does to `to_source_pos`. Both are `#[cube]`,
    /// so the round trip is checked here on the digit positions they are built from.
    #[test]
    fn fold_physical_digits_invert_to_source_pos_digits() {
        use crate::PhysicalAxis;

        let shape = [3usize, 8];
        let layout = ConcreteLayout::new(&[
            PhysicalAxis::new(A, shape[0]),
            PhysicalAxis::new(A, shape[1]),
        ]);
        let p = Projection::of_layout(&layout);
        assert!(p.is_invertible());
        let block = |pa| {
            p.digit(pa, A)
                .0
                .iter()
                .map(|&q| shape[q])
                .product::<usize>()
        };

        for coord in [0usize, 1, 7, 8, 9, 19, 23] {
            let digit = |pa| match p.digit(pa, A).1 {
                Some(m) => (coord / block(pa)) % shape[m],
                None => coord / block(pa),
            };
            assert_eq!(digit(0) * block(0) + digit(1) * block(1), coord);
        }
    }

    #[test]
    fn is_invertible_false_for_a_stencil_map() {
        let p = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 3)]),
                PhysicalAxisMap::of(B),
            ],
        );
        assert!(!p.is_invertible());
        assert!(Projection::direct(&[A, B]).is_invertible());
    }
}
