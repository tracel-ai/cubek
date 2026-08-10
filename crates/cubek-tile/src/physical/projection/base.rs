//! An operand's logical axes mapped onto its buffer's physical axes: the affine combination the
//! module doc derives, assembled from one [`PhysicalAxisMap`] per physical axis.

use cubecl::zspace::SmallVec;

use crate::{Axis, ConcreteLayout, MAX_AXES, PhysicalAxisMap, Scale, StorageTiling};

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

    /// [`direct`](Projection::direct) with the axes storage-tiled per `tiling`: each axis labels as
    /// many physical axes as it has fragments, emitted in [`StorageTiling`]'s level-major order. A
    /// [`StorageTiling`] of all ones is [`direct`](Projection::direct) itself.
    ///
    /// The tiling lives in the repetition, so the projection alone says how many physical axes the
    /// buffer has and how a coordinate splits across them ([`digit`](Projection::digit)); no extent
    /// is read here and none is baked in.
    pub fn tiled(axes: &[Axis], tiling: StorageTiling) -> Self {
        let physical: Vec<PhysicalAxisMap> = tiling
            .order(axes)
            .into_iter()
            .map(PhysicalAxisMap::of)
            .collect();
        Projection::new(axes, &physical)
    }

    /// The same operand in *coordinate* space: an axis's storage fragments merged back into the one
    /// coordinate they are digits of, so there is one entry per coordinate a
    /// [`GmemLayout`](crate::GmemLayout) consumes rather than one per physical axis of the buffer.
    /// [`positional`](Projection::positional) is the other half of the pair, coordinate to
    /// physical; this one is logical to coordinate. An untiled operand, gathered or not, is its own
    /// coordinate map.
    pub fn untiled(&self) -> Projection {
        let carried = self.carried_groups();
        let physical: Vec<PhysicalAxisMap> = carried
            .iter()
            .map(|&pa| self.physical[pa].clone())
            .collect();
        Projection::new(&self.axes, &physical)
    }

    /// The same buffer addressed by physical position instead of by this operand's own axes: each
    /// physical axis relabeled with the synthetic [`Axis`] of the coordinate that addresses it, at
    /// coefficient `1`. Storage tiling survives (a tiled axis's fragments share one label, which is
    /// what makes them digits of one coordinate); a gather does not, since it is resolved one layer
    /// up, by [`AxisProjection`](crate::AxisProjection), and never reaches the layout.
    ///
    /// This is the map [`GmemLayout`](crate::GmemLayout) splits coordinates through, so a buffer
    /// only ever has to describe itself once: the operand's projection, relabeled.
    pub fn positional(&self) -> Projection {
        let carried = self.carried_groups();
        let axes: Vec<Axis> = (0..carried.len()).map(|p| Axis(p as u8)).collect();
        let physical: Vec<PhysicalAxisMap> = self
            .physical
            .iter()
            .map(|map| {
                let at = carried
                    .iter()
                    .position(|&pa| self.physical[pa].terms()[0].axis == map.terms()[0].axis)
                    .expect("collected above");
                PhysicalAxisMap::of(axes[at])
            })
            .collect();
        Projection::new(&axes, &physical)
    }

    /// One physical axis per *coordinate* this operand is addressed by: the first fragment of each
    /// distinct leading axis, in buffer order. The two collapsing views ([`untiled`],
    /// [`positional`]) share it, since both fold an axis's fragments back into the one coordinate
    /// they are digits of.
    ///
    /// Identifying a group by its leading term is only an identity when a physical axis carries
    /// one logical axis, which is exactly what storage tiling produces: an affine map contributes
    /// its whole cell as one coordinate, so a *gathered* projection must be untiled for this to
    /// mean anything. [`validate`](Projection::validate) pins that down at construction, and the
    /// assert here keeps a hand-built projection from silently losing a physical axis.
    fn carried_groups(&self) -> Vec<usize> {
        assert!(
            self.is_invertible() || !self.is_tiled(),
            "Projection: an affine map cannot also be storage-tiled; its physical axes do not \
             group into coordinates"
        );
        let mut carried: Vec<usize> = Vec::new();
        for (pa, map) in self.physical.iter().enumerate() {
            let axis = map.terms()[0].axis;
            if !carried
                .iter()
                .any(|&q| self.physical[q].terms()[0].axis == axis)
            {
                carried.push(pa);
            }
        }
        carried
    }

    /// [`GmemLayout`](crate::GmemLayout)'s own physical-position map: coordinate `p`
    /// (`0..tiling.rank()`) labeled by the synthetic axis `Axis(p)`, split per `tiling`. A
    /// `GmemLayout` addresses its buffer by physical position, already resolved past any gather one
    /// layer up, so it never needs the operand's real axis labels.
    pub fn of_tiling(tiling: StorageTiling) -> Projection {
        let axes: Vec<Axis> = (0..tiling.rank()).map(|p| Axis(p as u8)).collect();
        Projection::tiled(&axes, tiling)
    }

    /// How many fragments each logical axis is split across, counted off the physical map. A
    /// gathered one reports one fragment per axis, since several of its axes share one physical
    /// axis rather than one axis spanning several: it is not tiled, and its physical rank does not
    /// follow from this.
    ///
    /// Counts only, not an order: [`tiled`](Projection::tiled) reconstructs this projection from
    /// them exactly when it is [`level_major`](Projection::is_level_major), which is every
    /// projection [`tiled`](Projection::tiled) itself builds but not every one
    /// [`of_layout`](Projection::of_layout) can read off a real buffer (`[A, A, B]` counts as
    /// `[2, 1]`, whose level-major order is `[A, B, A]`). [`is_tiled`](Projection::is_tiled) asks
    /// only whether some count exceeds one, which the order cannot change.
    pub fn tiling(&self) -> StorageTiling {
        StorageTiling::per_axis(
            &self
                .axes
                .iter()
                .map(|&axis| self.carriers(axis).len())
                .collect::<Vec<_>>(),
        )
    }

    /// Whether the physical axes run in [`StorageTiling`]'s level-major order, so
    /// [`tiling`](Projection::tiling) describes this projection whole and
    /// [`tiled`](Projection::tiled) rebuilds it. True by construction for
    /// [`tiled`](Projection::tiled) and [`of_tiling`](Projection::of_tiling), and for any untiled
    /// projection trivially; a buffer that groups an axis's fragments together
    /// (`[A, A, B]`) is a layout the counts alone do not pin down.
    pub fn is_level_major(&self) -> bool {
        self.is_invertible()
            && self.tiling().order(&self.axes)
                == self
                    .physical
                    .iter()
                    .map(|m| m.terms()[0].axis)
                    .collect::<Vec<_>>()
    }

    /// Whether some axis is storage-tiled: split across several physical fragments, so a
    /// coordinate along it decomposes into one digit per fragment. Not a rank comparison, which a
    /// gather also fails (its logical rank exceeds its physical one without any axis being split).
    pub fn is_tiled(&self) -> bool {
        self.tiling().is_tiled()
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
    /// ([`logical_extent`](crate::logical_extent)). Never empty: every caller decomposes a coordinate along `axis`, and an
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
    /// physical coordinates uniquely determine the logical ones and
    /// [`fold_physical`](crate::fold_physical) can invert `GmemLayout`'s `to_source_pos`. True for
    /// [`of_layout`](Projection::of_layout) and [`of_tiling`](Projection::of_tiling); false for an
    /// affine (gather/stencil) map, which mixes several logical coordinates into one physical cell.
    pub fn is_invertible(&self) -> bool {
        self.physical
            .iter()
            .all(|m| matches!(m.terms(), [t] if matches!(t.scale, Scale::Static(1))))
    }

    /// The phase-1 contract for a *gathered* (affine) projection. A plain one, storage-tiled or
    /// not, is unconstrained: it is what the engine has always done.
    pub fn validate(&self) {
        assert!(
            !self
                .physical
                .iter()
                .any(|m| m.terms().iter().any(|t| t.scale.is_dynamic())),
            "Projection: Dynamic scales are reserved for a runtime stride/dilation and are not \
             implemented yet"
        );
        // Every physical axis carrying one logical axis at coefficient 1 is exactly "no gather",
        // whatever the ranks: `direct`, and `tiled`, which repeats a label rather than scaling it.
        if self.is_invertible() {
            return;
        }
        assert!(
            !self.physical.is_empty() && !self.axes.is_empty(),
            "Projection: an operand must span at least one logical and one physical axis"
        );
        // The innermost physical axis is addressed in *lines*, not elements: `MemData::at` divides
        // its edge by the vector size and `of_impl` counts its physical shape in lines. That
        // arithmetic is only sound when it is one logical axis at coefficient 1.
        //
        // The other direction, a *coarser* physical axis also scaling that same logical axis,
        // would mix lines into an element count.
        let innermost = self.axes[self.axes.len() - 1];
        assert!(
            self.physical[self.physical.len() - 1].is_identity(innermost),
            "Projection: the innermost physical axis must be the operand's last logical axis at \
             coefficient 1 (it is addressed in vector lines)"
        );
        for &axis in self.axes.iter() {
            let count = self.physical.iter().filter(|m| m.scale(axis) != 0).count();
            assert!(
                count > 0,
                "Projection: logical axis {axis:?} addresses no physical axis"
            );
            // A gathered operand must be untiled gmem, so each logical axis can map to at most one physical axis.
            assert!(
                count == 1,
                "Projection: logical axis {axis:?} addresses several physical axes, so it is \
                 either storage-tiled (a gathered operand must be untiled gmem) or read off two \
                 places at once"
            );
        }
    }
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
    fn direct_span_is_the_edge() {
        let p = Projection::direct(&[A, B]);
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
        assert_eq!(p.scale(0, A), 2);
        assert_eq!(p.scale(0, R), 3);
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

    /// The innermost axis is addressed in lines, so a coarser physical axis scaling it would mix
    /// line and element units. Carrying it twice is what storage tiling *is*, so the tiling refusal
    /// is what rules the shape out. `A <- A*1 + B*2` over logical `[A, B]` with `B` innermost.
    #[test]
    #[should_panic(expected = "addresses several physical axes")]
    fn innermost_axis_rides_no_coarser_physical_axis() {
        Projection::new(
            &[A, B],
            &[
                PhysicalAxisMap::affine(&[(A, 1), (B, 2)]),
                PhysicalAxisMap::of(B),
            ],
        )
        .validate();
    }

    #[test]
    #[should_panic(expected = "innermost physical axis")]
    fn innermost_must_be_identity() {
        Projection::new(
            &[A, R],
            &[PhysicalAxisMap::of(A), PhysicalAxisMap::affine(&[(R, 2)])],
        )
        .validate();
    }

    /// A gather cannot ride a `[grid…, tile…]` buffer: `B` here is storage-tiled (two fragments)
    /// while `Ih <- A*2 + R*3` gathers, and one advance cannot be both split into digits and
    /// scaled.
    #[test]
    #[should_panic(expected = "addresses several physical axes")]
    fn a_gather_rejects_tiled_storage() {
        Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 3)]),
                PhysicalAxisMap::of(B),
                PhysicalAxisMap::of(B),
            ],
        )
        .validate();
    }

    #[test]
    #[should_panic(expected = "addresses no physical axis")]
    fn every_axis_must_address_a_physical_axis() {
        Projection::new(
            &[A, R, B],
            &[PhysicalAxisMap::affine(&[(A, 2)]), PhysicalAxisMap::of(B)],
        )
        .validate();
    }

    /// A plain projection is never constrained: storage tiling is exactly what it is for.
    #[test]
    fn tiled_is_not_a_gather() {
        let p = Projection::tiled(&[A, B], StorageTiling::uniform(2, 1));
        assert!(!p.is_direct());
        assert!(p.is_tiled());
        assert!(p.is_invertible());
        p.validate();
    }

    /// A gather has fewer physical axes than logical ones without any axis being split, so the
    /// rank comparison it used to be tested by does not answer this.
    #[test]
    fn a_gather_is_not_tiled() {
        let p = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 3)]),
                PhysicalAxisMap::of(B),
            ],
        );
        assert!(!p.is_tiled());
        p.validate();
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

        let p = Projection::of_tiling(StorageTiling::suffix(3, 1, 2));
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

    /// A spec built from a realized tiled layout is honest about its buffer: its physical rank
    /// *is* the tensor's rank, which is what `Tile::of` reads its shape and strides over, and its
    /// positional relabeling is the synthetic map the layout is addressed through. The declared
    /// twin (`TileSpec::new` plus a tiled `Storage`) describes the same buffer.
    #[test]
    fn a_tiled_spec_matches_its_buffer() {
        use crate::{PhysicalAxis, TileSpec};

        const BATCH: Axis = Axis(3);
        let layout = ConcreteLayout::new(&[
            PhysicalAxis::new(BATCH, 2),
            PhysicalAxis::new(A, 4),
            PhysicalAxis::new(B, 4),
            PhysicalAxis::new(A, 8),
            PhysicalAxis::new(B, 8),
        ]);
        let spec = TileSpec::from_concrete(&layout, false, 0);

        assert_eq!(spec.projection.physical_rank(), layout.axes().len());
        assert_eq!(
            spec.projection.positional(),
            Projection::of_tiling(StorageTiling::suffix(3, 1, 1))
        );
        assert_eq!(
            Projection::tiled(&[BATCH, A, B], StorageTiling::suffix(3, 1, 1)),
            spec.projection
        );
        // The tiling is readable straight back off the realized buffer's map.
        assert_eq!(spec.projection.tiling(), StorageTiling::suffix(3, 1, 1));
    }

    /// An untiled operand is its own positional map, up to the relabeling.
    #[test]
    fn positional_of_a_plain_operand_is_the_identity() {
        assert_eq!(
            Projection::direct(&[B, A]).positional(),
            Projection::direct(&[Axis(0), Axis(1)])
        );
    }

    /// A gather resolves one layer up, so its layout sees one coordinate per physical axis: the
    /// two logical axes sharing physical axis 0 collapse into that one position.
    #[test]
    fn positional_of_a_gather_drops_the_affine_terms() {
        let p = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 3)]),
                PhysicalAxisMap::of(B),
            ],
        );
        assert_eq!(
            p.positional(),
            Projection::of_tiling(StorageTiling::uniform(2, 0))
        );
    }

    /// Axes tiled to different depths, which no start/depth pair can describe: `A` is split three
    /// ways and `B` two, so `A` alone appears at the finest level. Each fragment still strips
    /// exactly the finer ones of its own axis.
    #[test]
    fn a_ragged_tiling_orders_level_major() {
        let p = Projection::tiled(&[A, B], StorageTiling::per_axis(&[3, 2]));
        assert_eq!(p.physical_rank(), 5);
        // `[A0, B0, A1, B1, A2]`.
        assert_eq!(p.carriers(A).as_slice(), &[0, 2, 4]);
        assert_eq!(p.carriers(B).as_slice(), &[1, 3]);
        assert_eq!(p.digit(0, A), (SmallVec::from_slice(&[2, 4]), None));
        assert_eq!(p.digit(2, A), (SmallVec::from_slice(&[4]), Some(2)));
        assert_eq!(p.digit(4, A), (SmallVec::new(), Some(4)));
        assert_eq!(p.digit(1, B), (SmallVec::from_slice(&[3]), None));
        assert!(p.is_invertible());
        p.validate();
    }

    /// The fragment counts pin a buffer down only up to the level-major order. A buffer that keeps
    /// an axis's fragments adjacent counts the same as the level-major one it is not, so `tiling`
    /// round-trips through `tiled` for exactly the projections that report `is_level_major`.
    #[test]
    fn tiling_describes_only_a_level_major_buffer() {
        use crate::PhysicalAxis;

        let grouped = Projection::of_layout(&ConcreteLayout::new(&[
            PhysicalAxis::new(A, 4),
            PhysicalAxis::new(A, 8),
            PhysicalAxis::new(B, 4),
        ]));
        let level_major = Projection::tiled(&[A, B], StorageTiling::per_axis(&[2, 1]));

        // Same axes, same counts, different buffers.
        assert_eq!(grouped.logical_axes(), level_major.logical_axes());
        assert_eq!(grouped.tiling(), level_major.tiling());
        assert_ne!(grouped, level_major);

        assert!(!grouped.is_level_major());
        assert!(level_major.is_level_major());
        // What the counts do settle, whatever the order.
        assert!(grouped.is_tiled() && level_major.is_tiled());
        assert_eq!(grouped.carriers(A).as_slice(), &[0, 1]);
        assert_eq!(level_major.carriers(A).as_slice(), &[0, 2]);
    }

    /// A gathered projection is never level-major: its counts do not describe its physical rank at
    /// all, so there is no order for them to be right or wrong about.
    #[test]
    fn a_gather_is_not_level_major() {
        let p = Projection::new(
            &[A, R, B],
            &[
                PhysicalAxisMap::affine(&[(A, 2), (R, 3)]),
                PhysicalAxisMap::of(B),
            ],
        );
        assert!(!p.is_level_major());
        assert!(Projection::direct(&[A, B]).is_level_major());
    }

    /// `tiling` inverts `tiled` for every projection storage tiling can build, so the two are one
    /// description read in either direction.
    #[test]
    fn tiling_round_trips_through_tiled() {
        for tiling in [
            StorageTiling::uniform(2, 0),
            StorageTiling::uniform(3, 2),
            StorageTiling::suffix(3, 1, 1),
            StorageTiling::per_axis(&[3, 1, 2]),
        ] {
            let axes: Vec<Axis> = (0..tiling.rank()).map(|p| Axis(p as u8)).collect();
            let p = Projection::tiled(&axes, tiling.clone());
            assert_eq!(p.tiling(), tiling);
            assert_eq!(p.physical_rank(), tiling.physical_rank());
            assert_eq!(p.is_tiled(), tiling.is_tiled());
            assert!(p.is_level_major());
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
