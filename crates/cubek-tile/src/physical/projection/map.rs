//! One physical axis of a buffer as an affine combination of the operand's logical axes, and the
//! coefficients that combination is built from: the parts [`Projection`](crate::Projection)
//! assembles, one per physical axis.

use cubecl::zspace::SmallVec;

use crate::{Axis, MAX_AXES};

/// How far one unit of a logical axis's coordinate moves along one physical axis. Mirrors
/// [`Extent`](crate::Extent): `Static` is a comptime constant so the advance folds the way
/// [`window_start`](crate::MemData) needs, `Dynamic` is a runtime stride or dilation whose value
/// rides the tile ([`Tile::of_gathered`](crate::Tile::of_gathered)) instead of the kernel.
///
/// A `Dynamic` coefficient still declares `max`, the largest value the launch may pass. The exact
/// receptive field is then a runtime value but its *bound* is not, which is all a stage needs: the
/// smem box is sized at `max` and the fill occupies as much of it as the runtime coefficient
/// reaches. Overshoot is dead space, so keep `max` tight; it is part of the kernel's identity, so a
/// loose one only costs occupancy, never correctness.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Scale {
    Static(usize),
    Dynamic { max: usize },
}

impl Scale {
    /// The comptime coefficient; panics on `Dynamic`.
    pub fn get(self) -> usize {
        match self {
            Scale::Static(n) => n,
            Scale::Dynamic { .. } => {
                panic!(
                    "Scale::get: this coefficient is Dynamic; its value is only known at runtime"
                )
            }
        }
    }

    /// The largest value this coefficient can take: itself when `Static`, the declared `max` when
    /// `Dynamic`. What [`span`](crate::Projection::span) and [`Compaction`](crate::Compaction) size
    /// a window by, since a receptive field grows with its coefficients.
    pub fn bound(self) -> usize {
        match self {
            Scale::Static(n) => n,
            Scale::Dynamic { max } => max,
        }
    }

    pub fn is_dynamic(self) -> bool {
        matches!(self, Scale::Dynamic { .. })
    }
}

/// The constant term of one physical axis's affine combination. Mirrors [`Scale`]: `Static` is a
/// comptime constant so [`may_underflow`](crate::Projection::may_underflow) can track whether it
/// is negative, `Dynamic` is a runtime padding or placement whose value rides the tile's signed
/// offset carrier instead of the kernel.
///
/// Unlike [`Scale::Dynamic`], an `Offset::Dynamic` needs no bound to be staged: `span` is
/// offset-invariant, and [`Compaction`](crate::Compaction) drops the offset entirely, so it costs
/// no window geometry at all rather than merely a conservative one. The only cost is that
/// [`may_underflow`](crate::Projection::may_underflow) cannot prove non-negativity and
/// conservatively arms the signed guard.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Offset {
    Static(isize),
    Dynamic,
}

impl Offset {
    /// The comptime offset; panics on `Dynamic`.
    pub fn get(self) -> isize {
        match self {
            Offset::Static(n) => n,
            Offset::Dynamic => {
                panic!("Offset::get: this offset is Dynamic; its value is only known at runtime")
            }
        }
    }

    pub fn is_dynamic(self) -> bool {
        matches!(self, Offset::Dynamic)
    }
}

impl From<isize> for Offset {
    fn from(n: isize) -> Self {
        Offset::Static(n)
    }
}

/// What one physical axis's whole affine combination is divided by, floored: the denominator of a
/// *rational* mapping, `(Σ digit * scale + offset) / divisor`. `Static(1)` is the integer mapping
/// every operand but a fractionally scaled one carries, and is the identity everywhere below.
///
/// One per physical axis rather than one per term, because the floor does not distribute over the
/// sum: `Σ ⌊tᵢ⌋` is not `⌊Σ tᵢ⌋`, and it is the whole numerator (the offset included) the resample
/// mapping `⌊(o·Wᵢₙ + offset) / Wₒᵤₜ⌋` divides. A term meant to stay integral is spelled by giving
/// it the divisor as its coefficient, which the division then cancels exactly.
///
/// A divisor every coefficient cancels is not a division at all, and
/// [`over`](PhysicalAxisMap::over) reduces it away rather than carrying it: `⌊(2o + 4r)/2⌋` is
/// spelled as a fraction but steps like the integer map `o + 2r`, and is stored as one.
///
/// Like [`Scale::Dynamic`], a `Dynamic` divisor declares a bound rather than a value, and it is a
/// *lower* one: a window shrinks as its divisor grows, so the widest field is the one the smallest
/// divisor spans and `min` is what sizes the stage. It still costs an in-kernel integer division
/// per read, where a `Static` one folds. Any rational mapping stages uncompacted (step 1) with a
/// conservative comptime extent.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Divisor {
    Static(usize),
    Dynamic { min: usize },
}

impl Divisor {
    pub fn is_dynamic(self) -> bool {
        matches!(self, Divisor::Dynamic { .. })
    }

    /// The smallest value this divisor can take, which is the one spanning the widest window:
    /// itself when `Static`, the declared `min` when `Dynamic`.
    pub fn bound(self) -> usize {
        match self {
            Divisor::Static(d) => d,
            Divisor::Dynamic { min } => min,
        }
    }

    /// Whether this divides by `1`, i.e. the mapping is integer and every rational path below is
    /// the identity. A `Dynamic` divisor is never unit whatever its bound: the launch may still
    /// pass anything at or above it, so the division has to stay.
    pub fn is_unit(self) -> bool {
        self == Divisor::Static(1)
    }
}

impl From<usize> for Divisor {
    fn from(n: usize) -> Self {
        Divisor::Static(n)
    }
}

/// One logical axis's contribution to one physical axis: `digit * scale`, where the digit is the
/// whole coordinate unless the axis is spread over several physical axes, which
/// [`Projection::digit`](crate::Projection::digit) reads off the map's own shape rather than off a
/// stored constant.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AxisTerm {
    pub axis: Axis,
    pub scale: Scale,
}

/// Whether a physical axis's terms can land on the same cell.
///
/// The coefficients alone cannot say. `affine(&[(A, 2), (B, 1)])` partitions the axis when `B` has
/// extent `2` and is a stride-2 stencil when it has extent `3` — same map, different structure —
/// so the caller states which it means and [`validate_composition`](crate::Projection) checks the
/// claim against the extents.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Composition {
    /// No two logical positions share a cell: each coefficient is the product of the finer axes'
    /// extents, so the terms partition the axis and the map is a bijection. Every window stays a
    /// dense box, exactly as for the one-term identity, which is the degenerate case of this.
    ///
    /// The mirror of storage tiling, which spreads one logical axis over several physical ones.
    /// Here several logical axes share one physical axis, for the opposite reason: the operands
    /// tell them apart. A quantization block is this — one axis for the block index, one for the
    /// position inside it — because the scales vary over the first and not the second.
    Disjoint,
    /// Two positions may land on the same cell, so a cell does not determine the position: a
    /// stencil's taps, a resample's ratio.
    Overlapping,
}

/// One [`PhysicalAxis`](crate::PhysicalAxis) as an affine combination of logical axes' digits
/// plus a constant term, over a divisor: `physical = (Σ digit(axis) * scale + offset) / divisor`,
/// floored. The divisor is `1` for every mapping but a [rational](Divisor) one, which is the only
/// case any of the arithmetic below is not the plain affine sum.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PhysicalAxisMap {
    terms: SmallVec<[AxisTerm; MAX_AXES]>,
    offset: Offset,
    divisor: Divisor,
    composition: Composition,
}

impl PhysicalAxisMap {
    /// The identity map: this physical axis *is* `axis`, coefficient `1`, offset `0`. What every
    /// operand of a non-gather operation uses on every physical axis.
    pub fn of(axis: Axis) -> Self {
        PhysicalAxisMap {
            terms: SmallVec::from_slice(&[AxisTerm {
                axis,
                scale: Scale::Static(1),
            }]),
            offset: Offset::Static(0),
            divisor: Divisor::Static(1),
            composition: Composition::Disjoint,
        }
    }

    /// This physical axis partitioned by several logical ones, coarsest first:
    /// `disjoint(&[(KB, 32), (KI, 1)])` is `k = kb * 32 + ki`, the axis cut into 32-wide blocks
    /// with one axis counting blocks and one counting inside them.
    ///
    /// The same arithmetic as [`affine`](Self::affine), and deliberately a different constructor:
    /// this one claims no two positions share a cell, which keeps every window dense and every
    /// read on the direct path, and a claim the extents contradict is refused at
    /// [`Tile::of`](crate::Tile::of).
    pub fn disjoint(terms: &[(Axis, usize)]) -> Self {
        let mut map = Self::affine(terms);
        map.composition = Composition::Disjoint;
        map
    }

    /// An affine combination with zero offset, e.g. `affine(&[(Oh, stride), (Rh, dilation)])`.
    pub fn affine(terms: &[(Axis, usize)]) -> Self {
        Self::affine_with_offset(terms, 0)
    }

    /// An affine combination with a signed constant or dynamic offset, e.g.
    /// `affine_with_offset(&[(Oh, stride), (Rh, dilation)], -padding)` or
    /// `affine_with_offset(&[(Oh, stride), (Rh, dilation)], Offset::Dynamic)`.
    pub fn affine_with_offset(terms: &[(Axis, usize)], offset: impl Into<Offset>) -> Self {
        let terms: SmallVec<[(Axis, Scale); MAX_AXES]> = terms
            .iter()
            .map(|&(axis, scale)| (axis, Scale::Static(scale)))
            .collect();
        Self::scaled_with_offset(&terms, offset)
    }

    /// [`affine`](Self::affine) over explicit [`Scale`]s, which is how a coefficient only known at
    /// runtime is spelled: `scaled(&[(Oh, Scale::Dynamic { max: 2 }), (Rh, Scale::Static(1))])`.
    pub fn scaled(terms: &[(Axis, Scale)]) -> Self {
        Self::scaled_with_offset(terms, 0)
    }

    /// [`scaled`](Self::scaled) with a signed constant or dynamic offset.
    pub fn scaled_with_offset(terms: &[(Axis, Scale)], offset: impl Into<Offset>) -> Self {
        for &(axis, scale) in terms {
            // A coefficient that can only ever be 0 addresses nothing, which `Projection::validate`
            // reads as the axis being absent rather than as a degenerate term.
            assert!(
                scale.bound() > 0,
                "PhysicalAxisMap: {axis:?}'s coefficient is bounded at 0, so it addresses no cell; \
                 drop the term instead"
            );
        }
        let offset = offset.into();
        // One axis stepping by one is the identity however it was spelled, and the identity
        // partitions its axis. Anything else is ambiguous from the coefficients alone, so it takes
        // the [`Overlapping`](Composition::Overlapping) reading unless a caller claims
        // otherwise ([`disjoint`](Self::disjoint)).
        let composition = match (terms, offset) {
            ([(_, Scale::Static(1))], Offset::Static(0)) => Composition::Disjoint,
            _ => Composition::Overlapping,
        };
        PhysicalAxisMap {
            terms: terms
                .iter()
                .map(|&(axis, scale)| AxisTerm { axis, scale })
                .collect(),
            offset,
            divisor: Divisor::Static(1),
            composition,
        }
    }

    /// The same combination read as a fraction: `(Σ digit * scale + offset) / divisor`, floored.
    /// The rational spelling of a fractional scale, e.g. resizing `w_in` cells to `w_out` is
    /// `affine_with_offset(&[(O, w_in), (R, w_out)], offset).over(w_out)`, where `R`'s coefficient
    /// is the divisor precisely so the tap index survives the division whole.
    ///
    /// A divisor the coefficients all cancel is [reduced](Self::reduced) away here, so
    /// [`is_rational`](Self::is_rational) means the mapping genuinely divides rather than merely
    /// being spelled as a fraction.
    pub fn over(mut self, divisor: impl Into<Divisor>) -> Self {
        let divisor = divisor.into();
        assert!(
            divisor.bound() > 0,
            "PhysicalAxisMap::over: a divisor of 0 does not map anywhere"
        );
        // Setting, not composing: a second `over` would drop a divisor the first could not reduce
        // away, so spell the product instead.
        assert!(
            self.divisor.is_unit(),
            "PhysicalAxisMap::over: this map already divides by {:?}; state the whole divisor once",
            self.divisor
        );
        self.divisor = divisor;
        // A division coarsens: several logical positions land on one physical cell, which is the
        // one thing a partition never does.
        self.composition = Composition::Overlapping;
        self.reduced()
    }

    /// The same mapping with a divisor its own coefficients cancel spelled without one:
    /// `⌊(Σ digit * s + o) / d⌋` *is* `Σ digit * (s/d) + ⌊o/d⌋`, exactly, when `d` divides every
    /// coefficient. The sum is then a multiple of `d` whatever the digits are, so the floor only
    /// ever acts on the constant term, and the constant term is the window origin's business.
    ///
    /// Worth taking out because a divisor is not free: a rational axis reads through an in-kernel
    /// division, and its uncompacted window carries a conservative extent. Reduced here, a fractionally
    /// *spelled* but integrally *stepping* mapping keeps both compact step and exact extent.
    ///
    /// Only a fully comptime numerator reduces. A [`Dynamic`](Scale::Dynamic) coefficient cannot be
    /// shown divisible; a [`Dynamic`](Offset::Dynamic) offset could still cancel, but the carrier
    /// holds the offset itself and the reduced map would need its quotient, which is the caller's
    /// to pass ([`Tile::of_gathered`](crate::Tile::of_gathered)) and not this one's to rewrite.
    fn reduced(mut self) -> Self {
        let (Divisor::Static(d), Offset::Static(o)) = (self.divisor, self.offset) else {
            return self;
        };
        let common = self.terms.iter().try_fold(0, |g, t| match t.scale {
            Scale::Static(s) => Some(super::gcd(g, s)),
            Scale::Dynamic { .. } => None,
        });
        // `gcd(0, s)` seeds from the first coefficient, so an all-`0` combination (or none at all)
        // reports `0`, which every divisor divides: it has no digit to step by either way.
        if !matches!(common, Some(g) if g % d == 0) {
            return self;
        }
        for term in self.terms.iter_mut() {
            term.scale = Scale::Static(term.scale.get() / d);
        }
        self.offset = Offset::Static(o.div_euclid(d as isize));
        self.divisor = Divisor::Static(1);
        self
    }

    pub fn terms(&self) -> &[AxisTerm] {
        &self.terms
    }

    /// How this axis's terms sit on it ([`Composition`]).
    pub fn composition(&self) -> Composition {
        self.composition
    }

    /// The radices a [`Disjoint`](Composition::Disjoint) map claims, coarsest first: each term's
    /// coefficient must be the product of the finer axes' extents, and the finest must be `1`.
    /// Returns the axis whose extent each coefficient stands for, so the caller can check the
    /// claim against a space it holds.
    ///
    /// Empty for an [`Overlapping`](Composition::Overlapping) map: it claims nothing to check.
    pub fn claimed_radices(&self) -> SmallVec<[(Axis, usize); MAX_AXES]> {
        match self.composition {
            Composition::Overlapping => SmallVec::new(),
            Composition::Disjoint => self
                .terms
                .iter()
                .map(|t| (t.axis, t.scale.bound()))
                .collect(),
        }
    }

    /// The signed constant or dynamic offset of this physical axis.
    pub fn offset(&self) -> Offset {
        self.offset
    }

    /// What this axis's affine combination is divided by, [`Static(1)`](Divisor::Static) unless
    /// [`over`](Self::over) made it rational.
    pub fn divisor(&self) -> Divisor {
        self.divisor
    }

    /// The static physical step this term contributes outside this map's rational floor, when it
    /// can be factored exactly: `⌊(base + digit * scale) / divisor⌋` is
    /// `⌊base / divisor⌋ + digit * (scale / divisor)` for a static `scale` a static `divisor`
    /// divides. The per-term counterpart of [`reduced`](Self::reduced), which needs *every*
    /// coefficient divisible and takes the divisor away with them; this leaves the divisor in
    /// place for the terms that still need it.
    ///
    /// `Some` implies [`is_rational`](Self::is_rational): a unit divisor divides every coefficient
    /// and would empty the numerator, so readers that only add these back under a division are
    /// entitled to skip them elsewhere.
    pub(crate) fn static_offset_step(&self, term: usize) -> Option<usize> {
        match (self.divisor, self.terms[term].scale) {
            (Divisor::Static(d), Scale::Static(s)) if d > 1 && s.is_multiple_of(d) => Some(s / d),
            _ => None,
        }
    }

    /// Whether this axis divides by anything but `1`, so the plain affine sum is not the physical
    /// coordinate and every path below has to carry the division.
    pub fn is_rational(&self) -> bool {
        !self.divisor.is_unit()
    }

    /// The physical cell this axis's logical origin lands on: `⌊offset / divisor⌋`, the part of the
    /// offset a [`Window`](crate::Window) origin can absorb. `None` when either side is only known
    /// at runtime, which is the launch's to compute rather than the kernel's.
    pub fn origin(&self) -> Option<isize> {
        match (self.offset, self.divisor) {
            (Offset::Static(o), Divisor::Static(d)) => Some(o.div_euclid(d as isize)),
            _ => None,
        }
    }

    /// The phase the division starts at: `offset - divisor * origin`, in `0..divisor`. What the
    /// window origin cannot absorb, since `⌊(x + offset) / divisor⌋` is `⌊offset / divisor⌋ +
    /// ⌊(x + residue) / divisor⌋` and no further. `None` when either side is dynamic.
    pub fn residue(&self) -> Option<usize> {
        match (self.offset, self.divisor) {
            (Offset::Static(o), Divisor::Static(d)) => Some(o.rem_euclid(d as isize) as usize),
            _ => None,
        }
    }

    /// How many of this axis's coefficients are [`Dynamic`](Scale::Dynamic).
    pub fn dynamic_scale_count(&self) -> usize {
        self.terms.iter().filter(|t| t.scale.is_dynamic()).count()
    }

    /// Whether any of this axis's coefficients are [`Dynamic`](Scale::Dynamic).
    pub fn has_dynamic_scale(&self) -> bool {
        self.terms.iter().any(|t| t.scale.is_dynamic())
    }

    /// Whether `axis` addresses this physical axis at all, whatever its coefficient is and whether
    /// or not it is comptime. [`scale`](Self::scale)'s question minus the value.
    pub fn addresses(&self, axis: Axis) -> bool {
        self.terms.iter().any(|t| t.axis == axis)
    }

    /// `axis`'s coefficient, `0` when it does not address this physical axis. Panics when the
    /// coefficient is [`Dynamic`](Scale::Dynamic); [`addresses`](Self::addresses) is the question
    /// that survives one.
    pub fn scale(&self, axis: Axis) -> usize {
        self.terms
            .iter()
            .find(|t| t.axis == axis)
            .map_or(0, |t| t.scale.get())
    }

    /// The one [`Axis`] this map is the identity of, `None` when it combines several, scales,
    /// shifts or divides. An identity map reaches exactly as far as its own coordinate, which is
    /// what lets a caller prove it stays inside the buffer; anything else reaches further.
    pub fn identity_axis(&self) -> Option<Axis> {
        match self.terms.as_slice() {
            [
                AxisTerm {
                    axis,
                    scale: Scale::Static(1),
                },
            ] if self.offset == Offset::Static(0) && self.divisor.is_unit() => Some(*axis),
            _ => None,
        }
    }

    /// Whether this physical axis is exactly `axis` at coefficient `1` with zero offset and no
    /// division. Says nothing about digit extraction, which is a property of the whole
    /// [`Projection`](crate::Projection) (how many physical axes carry `axis`), not of one map.
    pub fn is_identity(&self, axis: Axis) -> bool {
        self.identity_axis() == Some(axis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Axis = Axis(0);
    const B: Axis = Axis(1);

    /// The identity map is the one every non-gather operand uses; a scaled or multi-term map is
    /// not it, and an axis the map does not address contributes nothing.
    #[test]
    fn identity_is_exactly_one_axis_at_coefficient_one() {
        let id = PhysicalAxisMap::of(A);
        assert!(id.is_identity(A));
        assert!(!id.is_identity(B));
        assert_eq!(id.scale(A), 1);
        assert_eq!(id.scale(B), 0);
        assert_eq!(id.offset(), Offset::Static(0));

        let affine = PhysicalAxisMap::affine(&[(A, 2), (B, 3)]);
        assert!(!affine.is_identity(A));
        assert_eq!(affine.scale(A), 2);
        assert_eq!(affine.scale(B), 3);
        assert_eq!(affine.offset(), Offset::Static(0));
        // A single term is still not the identity unless its coefficient is 1 and offset is 0.
        assert!(!PhysicalAxisMap::affine(&[(A, 2)]).is_identity(A));
        assert!(PhysicalAxisMap::affine(&[(A, 1)]).is_identity(A));

        let with_offset = PhysicalAxisMap::affine_with_offset(&[(A, 1)], -2);
        assert!(!with_offset.is_identity(A));
        assert_eq!(with_offset.scale(A), 1);
        assert_eq!(with_offset.offset(), Offset::Static(-2));

        let with_dynamic_offset = PhysicalAxisMap::affine_with_offset(&[(A, 1)], Offset::Dynamic);
        assert!(!with_dynamic_offset.is_identity(A));
        assert_eq!(with_dynamic_offset.offset(), Offset::Dynamic);
        assert!(with_dynamic_offset.offset().is_dynamic());
        assert!(!with_dynamic_offset.has_dynamic_scale());
        assert_eq!(with_dynamic_offset.dynamic_scale_count(), 0);
    }

    #[test]
    fn rational_axis_map_properties() {
        let map = PhysicalAxisMap::affine_with_offset(&[(A, 100)], -50).over(133);
        assert!(map.is_rational());
        assert_eq!(map.divisor(), Divisor::Static(133));
        assert_eq!(map.offset(), Offset::Static(-50));
        assert_eq!(map.origin(), Some((-50isize).div_euclid(133)));
        assert_eq!(map.residue(), Some((-50isize).rem_euclid(133) as usize));

        let dynamic_div = PhysicalAxisMap::affine(&[(A, 100)]).over(Divisor::Dynamic { min: 3 });
        assert!(dynamic_div.is_rational());
        assert!(dynamic_div.divisor().is_dynamic());
        assert_eq!(dynamic_div.origin(), None);
        assert_eq!(dynamic_div.residue(), None);
    }

    /// A rational map need not reduce as a whole for some of its terms to be exact steps. This is
    /// the resampling shape: the spatial term stays under the floor, while a tap coefficient of
    /// the divisor advances one physical cell at a time.
    #[test]
    fn a_rational_map_factors_exact_static_terms_into_offsets() {
        let map = PhysicalAxisMap::affine_with_offset(&[(A, 5), (B, 6)], -2).over(6);

        assert!(map.is_rational());
        assert_eq!(map.static_offset_step(0), None);
        assert_eq!(map.static_offset_step(1), Some(1));
        for output in 0..8isize {
            for tap in 0..3isize {
                assert_eq!(
                    (output * 5 + tap * 6 - 2).div_euclid(6),
                    (output * 5 - 2).div_euclid(6) + tap
                );
            }
        }

        let strided_tap = PhysicalAxisMap::affine(&[(A, 5), (B, 12)]).over(6);
        assert_eq!(strided_tap.static_offset_step(1), Some(2));

        let fractional_tap = PhysicalAxisMap::affine(&[(A, 5), (B, 2)]).over(3);
        assert_eq!(fractional_tap.static_offset_step(1), None);

        let dynamic_divisor =
            PhysicalAxisMap::affine(&[(A, 5), (B, 6)]).over(Divisor::Dynamic { min: 3 });
        assert_eq!(dynamic_divisor.static_offset_step(1), None);
    }

    #[test]
    #[should_panic(expected = "divisor of 0 does not map anywhere")]
    fn divisor_zero_panics() {
        PhysicalAxisMap::affine(&[(A, 1)]).over(0);
    }

    /// `over` sets the divisor rather than composing with it, so a second one over a divisor the
    /// first could not reduce away would silently drop it.
    #[test]
    #[should_panic(expected = "state the whole divisor once")]
    fn a_second_over_is_refused() {
        PhysicalAxisMap::affine(&[(A, 4), (B, 6)]).over(4).over(3);
    }

    /// `⌊(8a + 4r)/4⌋` is `2a + r` exactly, so it is stored as that: an integer map, which stages.
    #[test]
    fn a_divisor_the_coefficients_cancel_reduces_away() {
        let map = PhysicalAxisMap::affine(&[(A, 8), (B, 4)]).over(4);
        assert!(!map.is_rational());
        assert_eq!(map.divisor(), Divisor::Static(1));
        assert_eq!(map.scale(A), 2);
        assert_eq!(map.scale(B), 1);
        assert_eq!(map.offset(), Offset::Static(0));

        // Down to the identity, which is what makes such a spelling `is_direct` again.
        assert!(PhysicalAxisMap::affine(&[(A, 4)]).over(4).is_identity(A));
    }

    /// The sum is a multiple of the divisor whatever the digits are, so the floor only ever acts on
    /// the constant term: `⌊(8a - 3)/4⌋` is `2a + ⌊-3/4⌋`, i.e. `2a - 1`.
    #[test]
    fn reducing_floors_the_offset() {
        let map = PhysicalAxisMap::affine_with_offset(&[(A, 8)], -3).over(4);
        assert!(!map.is_rational());
        assert_eq!(map.scale(A), 2);
        assert_eq!(map.offset(), Offset::Static(-1));
        assert_eq!(map.origin(), Some(-1));
        assert_eq!(map.residue(), Some(0));
    }

    /// `gcd(4, 6) = 2` does not divide `6`, so the mapping genuinely steps fractionally and keeps
    /// its divisor. A `Dynamic` coefficient cannot be shown divisible, and a `Dynamic` offset has
    /// no quotient to fold into, so neither reduces.
    #[test]
    fn a_divisor_the_coefficients_do_not_cancel_stays() {
        assert!(
            PhysicalAxisMap::affine(&[(A, 4), (B, 6)])
                .over(6)
                .is_rational()
        );
        assert!(
            PhysicalAxisMap::scaled(&[(A, Scale::Dynamic { max: 2 }), (B, Scale::Static(4))])
                .over(4)
                .is_rational()
        );
        assert!(
            PhysicalAxisMap::scaled_with_offset(&[(A, Scale::Static(4))], Offset::Dynamic)
                .over(4)
                .is_rational()
        );
    }

    /// A partition and a stencil can carry the same coefficients, so which one a map is comes
    /// from the constructor, not from the numbers.
    #[test]
    fn the_same_coefficients_partition_or_overlap() {
        assert_eq!(
            PhysicalAxisMap::disjoint(&[(A, 2), (B, 1)]).composition(),
            Composition::Disjoint
        );
        assert_eq!(
            PhysicalAxisMap::affine(&[(A, 2), (B, 1)]).composition(),
            Composition::Overlapping
        );
    }

    /// The identity partitions its axis however it was spelled: `of`, `affine`, `scaled`. A
    /// coefficient past one skips cells, so it does not.
    #[test]
    fn the_identity_partitions_whichever_door_minted_it() {
        assert_eq!(PhysicalAxisMap::of(A).composition(), Composition::Disjoint);
        assert_eq!(
            PhysicalAxisMap::affine(&[(A, 1)]).composition(),
            Composition::Disjoint
        );
        assert_eq!(
            PhysicalAxisMap::scaled(&[(A, Scale::Static(1))]).composition(),
            Composition::Disjoint
        );
        assert_eq!(
            PhysicalAxisMap::affine(&[(A, 2)]).composition(),
            Composition::Overlapping
        );
    }

    /// A division coarsens, so a rational map never partitions, whatever it was before.
    #[test]
    fn a_coarsening_is_never_a_partition() {
        assert_eq!(
            PhysicalAxisMap::of(A).over(4).composition(),
            Composition::Overlapping
        );
    }

    /// The radices a partition claims, coarsest first: the finest digit steps by one and each
    /// coarser one steps over everything below it.
    #[test]
    fn a_partition_claims_its_radices() {
        let map = PhysicalAxisMap::disjoint(&[(A, 32), (B, 1)]);
        assert_eq!(&map.claimed_radices()[..], &[(A, 32), (B, 1)]);
        assert!(
            PhysicalAxisMap::affine(&[(A, 32), (B, 1)])
                .claimed_radices()
                .is_empty()
        );
    }
}
