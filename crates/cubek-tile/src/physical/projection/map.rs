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
/// A `Dynamic` coefficient costs the comptime window geometry: the receptive field it spans is a
/// runtime value, so the operand cannot be staged.
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

/// The constant term of one physical axis's affine combination. Mirrors [`Scale`]: `Static` is a
/// comptime constant so [`may_underflow`](crate::Projection::may_underflow) can track whether it
/// is negative, `Dynamic` is a runtime padding or placement whose value rides the tile's signed
/// offset carrier instead of the kernel.
///
/// Unlike [`Scale::Dynamic`], an `Offset::Dynamic` costs no comptime window geometry: `span` is
/// offset-invariant, and [`Compaction`](crate::Compaction) drops the offset entirely, so an
/// operand carrying one can still be staged. The only cost is that
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
/// spelled as a fraction but steps like the integer map `o + 2r`, and is stored as one. So
/// [`is_rational`](PhysicalAxisMap::is_rational) marks the mappings that genuinely divide, whose
/// window is not a lattice and therefore cannot be staged.
///
/// Like [`Scale::Dynamic`], a `Dynamic` divisor costs the comptime window geometry: the receptive
/// field it spans is a runtime value, so the operand cannot be staged. It also costs an in-kernel
/// integer division per read, where a `Static` one folds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Divisor {
    Static(usize),
    Dynamic,
}

impl Divisor {
    pub fn is_dynamic(self) -> bool {
        matches!(self, Divisor::Dynamic)
    }

    /// Whether this divides by `1`, i.e. the mapping is integer and every rational path below is
    /// the identity.
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

/// One [`PhysicalAxis`](crate::PhysicalAxis) as an affine combination of logical axes' digits
/// plus a constant term, over a divisor: `physical = (Σ digit(axis) * scale + offset) / divisor`,
/// floored. The divisor is `1` for every mapping but a [rational](Divisor) one, which is the only
/// case any of the arithmetic below is not the plain affine sum.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PhysicalAxisMap {
    terms: SmallVec<[AxisTerm; MAX_AXES]>,
    offset: Offset,
    divisor: Divisor,
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
        }
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
    /// runtime is spelled: `scaled(&[(Oh, Scale::Dynamic), (Rh, Scale::Static(1))])`.
    pub fn scaled(terms: &[(Axis, Scale)]) -> Self {
        Self::scaled_with_offset(terms, 0)
    }

    /// [`scaled`](Self::scaled) with a signed constant or dynamic offset.
    pub fn scaled_with_offset(terms: &[(Axis, Scale)], offset: impl Into<Offset>) -> Self {
        PhysicalAxisMap {
            terms: terms
                .iter()
                .map(|&(axis, scale)| AxisTerm { axis, scale })
                .collect(),
            offset: offset.into(),
            divisor: Divisor::Static(1),
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
            divisor != Divisor::Static(0),
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
        self.reduced()
    }

    /// The same mapping with a divisor its own coefficients cancel spelled without one:
    /// `⌊(Σ digit * s + o) / d⌋` *is* `Σ digit * (s/d) + ⌊o/d⌋`, exactly, when `d` divides every
    /// coefficient. The sum is then a multiple of `d` whatever the digits are, so the floor only
    /// ever acts on the constant term, and the constant term is the window origin's business.
    ///
    /// Worth taking out because a divisor is not free: a rational axis reads through an in-kernel
    /// division, and its window is not a lattice, so it cannot be staged
    /// ([`Compaction`](crate::Compaction) has no step to compact by). Reduced here, a fractionally
    /// *spelled* but integrally *stepping* mapping keeps both.
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
            Scale::Dynamic => None,
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

    /// The signed constant or dynamic offset of this physical axis.
    pub fn offset(&self) -> Offset {
        self.offset
    }

    /// What this axis's affine combination is divided by, [`Static(1)`](Divisor::Static) unless
    /// [`over`](Self::over) made it rational.
    pub fn divisor(&self) -> Divisor {
        self.divisor
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

    /// Whether this physical axis is exactly `axis` at coefficient `1` with zero offset and no
    /// division. Says nothing about digit extraction, which is a property of the whole
    /// [`Projection`](crate::Projection) (how many physical axes carry `axis`), not of one map.
    pub fn is_identity(&self, axis: Axis) -> bool {
        self.offset == Offset::Static(0)
            && self.divisor.is_unit()
            && matches!(
                self.terms.as_slice(),
                [AxisTerm {
                    axis: a,
                    scale: Scale::Static(1)
                }] if *a == axis
            )
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

        let dynamic_div = PhysicalAxisMap::affine(&[(A, 100)]).over(Divisor::Dynamic);
        assert!(dynamic_div.is_rational());
        assert!(dynamic_div.divisor().is_dynamic());
        assert_eq!(dynamic_div.origin(), None);
        assert_eq!(dynamic_div.residue(), None);
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
            PhysicalAxisMap::scaled(&[(A, Scale::Dynamic), (B, Scale::Static(4))])
                .over(4)
                .is_rational()
        );
        assert!(
            PhysicalAxisMap::scaled_with_offset(&[(A, Scale::Static(4))], Offset::Dynamic)
                .over(4)
                .is_rational()
        );
    }
}
