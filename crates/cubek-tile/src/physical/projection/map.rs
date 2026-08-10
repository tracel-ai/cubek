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
/// plus a constant term: `physical = Σ digit(axis) * scale + offset`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PhysicalAxisMap {
    terms: SmallVec<[AxisTerm; MAX_AXES]>,
    offset: Offset,
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
        }
    }

    pub fn terms(&self) -> &[AxisTerm] {
        &self.terms
    }

    /// The signed constant or dynamic offset of this physical axis.
    pub fn offset(&self) -> Offset {
        self.offset
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

    /// Whether this physical axis is exactly `axis` at coefficient `1` with zero offset.
    /// Says nothing about digit extraction, which is a property of the whole
    /// [`Projection`](crate::Projection) (how many physical axes carry `axis`), not of one map.
    pub fn is_identity(&self, axis: Axis) -> bool {
        self.offset == Offset::Static(0)
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
}
