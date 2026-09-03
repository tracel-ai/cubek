//! Writing a [`Projection`] one buffer dim at a time.
//!
//! [`Projection::new`] takes two parallel lists — the logical axes and one map per physical
//! dim — and nothing at the call site says how they relate. This states the same value the
//! way a reader thinks of it: **one line per buffer dim, in buffer order, each saying what
//! that dim's index is computed from**. The logical axis list falls out of the dims.
//!
//! ```text
//! Projection::dims()
//!     .dim(B)                                        // dim 0 = b
//!     .dim(window(&[(OH, stride), (RH, dilation)]).pad(pad))   // dim 1 = oh·stride + rh·dilation − pad
//!     .dim(window(&[(OW, stride), (RW, dilation)]).pad(pad))   // dim 2
//!     .dim(C)                                        // dim 3 = c
//!     .build()
//! ```
//!
//! Three kinds of dim, and the argument says which:
//!
//! * an [`Axis`] — the dim *is* that coordinate;
//! * [`split`] — several axes cut one dim into blocks, `kb·block + ki`, and cannot land on the
//!   same cell. Stated in **extents**, coarsest first, so no coefficient is computed by hand;
//! * [`window`] — several axes slide over one dim, `oh·stride + rh·dilation`, and may land on
//!   the same cell. Stated in coefficients, because a stride and a dilation *are* the
//!   coefficients, with [`pad`](WindowDim::pad) for the constant term.
//!
//! A storage-tiled axis is the plain axis repeated, coarsest fragment first; the radix each
//! fragment steps by is read off the buffer at use time, not stated here.

use cubecl::zspace::SmallVec;

use crate::{Axis, Composition, MAX_AXES, PhysicalAxisMap, Projection};

impl Projection {
    /// Start writing a projection one buffer dim at a time. See the [module doc](self).
    pub fn dims() -> DimsBuilder {
        DimsBuilder {
            physical: SmallVec::new(),
            spanning: SmallVec::new(),
        }
    }
}

/// A [`Projection`] under construction: the dims stated so far, and any axis the operand is
/// defined over but addresses with no dim.
#[derive(Clone, Debug)]
pub struct DimsBuilder {
    physical: SmallVec<[PhysicalAxisMap; MAX_AXES]>,
    spanning: SmallVec<[Axis; MAX_AXES]>,
}

impl DimsBuilder {
    /// The next buffer dim, coarsest first: an [`Axis`], a [`split`], or a [`window`].
    pub fn dim(mut self, dim: impl Into<PhysicalAxisMap>) -> Self {
        self.physical.push(dim.into());
        self
    }

    /// An axis this operand is defined over and constant along — addressed by no dim. One scale
    /// covering a block of the contraction is this, and so is a grouped-query cache, which spans
    /// the query group without distinguishing its members.
    pub fn spanning(mut self, axis: Axis) -> Self {
        self.spanning.push(axis);
        self
    }

    /// The projection, with its logical axes derived from the dims.
    ///
    /// The order is the one [`Projection::validate`] requires and nothing else: the innermost
    /// dim's axes come **last**, because that dim is addressed in vector lines and the line runs
    /// along the operand's last logical axis. Every other axis follows first mention, and a
    /// [`spanning`](Self::spanning) axis sits after the addressed ones but before the innermost
    /// dim's, so it never takes the line position.
    ///
    /// # Panics
    ///
    /// On no dims at all, or on an axis that is both [`spanning`](Self::spanning) and addressed
    /// by a dim — the two are contradictory claims about the same coordinate.
    pub fn build(self) -> Projection {
        assert!(
            !self.physical.is_empty(),
            "Projection::dims: an operand has at least one dim"
        );
        let (innermost, outer) = self.physical.split_last().expect("checked non-empty");
        let mut axes: SmallVec<[Axis; MAX_AXES]> = SmallVec::new();
        let mention = |axis: Axis, axes: &mut SmallVec<[Axis; MAX_AXES]>| {
            if !axes.contains(&axis) {
                axes.push(axis);
            }
        };
        for map in outer {
            for term in map.terms() {
                mention(term.axis, &mut axes);
            }
        }
        for &axis in &self.spanning {
            assert!(
                !self.physical.iter().any(|map| map.addresses(axis)),
                "Projection::dims: {axis:?} is stated as spanning (addressed by no dim) but a dim \
                 addresses it"
            );
            mention(axis, &mut axes);
        }
        // Last, so the line runs along it. A term already mentioned by an outer dim is a
        // storage-tiled axis whose finest fragment this is; it keeps its earlier position, and
        // `validate` accepts that because such a projection is invertible.
        for term in innermost.terms() {
            mention(term.axis, &mut axes);
        }
        Projection::new(&axes, &self.physical)
    }
}

/// A dim that *is* one coordinate: `.dim(K)`.
impl From<Axis> for PhysicalAxisMap {
    fn from(axis: Axis) -> Self {
        PhysicalAxisMap::of(axis)
    }
}

/// One dim cut into blocks by several axes, stated in **extents**, coarsest first:
/// `split(&[(KB, blocks), (KI, block)])` is `kb·block + ki`.
///
/// The coefficients are derived — each axis steps by the product of the extents finer than it
/// — so the caller states the sizes it already knows and never a stride-within-a-dim. And
/// because they are derived that way, no two positions can share a cell: this is
/// [`Composition::Disjoint`] by construction, which is what keeps every window dense and every
/// read on the direct path.
///
/// # Panics
///
/// On no terms, or on an extent of zero.
pub fn split(extents: &[(Axis, usize)]) -> PhysicalAxisMap {
    assert!(
        !extents.is_empty(),
        "split: a dim is cut by at least one axis"
    );
    let mut terms: SmallVec<[(Axis, usize); MAX_AXES]> = SmallVec::new();
    let mut coefficient = 1;
    for &(axis, extent) in extents.iter().rev() {
        assert!(extent > 0, "split: {axis:?} has extent 0");
        terms.push((axis, coefficient));
        coefficient *= extent;
    }
    terms.reverse();
    PhysicalAxisMap::disjoint(&terms)
}

/// One dim several axes slide over, stated in **coefficients**:
/// `window(&[(OH, stride), (RH, dilation)])` is `oh·stride + rh·dilation`, and
/// [`.pad(p)`](WindowDim::pad) subtracts the padding.
///
/// Consecutive windows may overlap — that is what a receptive field is — so this is
/// [`Composition::Overlapping`], and the aliasing checks leave it alone.
pub fn window(coefficients: &[(Axis, usize)]) -> WindowDim {
    WindowDim {
        coefficients: SmallVec::from_slice(coefficients),
        pad: 0,
    }
}

/// A [`window`] before its padding is stated. Converts into the dim's map directly, at zero
/// padding, or after [`pad`](Self::pad).
#[derive(Clone, Debug)]
pub struct WindowDim {
    coefficients: SmallVec<[(Axis, usize); MAX_AXES]>,
    pad: usize,
}

impl WindowDim {
    /// How many cells before the buffer's first this window's origin sits: the padding, which
    /// a boundary guard reads as zero.
    pub fn pad(mut self, pad: usize) -> Self {
        self.pad = pad;
        self
    }
}

impl From<WindowDim> for PhysicalAxisMap {
    fn from(window: WindowDim) -> Self {
        let map = PhysicalAxisMap::affine_with_offset(&window.coefficients, -(window.pad as isize));
        debug_assert!(
            window.coefficients.len() == 1 || map.composition() == Composition::Overlapping,
            "a window over several axes takes the overlapping reading"
        );
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Offset;

    const B: Axis = Axis(0);
    const H: Axis = Axis(1);
    const G: Axis = Axis(2);
    const T: Axis = Axis(3);
    const D: Axis = Axis(4);
    const K: Axis = Axis(5);
    const N: Axis = Axis(6);
    const KB: Axis = Axis(7);
    const KI: Axis = Axis(8);
    const OH: Axis = Axis(9);
    const RH: Axis = Axis(10);
    const C: Axis = Axis(11);
    const M: Axis = Axis(12);

    /// One dim's index for a coordinate: `Σ axis·coefficient + offset` — the projection,
    /// evaluated, so each case below checks arithmetic rather than describing it.
    fn index(p: &Projection, dim: usize, coordinate: &[(Axis, usize)]) -> isize {
        let sum: usize = coordinate
            .iter()
            .map(|&(axis, at)| at * p.scale(dim, axis))
            .sum();
        let offset = match p.offset(dim) {
            Offset::Static(o) => o,
            Offset::Dynamic => unreachable!("nothing here builds a dynamic offset"),
        };
        sum as isize + offset
    }

    #[test]
    fn one_axis_per_dim_is_direct() {
        let p = Projection::dims().dim(K).dim(N).build();
        assert_eq!(p, Projection::direct(&[K, N]));
    }

    /// `kb·32 + ki`, stated as extents: 4 blocks of 32. The coefficients are derived, and
    /// `k = 100` lands at block 3, position 4.
    #[test]
    fn split_derives_the_coefficients_from_extents() {
        let p = Projection::dims()
            .dim(N)
            .dim(split(&[(KB, 4), (KI, 32)]))
            .build();

        assert_eq!(p.scale(1, KB), 32);
        assert_eq!(p.scale(1, KI), 1);
        assert_eq!(index(&p, 1, &[(KB, 3), (KI, 4)]), 100);
        assert_eq!(p.composition(), Composition::Disjoint);
        // The innermost dim's axes are last, its finest term last of all.
        assert_eq!(p.logical_axes(), &[N, KB, KI]);
    }

    /// A three-way split multiplies through: `(a, b, c)` over extents `(2, 3, 5)` steps by
    /// `15, 5, 1`.
    #[test]
    fn split_composes_more_than_two_axes() {
        let p = Projection::dims()
            .dim(split(&[(B, 2), (H, 3), (D, 5)]))
            .build();
        assert_eq!(p.scale(0, B), 15);
        assert_eq!(p.scale(0, H), 5);
        assert_eq!(p.scale(0, D), 1);
        assert_eq!(index(&p, 0, &[(B, 1), (H, 2), (D, 4)]), 29);
    }

    /// Split-merge partials, group-major: `g·splits + t`, so a group member's slices are
    /// adjacent. The other choice is a one-line change and a different kernel.
    #[test]
    fn a_partition_in_the_middle_keeps_the_innermost_last() {
        let (group, splits) = (4, 8);
        let p = Projection::dims()
            .dim(B)
            .dim(H)
            .dim(split(&[(G, group), (T, splits)]))
            .dim(D)
            .build();

        assert_eq!(index(&p, 2, &[(G, 2), (T, 5)]), 21);
        assert_eq!(index(&p, 2, &[(G, 2), (T, 6)]), 22);
        assert_eq!(p.logical_axes(), &[B, H, G, T, D]);
    }

    /// `oh·2 + rh·1 − 1`: adjacent output steps overlap, and `oh = 0` reaches into the pad.
    #[test]
    fn window_is_an_overlapping_affine_map_with_padding() {
        let p = Projection::dims()
            .dim(B)
            .dim(window(&[(OH, 2), (RH, 1)]).pad(1))
            .dim(C)
            .build();

        assert_eq!(index(&p, 1, &[(OH, 3), (RH, 2)]), 7);
        assert_eq!(index(&p, 1, &[(OH, 4), (RH, 0)]), 7);
        assert_eq!(index(&p, 1, &[(OH, 0), (RH, 0)]), -1);
        assert_eq!(p.composition(), Composition::Overlapping);
        assert_eq!(p.logical_axes(), &[B, OH, RH, C]);
    }

    /// A grouped-query cache: `g` is in the coordinate and in no dim, and it must not take the
    /// line position — the innermost dim's axis stays last.
    #[test]
    fn a_spanning_axis_sits_before_the_innermost() {
        let p = Projection::dims()
            .dim(B)
            .dim(H)
            .dim(T)
            .dim(D)
            .spanning(G)
            .build();

        assert!(!p.addresses(G));
        assert_eq!(p.logical_axes(), &[B, H, T, G, D]);
        // The rule `validate` enforces, checked here so a builder cannot produce a projection
        // the bind refuses.
        p.validate(4);
    }

    /// Storage tiling is the plain axis repeated, coarsest fragment first. The logical list
    /// keeps each axis once, at its first mention.
    #[test]
    fn a_repeated_axis_is_storage_tiled() {
        let p = Projection::dims()
            .dim(B)
            .dim(M)
            .dim(K)
            .dim(M)
            .dim(K)
            .dim(C)
            .build();

        assert_eq!(p.logical_axes(), &[B, M, K, C]);
        assert_eq!(p.tiling().fragments(1), 2);
        assert_eq!(p.tiling().fragments(3), 1);
    }

    #[test]
    #[should_panic(expected = "stated as spanning")]
    fn spanning_an_addressed_axis_is_refused() {
        let _ = Projection::dims().dim(B).dim(D).spanning(D).build();
    }
}
