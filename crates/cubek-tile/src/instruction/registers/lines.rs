//! What the register block reads: an operand's line at a position, and the scale line covering a
//! run of them.
//!
//! **A scale is never found from a value's position.** The walk holds the coarser coordinate and
//! builds the finer one out of it: a run names one scale line, a field of that line names the tile
//! it covers, and the value lines of that tile follow by multiplication. Nothing here divides a
//! coordinate, and no operand can be asked which scale covers a line — it is handed the scales it
//! is to work under.
//!
//! **The scales are read through this same trait**, so a scaled operand's scales are themselves an
//! operand: one scale for a tile of values, and every level coarser than that already folded into
//! the line it hands back ([`CombinedScales`]).

use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::*;

/// How an operand's value lines are grouped under its scale lines.
///
/// One read of the scales brings `fields` of them, and each covers `lines` value lines. A run of
/// the walk is one such read: `fields · lines` value lines, whose indices the walk multiplies out
/// of the run it is in. An operand carrying no scales is [`PLAIN`](Span::PLAIN) and groups nothing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Span {
    /// Scales one read of them serves.
    pub fields: usize,
    /// Value lines one scale covers.
    pub lines: usize,
}

impl Span {
    /// An operand with no scales: every line stands alone.
    pub const PLAIN: Span = Span {
        fields: 1,
        lines: 1,
    };

    /// Value lines one run covers.
    pub fn run(&self) -> usize {
        self.fields * self.lines
    }

    /// The run two factors walk together.
    ///
    /// One walk steps both, so either they group their lines the same way or one of them groups
    /// nothing. Two different groupings would need a position resolved per side, which is the
    /// division this module exists not to do.
    pub fn join(self, other: Span) -> Span {
        match (self == Span::PLAIN, other == Span::PLAIN) {
            (true, _) => other,
            (_, true) => self,
            _ => {
                assert!(
                    self == other,
                    "block::contract: one factor groups its lines {self:?} and the other \
                     {other:?}. A walk steps both, so a run is one grouping or none"
                );
                self
            }
        }
    }
}

/// The scale line covering one run, held by the walk that reads it.
///
/// Absent is a factor with no scales, and it emits nothing: its values go through as they lie.
#[derive(CubeType)]
pub struct Fold<S: Numeric, W: Size> {
    /// The scales of this run, where the factor carries any.
    line: ComptimeOption<Vector<S, W>>,
}

#[cube]
impl<S: Numeric, W: Size> Fold<S, W> {
    /// The scales one run is worked under.
    pub fn of(line: Vector<S, W>) -> Self {
        Fold::<S, W> {
            line: ComptimeOption::new_Some(line),
        }
    }

    /// A factor with no scales.
    pub fn none() -> Self {
        Fold::<S, W> {
            line: ComptimeOption::new_None(),
        }
    }

    /// `value` under field `w`, which is the scale covering the tile the walk is in. The field is
    /// a constant because the walk built it, not because anything checked.
    pub fn apply<E: Numeric, V: Size>(
        &self,
        value: Vector<E, V>,
        #[comptime] w: usize,
    ) -> Vector<E, V> {
        #[comptime]
        match &self.line {
            ComptimeOption::Some(line) => value * Vector::<E, V>::cast_from(line.extract(w)),
            ComptimeOption::None => value,
        }
    }
}

/// One operand's lines as the contraction reads them.
///
/// The element and width are the implementor's own, so a source can be wrapped by another source
/// that reads it — which is what makes a scale chain a chain rather than an arity.
#[cube]
pub trait Lines: CubeType {
    /// The element one line holds.
    type E: Numeric;
    /// Values per line.
    type V: Size;
    /// The element one scale holds.
    type S: Numeric;
    /// Scales per scale line.
    type W: Size;

    /// This operand's values at `pos`.
    fn line(&self, pos: Coords2d) -> Vector<Self::E, Self::V>;

    /// The scales at `pos`, or nothing where this operand carries none. `pos` is the walk's own:
    /// the row or column it is on, and the run it is in.
    fn fold(&self, pos: Coords2d) -> Fold<Self::S, Self::W>;

    /// How this operand's lines are grouped under its scales.
    fn span(&self) -> comptime_type!(Span);
}

#[cube]
impl<'a, E: Numeric, V: Size> Lines for MaskedView<'a, Vector<E, V>, Coords2d> {
    type E = E;
    type V = V;
    type S = E;
    type W = Const<1>;

    fn line(&self, pos: Coords2d) -> Vector<E, V> {
        self.read(pos)
    }

    fn fold(&self, _pos: Coords2d) -> Fold<E, Const<1>> {
        Fold::<E, Const<1>>::none()
    }

    fn span(&self) -> comptime_type!(Span) {
        Span::PLAIN
    }
}

/// An operand and the scales it is contracted under.
///
/// The values and the scales are each a [`Lines`], read at positions the walk builds: the values at
/// theirs, the scales at the run's. Which field of a scale line a value line takes is the walk's
/// too — it stepped the fields — so nothing here resolves one.
#[derive(CubeType)]
pub struct ScaledLines<V: Lines, S: Lines> {
    values: V,
    /// The scales, where this factor carries any. Absent is a plain operand, and every line of it
    /// reads exactly as the values' own.
    scales: ComptimeOption<S>,
    /// How the values' lines group under those scales.
    #[cube(comptime)]
    span: Span,
}

#[cube]
impl<V: Lines, S: Lines> ScaledLines<V, S> {
    /// This factor's values under `scales`, or the values alone where it carries none.
    pub fn new(values: V, scales: ComptimeOption<S>, #[comptime] span: Span) -> Self {
        comptime!(assert!(
            span.lines > 0,
            "ScaledLines: one scale covers less than a whole line of values, so a line straddles \
             two scales"
        ));
        ScaledLines::<V, S> {
            values,
            scales,
            span,
        }
    }
}

#[cube]
impl<V: Lines, S: Lines> Lines for ScaledLines<V, S> {
    type E = V::E;
    type V = V::V;
    type S = S::E;
    type W = S::V;

    fn line(&self, pos: Coords2d) -> Vector<V::E, V::V> {
        self.values.line(pos)
    }

    fn fold(&self, pos: Coords2d) -> Fold<S::E, S::V> {
        #[comptime]
        match &self.scales {
            ComptimeOption::Some(scales) => Fold::<S::E, S::V>::of(scales.line(pos)),
            ComptimeOption::None => Fold::<S::E, S::V>::none(),
        }
    }

    fn span(&self) -> comptime_type!(Span) {
        self.span
    }
}

/// Every level of a scales operand, read as one line.
///
/// The levels are a list, applied innermost first. Coarser levels are read one scale at a time and
/// broadcast across the inner line, which is what a level covering a tile of the inner level's tiles
/// *is*; the inner level is read at the width its own cut gives it.
///
/// **Every level is read at the same logical position.** A level resolves that position to its own
/// granularity through its own projection, which is what "one scale per block" already means, so
/// nothing here divides a coordinate or knows how many levels there are.
#[derive(CubeType)]
pub struct CombinedScales<'a, S: Numeric, W: Size> {
    inner: MaskedView<'a, Vector<S, W>, Coords2d>,
    /// Every coarser level, already met and carried as one value.
    ///
    /// Read once when this source is built, which is once per region rather than once per value:
    /// a coarser level does not change inside the region it covers, so there is nothing there for
    /// a per-value read to discover. That is the whole reason depth is cheap.
    coarser: Vector<S, Const<1>>,
}

#[cube]
impl<'a, S: Numeric, W: Size> CombinedScales<'a, S, W> {
    pub fn new(
        inner: MaskedView<'a, Vector<S, W>, Coords2d>,
        coarser: Vector<S, Const<1>>,
    ) -> Self {
        CombinedScales::<'a, S, W> { inner, coarser }
    }
}

#[cube]
impl<'a, S: Numeric, W: Size> Lines for CombinedScales<'a, S, W> {
    type E = S;
    type V = W;
    type S = S;
    type W = Const<1>;

    fn line(&self, pos: Coords2d) -> Vector<S, W> {
        self.inner.read(pos) * Vector::<S, W>::cast_from(self.coarser.extract(0usize))
    }

    /// A scales operand carries its own coarser levels in the line it hands back, so it folds
    /// nothing further.
    fn fold(&self, _pos: Coords2d) -> Fold<S, Const<1>> {
        Fold::<S, Const<1>>::none()
    }

    /// How a scale line is spent belongs to whoever reads it: this is the line, not the grouping.
    fn span(&self) -> comptime_type!(Span) {
        Span::PLAIN
    }
}
