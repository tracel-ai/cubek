//! What the register block reads: an operand's line at a position, and the scale line covering a
//! run of them.
//!
//! **A scale is never found from a value's position.** The walk holds the coarser coordinate and
//! builds the finer one out of it: a run names one scale line, a field of that line names the tile
//! it covers, and the value lines of that tile follow by multiplication. Nothing here divides a
//! coordinate, and no operand can be asked which scale covers a line — it is handed every scale
//! line a run needs, once, at the top of the run.
//!
//! **The scales are read through this same trait**, so a scaled operand's scales are themselves an
//! operand: one scale for a tile of values, and every level coarser than that already folded into
//! the line it hands back ([`CombinedScales`]).

use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::*;

/// Which edge of the contraction a factor's scale line runs along.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Along {
    /// The line's fields are consecutive blocks of the contraction: the lhs, and a rhs whose
    /// step folds a whole line.
    Contraction,
    /// The line's fields are consecutive blocks of the accumulator's columns: a rhs lining along
    /// them.
    Columns,
}

/// How an operand's value lines are grouped under its scale lines.
///
/// One read of the scales brings `fields` of them along [`along`](Span::along), each covering
/// `lines` value lines there; along the contraction, one scale holds for `steps` lines. A run of
/// the walk is `steps` lines of the contraction, and the walk multiplies every index out of the run
/// it is in. An operand carrying no scales is [`PLAIN`](Span::PLAIN) and groups nothing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Span {
    /// The edge the scale line runs along.
    pub along: Along,
    /// Scales one read of them serves.
    pub fields: usize,
    /// Value lines one scale covers along [`along`](Span::along).
    pub lines: usize,
    /// Lines of the contraction one scale holds for. Along the contraction this is
    /// `lines`; along the columns it is the block the scale covers there.
    pub steps: usize,
}

impl Span {
    /// An operand with no scales: every line stands alone.
    pub const PLAIN: Span = Span {
        along: Along::Contraction,
        fields: 1,
        lines: 1,
        steps: 1,
    };

    /// Whether this operand groups nothing.
    pub fn is_plain(&self) -> bool {
        self.fields == 1 && self.lines == 1 && self.steps == 1
    }

    /// Lines of the contraction one run covers: one scale line's worth where the line runs
    /// along it, one scale's worth where it runs along the columns.
    pub fn run(&self) -> usize {
        match self.along {
            Along::Contraction => self.fields * self.lines,
            Along::Columns => self.steps,
        }
    }

    /// Scale lines a factor needs for `majors` rows or columns of the block: one each where its
    /// line runs along the contraction, one per `fields · lines` columns where it runs along them.
    pub fn count(&self, majors: usize) -> usize {
        match self.along {
            Along::Contraction => majors,
            Along::Columns => {
                let covered = self.fields * self.lines;
                assert!(
                    majors.is_multiple_of(covered),
                    "block::contract: {majors} columns of a block do not divide into runs of \
                     {covered}, so one run would work under a scale that is not there"
                );
                majors / covered
            }
        }
    }

    /// The run two factors walk together along the contraction.
    ///
    /// One walk steps both, so either they group its lines the same way or one of them groups
    /// nothing. Two different groupings would need a position resolved per side, which is the
    /// division this module exists not to do.
    pub fn join_run(self, other: Span) -> usize {
        match (self.run(), other.run()) {
            (1, run) | (run, 1) => run,
            (a, b) => {
                assert!(
                    a == b,
                    "block::contract: one factor holds a scale for {a} lines of the contraction \
                     and the other for {b}. A walk steps both, so a run is one grouping or none"
                );
                a
            }
        }
    }
}

/// Every scale line one run holds, read once at the top of the run: one per row or column of
/// the block.
///
/// Absent is a factor with no scales, and it emits nothing: its values go through as they lie.
#[derive(CubeType)]
pub struct RunScales<S: Numeric, W: Size> {
    /// The scale lines of this run, where the factor carries any.
    lines: ComptimeOption<Array<Vector<S, W>>>,
}

#[cube]
impl<S: Numeric, W: Size> RunScales<S, W> {
    /// A factor with no scales.
    pub fn none() -> Self {
        RunScales::<S, W> {
            lines: ComptimeOption::new_None(),
        }
    }

    /// One scale line, for a run that needs a single one: the landing's, where the coordinate a
    /// scale is constant along is the run itself.
    pub fn of(line: Vector<S, W>) -> Self {
        let mut lines = Array::<Vector<S, W>>::new(1usize);
        lines[0usize] = line;
        RunScales::<S, W> {
            lines: ComptimeOption::new_Some(lines),
        }
    }

    /// `value` under field `field` of the scale line `major` holds, which is the scale covering
    /// the tile the walk is in. The field is a constant because the walk built it.
    pub fn apply<E: Numeric, V: Size>(
        &self,
        value: Vector<E, V>,
        major: usize,
        #[comptime] field: usize,
    ) -> Vector<E, V> {
        #[comptime]
        match &self.lines {
            ComptimeOption::Some(lines) => {
                value * Vector::<E, V>::cast_from(lines[major].extract(field))
            }
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

    /// The scale lines run `run` holds — `count` of them, one per row or column of the block
    /// ([`Span::count`]) — or nothing where this operand carries none.
    fn run_scales(&self, #[comptime] count: usize, run: u32) -> RunScales<Self::S, Self::W>;

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

    fn run_scales(&self, #[comptime] _count: usize, _run: u32) -> RunScales<E, Const<1>> {
        RunScales::<E, Const<1>>::none()
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

    /// A line along the contraction sits at `(major, run)`; one along the columns at
    /// `(run · steps, major)` — any row of the block the run covers names its scale.
    fn run_scales(&self, #[comptime] count: usize, run: u32) -> RunScales<S::E, S::V> {
        #[comptime]
        match &self.scales {
            ComptimeOption::Some(scales) => {
                let mut lines = Array::<Vector<S::E, S::V>>::new(count);
                #[unroll]
                for major in 0..count {
                    let at = comptime!(major as u32).runtime();
                    let pos = match comptime!(self.span.along) {
                        Along::Contraction => (at, run),
                        Along::Columns => (run * comptime!(self.span.steps as u32), at),
                    };
                    lines[major] = scales.line(pos);
                }
                RunScales::<S::E, S::V> {
                    lines: ComptimeOption::new_Some(lines),
                }
            }
            ComptimeOption::None => RunScales::<S::E, S::V>::none(),
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
    fn run_scales(&self, #[comptime] _count: usize, _run: u32) -> RunScales<S, Const<1>> {
        RunScales::<S, Const<1>>::none()
    }

    /// How a scale line is spent belongs to whoever reads it: this is the line, not the grouping.
    fn span(&self) -> comptime_type!(Span) {
        Span::PLAIN
    }
}
