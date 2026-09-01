//! What the register block reads: one line of an operand, at a position.
//!
//! A plain operand hands its line back. A scaled one multiplies its scales in first, and does it
//! here rather than under a view because a scale line covers several value lines: which lane of it
//! a line takes has to be a constant, and only a caller walking a comptime run knows that. `run`
//! is that position, and an operand with nothing to index by it ignores it.
//!
//! **The scales are read through this same trait**, so a scaled operand's scales may themselves be
//! scaled: one scale for a tile of values, a tile of scales for a tile of tiles of values. Depth is
//! whatever the type says, and every level reads the level above it the same way, so nothing here
//! counts levels or knows how many there are.

use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::*;

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

    /// The line at `pos`, folded with whatever this operand carries beside its values. `run` names
    /// this line's ordinal along the edge the fold is indexed by, so the lane it selects is a
    /// constant.
    fn line(&self, pos: Coords2d, #[comptime] run: usize) -> Vector<Self::E, Self::V>;

    /// How this operand's fold repeats across its lines ([`Reuse`]).
    fn reuse(&self) -> comptime_type!(Reuse);
}

/// How a loaded value is reused across the walk.
///
/// The two travel together because a caller needs both to walk correctly: it may roll a run of
/// `steps`, since every step in one takes the same value, but it must walk the values themselves
/// under a *constant* ordinal, because each is a lane of the read it arrived in and a lane index is
/// not addressable at runtime.
///
/// [`PER_STEP`](Reuse::PER_STEP) is an operand with nothing to reuse: it reads what it needs, every
/// step, so a caller walking it needs no constant at all.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Reuse {
    /// Values one read brings back, which is the operand's line width where it has one.
    pub per_load: usize,
    /// Steps one value serves before the next is wanted.
    pub steps: usize,
}

impl Reuse {
    /// An operand with nothing to reuse: it reads what it needs, every step.
    pub const PER_STEP: Reuse = Reuse {
        per_load: 1,
        steps: 1,
    };

    /// Steps walked before the pattern repeats.
    pub fn span(&self) -> usize {
        self.per_load * self.steps
    }

    /// This level's run against the one its own scales carry, in value lines.
    ///
    /// A level whose read serves a single value constrains no walk: every step takes the same lane
    /// whatever its ordinal, so it drops out here rather than widening the run its caller has to
    /// unroll. That is what keeps a coarse outer level free, and a per-tensor scale, which is one
    /// value covering everything, entirely invisible to the walk.
    ///
    /// `per_line` is how many value lines one line of *these* scales covers, which is what carries
    /// the level above into value-line units.
    pub fn compose(self, above: Reuse, per_line: usize) -> Reuse {
        let here = (self.per_load > 1).then_some(self);
        let above = (above.per_load > 1).then_some(Reuse {
            per_load: above.per_load,
            steps: above.steps * per_line,
        });
        match (here, above) {
            (None, None) => Reuse::PER_STEP,
            (Some(only), None) | (None, Some(only)) => only,
            // The finer run is the one a caller must hold constant, and the coarser is a multiple
            // of it, so they repeat together at the wider span.
            (Some(here), Some(above)) => {
                let steps = here.steps.min(above.steps);
                let span = lcm(here.span(), above.span());
                Reuse {
                    per_load: span / steps,
                    steps,
                }
            }
        }
    }
}

fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

fn gcd(a: usize, b: usize) -> usize {
    match b {
        0 => a,
        _ => gcd(b, a % b),
    }
}

// `per_load` alone decides how a caller must walk: at one, every step takes the same lane
// whatever its ordinal, so `steps` is the caller's own business. Only past one does the ordinal have
// to be a constant, which is what the walks below branch on.

#[cube]
impl<'a, E: Numeric, V: Size> Lines for MaskedView<'a, Vector<E, V>, Coords2d> {
    type E = E;
    type V = V;

    fn line(&self, pos: Coords2d, #[comptime] _run: usize) -> Vector<E, V> {
        self.read(pos)
    }

    fn reuse(&self) -> comptime_type!(Reuse) {
        Reuse::PER_STEP
    }
}

/// An operand read against its scales: the values' line, times the scale covering it.
///
/// The scales are read through their own matrix, whose columns count blocks where the values'
/// count lines. Which *lane* of a scale line a value line takes is a constant the caller knows and
/// the view could not: `run` is the value line's ordinal along the shared edge, and the lane falls
/// out of it. Which scale *line* to read is only an address, so it stays runtime.
///
/// **Both sides are a [`Lines`]**, so this one type serves every level: the values are an operand's
/// own lines at the level nearest them, and the level below's scales at any level above. The scales
/// likewise carry scales of their own. A second level is then this same type wrapping this same
/// type, which is why nothing here says how deep the chain goes.
#[derive(CubeType)]
pub struct ScaledLines<V: Lines, S: Lines> {
    values: V,
    scales: S,
    /// Value lines one scale covers along the shared edge.
    #[cube(comptime)]
    lines_per_scale: usize,
    /// Scales one line of them serves, which is the width the scales are read at.
    #[cube(comptime)]
    lanes: usize,
}

#[cube]
impl<V: Lines, S: Lines> ScaledLines<V, S> {
    pub fn new(
        values: V,
        scales: S,
        #[comptime] lines_per_scale: usize,
        #[comptime] lanes: usize,
    ) -> Self {
        comptime!(assert!(
            lines_per_scale > 0,
            "ScaledLines: one scale covers less than a whole line of values, so a line straddles \
             two scales"
        ));
        ScaledLines::<V, S> {
            values,
            scales,
            lines_per_scale,
            lanes,
        }
    }
}

#[cube]
impl<V: Lines, S: Lines> Lines for ScaledLines<V, S> {
    type E = V::E;
    type V = V::V;

    fn line(&self, pos: Coords2d, #[comptime] run: usize) -> Vector<V::E, V::V> {
        let value = self.values.line(pos, run);
        let (row, col) = pos;
        let per_line = comptime!(self.lines_per_scale * self.lanes);
        // One line of these scales covers `per_line` value lines, so that is both the column it
        // sits at and the ordinal it is read under: the level above indexes scale lines, not value
        // lines, and per_load its own scale in on the way back.
        let scale = self.scales.line(
            (row, col / comptime!(per_line as u32)),
            comptime!(run / per_line),
        );
        let lane = comptime!((run / self.lines_per_scale) % self.lanes);
        value * Vector::<V::E, V::V>::cast_from(scale.extract(lane))
    }

    fn reuse(&self) -> comptime_type!(Reuse) {
        let above = self.scales.reuse();
        comptime!(
            Reuse {
                per_load: self.lanes,
                steps: self.lines_per_scale,
            }
            .compose(above, self.lines_per_scale * self.lanes)
        )
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

    fn line(&self, pos: Coords2d, #[comptime] _run: usize) -> Vector<S, W> {
        self.inner.read(pos) * Vector::<S, W>::cast_from(self.coarser.extract(0usize))
    }

    fn reuse(&self) -> comptime_type!(Reuse) {
        // How a scale line is spent belongs to whoever reads it: this is the line, not the pattern.
        // The coarser levels add nothing either, being one value each over a span at least this
        // wide.
        Reuse::PER_STEP
    }
}
