//! What a scales operand is against the values it covers.
//!
//! A scale is a tile that spans fewer axes than the values it multiplies, and "one scale per block"
//! is what its axes say rather than what any arithmetic does. Reading that off takes four facts —
//! which side it folds into, which edge it shares with the values, how many of their lines one
//! scale covers, and how many scales arrive per read — and every one of them is derived here and
//! nowhere else. Two accumulators need them, and a decision re-derived at each caller is one that
//! has already drifted.
//!
//! What stays with each accumulator is its own geometry, which genuinely differs: the edges it can
//! walk, the widths it serves them at, and whether it steps them under an ordinal it knows
//! ([`ContractEdges`]).

use crate::*;

/// Which factor of the term a scales operand multiplies. Read off the axes it spans, never
/// stated: a scale over the accumulator's column axis is a fact about the rhs's columns and
/// nothing else could fold it in; anything else scales the lhs.
///
/// One verb, then, not two. `(a ⊗ s) · b` and `a · (b ⊗ s)` are the same sum of terms — the scale
/// is one more factor of each — and which operand it rides is only *where* it folds in cheapest:
/// once per `(row, k)` beside the lhs, or once per `(col, k)` beside the rhs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ScaleSide {
    /// The scale spans the accumulator's rows (or the contracted axis alone): folded into the
    /// lhs value before it forms its products.
    Lhs,
    /// The scale spans the accumulator's columns: folded into each rhs line.
    Rhs,
}

/// The side a scales operand multiplies on, from the axes it spans against the accumulator's.
///
/// A scale over neither matrix axis (per-tensor, or one value per block of `k`) is the same
/// number wherever it folds, so it takes the lhs side.
pub(crate) fn scale_side(scales: &Space, output: &Space, axes: MatrixAxes) -> ScaleSide {
    let group = |range: core::ops::Range<usize>| {
        range
            .filter(|&p| scales.contains(output.axis_at(p)))
            .map(|p| output.axis_at(p))
            .collect::<Vec<_>>()
    };
    let rows = group(axes.row_split..axes.col_split);
    let cols = group(axes.col_split..output.rank());
    assert!(
        rows.is_empty() || cols.is_empty(),
        "mm_scaled: a scales operand over the accumulator's rows {rows:?} and its columns \
         {cols:?} is a scale of the output, not a factor of either operand's term"
    );
    match cols.is_empty() {
        false => ScaleSide::Rhs,
        true => ScaleSide::Lhs,
    }
}

/// Refuse a scales operand that spells its granularity by dividing.
///
/// A scale covers a block because the operand has no axis to vary over inside one: the block is an
/// axis and the scales omit it. A rational axis (`PhysicalAxisMap::of(N).over(bn)`) states the same
/// granularity arithmetically, and then whether a line straddles a block stops being a fact about
/// the axes and becomes one about the line width, which no operand states. Split the axis instead,
/// and the invariance is structural.
pub(crate) fn check_scales_omit_rather_than_divide(scales: &Projection) {
    for pa in 0..scales.physical_rank() {
        let divisor = scales.divisor(pa).bound();
        assert!(
            divisor == 1,
            "mm_scaled: this scales operand divides a logical axis by {divisor} to reach its \
             block. Spell the block as an axis of its own and omit the position inside it, so one \
             scale per block is what the operand's axes say rather than what its arithmetic does"
        );
    }
}

/// Whether the caller walks the edge a scales operand shares with its values under an ordinal it
/// knows at comptime.
///
/// A scale line wider than one scale needs each value line's ordinal along that edge as a
/// constant: a fold is a lane of the read it arrived in, and a lane index is not addressable at
/// runtime. Whether the ordinal is a constant is a fact about how the caller steps, so the caller
/// states it and the rule reads off it — once, here, rather than as an exception each caller
/// spells for itself.
pub(crate) enum EdgeOrdinal {
    /// Each line's position along the shared edge is a constant, so the scales may be served
    /// several at a time.
    Constant,
    /// The edge is stepped at runtime, so only a scalar read is addressable. Carries what about
    /// this walk makes it so, for the refusal to quote.
    Runtime(String),
}

/// How a scale level is applied to what it covers.
///
/// Named rather than assumed. What the engine may do with several levels follows from this and from
/// nothing else, the same way [`Semiring`] licenses what the contraction may do: combining levels
/// before they meet the values is a reassociation, and only a verb can license one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Apply {
    /// The level below is multiplied by it.
    Product,
}

/// A caller's contraction geometry, as a scale level needs to see it: the two edges its values can
/// be read along, the row counts their matrices take, and the widths each edge is served at.
///
/// Stating this is each accumulator's own business — the memory nest reads it off a
/// [`ContractShape`](super::shape::ContractShape), the promoted block off its own cells — and
/// nothing here decides any of it. What is decided here is which of the two edges a given scales
/// operand shares, which is not a fact about the accumulator at all.
pub(crate) struct ContractEdges {
    /// Rows of the lhs's matrix.
    pub mr: usize,
    /// The contracted extent, which is the rhs's matrix rows.
    pub kc: usize,
    /// Columns of the accumulator.
    pub cols: usize,
    /// The contracted axes with their extents. The accumulator cannot size these: a contracted
    /// axis is by definition absent from it.
    pub reduce: Vec<(Axis, usize)>,
    /// The accumulator's column axes with theirs.
    pub columns: Vec<(Axis, usize)>,
    /// The width the lhs's lines are served at.
    pub lw: usize,
    /// The width the accumulator's lines are served at.
    pub aw: usize,
    /// Contracted values one step consumes. Past one, the step's own edge *is* the contraction,
    /// whichever side the scales ride.
    pub contracted_per_step: usize,
    /// How this caller steps the edge the scales share.
    pub ordinal: EdgeOrdinal,
}

/// One level of a scale hierarchy, against the values it covers.
///
/// The level below is whatever this one scales: the operand's values at the first level, and the
/// level beneath it at any level above. Nothing here knows which, or how many there are.
#[derive(Clone, Copy)]
pub(crate) struct ScaleLevel {
    /// What this level does to the level below it.
    pub apply: Apply,
    /// The scales' own matrix, as the level below reads it.
    pub axes: MatrixAxes,
    /// Lines of the level below that one scale covers, along the edge they share.
    pub lines_per_scale: usize,
    /// Scales one read of them serves.
    pub lanes: usize,
}

impl ScaleLevel {
    /// Read the level off the axes.
    ///
    /// Which edge the scales share is the side's to say, with one exception that is not one: a
    /// scale on the lhs varies along the contraction, and so does a scale a *folded* step reads,
    /// because a folded step's own edge is the contraction whichever side it came from. Only an
    /// unfolded rhs varies along the accumulator's columns.
    ///
    /// `lines_per_scale` then falls out of `invariant` — the edge axes the scales do not
    /// distinguish — rather than out of dividing one extent by another. The scale is constant
    /// along every axis it does not address, so one read of it serves every position of them, and
    /// no line can straddle a block whatever width it is served at.
    pub(crate) fn of(
        scales: &Space,
        edges: &ContractEdges,
        side: ScaleSide,
        invariant: &[Axis],
        lanes: usize,
    ) -> Self {
        let folded = edges.contracted_per_step > 1;
        let (rows, edge, value_width) = match (side, folded) {
            (ScaleSide::Lhs, _) => (edges.mr, &edges.reduce, edges.lw),
            (ScaleSide::Rhs, true) => (edges.cols, &edges.reduce, edges.contracted_per_step),
            (ScaleSide::Rhs, false) => (edges.kc, &edges.columns, edges.aw),
        };
        match &edges.ordinal {
            EdgeOrdinal::Constant => {}
            // One scale a read needs no ordinal at all: every line takes the same lane.
            EdgeOrdinal::Runtime(_) if lanes == 1 => {}
            EdgeOrdinal::Runtime(why) => panic!(
                "mm_scaled: {lanes} scales are served as one line, which needs each value line's \
                 ordinal along the edge they share as a constant. {why}; bind the scales scalar \
                 here"
            ),
        }
        let cols = scales.extent_at(scales.rank() - 1);
        let lines_per_scale = edge
            .iter()
            .filter(|(axis, _)| invariant.contains(axis))
            .map(|(_, extent)| *extent)
            .product::<usize>()
            / value_width;
        ScaleLevel {
            apply: Apply::Product,
            axes: MatrixAxes::of(scales, rows, cols),
            lines_per_scale,
            lanes,
        }
    }
}
