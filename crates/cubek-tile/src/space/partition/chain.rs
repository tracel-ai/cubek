//! The drawing two chained contractions print as: one figure a level, eight cells laid out so
//! every operand touches the ones it shares an axis with.
//!
//! What attention is. The first contraction takes the queries against the keys and the second
//! takes what the softmax made of that against the values, so the second's left operand is the
//! first's output and the axis between them is contracted once and kept once.
//!
//! ```text
//!    ·    keys   ·    values
//!  query  score  p    out
//! ```
//!
//! The bottom row all share the down axis, so one band of rows is one worker's queries, its
//! scores, its probabilities and its output. The keys sit over the score because they share the
//! axis the first contraction keeps, and the values over the output for the same reason. The two
//! cells of the top row that stay empty are the ones no operand spans.

use std::fmt::{self, Display, Formatter};

use super::quadrant::{GUTTER, Grid, MARGIN, side};
use crate::{Axis, Level, Partitioning, Space};

/// The four axes two chained contractions are read off. For attention they are the query
/// position, the head dim, the key position and the value dim.
///
/// `through` is the pair's hinge: the first contraction keeps it and the second contracts it, so
/// it is the axis the score and the probabilities are wide along and the values are deep along.
///
/// The one thing the drawing cannot derive. A [`Partitioning`] holds axes and levels and has no
/// notion of an operand, so which axis plays which part is the client's to say.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Chain {
    pub down: Axis,
    pub over: Axis,
    pub through: Axis,
    pub across: Axis,
}

/// A [`Partitioning`] read as two chained contractions: what [`Partitioning::chained`] returns,
/// and what prints one figure a level.
pub struct Chained<'a> {
    partitioning: &'a Partitioning,
    chain: Chain,
}

impl<'a> Chained<'a> {
    pub(crate) fn new(partitioning: &'a Partitioning, chain: Chain) -> Self {
        Chained {
            partitioning,
            chain,
        }
    }
}

/// The axes the figure does not span, multiplied: the drawing is one sheet of that many, and an
/// attention's batch and its query heads are the everyday ones.
fn sheets(level: &Level, space: &Space, chain: Chain) -> usize {
    space
        .axes()
        .filter(|&axis| {
            axis != chain.down
                && axis != chain.over
                && axis != chain.through
                && axis != chain.across
        })
        .map(|axis| side(level, space, axis))
        .product()
}

/// One cell of a row, blank past the grid's own height: the top row holds two grids of different
/// depths, and the shallower one runs out first.
fn cell(grid: &Grid, row: usize) -> String {
    match row < grid.height() {
        true => grid.line(row),
        false => " ".repeat(grid.width()),
    }
}

/// A cell no operand spans, which is a hole of its neighbour's width.
fn blank(width: usize) -> String {
    " ".repeat(width)
}

impl Display for Chained<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Chain {
            down,
            over,
            through,
            across,
        } = self.chain;
        let mut space = self.partitioning.space().clone();
        let mut lines: Vec<String> = Vec::new();

        for level in self.partitioning.levels() {
            let (rows, deep, wide, out) = (
                side(level, &space, down),
                side(level, &space, over),
                side(level, &space, through),
                side(level, &space, across),
            );
            let query = Grid { rows, cols: deep };
            let keys = Grid {
                rows: deep,
                cols: wide,
            };
            let score = Grid { rows, cols: wide };
            let probabilities = Grid { rows, cols: wide };
            let values = Grid {
                rows: wide,
                cols: out,
            };
            let output = Grid { rows, cols: out };

            if !lines.is_empty() {
                lines.push(String::new());
            }
            lines.push(match sheets(level, &space, self.chain) {
                1 => format!("{MARGIN}{}", glyph(level)),
                many => format!("{MARGIN}{} ×{many}", glyph(level)),
            });
            lines.push(String::new());
            for row in 0..keys.height().max(values.height()) {
                // The two grids of the top row run out at different depths, so the shallower
                // one leaves a hole the line has no reason to carry to its end.
                lines.push(
                    format!(
                        "{MARGIN}{}{GUTTER}{}{GUTTER}{}{GUTTER}{}",
                        blank(query.width()),
                        cell(&keys, row),
                        blank(probabilities.width()),
                        cell(&values, row),
                    )
                    .trim_end()
                    .to_string(),
                );
            }
            lines.push(String::new());
            for row in 0..score.height() {
                lines.push(format!(
                    "{MARGIN}{}{GUTTER}{}{GUTTER}{}{GUTTER}{}",
                    query.line(row),
                    score.line(row),
                    probabilities.line(row),
                    output.line(row),
                ));
            }

            space = level.child(&space);
        }
        write!(f, "{}", lines.join("\n"))
    }
}

/// The glyph a level prints as, which is the table's.
fn glyph(level: &Level) -> char {
    super::table::glyph(level.scope())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Space, Tiling};

    const B: Axis = Axis(0);
    const QP: Axis = Axis(1);
    const D: Axis = Axis(2);
    const S: Axis = Axis(3);
    const DV: Axis = Axis(4);

    const ATTENTION: Chain = Chain {
        down: QP,
        over: D,
        through: S,
        across: DV,
    };

    /// A cube per query block and batch, walking the cache in blocks.
    fn prefill() -> Partitioning {
        Partitioning::new(
            Space::new(&[(B, 2), (QP, 512), (D, 128), (S, 4096), (DV, 128)]),
            Tiling::leaf(&[(QP, 16), (D, 32), (S, 64), (DV, 32)])
                .lanes(&[(D, 4), (DV, 4)])
                .walk(&[(S, 4)])
                .planes(&[(QP, 4)])
                .cubes(&[QP])
                .batches(&[B])
                .levels(),
        )
    }

    /// The bottom row shares its rows, so one band is one worker's queries, scores,
    /// probabilities and output; the keys sit over the score and the values over the output.
    ///
    /// Each level shows the axes it cuts and no others: the cubes and the planes cut the query
    /// positions, the walk cuts the keys, and the lanes cut the two head dims. The score and
    /// the probabilities are the same two axes, so they draw alike at every level.
    #[test]
    fn the_operands_meet_where_they_share_an_axis() {
        assert_eq!(
            prefill().chained(ATTENTION).to_string(),
            [
                "  ▣ ×2",
                "",
                "      ██      ██",
                "",
                "  ██  ██  ██  ██",
                "  ░░  ░░  ░░  ░░",
                "  ░░  ░░  ░░  ░░",
                "  ░░  ░░  ░░  ░░",
                "  ░░  ░░  ░░  ░░",
                "  ░░  ░░  ░░  ░░",
                "  ░░  ░░  ░░  ░░",
                "  ░░  ░░  ░░  ░░",
                "",
                "  ▤",
                "",
                "      ██      ██",
                "",
                "  ██  ██  ██  ██",
                "  ░░  ░░  ░░  ░░",
                "  ░░  ░░  ░░  ░░",
                "  ░░  ░░  ░░  ░░",
                "",
                "  ↻",
                "",
                "      ██░░░░░░            ██",
                "                          ░░",
                "                          ░░",
                "                          ░░",
                "",
                "  ██  ██░░░░░░  ██░░░░░░  ██",
                "",
                "  ▪",
                "",
                "            ██      ██░░░░░░",
                "            ░░",
                "            ░░",
                "            ░░",
                "",
                "  ██░░░░░░  ██  ██  ██░░░░░░",
            ]
            .join("\n")
        );
    }
}
