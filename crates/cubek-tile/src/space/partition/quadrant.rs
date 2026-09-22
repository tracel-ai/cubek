//! The drawing a contraction's partitioning prints as: one figure a level, the three operands
//! laid out so their cuts line up.
//!
//! The lhs sits left of the out and the rhs above it, so the lhs's row cuts *are* the out's row
//! cuts and the rhs's column cuts its column cuts. One glyph a tile; the filled ones are one
//! worker's region at that level: a row band, a column band, and the out cell where they meet.
//!
//! Only three axes are injected. Everything else is read off the levels, and every axis the
//! figure does not span rides the level's header as a multiplier, since the drawing is one sheet
//! of however many that axis holds.

use std::fmt::{self, Display, Formatter};

use crate::{Axis, Level, Partitioning, Space};

/// Tiles a figure draws before the rest are elided, rows more tightly than columns: a row costs
/// a line of the terminal and a column costs two characters, and a walk over the contraction is
/// the everyday figure that is one tile wide and hundreds deep.
const ROWS_SHOWN: usize = 8;
const COLS_SHOWN: usize = 16;
/// One tile, and the marks an elided column and an elided row print instead.
const TILE: &str = "░░";
const TAKEN: &str = "██";
const COLUMNS: &str = " ⋯";
const ROWS: &str = " ⋮";
/// The left margin, and the space between the lhs and the out.
const MARGIN: &str = "  ";
const GUTTER: &str = "  ";

/// The three axes a contraction's operands are read off: the lhs is `down × over`, the rhs is
/// `over × across`, and the out is `down × across`. For a matmul they are `m`, `n` and `k`.
///
/// The one thing the drawing cannot derive. A [`Partitioning`] holds axes and levels and has no
/// notion of an operand, so which axis plays which part is the client's to say.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Contraction {
    pub down: Axis,
    pub across: Axis,
    pub over: Axis,
}

/// A [`Partitioning`] read as a contraction: what [`Partitioning::quadrant`] returns, and what
/// prints one figure a level.
pub struct Quadrant<'a> {
    partitioning: &'a Partitioning,
    contraction: Contraction,
}

impl<'a> Quadrant<'a> {
    pub(crate) fn new(partitioning: &'a Partitioning, contraction: Contraction) -> Self {
        Quadrant {
            partitioning,
            contraction,
        }
    }
}

/// One operand's grid at one level: the tiles the worker above it holds, of which the worker
/// takes the first — every other is its twin, so drawing one says what all of them are.
struct GridCount {
    rows: usize,
    cols: usize,
}

impl GridCount {
    /// How many tiles the grid draws along a side, and whether the rest were elided.
    fn shown(side: usize, cap: usize) -> (usize, bool) {
        (side.min(cap), side > cap)
    }

    fn width(&self) -> usize {
        GridCount::shown(self.cols, COLS_SHOWN).0 * TILE.chars().count()
    }

    fn height(&self) -> usize {
        GridCount::shown(self.rows, ROWS_SHOWN).0
    }

    /// Row `row` of the figure: the taken tile at the origin, the elision marks at the far edge.
    fn line(&self, row: usize) -> String {
        let (cols, elided) = GridCount::shown(self.cols, COLS_SHOWN);
        let (rows, deep) = GridCount::shown(self.rows, ROWS_SHOWN);
        (0..cols)
            .map(
                |col| match (deep && row + 1 == rows, elided && col + 1 == cols) {
                    (true, _) => ROWS,
                    (_, true) => COLUMNS,
                    _ if row == 0 && col == 0 => TAKEN,
                    _ => TILE,
                },
            )
            .collect()
    }
}

/// How many tiles `level` takes along `axis`, as a side of a figure: an axis it does not name is
/// one tile, and a count only the launch knows is drawn as elided, since nothing here can say how
/// many there are.
fn side(level: &Level, space: &Space, axis: Axis) -> usize {
    match level.count(axis) {
        None => 1,
        Some(_) => level.tiles_const(space, axis).unwrap_or(usize::MAX),
    }
}

/// The axes the figure does not span, multiplied: the drawing is one sheet of that many, and a
/// batch axis is the everyday one.
fn sheets(level: &Level, space: &Space, contraction: Contraction) -> usize {
    space
        .axes()
        .filter(|&axis| {
            axis != contraction.down && axis != contraction.across && axis != contraction.over
        })
        .map(|axis| side(level, space, axis))
        .product()
}

impl Display for Quadrant<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Contraction { down, across, over } = self.contraction;
        let mut space = self.partitioning.space().clone();
        let mut lines: Vec<String> = Vec::new();

        for level in self.partitioning.levels() {
            let (rows, cols, steps) = (
                side(level, &space, down),
                side(level, &space, across),
                side(level, &space, over),
            );
            let lhs = GridCount { rows, cols: steps };
            let rhs = GridCount { rows: steps, cols };
            let out = GridCount { rows, cols };
            let indent = " ".repeat(MARGIN.chars().count() + lhs.width() + GUTTER.chars().count());

            if !lines.is_empty() {
                lines.push(String::new());
            }
            lines.push(match sheets(level, &space, self.contraction) {
                1 => format!("{MARGIN}{}", glyph(level)),
                many => format!("{MARGIN}{} ×{many}", glyph(level)),
            });
            lines.push(String::new());
            lines.extend((0..rhs.height()).map(|row| format!("{indent}{}", rhs.line(row))));
            lines.push(String::new());
            lines.extend(
                (0..out.height())
                    .map(|row| format!("{MARGIN}{}{GUTTER}{}", lhs.line(row), out.line(row))),
            );

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
    use crate::{Levels, Space};

    const B: Axis = Axis(0);
    const M: Axis = Axis(1);
    const N: Axis = Axis(2);
    const K: Axis = Axis(3);

    const MNK: Contraction = Contraction {
        down: M,
        across: N,
        over: K,
    };

    /// A walk that streams the contraction, under a cube grid over a batched space.
    fn staged() -> Partitioning {
        Partitioning::new(
            Space::new(&[(B, 4), (M, 512), (N, 1024), (K, 4096)]),
            Levels::leaf(&[(M, 64), (N, 64), (K, 1024)])
                .walk(&[(K, 4)])
                .cubes(&[M, N])
                .batches(&[B])
                .build(),
        )
    }

    /// The lhs is a row band, the rhs a column band, the out the cell they meet in — and the
    /// batch axis the figure does not span rides the header.
    #[test]
    fn the_bands_meet_in_the_cell_the_worker_owns() {
        assert_eq!(
            staged().quadrant(MNK).to_string(),
            [
                "  ▣ ×4",
                "",
                "      ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░",
                "",
                "  ██  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░",
                "  ░░  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░",
                "  ░░  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░",
                "  ░░  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░",
                "  ░░  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░",
                "  ░░  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░",
                "  ░░  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░",
                "  ░░  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░",
                "",
                "  ↻",
                "",
                "            ██",
                "            ░░",
                "            ░░",
                "            ░░",
                "",
                "  ██░░░░░░  ██",
            ]
            .join("\n")
        );
    }

    /// A side past the cap keeps its shape and says the rest were left out.
    #[test]
    fn a_grid_too_wide_to_draw_elides_its_far_edge() {
        let partitioning = Partitioning::new(
            Space::new(&[(M, 64), (N, 64), (K, 4096)]),
            Levels::leaf(&[(M, 64), (N, 64), (K, 32)])
                .walk(&[(K, 128)])
                .build(),
        );
        let drawing = partitioning.quadrant(MNK).to_string();

        assert!(drawing.contains(" ⋯"), "{drawing}");
        assert!(drawing.lines().any(|line| line.contains(" ⋮")), "{drawing}");
    }

    /// An axis the launch stamps has no count here, so its side draws as elided rather than
    /// inventing one.
    #[test]
    fn a_dynamic_side_elides() {
        let partitioning = Partitioning::new(
            Space::new(&[(M, 64), (N, 64), (K, 4096)]).with_dynamic(&[K]),
            Levels::leaf(&[(M, 64), (N, 64), (K, 32)])
                .walk_every(&[K])
                .build(),
        );

        assert!(partitioning.quadrant(MNK).to_string().contains(" ⋯"));
    }
}
