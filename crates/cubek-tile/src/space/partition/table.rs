//! The table a [`Partitioning`] prints as: one row a level, leaf up.
//!
//! Two blocks of the same shape: the count block is how many tiles a level takes along each axis,
//! the tile block what one of those tiles holds. A row's tile is the row below times the count
//! beside it, axis by axis: a row that fails to multiply out is a partitioning that answers wrong.
//!
//! The level column is the coverage's glyph and what it covers: how many cubes, planes a cube or
//! lanes, or how many steps a walk, with the spread and the filling planes where stated.

use std::fmt::{self, Display, Formatter};

use crate::{Axis, ComputeScope, Coverage, Level, Partitioning, Space, Spread};

/// The leaf row's glyph: the tile the levels reach, which no level of its own cuts.
const LEAF: char = '◦';
/// The left margin, and the gap after the level's glyph, before what it covers.
const MARGIN: &str = "  ";
const LEVEL: &str = "  ";
/// Between two axes of a block, and between the blocks.
const TIMES: &str = " × ";
const GAP: &str = "    ";

/// How many tiles a level takes along one axis, as the table prints it. A level that does not
/// name the axis contributes a factor of one; one taking every tile of an extent the launch has
/// not stamped has a count only the launch knows, and says so rather than inventing one.
fn count(level: &Level, space: &Space, axis: Axis) -> String {
    match level.count(axis) {
        None => "·".to_string(),
        Some(_) => match level.tiles_const(space, axis) {
            Some(tiles) => tiles.to_string(),
            None => "?".to_string(),
        },
    }
}

/// A [`Partitioning`] with a name for each of its axes, which is the one thing the value cannot
/// supply: an [`Axis`] is a client-assigned index and the labels are the client's. An axis the
/// labels do not name prints that index.
pub struct LevelTable<'a> {
    partitioning: &'a Partitioning,
    labels: &'a [(Axis, &'a str)],
}

impl<'a> LevelTable<'a> {
    pub(crate) fn new(partitioning: &'a Partitioning, labels: &'a [(Axis, &'a str)]) -> Self {
        LevelTable {
            partitioning,
            labels,
        }
    }

    fn label(&self, axis: Axis) -> String {
        match self.labels.iter().find(|&&(named, _)| named == axis) {
            Some(&(_, name)) => name.to_string(),
            None => format!("a{}", axis.0),
        }
    }

    /// Every row of the table, leaf up: the leaf the levels reach, then each level from the one
    /// that reached it out to the grid.
    fn rows(&self) -> Vec<Row> {
        let axes: Vec<Axis> = self.partitioning.space().axes().collect();
        let mut space = self.partitioning.space().clone();
        let mut rows: Vec<Row> = self
            .partitioning
            .levels()
            .iter()
            .map(|level| {
                let row = Row::of(level, &space, &axes);
                space = level.child(&space);
                row
            })
            .collect();
        rows.reverse();
        rows.insert(0, Row::leaf(&space, &axes));
        rows
    }
}

/// The glyph a level's coverage prints as.
fn glyph(coverage: Coverage) -> char {
    match coverage {
        Coverage::Distribute(ComputeScope::Cube) => '▣',
        Coverage::Distribute(ComputeScope::Plane) => '▤',
        Coverage::Distribute(ComputeScope::Unit) => '▪',
        Coverage::Walk => '↻',
    }
}

/// What a level covers, in words: the instances the level distributes to (or the steps a walk
/// takes) multiplied over its axes, with the spread and the filling planes where it states them.
fn coverage(level: &Level, space: &Space) -> String {
    let counts: Vec<Option<usize>> = level
        .axes()
        .iter()
        .map(|&axis| level.tiles_const(space, axis))
        .collect();
    let total = counts
        .iter()
        .try_fold(1usize, |acc, count| count.map(|n| acc * n));
    let many = match total {
        Some(n) => n.to_string(),
        None => "?".to_string(),
    };
    let interleaved = level
        .axes()
        .iter()
        .any(|&axis| level.spread(axis) == Some(Spread::Interleaved));
    let mut note = match level.coverage() {
        Coverage::Distribute(ComputeScope::Cube) => match level.shared_by() {
            Some(cubes) => format!("{cubes} cubes sharing {many} boxes"),
            None => format!("{many} cubes"),
        },
        Coverage::Distribute(ComputeScope::Plane) => match level.shared_by() {
            Some(planes) => format!("{planes} planes sharing {many} boxes"),
            None => format!("{many} planes a cube"),
        },
        Coverage::Distribute(ComputeScope::Unit) => format!("{many} lanes"),
        Coverage::Walk => format!("{many} steps"),
    };
    if interleaved {
        note += " interleaved";
    }
    match level.fillers() {
        0 => {}
        1 => note += ", 1 filling plane",
        n => note += &format!(", {n} filling planes"),
    }
    note
}

/// One line of the table: a level's glyph, what it cuts each axis into, and what one of its
/// regions holds.
struct Row {
    glyph: char,
    coverage: String,
    counts: Vec<String>,
    tile: Vec<String>,
}

impl Row {
    fn of(level: &Level, space: &Space, axes: &[Axis]) -> Row {
        Row {
            glyph: glyph(level.coverage()),
            coverage: coverage(level, space),
            counts: axes.iter().map(|&axis| count(level, space, axis)).collect(),
            tile: axes.iter().map(|&axis| extent(space, axis)).collect(),
        }
    }

    /// The tile the levels reach, which no level cuts: every count is a one.
    fn leaf(space: &Space, axes: &[Axis]) -> Row {
        Row {
            glyph: LEAF,
            coverage: String::new(),
            counts: axes.iter().map(|_| "·".to_string()).collect(),
            tile: axes.iter().map(|&axis| extent(space, axis)).collect(),
        }
    }
}

/// `axis`'s extent, or the mark a dynamic one prints as: the launch stamps it, so nothing here
/// can name it.
fn extent(space: &Space, axis: Axis) -> String {
    match space.is_dynamic(axis) {
        true => "?".to_string(),
        false => space.extent(axis).to_string(),
    }
}

/// The widest cell of each column, the header's own label included.
fn widths(header: &[String], cells: impl Iterator<Item = Vec<String>>) -> Vec<usize> {
    cells.fold(
        header.iter().map(|label| label.chars().count()).collect(),
        |widest: Vec<usize>, row| {
            widest
                .iter()
                .zip(&row)
                .map(|(&widest, cell)| widest.max(cell.chars().count()))
                .collect()
        },
    )
}

/// One block of a line: every cell right-aligned in its column, `×` between.
fn block(cells: &[String], widths: &[usize]) -> String {
    cells
        .iter()
        .zip(widths)
        .map(|(cell, &width)| format!("{cell:>width$}"))
        .collect::<Vec<_>>()
        .join(TIMES)
}

/// The rule naming a block, drawn to the block's own width.
fn rule(name: &str, width: usize) -> String {
    format!("└─ {name} {}┘", "─".repeat(width - ruled(name) + 1))
}

/// The narrowest a block can print and still carry its rule, which a block of one short column
/// is not: the pad below makes up the difference rather than widening a column, so every cell
/// stays right-aligned where the header put it.
fn ruled(name: &str) -> usize {
    name.chars().count() + 6
}

impl Display for LevelTable<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let axes: Vec<Axis> = self.partitioning.space().axes().collect();
        let header: Vec<String> = axes.iter().map(|&axis| self.label(axis)).collect();
        let rows = self.rows();

        let counts = widths(&header, rows.iter().map(|row| row.counts.clone()));
        let tiles = widths(&header, rows.iter().map(|row| row.tile.clone()));
        let (counts_wide, counts_pad) = padded(&counts, "count");
        let (tiles_wide, tiles_pad) = padded(&tiles, "tile");
        // The coverage column, plus the gap before the count block.
        let coverage_wide = rows
            .iter()
            .map(|row| row.coverage.chars().count())
            .max()
            .unwrap_or(0)
            + GAP.chars().count();
        let indent = format!("{MARGIN} {LEVEL}{:coverage_wide$}", "");

        writeln!(
            f,
            "{indent}{counts_pad}{}{GAP}{tiles_pad}{}",
            block(&header, &counts),
            block(&header, &tiles)
        )?;
        writeln!(f)?;
        for row in &rows {
            writeln!(
                f,
                "{MARGIN}{}{LEVEL}{:coverage_wide$}{counts_pad}{}{GAP}{tiles_pad}{}",
                row.glyph,
                row.coverage,
                block(&row.counts, &counts),
                block(&row.tile, &tiles)
            )?;
        }
        writeln!(f)?;
        write!(
            f,
            "{indent}{}{GAP}{}",
            rule("count", counts_wide),
            rule("tile", tiles_wide)
        )
    }
}

/// How wide a block of these columns prints and the indent it takes to get there: its own
/// span, or its rule's width where that is wider.
fn padded(widths: &[usize], name: &str) -> (usize, String) {
    let wide = spanned(widths).max(ruled(name));
    (wide, " ".repeat(wide - spanned(widths)))
}

/// How wide a block of these columns prints, separators included.
fn spanned(widths: &[usize]) -> usize {
    widths.iter().sum::<usize>() + TIMES.chars().count() * (widths.len().saturating_sub(1))
}

/// A partitioning prints its table with no axis named, which is all a value holding
/// client-assigned indices can promise; [`Partitioning::table`] is the one worth reading.
impl Display for Partitioning {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.table(&[]).fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Levels, Space};

    const B: Axis = Axis(0);
    const M: Axis = Axis(1);
    const N: Axis = Axis(2);
    const K: Axis = Axis(3);

    /// The five levels a staged matmul states, leaf up: the instruction, its grid of fragments,
    /// its steps through the partition, the plane split, the stage walk, and the cube grid.
    fn staged() -> Partitioning {
        Partitioning::new(
            Space::new(&[(B, 4), (M, 512), (N, 1024), (K, 4096)]),
            Levels::leaf(&[(M, 16), (N, 16), (K, 16)])
                .walk(&[(M, 2), (N, 2)])
                .walk(&[(K, 2)])
                .planes(&[(M, 2), (N, 2)])
                .walk_every(&[K])
                .cubes(&[M, N])
                .batches(&[B])
                .build(),
        )
    }

    #[test]
    fn a_row_is_the_row_below_it_times_its_count() {
        let labels = [(B, "b"), (M, "m"), (N, "n"), (K, "k")];

        assert_eq!(
            staged().table(&labels).to_string(),
            [
                "                        b × m ×  n ×   k    b ×   m ×    n ×    k",
                "",
                "  ◦                     · × · ×  · ×   ·    1 ×  16 ×   16 ×   16",
                "  ↻  4 steps            · × 2 ×  2 ×   ·    1 ×  32 ×   32 ×   16",
                "  ↻  2 steps            · × · ×  · ×   2    1 ×  32 ×   32 ×   32",
                "  ▤  4 planes a cube    · × 2 ×  2 ×   ·    1 ×  64 ×   64 ×   32",
                "  ↻  128 steps          · × · ×  · × 128    1 ×  64 ×   64 × 4096",
                "  ▣  512 cubes          4 × 8 × 16 ×   ·    4 × 512 × 1024 × 4096",
                "",
                "                        └─ count ──────┘    └─ tile ────────────┘",
            ]
            .join("\n")
        );
    }

    /// An axis the labels do not name prints its index, so the table stands on its own.
    #[test]
    fn an_unlabelled_axis_prints_its_index() {
        let table = staged().to_string();

        assert!(table.contains("a0 × a1 × a2 ×  a3"), "{table}");
    }

    /// A count the launch decides is one no table can state: an every-level over an extent that
    /// is not stamped yet says so rather than inventing a number.
    #[test]
    fn a_dynamic_axis_prints_a_question() {
        let partitioning = Partitioning::new(
            Space::new(&[(M, 512), (K, 4096)]).with_dynamic(&[K]),
            Levels::leaf(&[(M, 64), (K, 32)])
                .walk_every(&[K])
                .cubes(&[M])
                .build(),
        );
        let table = partitioning.table(&[(M, "m"), (K, "k")]).to_string();

        assert!(table.lines().any(|line| line.contains("× ?")), "{table}");
    }

    /// A level whose grid is shared still prints as the cube grid, and says who shares it.
    #[test]
    fn a_shared_level_still_prints_as_the_cube_grid() {
        let partitioning = Partitioning::new(
            Space::new(&[(M, 512), (N, 1024)]),
            Levels::leaf(&[(M, 64), (N, 64)])
                .cubes(&[M, N])
                .shared_by(8)
                .build(),
        );

        assert!(partitioning.to_string().contains('▣'));
    }

    /// An overhanging axis counts its partial tile, so the row still multiplies past the extent
    /// rather than losing it — and a block too narrow for its own rule is padded rather than
    /// widened, which keeps every cell where the header put it.
    #[test]
    fn an_overhanging_axis_counts_its_partial_tile() {
        let partitioning = Partitioning::new(
            Space::new(&[(M, 500), (K, 4096)]),
            Levels::leaf(&[(M, 128), (K, 32)])
                .walk_every(&[K])
                .cubes(&[M])
                .build(),
        );

        assert_eq!(
            partitioning.table(&[(M, "m"), (K, "k")]).to_string(),
            [
                "                      m ×   k      m ×    k",
                "",
                "  ◦                   · ×   ·    128 ×   32",
                "  ↻  128 steps        · × 128    128 × 4096",
                "  ▣  4 cubes          4 ×   ·    500 × 4096",
                "",
                "                  └─ count ─┘    └─ tile ─┘",
            ]
            .join("\n")
        );
    }
}
