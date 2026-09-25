//! The order the cube level hands its boxes to the grid ([`CubeOrder`]), and the joint decode of
//! the two in-plane positions it permutes.
//!
//! A [`WalkOrder`](crate::WalkOrder) is a different thing, and the cube level shows it: every
//! instance takes one tile, so the walk's `total` is one and there is no step order to state. A
//! [`CubeOrder`] permutes which *instance* holds which box: the positions the walk decodes from
//! the hardware.

use crate::{AxisDistribution, Coords, CubeAxis, Integer, IntegerExpand, Level, Space};
use cubecl::prelude::*;
use cubecl::std::tensor::layout::Coords2d;

/// The order a cube level distributes its boxes to the grid.
///
/// The grid's own order runs a wave of cubes along one axis: a band of one operand as wide as the
/// whole other axis. A swizzle folds it into a square-ish patch, so cubes running together share
/// rows and columns in the last level cache: nothing while a band fits, everything once none does.
///
/// **Any width serves any grid.** Where it does not divide the cube count of the axis it strips,
/// the last strip is only as wide as the boxes left ([`swizzle_ragged`]), so a width can stay what
/// a plan asked for rather than one fitted to each grid's count: the order is part of what a kernel
/// compiles, and a width that moved with the grid would compile a kernel per grid.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum CubeOrder {
    /// The box at `(x, y)` goes to the cube at `(x, y)`.
    #[default]
    RowMajor,
    /// Cubes snake down the second axis in strips `width` wide along the first.
    SwizzleRow(usize),
    /// Cubes snake along the first axis in strips `width` wide along the second.
    SwizzleCol(usize),
}

impl CubeOrder {
    /// A strip one box wide is the grid's own order, reached without the arithmetic.
    pub fn canonicalize(self) -> Self {
        match self {
            CubeOrder::SwizzleRow(1) | CubeOrder::SwizzleCol(1) => CubeOrder::RowMajor,
            other => other,
        }
    }

    /// Whether this order permutes anything, which is what decides that the two in-plane axes
    /// are decoded together rather than each from its own hardware dimension.
    pub fn swizzles(self) -> bool {
        !matches!(self.canonicalize(), CubeOrder::RowMajor)
    }
}

/// The two in-plane positions this order gives the cube at flat dispatch index `flat`, over a
/// grid of `cubes` boxes.
///
/// `flat` is the grid's own linear order, the one the hardware starts cubes in, which is what
/// makes the permutation say anything about what runs together.
#[cube]
pub(crate) fn cube_positions(
    flat: usize,
    cubes: (usize, usize),
    #[comptime] order: CubeOrder,
) -> (usize, usize) {
    let (count_x, count_y) = cubes;
    match comptime!(order.canonicalize()) {
        CubeOrder::RowMajor => (flat.remainder(count_x), flat.divided_by(count_x)),
        CubeOrder::SwizzleRow(width) => {
            let (step, along) = swizzle_ragged(flat, count_y, comptime!(width as u32), count_x);
            (along as usize, step as usize)
        }
        CubeOrder::SwizzleCol(width) => {
            let (step, along) = swizzle_ragged(flat, count_x, comptime!(width as u32), count_y);
            (step as usize, along as usize)
        }
    }
}

/// [`swizzle`] over strips cut from an axis of `strip_axis` elements that `step_length` need not
/// divide: every strip is `step_length` wide but the last, which holds what is left of the axis
/// and snakes the same way. Where `step_length` divides the axis this is [`swizzle`].
///
/// `index` lies in the grid, below `num_steps * strip_axis`; past it the position lands off the
/// grid, without a division by zero or an overflow.
#[cube]
pub fn swizzle_ragged(
    index: usize,
    num_steps: usize,
    #[comptime] step_length: u32,
    strip_axis: usize,
) -> Coords2d {
    comptime!(assert!(
        step_length > 0,
        "swizzle: a strip holds at least one box"
    ));
    let full = num_steps as u32 * step_length;
    let index = index as u32;
    let strip_index = index / full;
    let strip_offset = step_length * strip_index;
    // What is left of the axis: fewer than a strip only in the ragged last one. An index past the
    // grid leaves none, and takes a whole width rather than divide by it.
    let left = (strip_axis as u32).saturating_sub(strip_offset);
    let width = if left < step_length && left > 0 {
        left
    } else {
        step_length.runtime()
    };
    snake(
        strip_index,
        index - strip_index * full,
        num_steps,
        width,
        strip_offset,
    )
}

#[cube]
/// Maps a linear `index` to 2D zigzag coordinates `(x, y)` within horizontal or vertical strips.
///
/// Each strip is made of `num_steps` steps, each of length `step_length`.
/// Strips alternate direction: even strips go top-down, odd strips bottom-up.
/// Steps alternate direction: even steps go left-to-right, odd steps right-to-left.
///
/// - Prefer **odd `num_steps`** for smoother transitions between strips.
///
/// # Parameters
///
/// - `index`: linear input index
/// - `num_steps`: number of snaking steps in a strip
/// - `step_length`: number of elements in each step (must be > 0)
///
/// # Returns
/// `(x, y)` coordinates after swizzling
pub fn swizzle(index: usize, num_steps: usize, #[comptime] step_length: u32) -> Coords2d {
    comptime!(assert!(
        step_length > 0,
        "swizzle: a strip holds at least one box"
    ));
    let full = num_steps as u32 * step_length;
    let index = index as u32;
    let strip_index = index / full;
    // Every strip whole, so the width is the comptime one and its divisions are by a constant.
    snake(
        strip_index,
        index - strip_index * full,
        num_steps,
        step_length,
        step_length * strip_index,
    )
}

/// The snake both swizzles walk: position `pos_in_strip` of strip `strip_index`, `width` boxes
/// wide from `strip_offset`, top-down or bottom-up by strip and left-right or right-left by step.
#[cube]
fn snake(
    strip_index: u32,
    pos_in_strip: u32,
    num_steps: usize,
    width: u32,
    strip_offset: u32,
) -> Coords2d {
    let abs_step_index = pos_in_strip / width;
    let abs_pos_in_step = pos_in_strip % width;

    // Top-down (0) or bottom-up (1), then left-right (0) or right-left (1).
    let strip_direction = strip_index % 2;
    let step_direction = abs_step_index % 2;

    let step_index = strip_direction * (num_steps as u32 - abs_step_index - 1)
        + (1 - strip_direction) * abs_step_index;
    let pos_in_step =
        step_direction * (width - abs_pos_in_step - 1) + (1 - step_direction) * abs_pos_in_step;

    (step_index, pos_in_step + strip_offset)
}

/// The two in-plane positions of a cube level that distributes its boxes in an order other than the
/// grid's own, decoded together from the flat dispatch index ([`CubeOrder`]).
///
/// Zeros where no order is stated, which is every other level and every walk: the branch is
/// comptime, so nothing of this reaches a kernel that did not ask for it.
#[cube]
pub(crate) fn swizzled_positions(
    #[comptime] space: Space,
    #[comptime] level: Level,
    instances: &Coords<usize>,
) -> Coords<usize> {
    let mut out = Coords::<usize>::new();
    if comptime!(level.order().swizzles()) {
        let (x_at, y_at) = comptime!(in_plane_axes(&space, &level));
        let (count_x, count_y) = (instances.at(x_at), instances.at(y_at));
        // The grid's own linear order, which is the order the hardware starts cubes in and so
        // the one a permutation of it can say anything about.
        let flat =
            CubeAxis::position(CubeAxis::X).plus(CubeAxis::position(CubeAxis::Y).times(count_x));
        let (x, y) = cube_positions(flat, (count_x, count_y), comptime!(level.order()));
        out.push(x);
        out.push(y);
    } else {
        out.push(0usize);
        out.push(0usize);
    }
    out
}

/// Where this space holds the cube level's two in-plane axes, `(x, y)`.
///
/// Refuses what the joint decode cannot state: an axis not owning its whole grid dimension. A
/// swizzle permutes the grid, and a dimension shared by axes ([`Level::shared_by`], a batch axis
/// folded onto an in-plane dimension) carries digits this has no way to put back.
pub(crate) fn in_plane_axes(space: &Space, level: &Level) -> (usize, usize) {
    let at = |wanted: CubeAxis| {
        let found: Vec<usize> = (0..space.rank())
            .filter(|&p| level.cube_axis(space.axis_at(p)) == Some(wanted))
            .collect();
        assert_eq!(
            found.len(),
            1,
            "Walk: a {:?} cube level distributes the grid's {wanted:?} dimension to {} of its axes, \
             and a swizzle can put back only an axis that owns a whole dimension",
            level.order(),
            found.len()
        );
        let p = found[0];
        assert_eq!(
            AxisDistribution::unspanned_weight(level, space, space.axis_at(p)),
            1,
            "Walk: a {:?} cube level shares the grid's {wanted:?} dimension with an axis this \
             space does not span, whose digit a swizzle has no way to put back",
            level.order()
        );
        p
    };
    (at(CubeAxis::X), at(CubeAxis::Y))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one property an order owes: over a grid, it is a permutation — every box is held by
    /// exactly one cube. A swizzle that repeats a box computes it twice and drops another, and
    /// that reads as a wrong product, not a slow one.
    ///
    /// Host-side, because the arithmetic is the device's only by where it runs. Over grids the
    /// widths divide and grids they do not, whose last strip is ragged.
    #[test]
    fn every_order_is_a_permutation_of_the_grid() {
        for &(x, y) in &[
            (8usize, 8usize),
            (16, 4),
            (4, 16),
            (32, 8),
            (6, 10),
            (1, 7),
            (7, 5),
            (10, 3),
            (3, 9),
            (13, 1),
        ] {
            for order in [
                CubeOrder::RowMajor,
                CubeOrder::SwizzleRow(2),
                CubeOrder::SwizzleRow(4),
                CubeOrder::SwizzleCol(2),
                CubeOrder::SwizzleCol(8),
            ] {
                let mut seen = vec![false; x * y];
                for flat in 0..x * y {
                    let (px, py) = positions_host(flat, (x, y), order);
                    assert!(
                        px < x && py < y,
                        "{order:?} on {x}x{y}: ({px}, {py}) is off the grid"
                    );
                    let at = py * x + px;
                    assert!(
                        !seen[at],
                        "{order:?} on {x}x{y}: two cubes hold ({px}, {py})"
                    );
                    seen[at] = true;
                }
            }
        }
    }

    /// A strip one box wide is the grid's own order, and says so rather than paying for the
    /// arithmetic that would produce it.
    #[test]
    fn a_strip_one_box_wide_is_the_grids_own_order() {
        assert_eq!(CubeOrder::SwizzleRow(1).canonicalize(), CubeOrder::RowMajor);
        assert_eq!(CubeOrder::SwizzleCol(1).canonicalize(), CubeOrder::RowMajor);
        assert!(!CubeOrder::SwizzleRow(1).swizzles());
        assert!(CubeOrder::SwizzleRow(4).swizzles());
    }

    /// An index past the grid lands off it, but the strips there leave no boxes, which must not
    /// become a width of zero to divide by nor an axis overdrawn: a dispatch larger than its grid
    /// is guarded by its caller, not trapped here.
    #[test]
    fn an_index_past_the_grid_lands_off_it() {
        // A 4x3 grid in strips of 2: index 12 is where a third strip would start, and 18 a fourth,
        // wholly past the axis.
        for index in [12, 18, 30] {
            let (_, along) = swizzle_ragged(index, 3, 2, 4);
            assert!(
                along >= 4,
                "index {index}, past the grid, landed on it at {along}"
            );
        }
    }

    /// Consecutive cubes stay close: what the order exists for, stated as the thing a cache
    /// would notice. Over a 16x16 grid, the first sixteen cubes of `SwizzleRow(4)` span four
    /// columns where the grid's own order spans sixteen.
    #[test]
    fn a_swizzle_keeps_the_cubes_that_run_together_close() {
        let span = |order: CubeOrder| {
            let boxes: Vec<_> = (0..16)
                .map(|f| positions_host(f, (16, 16), order))
                .collect();
            let xs = boxes.iter().map(|b| b.0);
            xs.clone().max().unwrap() - xs.min().unwrap() + 1
        };
        assert_eq!(span(CubeOrder::RowMajor), 16);
        assert_eq!(span(CubeOrder::SwizzleRow(4)), 4);
    }

    /// The host twin of [`cube_positions`], which is `#[cube]` and so cannot be called from a
    /// test — but [`swizzle_ragged`] can, so only the dispatch is written twice and never the
    /// arithmetic the property is about.
    fn positions_host(flat: usize, cubes: (usize, usize), order: CubeOrder) -> (usize, usize) {
        let (count_x, count_y) = cubes;
        match order.canonicalize() {
            CubeOrder::RowMajor => (flat % count_x, flat / count_x),
            CubeOrder::SwizzleRow(width) => {
                let (step, along) = swizzle_ragged(flat, count_y, width as u32, count_x);
                (along as usize, step as usize)
            }
            CubeOrder::SwizzleCol(width) => {
                let (step, along) = swizzle_ragged(flat, count_x, width as u32, count_y);
                (step as usize, along as usize)
            }
        }
    }
}
