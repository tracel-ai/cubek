//! The order the cube level hands its boxes to the grid ([`CubeOrder`]), and the joint decode of
//! the two in-plane positions it permutes.
//!
//! A [`WalkOrder`](crate::WalkOrder) is a different thing, and the cube level shows it: every
//! instance takes one tile, so the walk's `total` is one and there is no step order to state. A
//! [`CubeOrder`] permutes which *instance* holds which box: the positions the walk decodes from
//! the hardware.

use crate::{ComputeScope, Coords, CubeAxis, Known, KnownExpand, Level, Space, hardware_pos};
use cubecl::prelude::*;
use cubecl::std::tensor::layout::Coords2d;

/// The order a cube level deals its boxes to the grid.
///
/// The grid's own order runs a wave of cubes along one axis: a band of one operand as wide as the
/// whole other axis. A swizzle folds it into a square-ish patch, so cubes running together share
/// rows and columns in the last level cache: nothing while a band fits, everything once none does.
///
/// **The width must divide the cube count of the axis it strips**, or the last strip runs past
/// the grid and two boxes answer to one cube. The kernel cannot check it (the counts are the
/// launch's), so the caller states it, and that routine checks it ([`CubeOrder::divides`]).
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

    /// Whether this order is a bijection over a grid of `cubes` boxes on the axis it strips —
    /// `x` for [`SwizzleRow`](CubeOrder::SwizzleRow), `y` for the other — which is the one
    /// thing the kernel cannot check for itself.
    pub fn divides(self, cubes: (usize, usize)) -> bool {
        let (x, y) = cubes;
        match self.canonicalize() {
            CubeOrder::RowMajor => true,
            CubeOrder::SwizzleRow(width) => width > 0 && x.is_multiple_of(width),
            CubeOrder::SwizzleCol(width) => width > 0 && y.is_multiple_of(width),
        }
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
            let (step, along) = swizzle(flat, count_y, comptime!(width as u32));
            (along as usize, step as usize)
        }
        CubeOrder::SwizzleCol(width) => {
            let (step, along) = swizzle(flat, count_x, comptime!(width as u32));
            (step as usize, along as usize)
        }
    }
}

#[cube]
/// Maps a linear `index` to 2D zigzag coordinates `(x, y)` within horizontal or vertical strips.
///
/// Each strip is made of `num_steps` steps, each of length `step_length`.
/// Strips alternate direction: even strips go top-down, odd strips bottom-up.
/// Steps alternate direction: even steps go left-to-right, odd steps right-to-left.
///
/// - Prefer **odd `num_steps`** for smoother transitions between strips.
/// - Prefer **power-of-two `step_length`** for better performance.
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
    comptime!(assert!(step_length > 0));

    let num_elements_per_strip = num_steps * step_length as usize;
    let strip_index = (index / num_elements_per_strip) as u32;
    let pos_in_strip = (index % num_elements_per_strip) as u32;
    let strip_offset = step_length * strip_index;

    // Indices without regards to direction
    let abs_step_index = pos_in_strip / step_length;
    let abs_pos_in_step = pos_in_strip % step_length;

    // Top-down (0) or Bottom-up (1)
    let strip_direction = strip_index % 2;
    // Left-right (0) or Right-left (1)
    let step_direction = abs_step_index % 2;

    // Update indices with direction
    let step_index = strip_direction * (num_steps as u32 - abs_step_index - 1)
        + (1 - strip_direction) * abs_step_index;

    let pos_in_step = if step_length & (step_length - 1) == 0 {
        abs_pos_in_step ^ (step_direction * (step_length - 1))
    } else {
        step_direction * (step_length - abs_pos_in_step - 1)
            + (1 - step_direction) * abs_pos_in_step
    };

    (step_index, pos_in_step + strip_offset)
}

/// The two in-plane positions of a cube level that deals its boxes in an order other than the
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
        let flat = hardware_pos(comptime!(ComputeScope::Cube(CubeAxis::X)))
            .plus(hardware_pos(comptime!(ComputeScope::Cube(CubeAxis::Y))).times(count_x));
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
            .filter(|&p| {
                level.distribution(space.axis_at(p)).scope() == Some(ComputeScope::Cube(wanted))
            })
            .collect();
        assert_eq!(
            found.len(),
            1,
            "Walk: a {:?} cube level deals the grid's {wanted:?} dimension to {} of its axes, \
             and a swizzle can put back only an axis that owns a whole dimension",
            level.order(),
            found.len()
        );
        let p = found[0];
        assert_eq!(
            level.inner_weight_unspanned(space, space.axis_at(p)),
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
    /// Host-side, because the arithmetic is the device's only by where it runs.
    #[test]
    fn every_order_that_divides_is_a_permutation_of_the_grid() {
        for &(x, y) in &[(8usize, 8usize), (16, 4), (4, 16), (32, 8), (6, 10), (1, 7)] {
            for order in [
                CubeOrder::RowMajor,
                CubeOrder::SwizzleRow(2),
                CubeOrder::SwizzleRow(4),
                CubeOrder::SwizzleCol(2),
                CubeOrder::SwizzleCol(8),
            ] {
                if !order.divides((x, y)) {
                    continue;
                }
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

    /// The width divides the axis it strips, and only that axis: the other is walked whole
    /// inside a step, so nothing there has to divide.
    #[test]
    fn the_width_divides_the_axis_it_strips() {
        assert!(CubeOrder::SwizzleRow(4).divides((8, 6)));
        assert!(!CubeOrder::SwizzleRow(4).divides((6, 8)));
        assert!(CubeOrder::SwizzleCol(4).divides((6, 8)));
        assert!(!CubeOrder::SwizzleCol(4).divides((8, 6)));
        assert!(
            CubeOrder::RowMajor.divides((7, 13)),
            "the grid's own order always does"
        );
        assert!(
            !CubeOrder::SwizzleRow(0).divides((8, 8)),
            "a strip holds something"
        );
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
    /// test — but [`swizzle`] can, so only the dispatch is written twice and never the
    /// arithmetic the property is about.
    fn positions_host(flat: usize, cubes: (usize, usize), order: CubeOrder) -> (usize, usize) {
        let (count_x, count_y) = cubes;
        match order.canonicalize() {
            CubeOrder::RowMajor => (flat % count_x, flat / count_x),
            CubeOrder::SwizzleRow(width) => {
                let (step, along) = swizzle(flat, count_y, width as u32);
                (along as usize, step as usize)
            }
            CubeOrder::SwizzleCol(width) => {
                let (step, along) = swizzle(flat, count_x, width as u32);
                (step as usize, along as usize)
            }
        }
    }
}
