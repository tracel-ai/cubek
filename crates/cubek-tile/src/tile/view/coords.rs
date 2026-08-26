//! The coordinate arithmetic the tile's [`Layout`](cubecl::std::tensor::layout::Layout)s share:
//! unraveling a flat index over a group of extents, joining two groups, and the box test each
//! layout answers `is_in_bounds` with. Kept apart from the layouts themselves so a reshaper and a
//! projection reach the same `unravel` rather than each spelling one out.

use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords2d, CoordsDyn},
};

use crate::*;

/// Unravels a flat row-major index into coordinates over the given extents: entry `p` is
/// `i / extents[p+1..].product() % extents[p]`.
///
/// The leading entry skips the modulo: an index within the box never overflows it, and dropping
/// the operation lets the divide fold when the extents are constant.
#[cube]
pub(crate) fn unravel(extents: &Coords<u32>, i: u32) -> Coords<u32> {
    let n = extents.len();
    let mut out = Coords::<u32>::new();

    #[unroll]
    for p in 0..n {
        let digit = i.fdiv(extents.fproduct(comptime!(((p + 1)..n).collect::<Vec<_>>())));
        if comptime!(p == 0) {
            out.push(digit);
        } else {
            out.push(digit.frem(extents.at(p)));
        }
    }

    out
}

/// [`unravel`] over extents that are known at comptime.
///
/// The same arithmetic, but the divisor and the modulus are constants rather than the `runtime()`
/// entries a [`const_coords`] group carries, so each digit folds instead of being computed. What a
/// caller unraveling a comptime index nest wants; the runtime form stays for extents read off a
/// layout.
#[cube]
pub(crate) fn unravel_const(#[comptime] extents: Vec<usize>, i: u32) -> Coords<u32> {
    let n = comptime!(extents.len());
    let mut out = Coords::<u32>::new();

    #[unroll]
    for p in 0..n {
        let digit = i.fdiv(comptime!(extents[p + 1..].iter().product::<usize>() as u32));
        if comptime!(p == 0) {
            out.push(digit);
        } else {
            out.push(digit.frem(comptime!(extents[p] as u32)));
        }
    }

    out
}

/// Concatenates two coordinate lists into dynamic coordinates.
#[cube]
pub(crate) fn concat(leading: &Coords<u32>, trailing: &Coords<u32>) -> CoordsDyn {
    let mut out = CoordsDyn::new();

    #[unroll]
    for p in 0..leading.len() {
        out.push(leading.at(p));
    }
    #[unroll]
    for p in 0..trailing.len() {
        out.push(trailing.at(p));
    }

    out
}

/// Whether every coordinate of `pos` falls inside `shape`.
#[cube]
pub(crate) fn within(shape: &Coords<u32>, pos: CoordsDyn) -> bool {
    let mut valid = true;

    #[unroll]
    for p in 0..shape.len() {
        valid = valid && pos[p] < shape.at(p);
    }

    valid
}

/// [`within`] at rank two, where the edges are a pair rather than a list.
#[cube]
pub(crate) fn within_2d(pos: Coords2d, shape: Coords2d) -> bool {
    let (row, col) = pos;
    let (rows, cols) = shape;
    row < rows && col < cols
}

/// The coordinate `operand` is read at, one entry per axis of its own space: an axis present in
/// `acc` takes its coordinate from `acc_coords`, everything else (contracted by definition, since
/// every operand axis falls in one or the other) comes from `reduce_coords`.
///
/// `width` is the operand's line width; only its innermost axis is addressed in lines (matching an
/// `nd`/[`matrix_transparent`](crate::Tile::matrix_transparent) view), so it alone divides by
/// `width`. `scale_acc_branch` decides whether that division also applies when the fastest axis
/// falls in the acc branch: a caller whose `acc_coords` already hold raw element coordinates
/// (reduce's accumulator cell) needs it there too; a caller whose acc-branch coordinate arrives
/// pre-divided into a line index by construction (mma's `col`, the gather leaf's own `nr`-loop
/// step) does not, and must pass `false` to avoid dividing twice.
#[cube]
pub(crate) fn resolve_nd_coords(
    #[comptime] operand: Space,
    #[comptime] acc: Space,
    #[comptime] reduce: Vec<Axis>,
    acc_coords: &Coords<u32>,
    reduce_coords: &Coords<u32>,
    #[comptime] width: usize,
    #[comptime] scale_acc_branch: bool,
) -> CoordsDyn {
    let operand_rank = comptime!(operand.rank());
    let mut out = CoordsDyn::new();

    #[unroll]
    for p in 0..operand_rank {
        let axis = comptime!(operand.axis_at(p));
        let in_acc = comptime!(acc.contains(axis));
        let raw_coord = if comptime!(in_acc) {
            let pos = comptime!(acc.position(axis));
            acc_coords.at(comptime!(pos))
        } else {
            let pos = comptime!(reduce.iter().position(|&r| r == axis).unwrap());
            reduce_coords.at(comptime!(pos))
        };
        let divides =
            comptime!(p == operand_rank - 1 && width > 1 && (scale_acc_branch || !in_acc));
        let coord = if comptime!(divides) {
            raw_coord.fdiv(comptime!(width as u32))
        } else {
            raw_coord
        };
        out.push(coord);
    }

    out
}
