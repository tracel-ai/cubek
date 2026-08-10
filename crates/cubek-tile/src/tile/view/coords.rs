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
