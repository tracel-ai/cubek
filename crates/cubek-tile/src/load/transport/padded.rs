//! Assembling one destination line out of scalar source cells, which is what a stage served in
//! lines its source cannot hand out whole is filled by, and the coordinate arithmetic under it.

use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

use crate::*;

/// Read one destination line from the masked source view at `pos`: whole for a 1:1 copy, or
/// assembled unit by unit from scalar source cells for a padded stage ([`widen_line`]).
#[cube]
pub(crate) fn read_stage_line<I2: Numeric, WP2: Size, SW: Size>(
    s: &Masked<'_, Vector<I2, SW>, CoordsDyn>,
    pos: &CoordsDyn,
    #[comptime] padding: Option<Padding>,
) -> Vector<I2, WP2> {
    if comptime!(padding.is_some()) {
        widen_line::<I2, WP2, SW>(s, pos, comptime!(padding.unwrap()))
    } else {
        // The unpadded caller builds its view at the destination's own width, so `SW` *is* `WP2`
        // here and the cast is an identity the trace folds away; the two only differ as types.
        Vector::<I2, WP2>::cast_from(s.read(pos.clone()))
    }
}

/// The logical coordinate of physical line `i` in a `[grid…, tile…]` store: decode `i` into one
/// digit per physical axis ([`line_digit`]), then [`fold_physical`] folds a storage-tiled axis's
/// digits back into one off `projection`'s own div/modulo (`BufferLayout`'s map, invertible).
#[cube]
pub(crate) fn physical_pos(
    #[comptime] projection: Projection,
    #[comptime] rows: RowPlacement,
    i: usize,
    shape: &Coords<u32>,
) -> CoordsDyn {
    let x = i.cast::<u32>();
    let mut digits = Coords::<u32>::new();
    #[unroll]
    for j in 0..shape.len() {
        digits.push(line_digit(x, shape, j));
    }
    // Line `i` of a swizzled stage holds the line its row's key moved there; the XOR is its own
    // inverse, so the same placement finds it.
    fold_physical(comptime!(projection), &placed_digits(rows, &digits), shape)
}

/// Assemble one padded destination line from adjacent scalar source cells.
///
/// When `Padding::units` is `None` (a `Dynamic` innermost extent), the source window must be
/// bounds-checked so that reads past the extent return zero. When it is `Some(n)`, reads past `n`
/// are masked off explicitly so the padding units keep the zero they start at.
#[cube]
pub(crate) fn widen_line<T: Numeric, W: Size, SW: Size>(
    s: &Masked<'_, Vector<T, SW>, CoordsDyn>,
    pos: &CoordsDyn,
    #[comptime] padding: Padding,
) -> Vector<T, W> {
    let width = comptime!(padding.width);
    let rank = comptime!(padding.rank);
    comptime!(assert!(
        SW::try_value_const() == Some(1),
        "widen_line: a padded stage is filled from a scalar source, got a {:?}-wide one",
        SW::try_value_const()
    ));
    comptime!(assert!(
        W::try_value_const().is_none_or(|n| n == width),
        "widen_line: assembles {width} units into a {:?}-wide destination line",
        W::try_value_const()
    ));
    let last = comptime!(rank - 1);
    let line = pos[last];
    let mut out = Vector::<T, W>::cast_from(T::from_int(0));
    let guarded = comptime!(match padding.extent {
        Some(n) => !n.is_multiple_of(width),
        None => false,
    });
    #[unroll]
    for l in 0..width {
        let cell = line
            .times(comptime!(width as u32))
            .plus(comptime!(l as u32));
        let valid = if comptime!(guarded) {
            cell < comptime!(padding.extent.unwrap() as u32)
        } else {
            true.runtime()
        };
        if valid {
            out.insert(
                l,
                s.read(source_component(pos, comptime!(rank), cell))
                    .extract(0usize),
            );
        }
    }
    out
}

/// Replace the destination line coordinate with its scalar source-cell coordinate.
#[cube]
pub(crate) fn source_component(pos: &CoordsDyn, #[comptime] rank: usize, cell: u32) -> CoordsDyn {
    let mut out = CoordsDyn::new();
    #[unroll]
    for p in 0..rank {
        if comptime!(p == rank - 1) {
            out.push(cell);
        } else {
            out.push(pos[p]);
        }
    }
    out
}

/// Express a padded destination's innermost physical extent in scalar source elements.
#[cube]
pub(crate) fn widened_shape(
    shape: &Coords<u32>,
    #[comptime] rank: usize,
    #[comptime] width: usize,
) -> Coords<u32> {
    let mut out = Coords::<u32>::new();
    #[unroll]
    for p in 0..rank {
        if comptime!(p == rank - 1) {
            out.push(shape.at(p).times(comptime!(width as u32)));
        } else {
            out.push(shape.at(p));
        }
    }
    out
}
