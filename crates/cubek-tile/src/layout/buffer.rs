//! The runtime half of a [`Projection`] over one buffer: its shape and strides, dotted in-kernel.

use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords1d, CoordsDyn, Layout, LayoutExpand},
};

use crate::*;

/// In-kernel twin of cubecl's `TiledViewLayout`: splits each coordinate into the digits its
/// [`projection`](Projection) spreads over the physical axes, then dots the physical strides. The
/// arithmetic folds, so a static store dots by constants and an untiled projection is a plain dot.
///
/// `Coordinates` are already physical (a gather is resolved a layer up, by [`ProjectionInKernel`]), so
/// `projection` is [`Projection::positional`]'s synthetic per-position map, not the operand's own.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct BufferLayout {
    pub(crate) physical_shape: Coords<u32>,
    pub(crate) physical_strides: Coords<u32>,
    #[cube(comptime)]
    pub(crate) projection: Projection,
    /// Where a stage keeps each line of a block row; [`InOrder`](RowArrangement::InOrder) for every
    /// other buffer.
    #[cube(comptime)]
    pub(crate) rows: RowArrangement,
}

#[cube]
impl Layout for BufferLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let rank = comptime!(self.projection.physical_rank());
        let mut digits = Coords::<u32>::new();
        #[unroll]
        for pa in 0..rank {
            digits.push(self.physical_digit(&pos, pa));
        }
        let digits = self.placed(&digits);
        // Per-physical-axis terms, summed below (chained, so a static store's dot folds).
        let mut terms = Sequence::<u32>::new();
        #[unroll]
        for pa in 0..rank {
            terms.push(digits.at(pa).times(self.physical_strides.at(pa)));
        }
        terms
            .sum(comptime!((0..rank).collect::<Vec<_>>()))
            .cast::<usize>()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        logical_extent(comptime!(self.projection.clone()), &self.physical_shape).to_dyn()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let bounds = self.shape();
        let mut valid = true;

        #[unroll]
        for i in 0..bounds.len() {
            valid = valid && pos[i] < bounds[i];
        }

        valid
    }
}

#[cube]
impl BufferLayout {
    /// The logical coordinate line `i` of this buffer holds, `i` counting its lines over its
    /// physical extents: [`to_source_pos`](Layout::to_source_pos) run backwards, which is what a
    /// fill that walks the buffer's lines reads its source at. A swizzled row holds, at line `i`,
    /// the line its row's key moved there; the XOR is its own inverse, so the same arrangement
    /// finds it.
    pub(crate) fn line_coords(&self, i: usize) -> CoordsDyn {
        let digits = self.placed(&self.line_digits(i));
        fold_physical(
            comptime!(self.projection.clone()),
            &digits,
            &self.physical_shape,
        )
    }

    /// Where line `i` of this buffer sits, `i` counting its lines over its physical extents: `i`
    /// itself, unless its rows are followed by padding ([`RowArrangement::Padded`]), when the
    /// strides step past it.
    pub(crate) fn line_offset(&self, i: usize) -> usize {
        if comptime!(self.rows.is_pitched()) {
            let digits = self.line_digits(i);
            let mut offset = 0u32;
            #[unroll]
            for pa in 0..digits.len() {
                offset = offset.plus(digits.at(pa).times(self.physical_strides.at(pa)));
            }
            offset.cast::<usize>()
        } else {
            i
        }
    }

    /// The digits line `i` of this buffer sits at, one per physical axis, under the row-major
    /// suffix products of its extents.
    fn line_digits(&self, i: usize) -> Coords<u32> {
        let x = i.cast::<u32>();
        let mut digits = Coords::<u32>::new();
        #[unroll]
        for pa in 0..self.physical_shape.len() {
            digits.push(line_digit(x, &self.physical_shape, pa));
        }
        digits
    }

    /// `digits`, one per physical axis, with a block row's line moved to where this buffer's rows
    /// keep it: a swizzled stage's line digit XORed by its row's key, every other digit as it is.
    /// The one place a swizzle is applied, by the buffer's reads and its fills alike.
    fn placed(&self, digits: &Coords<u32>) -> Coords<u32> {
        let mut placed = Coords::<u32>::new();
        #[unroll]
        for pa in 0..digits.len() {
            let mut digit = digits.at(pa);
            // A `match`, not an `if let`: the cube macro branches on a comptime value through a
            // match.
            #[allow(clippy::single_match)]
            match comptime!(self.rows.swizzle_along(pa)) {
                Some(swizzle) => {
                    digit = swizzled_line(swizzle, digit, digits.at(comptime!(swizzle.row_axis())));
                }
                None => {}
            }
            placed.push(digit);
        }
        placed
    }

    /// The digit `pos` sits at along physical axis `pa`: each coordinate's digits spread over the
    /// physical axes it carries, scaled by its term.
    fn physical_digit(&self, pos: &CoordsDyn, #[comptime] pa: usize) -> u32 {
        let map = comptime!(self.projection.physical_axis(pa).clone());
        let picks = comptime!((0..map.terms().len()).collect::<Vec<_>>());
        // Almost always one term (only a gather layers several onto one physical axis, and
        // `BufferLayout`'s own map never does); summed the same way regardless.
        let mut parts = Sequence::<u32>::new();
        #[unroll]
        for t in 0..comptime!(map.terms().len()) {
            let term = comptime!(map.terms()[t]);
            let p = comptime!(self.projection.position(term.axis));
            let (finer, modulo) = comptime!(self.projection.digit(pa, term.axis));
            // Strip the finer digits, then take this one. The outermost fragment of an axis
            // (and any untiled axis) has no radix and keeps the full quotient.
            let quot = pos[p].divided_by(self.physical_shape.product(comptime!(finer.to_vec())));
            let digit = match comptime!(modulo) {
                Some(m) => quot.remainder(self.physical_shape.at(m)),
                None => quot,
            };
            parts.push(digit.times(comptime!(term.scale.get() as u32).runtime()));
        }
        parts.sum(picks)
    }
}
