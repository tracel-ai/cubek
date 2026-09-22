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
/// `projection` is [`Projection::of_tiling`]'s synthetic per-position map, not the operand's own.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct BufferLayout {
    pub(crate) physical_shape: Coords<u32>,
    pub(crate) physical_strides: Coords<u32>,
    #[cube(comptime)]
    pub(crate) projection: Projection,
}

#[cube]
impl Layout for BufferLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        // Per-physical-axis terms, summed below (chained, so a static store's dot folds).
        let mut terms = Sequence::<u32>::new();
        let rank = comptime!(self.projection.physical_rank());
        #[unroll]
        for pa in 0..rank {
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
                let quot =
                    pos[p].divided_by(self.physical_shape.product(comptime!(finer.to_vec())));
                let digit = match comptime!(modulo) {
                    Some(m) => quot.remainder(self.physical_shape.at(m)),
                    None => quot,
                };
                parts.push(digit.times(comptime!(term.scale.get() as u32).runtime()));
            }
            terms.push(parts.sum(picks).times(self.physical_strides.at(pa)));
        }
        terms
            .sum(comptime!((0..rank).collect::<Vec<_>>()))
            .retyped::<usize>()
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
