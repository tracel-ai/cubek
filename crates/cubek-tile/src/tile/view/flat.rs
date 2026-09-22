//! The flat 1-D view over a [`Tile`]. [`FlatLayout`] is a [`Layout`] that re-views the tile's N-D
//! [`Space`] as one row-major [`Coords1d`] index; [`Tile::flat`]/[`Tile::flat_mut`] wrap it as a
//! [`FlatView`]/[`FlatViewMut`] (a [`MaskedView`] with the comptime overhang-`check` flag).

use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords1d, CoordsDyn, Layout, LayoutExpand},
};

use crate::*;

/// A masked 1-D ([`FlatLayout`]) view: a flat row-major scan over a [`Tile`].
pub type FlatView<'a, T> = MaskedView<'a, T, Coords1d>;
/// The mutable twin of [`FlatView`].
pub type FlatViewMut<'a, T> = MaskedViewMut<'a, T, Coords1d>;

/// Maps a flat row-major index to an N-D coordinate over `shape` ([`unravel`]): the inverse of a
/// strided dot. Re-view a [`Window`]ed [`View`](cubecl::std::tensor::View) through this to walk it
/// linearly (`shape()` is the element count). A static window's extents make the divisors constant.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct FlatLayout {
    shape: Coords<u32>,
}

#[cube]
impl FlatLayout {
    pub fn new(shape: Coords<u32>) -> Self {
        FlatLayout { shape }
    }
}

#[cube]
impl Layout for FlatLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        unravel(&self.shape, pos as u32).to_dyn()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        let rank = self.shape.len();
        self.shape
            .fproduct(comptime!((0..rank).collect::<Vec<_>>()))
            .fcast::<usize>()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos < self.shape()
    }
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// A flat 1-D view over `Vector<T, W>` lines (`W` = [`vector_size`](Tile::vector_size)): a
    /// row-major scan over the tile's window, masking the overhang per its comptime `check` flag.
    /// A packed store is refused: it unpacks under the fill ([`Tile::copy_from`]) and packed views.
    pub fn flat<W: Size>(&self) -> FlatView<'_, Vector<T, W>> {
        let g = self.mem("flat");
        if comptime!(g.store.packing != Packing::Plain) {
            panic!("Tile::flat: a packed tile only unpacks under Tile::copy_from")
        }
        g.flat::<W>()
    }

    /// The mutable twin of [`flat`](Tile::flat). Public because a routine outside this crate that
    /// computes its cells (a fold's drain, a delta in place) has no other verb for "each unit
    /// writes its own flat positions"; [`copy_from`](Tile::copy_from) chooses the value per cell.
    pub fn flat_mut<W: Size>(&mut self) -> FlatViewMut<'_, Vector<T, W>> {
        self.mem_mut("flat_mut").flat_mut::<W>()
    }
}
