use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords1d, CoordsDyn, Layout, LayoutExpand},
};

/// Maps a flat row-major index to an N-D coordinate over `shape`: the inverse of a
/// strided dot. Re-view a [`Window`]ed [`View`] through this to walk it linearly
/// (`shape()` is the element count) without re-deriving strides in the kernel.
#[derive(CubeType, Clone)]
pub struct Linear {
    shape: CoordsDyn,
}

#[cube]
impl Linear {
    pub fn new(shape: CoordsDyn) -> Self {
        Linear { shape }
    }
}

#[cube]
impl Layout for Linear {
    type Coordinates = Coords1d;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut out = CoordsDyn::new();

        #[unroll]
        for p in 0..self.shape.len() {
            let mut stride = 1u32;

            #[unroll]
            for q in p + 1..self.shape.len() {
                stride *= self.shape[q];
            }

            out.push((pos as u32 / stride) % self.shape[p]);
        }

        out
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        let mut total = 1u32;

        #[unroll]
        for p in 0..self.shape.len() {
            total *= self.shape[p];
        }

        total as usize
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos < self.shape()
    }
}
