use cubecl::prelude::*;
use cubecl::std::tensor::layout::*;

use crate::launch::BatchedCoords;

/// Slice the layout at a specific batch, and reduce its dimensionality
/// Not general enough to be in cubecl-std
#[derive(CubeType, Clone, Copy)]
pub struct SliceIndex {
    offset: usize,
    shape: Coords2d,
}

#[cube]
impl SliceIndex {
    pub fn new(offset: usize, shape: BatchedCoords) -> Self {
        let (_, rows, cols) = shape;
        SliceIndex {
            offset,
            shape: (rows, cols),
        }
    }
}

#[cube]
impl Layout for SliceIndex {
    type Coordinates = Coords2d;
    type SourceCoordinates = (usize, u32, u32);

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (row, col) = pos;
        (self.offset, row, col)
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        // we don't check batch
        true.runtime()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape
    }
}
