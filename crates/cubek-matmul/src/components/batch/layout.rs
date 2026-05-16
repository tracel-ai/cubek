use cubecl::{prelude::*, std::tensor::layout::*};

use crate::launch::BatchedCoords;

/// 2D view onto a single batch slice, with bounds-checking against the
/// supplied shape. Used by the plane-parallel and outer-product kernels.
#[derive(CubeType, Clone, Copy)]
pub struct MatLayout {
    batch: usize,
    shape: Coords2d,
}

#[cube]
impl MatLayout {
    pub fn new(batch: usize, shape: Coords2d) -> Self {
        MatLayout { batch, shape }
    }
}

#[cube]
impl Layout for MatLayout {
    type Coordinates = Coords2d;
    type SourceCoordinates = (usize, u32, u32);

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        (self.batch, pos.0, pos.1)
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos.0 < self.shape.0 && pos.1 < self.shape.1
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape
    }
}

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
