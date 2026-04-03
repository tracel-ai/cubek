use cubecl::prelude::*;
use cubecl::std::tensor::layout::*;
use cubek_std::MatrixLayout;

#[derive(CubeType, Clone, Copy)]
pub struct VecLayout {
    batch: usize,
    shape: Coords1d,
}

#[cube]
impl VecLayout {
    pub fn new(batch: usize, shape: Coords1d) -> Self {
        VecLayout { batch, shape }
    }
}

#[cube]
impl Layout for VecLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = (usize, u32, u32);

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        (self.batch, 0, pos as u32)
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

#[derive(CubeType, Clone, Copy)]
pub struct MatLayout {
    batch: usize,
    shape: Coords2d,
    #[cube(comptime)]
    matrix_layout: MatrixLayout,
}

#[cube]
impl MatLayout {
    pub fn new(batch: usize, shape: Coords2d, #[comptime] matrix_layout: MatrixLayout) -> Self {
        MatLayout {
            batch,
            shape,
            matrix_layout,
        }
    }
}

#[cube]
impl Layout for MatLayout {
    type Coordinates = Coords2d;
    type SourceCoordinates = (usize, u32, u32);

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (ordinal, vector_pos) = pos;

        match comptime!(self.matrix_layout) {
            MatrixLayout::RowMajor => {
                // matvec style: row = ordinal, col = vector_pos
                (self.batch, ordinal, vector_pos)
            }
            MatrixLayout::ColMajor => {
                // vecmat style: row = vector_pos, col = ordinal
                (self.batch, vector_pos, ordinal)
            }
        }
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
