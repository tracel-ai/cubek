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

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos < self.shape
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
        let (mn_pos, k_pos) = pos;

        match comptime!(self.matrix_layout) {
            MatrixLayout::RowMajor => (self.batch, mn_pos, k_pos),
            MatrixLayout::ColMajor => (self.batch, k_pos, mn_pos),
        }
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (mn_bound, k_bound) = match comptime!(self.matrix_layout) {
            MatrixLayout::RowMajor => (self.shape.0, self.shape.1),
            MatrixLayout::ColMajor => (self.shape.1, self.shape.0),
        };

        pos.0 < mn_bound && pos.1 < k_bound
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape
    }
}
