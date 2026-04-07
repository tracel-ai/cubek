use cubecl::{prelude::*, std::tensor::layout::*};

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
