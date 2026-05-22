use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords1d, Layout, LayoutExpand},
};

#[derive(CubeType, Clone, Copy)]
pub struct RowMajorLayout {
    width: usize,
    height: usize,
    vector_size: usize,
}

#[cube]
impl RowMajorLayout {
    pub fn new(width: usize, height: usize, vector_size: usize) -> Self {
        RowMajorLayout {
            width,
            height,
            vector_size,
        }
    }
}

#[cube]
impl Layout for RowMajorLayout {
    type Coordinates = (usize, usize);
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        (self.width * pos.0 + pos.1) / self.vector_size
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let is_valid = pos.0 < self.height && pos.1 < self.width;
        (self.to_source_pos(pos), is_valid)
    }

    fn shape(&self) -> Self::Coordinates {
        (self.width, self.height)
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }
}
