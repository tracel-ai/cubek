use cubecl::prelude::*;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, CubeType)]
pub struct TileSize {
    w: usize,
    h: usize,
}

impl TileSize {
    pub fn new(w: usize, h: usize, row_vector: bool) -> Self {
        if row_vector {
            Self { w: w * h, h: 1 }
        } else {
            Self { w, h }
        }
    }

    pub fn width(&self) -> usize {
        self.w
    }

    pub fn height(&self) -> usize {
        self.h
    }

    pub fn area(&self) -> usize {
        self.w * self.h
    }

    pub fn is_row_vector(&self) -> bool {
        self.h == 1
    }
}
