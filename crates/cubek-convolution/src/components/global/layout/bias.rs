use cubecl::prelude::*;
use cubecl::std::tensor::layout::*;
use cubek_matmul::launch::BatchedCoords;

#[derive(CubeType, CubeLaunch)]
pub struct BiasLayout {
    shape: u32,
    #[cube(comptime)]
    line_size: u32,
}

#[cube]
impl Layout for BiasLayout {
    type Coordinates = BatchedCoords;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (_, _, n) = pos;
        (n / self.line_size) as usize
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (_, _, n) = pos;
        n < self.shape
    }

    fn shape(&self) -> Self::Coordinates {
        (1, 1, self.shape)
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}
