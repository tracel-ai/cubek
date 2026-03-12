use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords2d, Layout, LayoutExpand},
};
use cubek_matmul::launch::BatchedCoords;

use crate::components::ConvolutionProblem;

/// Weight backwards needs a consolidated layout to work properly across the combined `k` dimension.
/// Padding to an even tile shape on width isn't valid, because `im2col` doesn't do this.
/// Wouldn't be necessary with `im2colWide`, should investigate at some point.
#[derive(CubeType, CubeLaunch)]
pub struct TmaOutGradLayout {
    rows: u32,
    cols: u32,
}

#[cube]
impl Layout for TmaOutGradLayout {
    type Coordinates = BatchedCoords;
    type SourceCoordinates = Coords2d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (_, row, col) = pos;
        (row, col)
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }

    fn shape(&self) -> Self::Coordinates {
        (1, self.rows, self.cols)
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}

impl<R: Runtime> TmaOutGradLayoutLaunch<R> {
    pub fn from_problem(problem: &ConvolutionProblem) -> Self {
        TmaOutGradLayoutLaunch::new(problem.k as u32, problem.m as u32)
    }
}
