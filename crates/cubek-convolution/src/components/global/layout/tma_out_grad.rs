use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords2d, Coords3d, Layout, LayoutExpand},
};

/// Weight backwards needs a consolidated layout to work properly across the combined `k` dimension.
/// Padding to an even tile shape on width isn't valid, because `im2col` doesn't do this.
/// Wouldn't be necessary with `im2colWide`, should investigate at some point.
#[derive(CubeType, CubeLaunch)]
pub struct TmaOutGradLayout {}

#[cube]
impl Layout for TmaOutGradLayout {
    type Coordinates = Coords3d;
    type SourceCoordinates = Coords2d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (_, row, col) = pos;
        (row, col)
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }

    fn shape(&self) -> Self::Coordinates {
        (u32::MAX, u32::MAX, u32::MAX).runtime()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}
