//! The scales a quantized tensor is reconstructed from, as one thing.

use cubecl::prelude::*;
use cubecl::std::tensor::View;

use crate::layout::ScalesView;
use crate::scale::{CoarserScale, Scale};

/// The inner level's scale grid, plus the coarser level's when the scheme has one.
///
/// A level's grid holds one scale for the tensor or one per block; either way `read` answers for
/// a position and is the last place that knows how many levels the scheme spreads its scales
/// across.
#[derive(CubeType)]
pub struct Scales<'a, 'b, FS: CubePrimitive> {
    grid: View<'a, FS, usize>,
    coarser: ComptimeOption<ScalesView<'b, f32>>,
}

#[cube]
impl<'a, 'b, FS: CubePrimitive> Scales<'a, 'b, FS> {
    pub fn new(
        grid: View<'a, FS, usize>,
        coarser: ComptimeOption<ScalesView<'b, f32>>,
    ) -> Scales<'a, 'b, FS> {
        Scales::<'a, 'b, FS> { grid, coarser }
    }

    /// The factor the value at `pos` is reconstructed with.
    pub fn read(&self, pos: usize) -> Scale<FS> {
        CoarserScale::read(&self.coarser, pos).at::<FS>(self.grid.read(pos))
    }
}
