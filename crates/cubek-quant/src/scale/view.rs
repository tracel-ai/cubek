//! The scales a quantized tensor is reconstructed from, as one thing.

use cubecl::prelude::*;
use cubecl::std::tensor::{View, layout::linear::LinearView};

use crate::scale::{GlobalScale, Scale};

/// The scale grid the inner level lays down, plus the tensor's global scale when it has one.
///
/// Per-tensor grids one scale, per-block one per block; either way `read` answers for a position
/// and is the last place that knows how many levels the scheme spreads its scales across. The
/// global scale is read at construction, so a kernel reading several positions pays for it once.
#[derive(CubeType)]
pub struct Scales<'a, FS: CubePrimitive> {
    grid: &'a View<'a, FS, usize>,
    global: GlobalScale,
}

#[cube]
impl<'a, FS: CubePrimitive> Scales<'a, FS> {
    pub fn new(
        grid: &'a View<'a, FS, usize>,
        global: ComptimeOption<LinearView<'_, f32>>,
    ) -> Scales<'a, FS> {
        Scales::<'a, FS> {
            grid,
            global: GlobalScale::read(global),
        }
    }

    /// The factor the value at `pos` is reconstructed with.
    pub fn read(&self, pos: usize) -> Scale<FS> {
        self.global.at::<FS>(self.grid.read(pos))
    }
}
