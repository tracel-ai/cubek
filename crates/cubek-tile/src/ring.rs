//! A small ring of staged buffers a pipelined lowering ping-pongs through.

use cubecl::prelude::*;

use super::*;

/// A ring of staged buffer [`Tile`]s.
#[derive(CubeType)]
pub struct Ring<T: CubePrimitive> {
    buffers: Sequence<Tile<T>>,
}

#[cube]
impl<T: CubePrimitive> Ring<T> {
    pub fn new(buffers: Sequence<Tile<T>>) -> Ring<T> {
        Ring::<T> { buffers }
    }

    pub fn stage(&mut self, #[comptime] slot: usize, src: &Tile<T>) {
        self.buffers.index_mut(slot).stage(src);
    }

    pub fn get(&self, #[comptime] slot: usize) -> &Tile<T> {
        self.buffers.index(slot)
    }
}
