//! The copy reading of a [`Tile`](crate::Tile): `dst.copy(&src)` walks the levels that hand
//! regions to different cubes, and transports each region it lands on.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<T: Numeric> Tile<T> {
    /// `dst.copy(&src)`: walk while a level spreads regions across cubes, else transport.
    ///
    /// The transport ([`copy_from`](Tile::copy_from)) fills a whole tile with every unit of the
    /// cube, so a level that spreads *inside* a cube is already its own doing and ends the walk.
    /// A quantized source decodes on the way through, as it does under any read.
    pub fn copy(&mut self, src: &Tile<T>) {
        if comptime!(spreads_across_cubes(&self.space)) {
            let space = self.copy_space(src);
            for region in Walk::over(space) {
                let mut dst = self.at(&region);
                dst.copy(&src.at(&region));
            }
        } else {
            self.copy_from(src);
        }
    }

    /// The walk's space: this tile's, with every [`Dynamic`](crate::Extent) axis sized by
    /// whichever operand [`witnesses`](Tile::witnesses) it. The destination is asked first,
    /// like [`mma`](Tile::mma) asks its accumulator: an axis it spans is one it writes, so its
    /// bound is the extent the walk must cover.
    fn copy_space(&self, src: &Tile<T>) -> Space {
        witnessed_space(comptime!(self.space.clone()), self, src, src)
    }
}

/// Whether this level hands its regions to different cubes: some axis rides a cube dim and none
/// rides a plane or a unit, so the walk steps exactly what the launch grid separates.
fn spreads_across_cubes(space: &Space) -> bool {
    if space.is_final() {
        return false;
    }
    let mut across = false;
    for axis in space.axes() {
        match space.partitioner().distribution(axis).scope() {
            Some(ComputeScope::Cube(_)) => across = true,
            Some(_) => return false,
            None => {}
        }
    }
    across
}
