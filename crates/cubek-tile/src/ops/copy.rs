//! The copy reading of a [`Tile`](crate::Tile): `dst.copy(&src)` walks the levels the launch grid
//! separates, and transports each region it lands on.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<T: Numeric> Tile<T> {
    /// `dst.copy(&src)`: walk a level that separates cubes, transport anything else. Same shape as
    /// [`zero`](Tile::zero), a final tile at the bottom.
    ///
    /// The transport ([`copy_from`](Tile::copy_from)) fills a whole tile with every unit of the
    /// cube, so a level reaching inside a cube ([`Planes`](LevelScope::Planes),
    /// [`Lanes`](LevelScope::Lanes)) is already its own doing and ends the walk. A quantized source
    /// decodes on the way through, as it does under any read.
    pub fn copy(&mut self, src: &Tile<T>) {
        match comptime!(self.space.partitioner().clone()) {
            Partitioner::Final => self.copy_from(src),
            Partitioner::Level(level) => match comptime!(level.scope()) {
                LevelScope::Cubes => {
                    let space = self.copy_space(src);
                    for region in Walk::over(space) {
                        let mut dst = self.at(&region);
                        dst.copy(&src.at(&region));
                    }
                }
                LevelScope::Sequential | LevelScope::Planes | LevelScope::Lanes => {
                    self.copy_from(src)
                }
            },
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
