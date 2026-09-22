//! The copy reading of a [`Tile`](crate::Tile): `dst.copy(&src)` walks the levels the launch grid
//! separates, and transports each region it lands on.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<T: Numeric> Tile<T> {
    /// `dst.copy(&src)`: the transport ([`copy_from`](Tile::copy_from)), which fills a whole
    /// tile with every unit of the cube. The kernel's own loops split a copy across levels; this
    /// runs on the region they hand it. A quantized source decodes on the way through.
    pub fn copy(&mut self, src: &Tile<T>) {
        self.copy_from(src)
    }
}
