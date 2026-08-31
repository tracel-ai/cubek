//! `dst.mul(&a, &b)`: the elementwise product of two tiles, each broadcasting over the axes it
//! omits.
//!
//! Not a quantization verb. A scale is a tile that spans fewer axes than the values it multiplies,
//! and "one scale per block" is what its axes say rather than what any arithmetic does; dequantizing
//! is this operation with a packed operand on one side. Written alone, and tested alone, because a
//! mechanism that only works inside the contraction is not a mechanism.
//!
//! Both operands are read at the destination's logical coordinate through their own
//! [`Projection`](crate::Projection), so an axis an operand does not address costs it nothing and
//! spreads its value across every position of that axis. That is the whole of the broadcast.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<T: Numeric> Tile<T> {
    /// `dst = a ⊗ b`, elementwise over the destination's box.
    ///
    /// The walk mirrors [`copy`](Tile::copy): a level that separates cubes is walked, anything
    /// else transports, because the transport already fills a whole tile with every unit.
    pub fn mul<A: Numeric, B: Numeric>(&mut self, a: &Tile<A>, b: &Tile<B>) {
        match comptime!(self.space.partitioner().clone()) {
            Partitioner::Final => self.mul_from(a, b),
            Partitioner::Level(level) => match comptime!(level.scope()) {
                LevelScope::Cubes => {
                    let space = self.mul_space(a, b);
                    for region in Walk::over(space) {
                        let mut dst = self.at(&region);
                        dst.mul(&a.at(&region), &b.at(&region));
                    }
                }
                LevelScope::Sequential | LevelScope::Planes | LevelScope::Lanes => {
                    self.mul_from(a, b)
                }
            },
        }
    }

    /// The walk's space: this tile's, with every [`Dynamic`](crate::Extent) axis sized by
    /// whichever operand [`witnesses`](Tile::witnesses) it. The destination is asked first, as
    /// [`copy`](Tile::copy) asks its own: an axis it spans is one it writes.
    fn mul_space<A: Numeric, B: Numeric>(&self, a: &Tile<A>, b: &Tile<B>) -> Space {
        witnessed_space(comptime!(self.space.clone()), self, a, b)
    }

    /// The transport: every unit of the cube strides the destination's lines, reading each operand
    /// at that line's own coordinate.
    ///
    /// Each operand is addressed over *its own* axes, so an axis it does not span costs it nothing
    /// and one of its values serves every position of that axis. A packed operand serves a whole
    /// stored word per line, which is why the walk moves in lines at all: there is no cell-at-a-time
    /// reading of a tensor whose values share a word.
    ///
    /// The scale of a line is one value, because an operand that varies inside the line would have
    /// to span the line's own axis, and then it would not be a scale.
    fn mul_from<A: Numeric, B: Numeric>(&mut self, a: &Tile<A>, b: &Tile<B>) {
        let space = comptime!(self.space.clone());
        let width = self.vector_size();
        let size!(W) = width;
        let a_view = a.nd_packed::<W>(Guard::Checked);
        let b_view = b.nd_packed::<Const<1>>(Guard::Checked);

        // The destination's box in lines: its innermost axis alone counts them.
        let extents = const_coords(comptime!(line_extents(&space, width, 0, space.rank())));
        let total = comptime!(
            line_extents(&space, width, 0, space.rank())
                .iter()
                .product::<usize>()
        );
        let mut dst = self.nd_mut::<W>();
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            let line = unravel(&extents, i.fcast::<u32>());
            // The line index back in values, which is what each operand's own resolution divides
            // by its own width.
            let mut cells = Coords::<u32>::new();
            #[unroll]
            for p in 0..comptime!(space.rank()) {
                let scale = comptime!(match p == space.rank() - 1 {
                    true => width as u32,
                    false => 1u32,
                });
                cells.push(line.at(p).fmul(scale));
            }
            let empty = Coords::<u32>::new();
            let a_pos = resolve_nd_coords(
                comptime!(a.space.clone()),
                comptime!(space.clone()),
                comptime!(Vec::new()),
                &cells,
                &empty,
                width,
                true,
            );
            let b_pos = resolve_nd_coords(
                comptime!(b.space.clone()),
                comptime!(space.clone()),
                comptime!(Vec::new()),
                &cells,
                &empty,
                1usize,
                true,
            );
            let left = Vector::<T, W>::cast_from(a_view.read(a_pos));
            let right = Vector::<T, W>::cast_from(b_view.read(b_pos).extract(0usize));
            dst.write(line.to_dyn(), left * right);
            i += workers;
        }
    }
}
