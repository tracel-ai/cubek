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
    /// `b` is read at its binding's width, so one read of it serves several of the destination's
    /// runs. Which lane of that read a run takes has to be a constant — a lane index is not
    /// addressable at runtime — so the fold is the unrolled dimension of the walk and the runs
    /// under one fold are not.
    fn mul_from<A: Numeric, B: Numeric>(&mut self, a: &Tile<A>, b: &Tile<B>) {
        let space = comptime!(self.space.clone());
        let width = self.vector_size();
        let folds = b.vector_size();
        let size!(W) = width;
        let size!(F) = folds;
        let a_view = a.nd_packed::<W>(Guard::Checked);
        let b_view = b.nd_packed::<F>(Guard::Checked);

        let rank = comptime!(space.rank());
        // The axis `b`'s own lines run along, as a position in the destination's box.
        let fold_axis = comptime!(b.space.axis_at(b.space.rank() - 1));
        let fold_at = comptime!(space.position(fold_axis));
        comptime!(assert!(
            folds == 1 || fold_at != rank - 1,
            "Tile::mul: {folds} of `b` arrive per read along {fold_axis:?}, which is also the axis \
             the destination's own lines run along, so one line would straddle two of them"
        ));

        // The destination's box in lines, its innermost axis alone counting them, and then split
        // once more at `fold_at`: a group holds one read of `b`, and the folds inside it are
        // walked under a constant.
        let mut group = comptime!(line_extents(&space, width, 0, rank));
        comptime!(assert!(
            group[fold_at].is_multiple_of(folds),
            "Tile::mul: {} positions of {fold_axis:?} do not divide into reads of {folds}",
            group[fold_at]
        ));
        comptime!({
            group[fold_at] /= folds;
        });
        let total = comptime!(group.iter().product::<usize>());
        let extents = const_coords(comptime!(group.clone()));

        let mut dst = self.nd_mut::<W>();
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            let at = unravel(&extents, i.fcast::<u32>());
            let empty = Coords::<u32>::new();

            // The fold is a *lane* of `b`'s read, not a step of it: every run in this group reads
            // the same vector and differs only in which lane it takes. So the read happens here,
            // once, above the loop that takes them — rather than four identical reads left for a
            // compiler to notice are one.
            let mut group = Coords::<u32>::new();
            #[unroll]
            for p in 0..rank {
                let coord = match comptime!(p == fold_at) {
                    true => at.at(p).fmul(comptime!(folds as u32)),
                    false => at.at(p),
                };
                group.push(coord.fmul(comptime!(match p == rank - 1 {
                    true => width as u32,
                    false => 1u32,
                })));
            }
            let scales = b_view.read(resolve_nd_coords(
                comptime!(b.space.clone()),
                comptime!(space.clone()),
                comptime!(Vec::new()),
                &group,
                &empty,
                folds,
                true,
            ));

            #[unroll]
            for f in 0..folds {
                // This run's line, and the same coordinate back in values, which is what `a`'s own
                // resolution divides by its own width.
                let mut line = Coords::<u32>::new();
                let mut cells = Coords::<u32>::new();
                #[unroll]
                for p in 0..rank {
                    let coord = match comptime!(p == fold_at) {
                        true => at.at(p).fmul(comptime!(folds as u32)) + comptime!(f as u32),
                        false => at.at(p),
                    };
                    line.push(coord);
                    cells.push(coord.fmul(comptime!(match p == rank - 1 {
                        true => width as u32,
                        false => 1u32,
                    })));
                }
                let left = Vector::<T, W>::cast_from(a_view.read(resolve_nd_coords(
                    comptime!(a.space.clone()),
                    comptime!(space.clone()),
                    comptime!(Vec::new()),
                    &cells,
                    &empty,
                    width,
                    true,
                )));
                let right = Vector::<T, W>::cast_from(scales.extract(f));
                dst.write(line.to_dyn(), left * right);
            }
            i += workers;
        }
    }
}
