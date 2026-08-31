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

use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

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

    /// The transport: every unit of the cube strides the destination's groups, reading `b` once
    /// per group and taking a lane of it per fold.
    ///
    /// Each operand is addressed over *its own* axes, so an axis it does not span costs it nothing
    /// and one of its values serves every position of that axis. A packed operand serves a whole
    /// stored word per line, which is why the walk moves in lines at all: there is no
    /// cell-at-a-time reading of a tensor whose values share a word.
    fn mul_from<A: Numeric, B: Numeric>(&mut self, a: &Tile<A>, b: &Tile<B>) {
        let space = comptime!(self.space.clone());
        let width = self.vector_size();
        let folds = b.vector_size();
        let split = comptime!(FoldWalk::of(&space, &b.space, width, folds));
        let size!(W) = width;
        let size!(F) = folds;
        let a_reader = a.nd_split_packed::<W>();
        let b_reader = b.nd_split_packed::<F>();

        let a_fold_at = comptime!(a.space.position(split.axis));
        let extents = const_coords(comptime!(split.groups.clone()));
        let total = comptime!(split.groups.iter().product::<usize>());

        let mut dst = self.nd_mut::<W>();
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            let group = group_line(
                &unravel(&extents, i.fcast::<u32>()),
                comptime!(split.clone()),
            );
            let (a_base, b_base) = bases(
                &group,
                comptime!(space.clone()),
                comptime!(a.space.clone()),
                comptime!(b.space.clone()),
                width,
                folds,
            );

            // The fold is a *lane* of `b`'s read, not a step of it, so its address is the group's
            // and the read happens here rather than once per lane.
            let scales = b_reader
                .view
                .read(b_reader.map.anchor(b_base, comptime!(Vec::new())));
            // `a`'s address does step with the fold, so its map folds once here and each step is
            // the addition [`advance`](crate::AxisProjection::advance) puts back.
            let moving = comptime!(vec![split.axis]);
            let a_anchor = a_reader
                .map
                .anchor(a_base.clone(), comptime!(moving.clone()));

            #[unroll]
            for f in 0..comptime!(split.folds) {
                let value = a_reader.view.read(a_reader.map.advance(
                    &a_anchor,
                    stepped(&a_base, comptime!(a_fold_at), f),
                    comptime!(moving.clone()),
                ));
                let left = Vector::<T, W>::cast_from(value);
                let right = Vector::<T, W>::cast_from(scales.extract(f));
                dst.write(stepped(&group, comptime!(split.at), f), left * right);
            }
            i += workers;
        }
    }
}

/// How a product's walk divides: one read of the broadcast operand per group, and the folds inside
/// a group taken as lanes of it.
///
/// The fold is the walk's unrolled dimension because a lane index is not addressable at runtime,
/// and the runs under one fold are not, because only the lane has to be a constant.
#[derive(Clone, Debug)]
struct FoldWalk {
    /// The axis the fold steps: the one the broadcast operand's own lines run along.
    axis: Axis,
    /// Its position in the destination's box.
    at: usize,
    /// Folds one read of that operand serves.
    folds: usize,
    /// The destination's box in lines, cut at [`at`](Self::at) into groups of one read.
    groups: Vec<usize>,
}

impl FoldWalk {
    /// How `dst` divides against an operand spanning `b`, served `width` lines wide against
    /// `folds` of that operand per read.
    fn of(dst: &Space, b: &Space, width: usize, folds: usize) -> Self {
        let rank = dst.rank();
        let axis = b.axis_at(b.rank() - 1);
        let at = dst.position(axis);
        assert!(
            folds == 1 || at != rank - 1,
            "Tile::mul: {folds} of `b` arrive per read along {axis:?}, which is also the axis the \
             destination's own lines run along, so one line would straddle two of them"
        );
        let mut groups = line_extents(dst, width, 0, rank);
        assert!(
            groups[at].is_multiple_of(folds),
            "Tile::mul: {} positions of {axis:?} do not divide into reads of {folds}",
            groups[at]
        );
        groups[at] /= folds;
        FoldWalk {
            axis,
            at,
            folds,
            groups,
        }
    }
}

/// A group's own line coordinate in the destination's box: its position with the fold axis back at
/// the scale it was cut by.
#[cube]
fn group_line(at: &Coords<u32>, #[comptime] split: FoldWalk) -> CoordsDyn {
    let mut out = CoordsDyn::new();
    #[unroll]
    for p in 0..comptime!(split.groups.len()) {
        let coord = match comptime!(p == split.at) {
            true => at.at(p).fmul(comptime!(split.folds as u32)),
            false => at.at(p),
        };
        out.push(coord);
    }
    out
}

/// Each operand's own coordinate for a group, resolved over its own axes: an axis it does not span
/// drops out, and its innermost divides by the width it is read at.
#[cube]
fn bases(
    group: &CoordsDyn,
    #[comptime] dst: Space,
    #[comptime] a: Space,
    #[comptime] b: Space,
    #[comptime] width: usize,
    #[comptime] folds: usize,
) -> (CoordsDyn, CoordsDyn) {
    // Back in values, which is what each operand's own resolution divides by its own width.
    let mut cells = Coords::<u32>::new();
    #[unroll]
    for p in 0..comptime!(dst.rank()) {
        cells.push(group[p].fmul(comptime!(match p == dst.rank() - 1 {
            true => width as u32,
            false => 1u32,
        })));
    }
    let empty = Coords::<u32>::new();
    (
        resolve_nd_coords(
            a,
            comptime!(dst.clone()),
            comptime!(Vec::new()),
            &cells,
            &empty,
            width,
            true,
        ),
        resolve_nd_coords(b, dst, comptime!(Vec::new()), &cells, &empty, folds, true),
    )
}

/// `base` with the fold axis advanced by `f`: the one component a fold moves.
#[cube]
fn stepped(base: &CoordsDyn, #[comptime] at: usize, #[comptime] f: usize) -> CoordsDyn {
    let mut out = CoordsDyn::new();
    #[unroll]
    for p in 0..comptime!(base.len()) {
        let coord = match comptime!(p == at) {
            true => base[p] + comptime!(f as u32),
            false => base[p],
        };
        out.push(coord);
    }
    out
}
