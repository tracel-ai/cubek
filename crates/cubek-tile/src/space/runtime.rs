//! The runtime half of a space: the [`Dynamic`](crate::Extent) extents an operation's loops walk,
//! read off the operands that witness them.

use cubecl::prelude::*;

use crate::{Axis, Extent, Projection, Space, Tile};

/// The one physical dim whose bound is `axis`'s own extent: it carries `axis` alone, at
/// coefficient `1`. `None` for a gather (the dim holds a receptive field several axes reach over)
/// and for storage tiling (the extent is the product over the dims the axis is split across).
pub(crate) fn bound_states(projection: &Projection, axis: Axis) -> Option<usize> {
    // A broadcast axis has no dim to read a bound off: the operand is constant along it, so its
    // buffer holds nothing that sizes it.
    if !projection.addresses(axis) {
        return None;
    }
    match projection.carriers(axis)[..] {
        [pa] if projection.physical_axis(pa).is_identity(axis) => Some(pa),
        _ => None,
    }
}

/// The physical dim in this tile's window bounds that `axis`'s runtime extent is read off. A
/// direct operand maps each axis 1:1; anything else has to be answered by an operand of the same
/// operation that does ([`Tile::witnesses`]).
pub(crate) fn bound_position(projection: &Projection, axis: Axis) -> usize {
    bound_states(projection, axis).unwrap_or_else(|| {
        panic!(
            "Tile::runtime_extent: no bound of this operand is {axis:?}'s own extent (it gathers \
             over it, or splits it across storage fragments); ask an operand that witnesses it"
        )
    })
}

/// `space` with each [`Dynamic`](crate::Extent) axis sized by the first of `a`, `b`, `c` that
/// [`witnesses`](Tile::witnesses) it: the runtime space an operation's loops walk. A fully-`Static`
/// space short-circuits. One tile may stand for all three ([`runtime_space`](Tile::runtime_space)).
#[cube]
pub(crate) fn witnessed_space<A: Numeric, B: Numeric, C: Numeric>(
    #[comptime] space: Space,
    a: &Tile<A>,
    b: &Tile<B>,
    c: &Tile<C>,
) -> Space {
    let mut sizes = Sequence::<usize>::new();
    if comptime!(!space.is_static()) {
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            let axis = comptime!(space.axis_at(p));
            // `sizes` is positional, so every axis pushes, but [`Extents::count`] folds a `Static`
            // axis to its comptime extent. Fold it here too rather than asking an operand: one
            // `Dynamic` axis must not make the `Static` ones unreadable on a tile with no bound.
            let size = match comptime!(space.extent_raw(axis)) {
                Extent::Static(n) => comptime!(n).runtime(),
                Extent::Dynamic => {
                    let by_a = a.witnesses(axis);
                    let by_b = b.witnesses(axis);
                    let by_c = c.witnesses(axis);
                    if comptime!(by_a) {
                        a.runtime_extent(axis)
                    } else if comptime!(by_b) {
                        b.runtime_extent(axis)
                    } else if comptime!(by_c) {
                        c.runtime_extent(axis)
                    } else {
                        panic!(
                            "witnessed_space: {axis:?} is Dynamic and no operand states its size; \
                             every operand spanning it gathers over it, holds it Static, or is a \
                             fragment. Keep it Static in the kernel space, or give the operation \
                             an operand that maps it identically"
                        )
                    }
                }
            };
            sizes.push(size);
        }
    }
    Space::with_sizes(space, sizes)
}
