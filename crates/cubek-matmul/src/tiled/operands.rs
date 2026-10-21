//! The axes every tiled routine builds its space over.

use cubecl::tile::{Axis, Space};
pub(crate) use cubek_std::matmul_axes::{K, M, N, batch_axis};

/// The space a routine's levels are stated over before any problem: every axis dynamic, so the
/// counts a shape decides print as unknown and the tiling itself still reads. The axis order is
/// the launch's — batch axes first, then the matrix dims and the contraction — since that order
/// is what a [`Region`](cubecl::tile::Region)'s coordinates come in.
///
/// Nothing may ask this space for a grid: [`Partitioning::cube_count`] needs an extent the
/// launch has not stamped yet. Printing is what it is for, and only tests print one today. It
/// widens when a caller does.
#[cfg(test)]
pub(crate) fn form_space(batches: usize) -> Space {
    let axes: Vec<Axis> = (0..batches).map(batch_axis).chain([M, N, K]).collect();
    Space::dynamic(&axes)
}

/// What a partitioning over these axes prints as
/// ([`Partitioning::labelled`]): an [`Axis`] is an index, and only the routine that assigned it
/// knows what it stands for. Read off the space rather than off a batch list, so any caller
/// holding one can name its axes.
pub(crate) fn labels(space: &Space) -> Vec<(Axis, &'static str)> {
    const BATCHES: [&str; 6] = ["b0", "b1", "b2", "b3", "b4", "b5"];
    let mut batches = 0;
    space
        .axes()
        .map(|axis| {
            let name = if axis == M {
                "m"
            } else if axis == N {
                "n"
            } else if axis == K {
                "k"
            } else {
                batches += 1;
                BATCHES[batches - 1]
            };
            (axis, name)
        })
        .collect()
}
