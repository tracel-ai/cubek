//! How many physical fragments each of an operand's logical axes is split across: the whole
//! description of storage tiling, and the order it lays a buffer's physical axes out in.

use cubecl::zspace::SmallVec;

use crate::{Axis, MAX_AXES};

/// How many physical fragments each logical axis is split across, in the operand's own axis order.
/// One fragment is an untiled axis; `n` fragments make a coordinate along it an `n`-digit mixed
/// radix number ([`Projection::digit`](crate::Projection::digit)), which is the whole encoding of
/// storage tiling. No extent
/// appears here: the radices are read off the buffer's `physical_shape` at use time.
///
/// The physical order this induces is level-major, coarsest first: every axis contributes its
/// level-0 fragment, then every axis still deep enough contributes its level-1 fragment, down to
/// the tile fragments. For the common case of untiled leading axes over a uniformly tiled block
/// that is the buffer's `[pre…, grid…, tile…]` order, with each untiled axis dropping out after
/// the coarsest level. Per-axis counts rather than a start/depth pair, so a buffer whose axes are
/// tiled to different depths needs no new shape of description.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct StorageTiling {
    fragments: SmallVec<[usize; MAX_AXES]>,
}

impl StorageTiling {
    /// Every axis split `levels + 1` ways. `levels = 0` is the untiled projection.
    pub fn uniform(rank: usize, levels: usize) -> Self {
        StorageTiling::per_axis(&vec![levels + 1; rank])
    }

    /// Axes from `start_axis` on split `levels + 1` ways, the ones before it untiled: a buffer
    /// carrying batch (or otherwise passthrough) axes ahead of its tiled block.
    pub fn suffix(rank: usize, start_axis: usize, levels: usize) -> Self {
        assert!(
            start_axis <= rank,
            "StorageTiling::suffix: start_axis {start_axis} past rank {rank}"
        );
        StorageTiling::per_axis(
            &(0..rank)
                .map(|i| if i < start_axis { 1 } else { levels + 1 })
                .collect::<Vec<_>>(),
        )
    }

    /// An explicit fragment count per axis, in the operand's axis order.
    pub fn per_axis(fragments: &[usize]) -> Self {
        assert!(
            fragments.iter().all(|&n| n >= 1),
            "StorageTiling: every axis carries at least one physical fragment, got {fragments:?}"
        );
        StorageTiling {
            fragments: SmallVec::from_slice(fragments),
        }
    }

    pub fn rank(&self) -> usize {
        self.fragments.len()
    }

    /// The number of physical fragments the axis at logical position `i` is split across.
    pub fn fragments(&self, i: usize) -> usize {
        self.fragments[i]
    }

    /// The deepest axis's fragment count, which is how many levels the physical order runs over.
    /// A *count*, not a depth: an untiled axis carries one fragment, so this is the `levels + 1`
    /// the rest of the crate spells out, never `levels` itself.
    pub fn max_fragments(&self) -> usize {
        self.fragments.iter().copied().max().unwrap_or(0)
    }

    /// The physical rank this induces, which is the buffer's own rank.
    pub fn physical_rank(&self) -> usize {
        self.fragments.iter().sum()
    }

    /// Whether any axis is split at all.
    pub fn is_tiled(&self) -> bool {
        self.fragments.iter().any(|&n| n > 1)
    }

    /// The physical axis labels this tiling induces over `axes`, in buffer order: the level-major
    /// emission itself, one entry per physical axis, a tiled axis appearing once per fragment. The
    /// single place the order is defined, shared by [`Projection::tiled`](crate::Projection::tiled)
    /// and by callers labeling a binding's dims ([`ConcreteLayout`](crate::ConcreteLayout)).
    pub fn order(&self, axes: &[Axis]) -> Vec<Axis> {
        assert_eq!(
            self.rank(),
            axes.len(),
            "StorageTiling::order: the tiling describes {} axes but {} were given",
            self.rank(),
            axes.len()
        );
        let mut order = Vec::with_capacity(self.physical_rank());
        for level in 0..self.max_fragments() {
            for (i, &axis) in axes.iter().enumerate() {
                if level < self.fragments(i) {
                    order.push(axis);
                }
            }
        }
        order
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    const A: Axis = Axis(0);
    const B: Axis = Axis(1);
    const R: Axis = Axis(2);

    /// The uniform and suffix tilings are the two the engine builds today; both are special cases
    /// of the per-axis counts.
    #[test]
    fn shorthands_agree_with_explicit_counts() {
        assert_eq!(
            StorageTiling::uniform(3, 1),
            StorageTiling::per_axis(&[2, 2, 2])
        );
        assert_eq!(
            StorageTiling::suffix(3, 1, 1),
            StorageTiling::per_axis(&[1, 2, 2])
        );
        // A start past the last axis tiles nothing.
        assert_eq!(
            StorageTiling::suffix(2, 2, 3),
            StorageTiling::per_axis(&[1, 1])
        );
        assert_eq!(
            StorageTiling::uniform(2, 0),
            StorageTiling::per_axis(&[1, 1])
        );
    }

    /// The level-major emission: every axis contributes its coarsest fragment, then every axis
    /// still deep enough contributes the next, so an untiled axis drops out after level 0 and a
    /// deeper axis appears alone at the finest level.
    #[test]
    fn order_emits_level_major_coarsest_first() {
        assert_eq!(StorageTiling::uniform(2, 0).order(&[A, B]), vec![A, B]);
        assert_eq!(
            StorageTiling::uniform(2, 1).order(&[A, B]),
            vec![A, B, A, B]
        );
        // `[batch, A_grid, B_grid, A_tile, B_tile]`.
        assert_eq!(
            StorageTiling::suffix(3, 1, 1).order(&[R, A, B]),
            vec![R, A, B, A, B]
        );
        // Ragged: `A` split three ways, `B` two.
        assert_eq!(
            StorageTiling::per_axis(&[3, 2]).order(&[A, B]),
            vec![A, B, A, B, A]
        );
    }

    #[test]
    fn counts_describe_the_buffer_rank() {
        let t = StorageTiling::per_axis(&[3, 1, 2]);
        assert_eq!(t.rank(), 3);
        assert_eq!(t.physical_rank(), 6);
        assert_eq!(t.max_fragments(), 3);
        assert_eq!(t.fragments(1), 1);
        assert!(t.is_tiled());
        assert!(!StorageTiling::uniform(4, 0).is_tiled());
    }

    #[test]
    #[should_panic(expected = "at least one physical fragment")]
    fn an_axis_carrying_nothing_is_rejected() {
        StorageTiling::per_axis(&[2, 0]);
    }
}
