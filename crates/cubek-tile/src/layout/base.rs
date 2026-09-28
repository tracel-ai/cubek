//! [`Layout`]: where a buffer's values sit relative to one another, stated as counts from the
//! finest up.
//!
//! A buffer's dense part is one mixed-radix number: the finest entry says how many values run
//! along its axis before the next entry steps, the next says how many of those runs it holds,
//! and so on. Every size is a product of counts, so between two entries nothing divides.
//!
//! A word and a read are *cuts* of that order, not levels of it: a `u32` holding eight `e2m1`
//! fields is the first eight values, a four-word read the first thirty-two. The format fixes the
//! first, the device the second, and both must land on whole entries, splitting at most the last
//! one they reach at a divisor of its count.
//!
//! A layout is read off a binding ([`Layout::of`]), or stated leaf-up for a buffer about to be
//! written ([`Layout::tile`]): each tile made of the one below it, then the order of the grid of
//! tiles.

use core::fmt::{self, Display, Formatter};

use cubecl::zspace::SmallVec;

use crate::{Axis, Geometry, Space, StorageTiling};

/// Where a buffer's values sit relative to one another: its dense part as `(axis, count)`
/// entries, finest first, and the strides of whatever lies past it.
///
/// Each entry's stride is the product of the counts before it, which is what makes the part
/// dense; a dim that breaks that (a gap, a broadcast, a dim no axis labels) ends it, and its
/// stride and every coarser one are kept apart, since a cut has to divide them.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Layout {
    /// The dense part, finest first: an axis and how many of the entry below one step holds.
    dense: SmallVec<[(Axis, usize); Space::MAX_RANK]>,
    /// The strides past the dense part, finest first, in values.
    outer: SmallVec<[usize; Space::MAX_RANK]>,
}

impl Layout {
    /// The layout a buffer is stored in: `geometry`'s dims read from the innermost up, while each
    /// one steps by the product of the extents finer than it.
    ///
    /// `labels` name the trailing dims, right-aligned to the geometry: a leading dim they leave
    /// out ends the dense part, as does a dim that does not step by that product.
    pub(crate) fn of(geometry: &Geometry, labels: &[Axis]) -> Self {
        let rank = geometry.rank();
        let unlabelled = rank.saturating_sub(labels.len());
        let mut dense = SmallVec::new();
        let mut outer = SmallVec::new();
        let mut run = 1;
        let dims: Vec<(usize, usize)> = geometry.dims().collect();
        for (dim, &(extent, stride)) in dims.iter().enumerate().rev() {
            let continues = outer.is_empty() && dim >= unlabelled && stride == run;
            match continues {
                true => {
                    dense.push((labels[dim - unlabelled], extent));
                    run *= extent;
                }
                false => outer.push(stride),
            }
        }
        Layout { dense, outer }
    }

    /// A layout stated leaf-up for a buffer about to be written: `tile` is its finest tile,
    /// `(axis, count)` finest first, in values.
    pub fn tile(tile: &[(Axis, usize)]) -> LayoutBuilder {
        LayoutBuilder {
            levels: vec![tile.to_vec()],
        }
    }

    /// Whether the first `values` values are a cut this layout can serve: a word of that many
    /// fields, or a read of that many values.
    ///
    /// The cut runs inside the finest entry, and splits it at a divisor of its count; every
    /// stride past the dense part is a whole number of cuts, or a coarser step would land inside
    /// one.
    pub(crate) fn cut(&self, values: usize) -> Result<(), LineMisfit> {
        if values == 1 {
            return Ok(());
        }
        let Some(&(_, finest)) = self.dense.first() else {
            return Err(match self.outer.first() {
                Some(&stride) => LineMisfit::InnermostStrided(stride),
                None => LineMisfit::NoDims,
            });
        };
        if !finest.is_multiple_of(values) {
            return Err(LineMisfit::PartialLine(finest));
        }
        match self
            .outer
            .iter()
            .rev()
            .find(|stride| !stride.is_multiple_of(values))
        {
            Some(&stride) => Err(LineMisfit::StrideInsideLine(stride)),
            None => Ok(()),
        }
    }

    /// The buffer this layout stores `axes` in: its physical dims in the order cubecl's storage
    /// tiling lists them (level-major, coarsest first), each `(extent, stride)`, and the fragment
    /// count per axis that [`StorageTiling`] reads back.
    ///
    /// # Panics
    ///
    /// When the layout names an axis `axes` does not, or leaves one of them out: a buffer stores
    /// every axis it stands for.
    pub fn physical(&self, axes: &[Axis]) -> (Geometry, StorageTiling) {
        assert!(
            self.outer.is_empty(),
            "Layout::physical: only a dense layout describes a buffer to write"
        );
        let fragments: Vec<usize> = axes
            .iter()
            .map(|&axis| self.dense.iter().filter(|&&(a, _)| a == axis).count())
            .collect();
        assert!(
            self.dense.iter().all(|(a, _)| axes.contains(a)) && fragments.iter().all(|&f| f >= 1),
            "Layout::physical: the layout names {:?} but the buffer stands for {axes:?}",
            self.dense.iter().map(|&(a, _)| a).collect::<Vec<_>>()
        );
        let tiling = StorageTiling::per_axis(&fragments);
        // Each entry's stride is the product of the counts finer than it.
        let strides: Vec<usize> = self
            .dense
            .iter()
            .scan(1, |run, &(_, count)| {
                let stride = *run;
                *run *= count;
                Some(stride)
            })
            .collect();
        // An axis's entries coarsest first, so fragment `level` of `axis` is its `level`th from
        // the top, the order the tiling's level-major emission counts them in.
        let of_axis = |axis: Axis| -> Vec<usize> {
            (0..self.dense.len())
                .rev()
                .filter(|&e| self.dense[e].0 == axis)
                .collect()
        };
        let per_axis: Vec<Vec<usize>> = axes.iter().map(|&axis| of_axis(axis)).collect();
        let mut dims = Vec::with_capacity(self.dense.len());
        for level in 0..tiling.max_fragments() {
            for entries in &per_axis {
                if let Some(&e) = entries.get(level) {
                    dims.push((self.dense[e].1, strides[e]));
                }
            }
        }
        (Geometry::new(&dims), tiling)
    }
}

/// A [`Layout`] being stated leaf-up: its tiles, finest first, each made of the one below it,
/// before the grid says in which order the coarsest follow one another.
#[derive(Clone, Debug)]
pub struct LayoutBuilder {
    levels: Vec<Vec<(Axis, usize)>>,
}

impl LayoutBuilder {
    /// A coarser tile: `(axis, count)` finest first, each count how many of the tile below one
    /// step holds.
    pub fn tile(mut self, level: &[(Axis, usize)]) -> Self {
        self.levels.push(level.to_vec());
        self
    }

    /// The grid of tiles, its axes finest first: `[K, N]` puts the next tile along `K` right
    /// after this one. Its counts are the tensor's, so the layout is settled against the extents
    /// ([`over`](GridLayout::over)).
    pub fn grid(self, order: &[Axis]) -> GridLayout {
        GridLayout {
            levels: self.levels,
            order: order.to_vec(),
        }
    }
}

/// A stated layout short of the tensor's extents: the grid's counts are what is left of each
/// axis once the tile's levels have taken theirs.
#[derive(Clone, Debug)]
pub struct GridLayout {
    levels: Vec<Vec<(Axis, usize)>>,
    order: Vec<Axis>,
}

impl GridLayout {
    /// The layout over a tensor of these extents: the grid takes, along each axis, the extent over
    /// the tile's size, which is the one division, and has to be whole.
    ///
    /// # Errors
    ///
    /// An axis whose extent is not a whole number of tiles, or a grid that names an axis twice or
    /// leaves one out.
    pub fn over(self, extents: &[(Axis, usize)]) -> Result<Layout, LayoutMisfit> {
        let mut dense: SmallVec<[(Axis, usize); Space::MAX_RANK]> =
            self.levels.into_iter().flatten().collect();
        let mut order = self.order.clone();
        order.sort_by_key(|a| a.0);
        order.dedup();
        let mut named: Vec<Axis> = extents.iter().map(|&(a, _)| a).collect();
        named.sort_by_key(|a| a.0);
        if order.len() != self.order.len() || order != named {
            return Err(LayoutMisfit::Grid {
                order: self.order,
                axes: extents.iter().map(|&(a, _)| a).collect(),
            });
        }
        for &axis in &self.order {
            let extent = extents
                .iter()
                .find(|&&(a, _)| a == axis)
                .map(|&(_, e)| e)
                .expect("checked against the grid's axes above");
            let tile: usize = dense
                .iter()
                .filter(|&&(a, _)| a == axis)
                .map(|&(_, c)| c)
                .product();
            if !extent.is_multiple_of(tile) {
                return Err(LayoutMisfit::PartialTile { axis, extent, tile });
            }
            dense.push((axis, extent / tile));
        }
        Ok(Layout {
            dense,
            outer: SmallVec::new(),
        })
    }
}

/// Why a stated layout does not describe a tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutMisfit {
    /// The tiles along `axis` do not close its extent.
    PartialTile {
        axis: Axis,
        extent: usize,
        tile: usize,
    },
    /// The grid does not name each of the tensor's axes exactly once.
    Grid { order: Vec<Axis>, axes: Vec<Axis> },
}

impl Display for LayoutMisfit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::PartialTile { axis, extent, tile } => write!(
                f,
                "axis {axis:?} runs {extent}, which is not a whole number of {tile}-wide tiles"
            ),
            Self::Grid { order, axes } => write!(
                f,
                "the grid orders {order:?}, but the tensor's axes are {axes:?}: it names each once"
            ),
        }
    }
}

/// Why a [`Layout`] cannot serve a cut of some width: the value that decided it, so a message
/// names the number a reader has to go looking for otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineMisfit {
    /// The innermost dim's own stride, when it is not 1: consecutive values are not one line.
    InnermostStrided(usize),
    /// The finest entry's count, when the cut does not divide it.
    PartialLine(usize),
    /// A stride past the dense part, when re-expressing it as `stride / width` would land inside
    /// a line.
    StrideInsideLine(usize),
    /// No dims at all: there is no innermost extent to count in lines, so no width but `1`
    /// describes it. Carries nothing, because the misfit is the absence.
    NoDims,
}

impl Display for LineMisfit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InnermostStrided(stride) => write!(
                f,
                "its innermost dim steps by {stride} rather than 1, so consecutive values are \
                 not one line"
            ),
            Self::PartialLine(extent) => write!(
                f,
                "its innermost extent is {extent}, which is not a whole number of lines"
            ),
            Self::StrideInsideLine(stride) => write!(
                f,
                "its stride {stride} is not a whole number of lines, so a coarser step lands \
                 inside a line"
            ),
            Self::NoDims => write!(
                f,
                "it has no dims, so it has no innermost extent to count in lines"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// The rule the cut replaces, kept here as the gate that the replacement accepts what it
    /// accepted: the innermost dim steps by one and holds whole lines, and every coarser stride
    /// is a whole number of lines.
    fn served_before(geometry: &Geometry, width: usize) -> bool {
        if width == 1 {
            return true;
        }
        let Some(last) = geometry.rank().checked_sub(1) else {
            return false;
        };
        geometry.strides()[last] == 1
            && geometry.shape()[last].is_multiple_of(width)
            && geometry.strides()[..last]
                .iter()
                .all(|s| s.is_multiple_of(width))
    }

    /// Over plain, padded, transposed, broadcast and storage-tiled geometries, at every width
    /// a device reads, the cut accepts exactly what the rule it replaces did.
    #[test]
    fn a_cut_serves_what_the_line_rule_served() {
        let geometries: Vec<(Geometry, Vec<Axis>)> = vec![
            (Geometry::new(&[(64, 128), (128, 1)]), vec![M, N]),
            (Geometry::new(&[(64, 136), (128, 1)]), vec![M, N]),
            (Geometry::new(&[(64, 130), (130, 1)]), vec![M, N]),
            (Geometry::new(&[(64, 1), (128, 64)]), vec![M, N]),
            (Geometry::new(&[(3, 0), (64, 128), (128, 1)]), vec![M, N]),
            (Geometry::new(&[(1, 5), (64, 128), (128, 1)]), vec![M, N]),
            (Geometry::new(&[(2, 4096), (32, 1)]), vec![K, N]),
            (
                Geometry::new(&[(4, 2048), (4, 512), (16, 32), (32, 1)]),
                vec![K, N, K, N],
            ),
            (Geometry::new(&[(6, 1)]), vec![N]),
            (Geometry::new(&[]), vec![]),
        ];
        for (geometry, labels) in &geometries {
            let layout = Layout::of(geometry, labels);
            for width in [1, 2, 3, 4, 8, 16, 32] {
                assert_eq!(
                    layout.cut(width).is_ok(),
                    served_before(geometry, width),
                    "{geometry:?} at {width}"
                );
            }
        }
    }

    /// A storage-tiled `[k, n]` in a row-major grid read back: the tile's dims are the finest
    /// entries and the grid's the coarsest, though the tiling lists the grid first. A grid in
    /// another order ends the dense part at the grid, as the physical dims are read in the order
    /// they are listed; a cut stays inside the finest entry and never reaches it.
    #[test]
    fn a_tiled_binding_reads_as_its_entries() {
        // 64 x 64 in 16 x 32 tiles, row-major tiles: [k/16, n/32, 16, 32].
        let geometry = Geometry::new(&[(4, 1024), (2, 512), (16, 32), (32, 1)]);
        let layout = Layout::of(&geometry, &[K, N, K, N]);
        assert_eq!(&layout.dense[..], &[(N, 32), (K, 16), (N, 2), (K, 4)]);
    }

    /// Stated leaf-up and settled against the tensor, a layout describes the buffer cubek has
    /// always written for a row-major grid of tiles: the same dims, the same strides.
    #[test]
    fn a_row_major_grid_is_the_buffer_cubek_writes() {
        let layout = Layout::tile(&[(N, 32), (M, 16)])
            .grid(&[N, M])
            .over(&[(M, 64), (N, 64)])
            .unwrap();
        let (geometry, tiling) = layout.physical(&[M, N]);
        assert_eq!(
            geometry,
            Geometry::new(&[(4, 1024), (2, 512), (16, 32), (32, 1)])
        );
        assert_eq!(tiling, StorageTiling::per_axis(&[2, 2]));
        assert_eq!(Layout::of(&geometry, &tiling.order(&[M, N])), layout);
    }

    /// A grid ordered `K` first puts the next tile along `K` right after this one: the physical
    /// dims keep cubecl's order and only the strides move.
    #[test]
    fn a_k_first_grid_moves_the_strides_not_the_dims() {
        let layout = Layout::tile(&[(N, 16), (K, 32)])
            .grid(&[K, N])
            .over(&[(K, 128), (N, 64)])
            .unwrap();
        let (geometry, _) = layout.physical(&[K, N]);
        // [k/32, n/16, 32, 16]: a tile is 512 values, the next along K 512 on, along N 4 tiles on.
        assert_eq!(
            geometry,
            Geometry::new(&[(4, 512), (4, 2048), (32, 16), (16, 1)])
        );
    }

    /// Two tiles and a grid: every size a product, the extents divided once, at the top.
    #[test]
    fn levels_multiply_and_only_the_grid_divides() {
        let layout = Layout::tile(&[(K, 8)])
            .tile(&[(K, 4), (N, 16)])
            .grid(&[K, N])
            .over(&[(K, 256), (N, 32)])
            .unwrap();
        assert_eq!(
            &layout.dense[..],
            &[(K, 8), (K, 4), (N, 16), (K, 8), (N, 2)]
        );
        let refused = Layout::tile(&[(K, 8)])
            .grid(&[K, N])
            .over(&[(K, 12), (N, 4)]);
        assert_eq!(
            refused,
            Err(LayoutMisfit::PartialTile {
                axis: K,
                extent: 12,
                tile: 8
            })
        );
    }
}
