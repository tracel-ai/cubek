//! A [`StoragePartitioning`] stated leaf-up for a buffer about to be written: its tiles, finest first, then
//! the grid of tiles, settled against the tensor's extents.

use core::fmt::{self, Display, Formatter};

use cubecl::zspace::SmallVec;

use super::base::Piece;
use crate::{Axis, Space, StoragePartitioning};

/// A [`StoragePartitioning`] being stated leaf-up: its tiles, finest first, each made of the one below it,
/// before the grid says in which order the coarsest follow one another.
#[derive(Clone, Debug)]
pub struct StorageLevels {
    levels: Vec<Vec<(Axis, usize)>>,
}

impl StorageLevels {
    /// A layout stated leaf-up for a buffer about to be written: `tile` is its finest tile,
    /// `(axis, count)` finest first, in values.
    pub fn new(tile: &[(Axis, usize)]) -> Self {
        Self {
            levels: vec![tile.to_vec()],
        }
    }

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
    ///
    /// A tile's piece of one holds nothing and is dropped; the grid keeps one piece per axis,
    /// however few tiles it counts, since the buffer stores every axis it stands for.
    pub fn over(self, extents: &[(Axis, usize)]) -> Result<StoragePartitioning, LayoutMisfit> {
        let mut dense: SmallVec<[(Axis, usize); Space::MAX_RANK]> = self
            .levels
            .into_iter()
            .flatten()
            .filter(|&(_, count)| count > 1)
            .collect();
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
        // The tiles were stored; the grid's counts are the extents what is left of each axis.
        let tiles = dense.len() - self.order.len();
        let dense = dense
            .into_iter()
            .enumerate()
            .map(|(i, (axis, count))| Piece {
                axis,
                count,
                stored: i < tiles,
            })
            .collect();
        Ok(StoragePartitioning {
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
