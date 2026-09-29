//! [`StoragePartitioning`]: how a tensor's values are laid down in memory, as nested tiles.

use core::fmt::{self, Display, Formatter};

use crate::{Axis, Geometry, StorageTiling};

/// How a tensor's values are laid down: stated tiles, finest first, then the rest's order.
///
/// ```ignore
/// // 32 x 32 tiles of a [k, n] weight, stored for a four-wide read, the tiles along N
/// let storage = StorageLevels::new(&[(N, 4)])
///     .tile(&[(N, 8), (K, 32)])
///     .grid(&[N, K]);
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct StoragePartitioning {
    /// Stated tiles, finest first: each `(axis, count)` holds `count` of the piece below it.
    tiles: Vec<(Axis, usize)>,
    /// The rest of each axis, finest first; its counts are what the extents leave.
    order: Vec<Axis>,
}

impl StoragePartitioning {
    /// The statement a bound buffer was written from, read off its dims by stride; `labels` name
    /// the trailing dims. `None` where the buffer's order is no partitioning.
    pub fn new(geometry: &Geometry, labels: &[Axis]) -> Option<Self> {
        let rank = geometry.rank();
        let unlabelled = rank.saturating_sub(labels.len());
        let tiling = geometry.tiling();
        // Dims past the logical rank are stated pieces.
        let first_stated = match tiling.is_tiled() {
            true => tiling.logical_rank(rank).unwrap_or(rank),
            false => rank,
        };
        let mut dims: Vec<(usize, usize)> = geometry
            .dims()
            .enumerate()
            .filter(|&(dim, (extent, stride))| dim >= unlabelled && extent > 1 && stride > 0)
            .map(|(dim, (_, stride))| (dim, stride))
            .collect();
        dims.sort_by_key(|&(dim, stride)| (stride, core::cmp::Reverse(dim)));
        let (stated, rest): (Vec<_>, Vec<_>) =
            dims.iter().partition(|&&(dim, _)| dim >= first_stated);
        let finest_rest = rest.first().map_or(usize::MAX, |&(_, stride)| stride);
        if stated.iter().any(|&(_, stride)| stride > finest_rest) {
            return None;
        }
        Some(Self {
            tiles: stated
                .iter()
                .map(|&(dim, _)| (labels[dim - unlabelled], geometry.shape()[dim]))
                .collect(),
            order: rest
                .iter()
                .map(|&(dim, _)| labels[dim - unlabelled])
                .collect(),
        })
    }

    /// The stated tiles, finest first.
    pub fn tiles(&self) -> &[(Axis, usize)] {
        &self.tiles
    }

    /// The order the rest of each axis follows, finest first.
    pub fn order(&self) -> &[Axis] {
        &self.order
    }

    /// Whether `tile`, finest first, is held by this partitioning over a tensor of `extents`.
    /// Stated tiles are taken whole; the rest of an axis may be divided.
    pub fn holds(
        &self,
        tile: &[(Axis, usize)],
        extents: &[(Axis, usize)],
    ) -> Result<(), TileMisfit> {
        let mut pieces = self.pieces(extents)?;
        pieces.reverse();
        for &(axis, count) in tile {
            let mut run = 1;
            while run < count {
                let Some(mut piece) = pieces.pop() else {
                    return Err(TileMisfit::RunsOut { wanted: axis });
                };
                if piece.count == 1 {
                    continue;
                }
                if piece.axis != axis {
                    return Err(TileMisfit::Interleaved {
                        wanted: axis,
                        found: piece.axis,
                    });
                }
                let missing = count / run;
                if missing.is_multiple_of(piece.count) {
                    run *= piece.count;
                    continue;
                }
                if piece.stated || !piece.count.is_multiple_of(missing) {
                    return Err(TileMisfit::Overshoots {
                        axis,
                        wanted: count,
                        piece: piece.count,
                    });
                }
                piece.count /= missing;
                pieces.push(piece);
                run = count;
            }
        }
        Ok(())
    }

    /// The tiles, finest first, a window can be addressed inside with one stride per axis.
    pub(crate) fn contiguous_tiles(&self, extents: &[(Axis, usize)]) -> Vec<Vec<(Axis, usize)>> {
        let Ok(pieces) = self.pieces(extents) else {
            return Vec::new();
        };
        let mut tiles = Vec::new();
        let mut tile: Vec<(Axis, usize)> = Vec::new();
        for piece in pieces.into_iter().filter(|piece| piece.count > 1) {
            let continues = tile.last().is_some_and(|&(axis, _)| axis == piece.axis);
            match tile.iter_mut().find(|(axis, _)| *axis == piece.axis) {
                // An axis met again after another one sat between: no longer one run per axis.
                Some(_) if !continues => break,
                Some((_, extent)) => *extent *= piece.count,
                None => tile.push((piece.axis, piece.count)),
            }
            tiles.push(tile.clone());
        }
        tiles
    }

    /// The buffer storing a tensor of `extents`: level-major `(extent, stride)` dims and tiling.
    pub fn physical(
        &self,
        extents: &[(Axis, usize)],
    ) -> Result<(Geometry, StorageTiling), StorageMisfit> {
        let axes: Vec<Axis> = extents.iter().map(|&(axis, _)| axis).collect();
        let named = |axis: &Axis| self.order.iter().filter(|&a| a == axis).count() == 1;
        if self.order.len() != axes.len() || !axes.iter().all(named) {
            return Err(StorageMisfit::Order {
                order: self.order.clone(),
                axes,
            });
        }
        let pieces = self.pieces(extents).map_err(|misfit| match misfit {
            TileMisfit::Unwhole { axis, extent, tile } => {
                StorageMisfit::PartialTile { axis, extent, tile }
            }
            _ => unreachable!("only the rest of an axis is computed against its extent"),
        })?;
        let fragments: Vec<usize> = axes
            .iter()
            .map(|&axis| pieces.iter().filter(|p| p.axis == axis).count())
            .collect();
        let tiling = StorageTiling::per_axis(&fragments);
        let strides: Vec<usize> = pieces
            .iter()
            .scan(1, |run, piece| {
                let stride = *run;
                *run *= piece.count;
                Some(stride)
            })
            .collect();
        // Each axis's pieces coarsest first, matching the tiling's level-major order.
        let per_axis: Vec<Vec<usize>> = axes
            .iter()
            .map(|&axis| {
                (0..pieces.len())
                    .rev()
                    .filter(|&p| pieces[p].axis == axis)
                    .collect()
            })
            .collect();
        let mut dims = Vec::with_capacity(pieces.len());
        for level in 0..tiling.max_fragments() {
            for of_axis in &per_axis {
                if let Some(&p) = of_axis.get(level) {
                    dims.push((pieces[p].count, strides[p]));
                }
            }
        }
        Ok((Geometry::new(&dims), tiling))
    }

    /// Every piece over a tensor of `extents`, finest first: stated tiles, then each rest.
    fn pieces(&self, extents: &[(Axis, usize)]) -> Result<Vec<Piece>, TileMisfit> {
        let mut pieces: Vec<Piece> = self
            .tiles
            .iter()
            .map(|&(axis, count)| Piece {
                axis,
                count,
                stated: true,
            })
            .collect();
        for &axis in &self.order {
            let extent = extents
                .iter()
                .find(|&&(a, _)| a == axis)
                .map_or(1, |&(_, extent)| extent);
            let tile: usize = self
                .tiles
                .iter()
                .filter(|&&(a, _)| a == axis)
                .map(|&(_, count)| count)
                .product();
            if !extent.is_multiple_of(tile) {
                return Err(TileMisfit::Unwhole { axis, extent, tile });
            }
            pieces.push(Piece {
                axis,
                count: extent / tile,
                stated: false,
            });
        }
        Ok(pieces)
    }
}

/// A [`StoragePartitioning`] being stated leaf-up, until [`grid`](Self::grid).
#[derive(Clone, Debug)]
pub struct StorageLevels {
    tiles: Vec<(Axis, usize)>,
}

impl StorageLevels {
    /// The finest tile, `(axis, count)` finest first, in values.
    pub fn new(tile: &[(Axis, usize)]) -> Self {
        Self {
            tiles: tile.to_vec(),
        }
    }

    /// A coarser tile, each count how many of the tile below it holds.
    pub fn tile(mut self, level: &[(Axis, usize)]) -> Self {
        self.tiles.extend_from_slice(level);
        self
    }

    /// The partitioning, the rest of each axis following `order`, finest first.
    pub fn grid(self, order: &[Axis]) -> StoragePartitioning {
        StoragePartitioning {
            tiles: self
                .tiles
                .into_iter()
                .filter(|&(_, count)| count > 1)
                .collect(),
            order: order.to_vec(),
        }
    }
}

/// Where a tile stops being held by a [`StoragePartitioning`] ([`StoragePartitioning::holds`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileMisfit {
    /// A piece of another axis sits inside a wanted piece: the wanted tile is not one run.
    Interleaved { wanted: Axis, found: Axis },
    /// A piece reaches past the wanted count.
    Overshoots {
        axis: Axis,
        wanted: usize,
        piece: usize,
    },
    /// The partitioning ends before the wanted piece closes.
    RunsOut { wanted: Axis },
    /// The stated tiles do not close the axis's extent.
    Unwhole {
        axis: Axis,
        extent: usize,
        tile: usize,
    },
}

impl Display for TileMisfit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Interleaved { wanted, found } => write!(
                f,
                "a piece along {found:?} sits inside the one wanted along {wanted:?}"
            ),
            Self::Overshoots {
                axis,
                wanted,
                piece,
            } => write!(
                f,
                "a piece of {piece} along {axis:?} reaches past the {wanted} wanted, and is not \
                 the rest of the axis, which it would divide"
            ),
            Self::RunsOut { wanted } => {
                write!(f, "nothing is left along {wanted:?} for the piece wanted")
            }
            Self::Unwhole { axis, extent, tile } => write!(
                f,
                "axis {axis:?} runs {extent}, which is not a whole number of {tile}-wide tiles"
            ),
        }
    }
}

/// Why a [`StoragePartitioning`] cannot store a tensor ([`StoragePartitioning::physical`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StorageMisfit {
    /// The tiles along `axis` do not close its extent.
    PartialTile {
        axis: Axis,
        extent: usize,
        tile: usize,
    },
    /// The order does not name each of the tensor's axes once.
    Order { order: Vec<Axis>, axes: Vec<Axis> },
}

impl Display for StorageMisfit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::PartialTile { axis, extent, tile } => write!(
                f,
                "axis {axis:?} runs {extent}, which is not a whole number of {tile}-wide tiles"
            ),
            Self::Order { order, axes } => write!(
                f,
                "the tiles follow {order:?}, but the tensor's axes are {axes:?}: the order names \
                 each once"
            ),
        }
    }
}

/// One piece of a partitioning over a tensor: `count` of the piece below it along `axis`.
#[derive(Clone, Copy, Debug)]
struct Piece {
    axis: Axis,
    count: usize,
    /// Whether the count was stated, so a reader must take the piece whole.
    stated: bool,
}
