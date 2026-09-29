//! [`StoragePartitioning`]: how a tensor's values are laid down in memory.
//!
//! Memory is one line of values, and a tensor is folded onto it the way a kernel's space is cut
//! for its workers: into nested tiles, leaf up, each holding `count` of the one below along its
//! axis, the outermost of every axis holding whatever the extent leaves. A partitioning's digits
//! say who takes a tile; these say where it sits.
//!
//! It holds only what was stated: the tiles, and the order the rest of each axis follows. The
//! extents are the tensor's, and every question that needs them takes them, so one statement
//! describes a tensor of any size ([`StorageLevels`]), and a bound buffer reads back as the
//! statement it was written from ([`StoragePartitioning::new`]).
//!
//! A reader asks whether a tile it needs is [held](StoragePartitioning::holds): made of whole
//! stated tiles, in order, cutting only the rest of an axis. A packed word, a vector read and a
//! stage are all asked the same way. Where a tile sits in memory, dense or padded, is the
//! [`Geometry`]'s to answer ([`Geometry::serves`]).
//!
//! The order of a level's tiles is an order of axes: the strides a buffer carries state no other.
//! A swizzled or space-filling order is not expressible.

use core::fmt::{self, Display, Formatter};

use cubecl::zspace::{MetadataError, Tiling};

use crate::{Axis, Geometry};

/// How a tensor's values are laid down in memory: its stated tiles, finest first, then the order
/// the rest of each axis follows.
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
    /// The rest of each axis, finest first: which way the next outermost tile goes. Its counts
    /// are what the tensor's extents leave.
    order: Vec<Axis>,
}

impl StoragePartitioning {
    /// The statement a bound buffer was written from, read off its dims by stride: every piece of
    /// a storage-tiled axis but its coarsest was stated, and the coarsest, like every untiled dim,
    /// is the rest of its axis. `labels` name the trailing dims, right-aligned to the geometry
    /// ([`Geometry::labels`]); a dim they leave out, a broadcast one and the rest of an axis of
    /// extent one say nothing of the order. A stated piece of one is kept, since it is a dim of
    /// the buffer: its stride says nothing, so it reads back as the finest of the pieces sharing
    /// it, where [`physical`](Self::physical) writes it.
    ///
    /// `None` where the buffer's order is no partitioning: a stated piece coarser than the rest of
    /// an axis.
    pub fn new(geometry: &Geometry, labels: &[Axis]) -> Option<Self> {
        let rank = geometry.rank();
        let unlabelled = rank.saturating_sub(labels.len());
        let tiling = geometry.tiling();
        // A tiling lists every logical dim's coarsest piece first, so the dims past the logical
        // rank were stated.
        let first_stated = match tiling.is_tiled() {
            true => tiling.logical_rank(rank).unwrap_or(rank),
            false => rank,
        };
        let mut dims: Vec<(usize, usize, usize)> = geometry
            .dims()
            .enumerate()
            .filter(|&(dim, (extent, stride))| {
                let piece_of_one = extent == 1 && dim >= first_stated;
                dim >= unlabelled && (piece_of_one || extent > 1 && stride > 0)
            })
            .map(|(dim, (extent, stride))| (dim, extent, stride))
            .collect();
        dims.sort_by_key(|&(dim, extent, stride)| (stride, extent > 1, core::cmp::Reverse(dim)));
        let (stated, rest): (Vec<_>, Vec<_>) =
            dims.iter().partition(|&&(dim, _, _)| dim >= first_stated);
        let finest_rest = rest.first().map_or(usize::MAX, |&(_, _, stride)| stride);
        if stated
            .iter()
            .any(|&(_, extent, stride)| extent > 1 && stride > finest_rest)
        {
            return None;
        }
        Some(Self {
            tiles: stated
                .iter()
                .map(|&(dim, extent, _)| (labels[dim - unlabelled], extent))
                .collect(),
            order: rest
                .iter()
                .map(|&(dim, _, _)| labels[dim - unlabelled])
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

    /// The buffer's dims, coarsest first, each named by the axis it is a piece of, `axes` in the
    /// tensor's own order: what [`Projection::tiled`](crate::Projection::tiled) maps.
    pub fn labels(&self, axes: &[Axis]) -> Vec<Axis> {
        StoragePartitioning::level_major(axes, &self.fragments(axes))
    }

    /// The dims of a buffer storing `axes` (in the tensor's own order) as `fragments` pieces each,
    /// named by their axis, in the order cubecl's [`Tiling`] lists them: every axis's coarsest
    /// piece in the tensor's order, then every axis still split further gives its next, down to
    /// the finest. `[k, n]` stored `[k/32, n/32, 32, 32]` is `[K, N, K, N]`. The one place the order
    /// is written.
    pub(crate) fn level_major(axes: &[Axis], fragments: &[usize]) -> Vec<Axis> {
        assert_eq!(
            axes.len(),
            fragments.len(),
            "StoragePartitioning::level_major: {} axes but {} piece counts",
            axes.len(),
            fragments.len()
        );
        let depth = fragments.iter().copied().max().unwrap_or(0);
        (0..depth)
            .flat_map(|level| {
                axes.iter()
                    .zip(fragments)
                    .filter(move |&(_, &pieces)| level < pieces)
                    .map(|(&axis, _)| axis)
            })
            .collect()
    }

    /// Whether `tile`, finest first, is held by this partitioning over a tensor of `extents`:
    /// each of its pieces is this partitioning's next pieces along the same axis, their counts
    /// multiplying to exactly its count. A stated tile is taken whole or not at all; the rest of an
    /// axis may have the wanted count cut out of it, the one division, which has to come out whole.
    ///
    /// # Errors
    ///
    /// Where the walk first breaks: another axis inside a wanted piece, a stated tile it would
    /// split, the rest of an axis it does not divide, or nothing left to take.
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
                // What the wanted piece still lacks: a whole number, since `run` only ever grows
                // by counts that divide it.
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

    /// The tiles, finest first, a window can be addressed inside with one stride per axis: every
    /// run of pieces over a tensor of `extents` from the finest up, stated or the rest of an axis,
    /// in which each axis's pieces sit next to one another, so a coordinate along it is one
    /// offset times one stride. Each tile is its extent per axis it spans.
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

    /// The buffer this partitioning stores a tensor of `extents` in, the axes in its logical order:
    /// its physical dims in [`labels`](Self::labels)' order, each `(extent, stride)`, and the
    /// [`Tiling`] its metadata records.
    ///
    /// # Errors
    ///
    /// An axis the tiles do not close in whole tiles, an order that does not name each of the
    /// tensor's axes once, or more pieces than a [`Tiling`] records.
    pub fn physical(&self, extents: &[(Axis, usize)]) -> Result<Geometry, StorageMisfit> {
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
        let tiling = Tiling::new(&self.fragments(&axes)).map_err(StorageMisfit::Unrecordable)?;
        // Each piece's stride is the product of the counts finer than it.
        let strides: Vec<usize> = pieces
            .iter()
            .scan(1, |run, piece| {
                let stride = *run;
                *run *= piece.count;
                Some(stride)
            })
            .collect();
        // Each dim takes its axis's next piece from the top: `labels` names an axis's pieces
        // coarsest first, and `pieces` lists them finest first.
        let mut taken = vec![false; pieces.len()];
        let dims: Vec<(usize, usize)> = self
            .labels(&axes)
            .iter()
            .map(|&axis| {
                let p = (0..pieces.len())
                    .rev()
                    .find(|&p| !taken[p] && pieces[p].axis == axis)
                    .expect("an axis has as many pieces as it has dims");
                taken[p] = true;
                (pieces[p].count, strides[p])
            })
            .collect();
        Ok(Geometry::new(&dims).with_tiling(tiling))
    }

    /// How many buffer dims each of `axes` is stored as: one per stated tile along it, and one for
    /// the rest of it.
    fn fragments(&self, axes: &[Axis]) -> Vec<usize> {
        axes.iter()
            .map(|&axis| 1 + self.tiles.iter().filter(|&&(a, _)| a == axis).count())
            .collect()
    }

    /// Every piece over a tensor of `extents`, finest first: the stated tiles, then the rest of
    /// each axis in order, its count what the extent leaves.
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

/// A [`StoragePartitioning`] being stated leaf-up: its tiles, finest first, each holding `count`
/// of the one below, until [`grid`](Self::grid) says the order the rest of each axis follows.
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

    /// The partitioning, the rest of each axis following `order`, finest first: `[K, N]` puts the
    /// next tile along `K` right after this one. A tile's piece of one is kept: it is a dim of the
    /// buffer, as stated.
    pub fn grid(self, order: &[Axis]) -> StoragePartitioning {
        StoragePartitioning {
            tiles: self.tiles,
            order: order.to_vec(),
        }
    }
}

/// Where a tile stops being held by a [`StoragePartitioning`] ([`StoragePartitioning::holds`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileMisfit {
    /// A piece of another axis sits inside a wanted piece: the wanted tile is not one run.
    Interleaved { wanted: Axis, found: Axis },
    /// A piece reaches past the wanted count: a stated tile no reader may split, or the rest of an
    /// axis the wanted count does not divide.
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
    /// More pieces than the buffer's metadata records ([`Tiling`]).
    Unrecordable(MetadataError),
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
            Self::Unrecordable(why) => {
                write!(f, "the buffer's metadata cannot record its pieces: {why:?}")
            }
        }
    }
}

/// One piece of a partitioning over a tensor: `count` of the piece below it along `axis`.
#[derive(Clone, Copy, Debug)]
struct Piece {
    axis: Axis,
    count: usize,
    /// Whether the count was stated, so a reader takes the piece whole; `false` for the rest of an
    /// axis, which a reader may cut its own tile out of.
    stated: bool,
}
