//! [`Layout`]: where a buffer's values sit relative to one another, stated as counts from the
//! finest up.
//!
//! A buffer's dense part is one mixed-radix number: the finest piece says how many values run
//! along its axis before the next piece steps, the next says how many of those runs it holds,
//! and so on. Every size is a product of counts, so between two pieces nothing divides.
//!
//! A layout is read off a binding ([`Layout::of`]), or stated leaf-up for a buffer about to be
//! written ([`Layout::tile`]): each tile made of the one below it, then the order of the grid of
//! tiles.
//!
//! A reader states the coarsest layout it needs, and asks whether the stored one
//! [refines](Layout::refines) it: whether every boundary it names is a boundary of the stored
//! layout. Pieces it does not name fuse, which only multiplies. A packed word, a vector read and
//! a stage tile are all asked the same way; the one division left is a reader cutting its tile
//! out of an extent, a count the tensor's size decided rather than a tile someone stored.

use core::fmt::{self, Display, Formatter};

use cubecl::zspace::SmallVec;

use super::stated::LayoutBuilder;
use crate::{Axis, Geometry, Space, StorageTiling};

/// One piece of a layout: `count` of the piece below it, stepping along `axis`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) struct Piece {
    pub(super) axis: Axis,
    pub(super) count: usize,
    /// Whether someone stored this piece as a tile, so a reader takes it whole; `false` for an
    /// extent, a count the tensor's size decided (a grid, an untiled dim), which a reader may
    /// cut its own tile out of.
    pub(super) stored: bool,
}

/// Where a buffer's values sit relative to one another: its dense part as pieces, finest first,
/// and the strides of whatever lies past it.
///
/// Each piece's stride is the product of the counts before it, which is what makes the part
/// dense; a dim that breaks that (a gap, a broadcast, a dim no axis labels) ends it, and its
/// stride and every coarser one are kept apart, since a read has to be a whole number of them.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Layout {
    /// The dense part, finest first.
    pub(super) dense: SmallVec<[Piece; Space::MAX_RANK]>,
    /// The strides past the dense part, finest first, in values.
    pub(super) outer: SmallVec<[usize; Space::MAX_RANK]>,
}

impl Layout {
    /// The layout a buffer is stored in: `geometry`'s dims from the finest up, while each one
    /// steps by the product of the extents finer than it.
    ///
    /// `labels` name the trailing dims, right-aligned to the geometry: a leading dim they leave
    /// out ends the dense part, as does a dim that does not step by that product. An untiled
    /// buffer is read innermost dim first; a storage-tiled one by stride, since its tiling lists
    /// its pieces coarsest first rather than in the order they sit in memory. Every piece of a
    /// tiled dim but its coarsest was stored; the coarsest, and every untiled dim, is an extent.
    pub fn of(geometry: &Geometry, labels: &[Axis]) -> Self {
        let rank = geometry.rank();
        let unlabelled = rank.saturating_sub(labels.len());
        let tiling = geometry.tiling();
        // A tiling lists every logical dim's coarsest piece first, so the dims past the logical
        // rank are the stored ones.
        let first_stored = match tiling.is_tiled() {
            true => tiling.logical_rank(rank).unwrap_or(rank),
            false => rank,
        };
        let mut dims: Vec<(usize, usize, usize)> = geometry
            .dims()
            .enumerate()
            .map(|(dim, (extent, stride))| (dim, extent, stride))
            .collect();
        match tiling.is_tiled() {
            true => dims.sort_by_key(|&(dim, _, stride)| (stride, core::cmp::Reverse(dim))),
            false => dims.reverse(),
        }
        let mut dense = SmallVec::new();
        let mut outer = SmallVec::new();
        let mut run = 1;
        for (dim, extent, stride) in dims {
            let continues = outer.is_empty() && dim >= unlabelled && stride == run;
            match continues {
                true => {
                    dense.push(Piece {
                        axis: labels[dim - unlabelled],
                        count: extent,
                        stored: dim >= first_stored,
                    });
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

    /// The layout a reader needs, stated finest first: its own tiles, before anything it does not
    /// care how is laid out. A read of `4` along `N` is `Layout::wanted(&[(N, 4)])`.
    pub fn wanted(pieces: &[(Axis, usize)]) -> Self {
        Layout {
            dense: pieces
                .iter()
                .map(|&(axis, count)| Piece {
                    axis,
                    count,
                    stored: true,
                })
                .collect(),
            outer: SmallVec::new(),
        }
    }

    /// This layout seen at `wanted`'s pieces: whether every boundary `wanted` names is one of
    /// this layout's, walking both finest first.
    ///
    /// Each of `wanted`'s pieces takes this layout's next pieces on the same axis until their
    /// counts multiply to exactly its count, fusing them. A stored piece is taken whole or not
    /// at all; an extent may be cut, `wanted`'s count split out of it, which is the only division
    /// and has to come out whole. What `wanted` does not reach is kept as it is.
    ///
    /// # Errors
    ///
    /// Where the walk first breaks: another axis inside a wanted piece, a stored piece that
    /// overshoots it, an extent it does not divide, or no pieces left.
    pub fn refines(&self, wanted: &Layout) -> Result<Layout, Unrefined> {
        let mut rest: Vec<Piece> = self.dense.iter().rev().copied().collect();
        let mut fused: SmallVec<[Piece; Space::MAX_RANK]> = SmallVec::new();
        for &want in &wanted.dense {
            let mut run = 1;
            while run < want.count {
                let Some(mut piece) = rest.pop() else {
                    return Err(Unrefined::RunsOut { wanted: want.axis });
                };
                if piece.count == 1 {
                    continue;
                }
                if piece.axis != want.axis {
                    return Err(Unrefined::Interleaved {
                        wanted: want.axis,
                        found: piece.axis,
                    });
                }
                // What the wanted piece still lacks, a whole number since `run` only ever grows
                // by counts that divide it.
                let missing = want.count / run;
                if missing.is_multiple_of(piece.count) {
                    run *= piece.count;
                    continue;
                }
                // The piece reaches past the wanted boundary, or across it: only an extent may
                // have the rest cut out of it, and only a whole number of times.
                if piece.stored || !piece.count.is_multiple_of(missing) {
                    return Err(Unrefined::Overshoots {
                        axis: want.axis,
                        wanted: want.count,
                        piece: piece.count,
                    });
                }
                piece.count /= missing;
                rest.push(piece);
                run = want.count;
            }
            fused.push(Piece {
                stored: true,
                ..want
            });
        }
        fused.extend(rest.into_iter().rev());
        Ok(Layout {
            dense: fused,
            outer: self.outer.clone(),
        })
    }

    /// Whether this buffer can be read `values` at a time along its finest axis: the read
    /// [refines](Self::refines) it, and every stride past the dense part is a whole number of
    /// reads, or a coarser step would land inside one.
    pub(crate) fn serves(&self, values: usize) -> Result<(), LineMisfit> {
        if values == 1 {
            return Ok(());
        }
        let Some(finest) = self.dense.first() else {
            return Err(match self.outer.first() {
                Some(&stride) => LineMisfit::InnermostStrided(stride),
                None => LineMisfit::NoDims,
            });
        };
        if let Err(why) = self.refines(&Layout::wanted(&[(finest.axis, values)])) {
            return Err(LineMisfit::Unrefined(why));
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
    /// tiling lists them (level-major, coarsest first), each `(extent, stride)`, and the piece
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
            .map(|&axis| self.dense.iter().filter(|p| p.axis == axis).count())
            .collect();
        assert!(
            self.dense.iter().all(|p| axes.contains(&p.axis)) && fragments.iter().all(|&f| f >= 1),
            "Layout::physical: the layout names {:?} but the buffer stands for {axes:?}",
            self.dense.iter().map(|p| p.axis).collect::<Vec<_>>()
        );
        let tiling = StorageTiling::per_axis(&fragments);
        // Each piece's stride is the product of the counts finer than it.
        let strides: Vec<usize> = self
            .dense
            .iter()
            .scan(1, |run, piece| {
                let stride = *run;
                *run *= piece.count;
                Some(stride)
            })
            .collect();
        // An axis's pieces coarsest first, so fragment `level` of `axis` is its `level`th from
        // the top, the order the tiling's level-major emission counts them in.
        let of_axis = |axis: Axis| -> Vec<usize> {
            (0..self.dense.len())
                .rev()
                .filter(|&e| self.dense[e].axis == axis)
                .collect()
        };
        let per_axis: Vec<Vec<usize>> = axes.iter().map(|&axis| of_axis(axis)).collect();
        let mut dims = Vec::with_capacity(self.dense.len());
        for level in 0..tiling.max_fragments() {
            for pieces in &per_axis {
                if let Some(&e) = pieces.get(level) {
                    dims.push((self.dense[e].count, strides[e]));
                }
            }
        }
        (Geometry::new(&dims), tiling)
    }

    /// The pieces, finest first, as `(axis, count)`.
    pub fn pieces(&self) -> Vec<(Axis, usize)> {
        self.dense.iter().map(|p| (p.axis, p.count)).collect()
    }
}

/// Where a stored layout stops refining a wanted one ([`Layout::refines`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unrefined {
    /// A piece of another axis sits inside a wanted piece: the wanted tile is not contiguous.
    Interleaved { wanted: Axis, found: Axis },
    /// A piece reaches past the wanted count: a stored tile no reader may cut, or an extent the
    /// wanted count does not divide.
    Overshoots {
        axis: Axis,
        wanted: usize,
        piece: usize,
    },
    /// The dense part ends before the wanted piece closes.
    RunsOut { wanted: Axis },
}

impl Display for Unrefined {
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
                 an extent it divides"
            ),
            Self::RunsOut { wanted } => {
                write!(
                    f,
                    "the dense part ends before the piece wanted along {wanted:?}"
                )
            }
        }
    }
}

/// Why a [`Layout`] cannot serve a cut of some width: the value that decided it, so a message
/// names the number a reader has to go looking for otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineMisfit {
    /// The innermost dim's own stride, when it is not 1: consecutive values are not one line.
    InnermostStrided(usize),
    /// The read does not refine the buffer's layout: see [`Unrefined`].
    Unrefined(Unrefined),
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
            Self::Unrefined(why) => write!(f, "a line does not fit its layout: {why}"),
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
