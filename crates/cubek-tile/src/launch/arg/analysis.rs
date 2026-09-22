//! What the [`Arg`](super::Arg) builder decides about an operand, one named step each, and the
//! one [`Refusal`] any of them can answer with. Every step is a value a test can build without a
//! client.

use core::fmt::{self, Display, Formatter};

use cubecl::zspace::{SmallVec, Tiling};

use super::BoundaryPolicy;
use crate::{
    Axis, Boundary, Geometry, Launcher, Level, LineMisfit, PhysicalAxisMap, Projection, Space,
    Storage, StorageTiling,
};

/// Why an operand cannot be bound as described.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Refusal {
    /// [`in_stride_order`](super::Arg::in_stride_order) on a gathered mapping or a storage-tiled
    /// binding: the order a buffer steps is read off strides, which a mapping states for itself
    /// and a tiled binding stores in tiles.
    StrideOrderOfAStatedLayout,
    /// A gathered mapping addresses the buffer's own dims, so a storage-tiled binding has no
    /// reading under it.
    GatherOfATiledBinding(Tiling),
    /// A storage-tiled operand's storage tile is matched against the kernel's levels, which this
    /// launch does not state.
    TiledWithoutLevels,
    /// The operand is stored in `tile` storage tiles, which is the tile of no level of the kernel's
    /// nest (the levels cut it to `cuts`); the space owns the storage tile's size.
    StorageTileOfNoLevel {
        tile: Vec<(Axis, usize)>,
        cuts: Vec<Vec<(Axis, usize)>>,
    },
    /// [`gathered`](super::Arg::gathered) states the mapping outright, so `axes` and `batches`
    /// have nothing left to describe.
    GatherWithLabels,
    /// The mapping addresses `mapped` dims but the operand has `rank`.
    GatherRankMismatch { mapped: usize, rank: usize },
    /// The mapping spans an axis the launched space does not have.
    GatherOfAnUnknownAxis(Axis),
    /// The tiling describes `tiled` axes but `labelled` were stated.
    TilingRankMismatch { tiled: usize, labelled: usize },
    /// The operand's rank is smaller than its labelled block of dims.
    RankBelowLabels { rank: usize, block: usize },
    /// More leading dims than batch axes to label them.
    UnlabelledBatchDims { dims: usize, axes: usize },
    /// An operand must address at least one coordinate.
    NoCoordinate,
    /// The buffer cannot be served `width` wide.
    WidthNotServed { width: usize, why: LineMisfit },
    /// The innermost axis is served in vector lines but is not provably in bounds (it overhangs
    /// its tiling, or its map is affine and reaches past any extent stated here); serve it scalar,
    /// or state [`BoundaryPolicy::Unchecked`] if the launch proves its lines are in bounds.
    UncheckableVectorEdge,
    /// A gathered operand cannot be quantized: its scale grid is shaped over its logical axes,
    /// which its buffer's dims no longer match.
    QuantizedGather,
}

impl Display for Refusal {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::StrideOrderOfAStatedLayout => write!(
                f,
                "Arg::in_stride_order: the order a buffer steps is read off strides, which a \
                 gathered mapping states for itself and a storage-tiled binding stores in tiles"
            ),
            Refusal::GatherOfATiledBinding(tiling) => write!(
                f,
                "Arg::gathered: the mapping addresses the buffer's own dims, so a storage-tiled \
                 binding ({tiling:?}) has no reading here"
            ),
            Refusal::TiledWithoutLevels => write!(
                f,
                "Arg: a storage-tiled operand's storage tile is the tile of one of the kernel's \
                 levels, which this launch does not state"
            ),
            Refusal::StorageTileOfNoLevel { tile, cuts } => write!(
                f,
                "Arg: this operand is stored in {tile:?} storage tiles, which is the tile of no \
                 level of the kernel's nest (the levels cut it to {cuts:?}); the space owns the \
                 storage tile's size, so pack the tensor to one of its tiles"
            ),
            Refusal::GatherWithLabels => write!(
                f,
                "Arg::gathered: the mapping is stated outright, so `axes` and `batches` have \
                 nothing left to describe"
            ),
            Refusal::GatherRankMismatch { mapped, rank } => write!(
                f,
                "Arg::gathered: the mapping addresses {mapped} dims but the operand has {rank}"
            ),
            Refusal::GatherOfAnUnknownAxis(axis) => write!(
                f,
                "Arg::gathered: the mapping spans {axis:?}, which the launched space does not have"
            ),
            Refusal::TilingRankMismatch { tiled, labelled } => write!(
                f,
                "Arg: the tiling describes {tiled} axes but {labelled} were labelled"
            ),
            Refusal::RankBelowLabels { rank, block } => write!(
                f,
                "Arg: operand rank {rank} is smaller than its labelled block of {block} dims"
            ),
            Refusal::UnlabelledBatchDims { dims, axes } => {
                write!(f, "Arg: {dims} batch dims but only {axes} batch axes given")
            }
            Refusal::NoCoordinate => {
                write!(f, "Arg: an operand must address at least one coordinate")
            }
            Refusal::WidthNotServed { width, why } => {
                write!(
                    f,
                    "Arg::vectorize: this operand cannot be served {width} wide: {why}"
                )
            }
            Refusal::UncheckableVectorEdge => write!(
                f,
                "Arg: the innermost axis is served in vector lines but is not provably in bounds \
                 (it overhangs its tiling, or its map is affine and reaches past any extent stated \
                 here); serve it scalar, or state BoundaryPolicy::Unchecked if the launch proves \
                 its vector lines are in bounds"
            ),
            Refusal::QuantizedGather => write!(
                f,
                "Arg::quantized: a gathered operand cannot be quantized; its scale grid is shaped \
                 over its logical axes, which its buffer's dims no longer match"
            ),
        }
    }
}

impl std::error::Error for Refusal {}

/// The labelled reading of a buffer: its dims named by the operand's axes (leading dims by the
/// batch axes, right-aligned, size-1 broadcast dims dropped; trailing dims by the labelled axes in
/// the storage tiling's level-major order), as a [`Projection`] and the settled geometry.
pub(crate) struct Labels {
    pub(crate) geometry: Geometry,
    pub(crate) projection: Projection,
    /// Every logical axis the buffer carries, before broadcast dims dropped: what the bounds-check
    /// is derived over.
    pub(crate) addressed: Vec<Axis>,
}

impl Labels {
    pub(crate) fn new(
        geometry: &Geometry,
        axes: &[Axis],
        batches: &[Axis],
        tiling: Option<StorageTiling>,
    ) -> Result<Self, Refusal> {
        let rank = geometry.rank();
        let tiling = tiling.unwrap_or_else(|| StorageTiling::uniform(axes.len(), 0));
        if tiling.rank() != axes.len() {
            return Err(Refusal::TilingRankMismatch {
                tiled: tiling.rank(),
                labelled: axes.len(),
            });
        }
        let block = tiling.order(axes);
        if rank < block.len() {
            return Err(Refusal::RankBelowLabels {
                rank,
                block: block.len(),
            });
        }
        let batch_dims = rank - block.len();
        if batch_dims > batches.len() {
            return Err(Refusal::UnlabelledBatchDims {
                dims: batch_dims,
                axes: batches.len(),
            });
        }
        let addressed: Vec<Axis> = batches[batches.len() - batch_dims..]
            .iter()
            .chain(&block)
            .copied()
            .collect();

        let mut maps = Vec::new();
        let mut logical: Vec<Axis> = Vec::new();
        let mut dims = Vec::new();
        for (&axis, (extent, stride)) in addressed.iter().zip(geometry.dims()) {
            // A labelled axis never drops out, however small, since the tile is shaped over it.
            if batches.contains(&axis) && extent == 1 && !axes.contains(&axis) {
                continue;
            }
            maps.push(PhysicalAxisMap::of(axis));
            if !logical.contains(&axis) {
                logical.push(axis);
            }
            dims.push((extent, stride));
        }
        Ok(Labels {
            geometry: Geometry::new(&dims),
            projection: Projection::new(&logical, &maps),
            addressed,
        })
    }

    /// A stated gathered mapping, checked against the binding and the launch: it refuses a
    /// storage-tiled binding, nothing else labels it, it addresses every dim, and it spans only
    /// axes the kernel's space has. The geometry is kept as it stands.
    pub(crate) fn stated(
        geometry: &Geometry,
        launch: &Launcher,
        projection: Projection,
        axes: &[Axis],
        batches: &[Axis],
        stored: Tiling,
    ) -> Result<Self, Refusal> {
        if stored.is_tiled() {
            return Err(Refusal::GatherOfATiledBinding(stored));
        }
        if !(axes.is_empty() && batches.is_empty()) {
            return Err(Refusal::GatherWithLabels);
        }
        if projection.physical_rank() != geometry.rank() {
            return Err(Refusal::GatherRankMismatch {
                mapped: projection.physical_rank(),
                rank: geometry.rank(),
            });
        }
        let space = launch.partitioning().space();
        if let Some(&axis) = projection
            .logical_axes()
            .iter()
            .find(|&&axis| !space.contains(axis))
        {
            return Err(Refusal::GatherOfAnUnknownAxis(axis));
        }
        Ok(Labels {
            geometry: geometry.clone(),
            addressed: projection.logical_axes().to_vec(),
            projection,
        })
    }
}

/// Which level of the kernel's nest a storage-tiled operand's storage tile is the tile of. A tensor
/// stored tiles-of-tiles deep names one level per nesting, coarse to fine; [`at`](crate::Tile::at)
/// descends to the innermost. Matched on the labelled axes alone: a batch dim is one physical dim.
pub(crate) struct StorageLevel {
    innermost: usize,
}

impl StorageLevel {
    pub(crate) fn new(
        geometry: &Geometry,
        axes: &[Axis],
        tiling: &StorageTiling,
        space: &Space,
        levels: &[Level],
    ) -> Result<Self, Refusal> {
        if levels.is_empty() {
            return Err(Refusal::TiledWithoutLevels);
        }
        let fragments = Self::fragments(geometry, axes, tiling);
        let tile_of = |i: usize| -> Vec<(Axis, usize)> {
            let child = space.leaf(&levels[..=i]);
            axes.iter()
                .map(|&axis| (axis, child.extent(axis)))
                .collect()
        };
        let mut innermost = None;
        let mut from = 0;
        for nesting in 0..tiling.max_fragments() - 1 {
            // The storage tile at this nesting: what its finer fragments multiply to, per axis. An
            // axis stored as one fragment is whole; one stored shallower than the nesting reaches
            // has no storage tile here, and its edge of one matches no tile.
            let tile: Vec<(Axis, usize)> = axes
                .iter()
                .zip(&fragments)
                .map(|(&axis, extents)| match extents.len() {
                    1 => (axis, space.extent(axis)),
                    _ => (
                        axis,
                        extents[(nesting + 1).min(extents.len())..].iter().product(),
                    ),
                })
                .collect();
            let level = (from..levels.len())
                .find(|&i| tile_of(i) == tile)
                .ok_or_else(|| Refusal::StorageTileOfNoLevel {
                    tile: tile.clone(),
                    cuts: (from..levels.len()).map(tile_of).collect(),
                })?;
            innermost = Some(level);
            from = level + 1;
        }
        Ok(StorageLevel {
            innermost: innermost.expect("a tiled operand has at least one nesting"),
        })
    }

    pub(crate) fn storage(&self) -> Storage {
        Storage::Tiled(self.innermost)
    }

    /// Each axis's fragment extents, coarsest first, read off the trailing (labelled) dims.
    fn fragments(geometry: &Geometry, axes: &[Axis], tiling: &StorageTiling) -> Vec<Vec<usize>> {
        let order = tiling.order(axes);
        let batch_dims = geometry.rank() - order.len();
        let dims = &geometry.shape()[batch_dims..];
        axes.iter()
            .map(|&axis| {
                order
                    .iter()
                    .zip(dims)
                    .filter(|&(&a, _)| a == axis)
                    .map(|(_, &extent)| extent)
                    .collect()
            })
            .collect()
    }
}

impl Boundaries {
    /// The mode: the stated policy, else whether the operand overhangs or underflows.
    fn mode(
        policy: BoundaryPolicy,
        projection: &Projection,
        addressed: &[Axis],
        concrete: &Space,
        overhangs: &[Axis],
    ) -> Option<Boundary> {
        match policy {
            BoundaryPolicy::Unchecked => None,
            BoundaryPolicy::Every(boundary) => Some(boundary),
            BoundaryPolicy::Derived => {
                let overhangs = addressed
                    .iter()
                    .filter(|&&axis| concrete.contains(axis))
                    .any(|axis| overhangs.contains(axis));
                (overhangs || projection.may_underflow()).then_some(Boundary::Zero)
            }
        }
    }
}

/// Where the bounds-check lands: on the coordinate axes that can leave the buffer, and only those.
/// A settled axis would pay for a mask that can never fire, and a settled *innermost* axis must be
/// left alone outright, since a window clamps in lines and would alias the edge line.
pub(crate) struct Boundaries {
    /// One mode per coordinate axis; empty when nothing is checked.
    pub(crate) modes: SmallVec<[Option<Boundary>; Space::MAX_RANK]>,
}

impl Boundaries {
    /// `concrete` is the launch's real-extent space and `overhangs` the axes its tiles reach
    /// past; an operand built over geometry alone is judged by the same launch.
    pub(crate) fn new(
        policy: BoundaryPolicy,
        projection: &Projection,
        addressed: &[Axis],
        concrete: &Space,
        overhangs: &[Axis],
        width: usize,
    ) -> Result<Self, Refusal> {
        let boundary = Self::mode(policy, projection, addressed, concrete, overhangs);
        let coords = projection.untiled();
        let coord_rank = projection.coordinate_rank();
        if coord_rank == 0 {
            return Err(Refusal::NoCoordinate);
        }

        // Whether coordinate axis `pa` is inside the buffer by construction. Only an identity map
        // can be: it reaches exactly as far as its own coordinate, so it stays inside whenever its
        // tiling divides, and its zero offset keeps it clear of the underflow half above.
        //
        // A non-identity map (a negative or `Dynamic` offset, an affine reach past any stated
        // extent) is the caller's to size the buffer for, the trust the derivation runs on; no
        // proof here can retire its policy. An axis the concrete space does not describe is
        // unproven, not proven.
        let settled = |pa: usize| match coords.physical_axis(pa).identity_axis() {
            Some(axis) => concrete.contains(axis) && !overhangs.contains(&axis),
            None => false,
        };

        // A vector line only needs a scalar fallback when its own innermost axis is unsettled.
        // Other coordinate axes may still be masked or clamped independently (NHWC interpolation
        // clamps H/W while serving a contiguous C line).
        if boundary.is_some() && width > 1 && !settled(coord_rank - 1) {
            return Err(Refusal::UncheckableVectorEdge);
        }

        let modes: SmallVec<[Option<Boundary>; Space::MAX_RANK]> = (0..coord_rank)
            .map(|pa| boundary.filter(|_| !settled(pa)))
            .collect();
        // An all-`None` list collapses to the empty one, so "nothing is checked" has one form.
        Ok(match modes.iter().any(Option::is_some) {
            true => Boundaries { modes },
            false => Boundaries {
                modes: SmallVec::new(),
            },
        })
    }
}
