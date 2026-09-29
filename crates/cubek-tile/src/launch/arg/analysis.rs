//! The [`Arg`](super::Arg) builder's derivation steps and the [`Refusal`] they can return.

use core::fmt::{self, Display, Formatter};

use cubecl::zspace::{SmallVec, Tiling};

use super::BoundaryPolicy;
use crate::{
    Axis, Boundary, Geometry, Launcher, LineMisfit, PhysicalAxisMap, Projection, Space,
    StorageTiling,
};

/// Why an operand cannot be bound as described.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Refusal {
    /// [`in_stride_order`](super::Arg::in_stride_order) on a gathered or storage-tiled operand.
    StrideOrderOfAStatedLayout,
    /// A gathered mapping over a storage-tiled binding.
    GatherOfATiledBinding(Tiling),
    /// [`gathered`](super::Arg::gathered) combined with `axes` or `batches`.
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
    /// The innermost axis is served in vector lines but is not provably in bounds.
    UncheckableVectorEdge,
    /// A TMA box edge past the descriptor's per-axis limit `most`.
    BoxPastDescriptor {
        axis: Axis,
        edge: usize,
        most: usize,
    },
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
            Refusal::BoxPastDescriptor { axis, edge, most } => box_past(f, *axis, *edge, *most),
        }
    }
}

/// [`Refusal::BoxPastDescriptor`]'s message.
fn box_past(f: &mut Formatter<'_>, axis: Axis, edge: usize, most: usize) -> fmt::Result {
    write!(
        f,
        "TMA: {edge} along {axis:?} exceeds the {most}-per-axis box limit"
    )
}

impl std::error::Error for Refusal {}

/// The labelled reading of a buffer, as a [`Projection`] and the settled geometry.
pub(crate) struct Labels {
    pub(crate) geometry: Geometry,
    pub(crate) projection: Projection,
    /// Every logical axis the buffer carries, before broadcast dims dropped.
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
            // A labelled axis never drops out: the tile is shaped over it.
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

    /// A stated gathered mapping, checked against the binding and the launch.
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

/// Where the bounds-check lands: on the coordinate axes that can leave the buffer, only those.
pub(crate) struct Boundaries {
    /// One mode per coordinate axis; empty when nothing is checked.
    pub(crate) modes: SmallVec<[Option<Boundary>; Space::MAX_RANK]>,
}

impl Boundaries {
    /// `concrete` is the launch's real-extent space and `overhangs` the axes its tiles reach past.
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

        // Whether coordinate axis `pa` is in bounds by construction; only an identity map can be.
        let settled = |pa: usize| match coords.physical_axis(pa).identity_axis() {
            Some(axis) => concrete.contains(axis) && !overhangs.contains(&axis),
            None => false,
        };

        if boundary.is_some() && width > 1 && !settled(coord_rank - 1) {
            return Err(Refusal::UncheckableVectorEdge);
        }

        let modes: SmallVec<[Option<Boundary>; Space::MAX_RANK]> = (0..coord_rank)
            .map(|pa| boundary.filter(|_| !settled(pa)))
            .collect();
        // "Nothing is checked" has one form: the empty list.
        Ok(match modes.iter().any(Option::is_some) {
            true => Boundaries { modes },
            false => Boundaries {
                modes: SmallVec::new(),
            },
        })
    }
}
