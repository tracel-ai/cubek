//! [`Refusal`]: why a launch cannot run as described, or an operand cannot be bound to it.

use core::fmt::{self, Display, Formatter};

use cubecl::zspace::Tiling;

use crate::{Axis, LineMisfit};

/// Why a launch cannot run as described, or an operand cannot be bound to it.
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
    /// A gathered mapping read in vector lines whose innermost physical axis does not step by the
    /// operand's last logical axis at coefficient 1.
    GatherInnermostNotInLines,
    /// A gathered mapping that reads `axis` off several physical axes: the operand is
    /// storage-tiled, or reads one axis from two places.
    GatherAxisAddressedTwice(Axis),
    /// A grid read off the levels ([`Grid::FromLevels`](crate::launch::Grid::FromLevels)) needs a
    /// units level of one unit or the whole `plane`; this one holds `units`.
    GridNotReadOffLevels { units: u32, plane: u32 },
    /// A cube of `units` units, past the `most` the device holds.
    CubePastDevice { units: u32, most: u32 },
    /// `fillers` planes are set aside to fill a walk's stages, and the device has no barrier type
    /// for the two roles to meet on.
    FillersWithoutBarriers { fillers: u32 },
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
            Refusal::GatherInnermostNotInLines => write!(
                f,
                "Arg::gathered: the innermost physical axis must step by the operand's last \
                 logical axis at coefficient 1 (it is addressed in vector lines)"
            ),
            Refusal::GatherAxisAddressedTwice(axis) => write!(
                f,
                "Arg::gathered: logical axis {axis:?} addresses several physical axes, so it is \
                 either storage-tiled (a gathered operand must be untiled gmem) or read off two \
                 places at once"
            ),
            Refusal::GridNotReadOffLevels { units, plane } => write!(
                f,
                "Launcher: a grid read off the levels needs a units level of one unit or the \
                 whole plane ({plane}), got {units}"
            ),
            Refusal::CubePastDevice { units, most } => write!(
                f,
                "Launcher: a cube of {units} units, but the device holds at most {most}"
            ),
            Refusal::FillersWithoutBarriers { fillers } => write!(
                f,
                "Launcher: {fillers} plane(s) are set aside to fill a walk's stages, and this \
                 device carries no barrier type for the two roles to meet on"
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
