//! How an operand's bytes move: the [`Delivery`] (strided cooperative copy or TMA bulk
//! copy) and its type-level twin [`DeliveryFamily`], which lets one kernel body serve
//! both argument types.

use cubecl::prelude::*;

use crate::{Space, Tile, TileSpec, TmaData, TmaTileArg};

/// How an operand's bytes move out of it: a strided cooperative copy or a TMA hardware
/// bulk copy. Read off a tile via [`delivery`](crate::Tile::delivery); the staging sync
/// comes from it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Delivery {
    #[default]
    Strided,
    Tma,
}

/// CUDA caps each TMA box dimension at 256; a bulk copy fills one smem stage, so the
/// stage edges are the box dims.
const TMA_MAX_BOX_DIM: usize = 256;

impl Delivery {
    pub fn is_tma(&self) -> bool {
        matches!(self, Delivery::Tma)
    }

    /// Reject a plan the TMA descriptor path can't encode, so a bad plan fails here as a
    /// clean error instead of at descriptor encoding on the driver. `boxes` are the
    /// bulk-copy box dims (one stage per box); `batched` = any surviving batch dim.
    /// A no-op unless this is [`Delivery::Tma`].
    pub fn validate_tma(&self, boxes: &[usize], batched: bool) -> Result<(), String> {
        if !self.is_tma() {
            return Ok(());
        }
        // The descriptor is 3-D `(batch, row, col)`; surviving batch dims need a
        // batch-aware descriptor path not wired yet.
        if batched {
            return Err("TMA: batched problems are not supported yet".to_string());
        }
        if let Some(&max) = boxes.iter().max()
            && max > TMA_MAX_BOX_DIM
        {
            return Err(format!(
                "TMA: box {boxes:?} exceeds the {TMA_MAX_BOX_DIM}-per-axis box limit"
            ));
        }
        Ok(())
    }
}

/// How a derived smem stage lays out its buffer: storage-tiled at the final tile (one
/// contiguous block per fragment) or plain strided rows (legacy `sync_full_strided`).
/// A per-operand comptime plan config ([`stage`](crate::StridedTileSource::stage)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum StageStorage {
    Tiled,
    Strided,
}

impl StageStorage {
    /// The safe default: a cmma leaf reads a whole fragment per transaction, so tile its stages.
    /// Anything else keeps plain strided rows, the manual-mma leaf included: it addresses each
    /// element by computed offset, so contiguity buys it nothing.
    pub fn for_space(space: &Space) -> Self {
        if space.partitioner().leaf().is_cmma() {
            StageStorage::Tiled
        } else {
            StageStorage::Strided
        }
    }
}

/// How an operand's shared-memory stages are laid out and cooperatively filled: the tile
/// `layout` and the launch's `units` (cube size). One comptime value threaded from the
/// operand's [`Storage`](crate::Storage) to every stage derived from it, so a fill never
/// re-derives either.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StagePlan {
    pub layout: StageStorage,
    /// The launch's cube size (units per cube), `0` when unknown. A comptime worker count
    /// lets a fill emit straight-line tasks instead of a rolled loop whose runtime
    /// `CUBE_DIM` stride blocks unrolling; `0` falls back to the rolled loop.
    pub units: usize,
}

impl StagePlan {
    /// The default layout for `space` (tiled for a cmma leaf, else strided) with an unknown
    /// worker count. A [`Launcher`](crate::Launcher) stamps `units` on top.
    pub fn for_space(space: &Space) -> Self {
        StagePlan {
            layout: StageStorage::for_space(space),
            units: 0,
        }
    }

    /// A plain strided stage with an unknown worker count.
    pub fn strided() -> Self {
        StagePlan {
            layout: StageStorage::Strided,
            units: 0,
        }
    }
}

impl Default for StagePlan {
    fn default() -> Self {
        StagePlan::strided()
    }
}

/// [`Delivery`]'s type-level twin: which launchable argument carries an operand and how a
/// kernel serves that argument as a [`Tile`]. The operand is a plain tensor whose element
/// type carries the served width (`Vector<E, V>`, `V` a launch-fed [`Size`]), and the
/// comptime [`TileSpec`] arrives through the seam. A kernel body written over
/// `D: DeliveryFamily` runs strided or TMA unchanged; the launch entry picks the family.
/// One family covers both operands, since [`Sync::of`](crate::Sync::of) rejects a mixed
/// pair anyway.
#[cube]
pub trait DeliveryFamily: Send + core::marker::Sync + 'static {
    /// The launchable argument carrying one operand. `?Sized` because a plain
    /// `Tensor` is a slice-backed unsized type; kernels take it by reference.
    type Arg<E: Numeric, V: Size>: LaunchArg + CubeType + ?Sized;

    /// Serve the argument as a [`Tile`]: the kernel's one `space` projected onto the
    /// operand's `spec.axes`.
    fn tile<E: Numeric, V: Size>(
        arg: &Self::Arg<E, V>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> Tile<E>;
}

/// [`Delivery::Strided`]'s family: a plain `Tensor<Vector<E, V>>`, tiled in-kernel by
/// [`Tile::of`].
pub struct Strided;

/// [`Delivery::Tma`]'s family: a tensor map ([`TmaTileArg`]), hardware bulk-copied.
pub struct Tma;

#[cube]
impl DeliveryFamily for Strided {
    type Arg<E: Numeric, V: Size> = Tensor<Vector<E, V>>;

    fn tile<E: Numeric, V: Size>(
        arg: &Self::Arg<E, V>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> Tile<E> {
        Tile::of(arg, space, spec)
    }
}

#[cube]
impl DeliveryFamily for Tma {
    type Arg<E: Numeric, V: Size> = TmaTileArg<E>;

    fn tile<E: Numeric, V: Size>(
        arg: &Self::Arg<E, V>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> Tile<E> {
        // The width and storage don't apply to a tensor-map operand; only the projection.
        TmaData::from_tensor_map(arg.view.clone(), comptime!(space.project(&spec.axes)))
    }
}
