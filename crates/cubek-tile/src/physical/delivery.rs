//! Who moves an operand's bytes: the [`Delivery`] (the cube's own units, or the TMA engine) and
//! its type-level twin [`DeliveryFamily`], which lets one kernel body serve every argument type.
//!
//! How the operand is *stored* is a separate fact, and it rides the spec's
//! [`Storage`](crate::Storage): a storage-tiled operand states the level its tile is the tile of,
//! and every mover here serves it. The two are orthogonal on purpose, so a weight packed to the
//! stage can move under the TMA engine as well as under the units.

use cubecl::prelude::*;

use crate::{Partitioning, Storage, StridedOperand, Sync, Tile, TileArg, TmaTileArg};

/// Who moves an operand into a stage: the cube's own units (a cooperative buffer copy, or a
/// coordinate-backed materialization with no buffer at all), or the TMA engine. Read off a tile
/// via [`delivery`](crate::Tile::delivery); the staging sync comes from it.
///
/// Storage-tiledness is not a variant here. A storage tile is a fact of the data, stated by the
/// spec's [`Storage`], and it only decides how wide a run each stage is: under `Copy` the units
/// copy that run, under `Tma` it is the box the engine fetches.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Delivery {
    #[default]
    Copy,
    Procedural,
    Tma,
}

/// CUDA caps each TMA box dimension at 256; a bulk copy fills one smem stage, so the
/// stage edges are the box dims.
const TMA_MAX_BOX_DIM: usize = 256;

impl Delivery {
    pub fn is_tma(&self) -> bool {
        matches!(self, Delivery::Tma)
    }

    /// The synchronization required to materialize this source in a staging slot.
    pub(crate) fn rendezvous(&self) -> Sync {
        match self {
            Delivery::Copy | Delivery::Procedural => Sync::Cube,
            Delivery::Tma => Sync::Barrier,
        }
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

/// [`Delivery`]'s type-level twin: which launchable argument carries an operand and how a
/// kernel serves that argument as a [`Tile`]. Each argument bundles its own comptime
/// [`TileSpec`] ([`TileArg`] strided or storage-tiled, [`TmaTileArg`] tensor map), so a tensor
/// can never pair with another operand's spec; only the kernel's one [`Space`] crosses the
/// seam. A kernel body written over `D: DeliveryFamily` runs strided, storage-tiled or TMA
/// unchanged; the launch entry picks the family. One family covers both operands, since
/// [`Sync::for_deliveries`](crate::Sync::for_deliveries) rejects a mixed pair anyway.
#[cube]
pub trait DeliveryFamily: Send + core::marker::Sync + 'static {
    /// The launchable argument carrying one operand and its spec.
    type Arg<E: Numeric, V: Size>: LaunchArg + CubeType;

    /// Serve the argument as a [`Tile`]: the kernel's one `space` projected onto the
    /// argument's own spec axes.
    fn tile<E: Numeric, V: Size>(arg: &Self::Arg<E, V>, #[comptime] space: Partitioning)
    -> Tile<E>;
}

/// The families whose argument is a plain tensor + its spec ([`TileArg`]): what a built
/// [`StridedOperand`] launches as. Lets a launch entry written over `D` hand its operands to
/// the kernel without naming the erased element types the kernel's launch is spelled in.
pub trait TensorDelivery: DeliveryFamily {
    /// The operand as this family's launch argument.
    fn operand<E: Numeric, V: Size>(
        operand: StridedOperand,
    ) -> <Self::Arg<E, V> as LaunchArg>::RuntimeArg;
}

impl TensorDelivery for Cooperative {
    fn operand<E: Numeric, V: Size>(
        operand: StridedOperand,
    ) -> <Self::Arg<E, V> as LaunchArg>::RuntimeArg {
        operand.arg::<E, V>()
    }
}

/// [`Delivery::Copy`]'s family: a tensor + spec ([`TileArg`]), the cube's units moving it, tiled
/// in-kernel by [`Tile::of`]. Serves a plain operand and a storage-tiled one alike: the spec's
/// [`Storage`] says which, and a stated storage tile only makes each stage one contiguous run
/// instead of a row at a time. So an activation and a weight packed to the stage ride here
/// together.
pub struct Cooperative;

/// [`Delivery::Tma`]'s family: a tensor map ([`TmaTileArg`]), hardware bulk-copied.
pub struct Tma;

#[cube]
impl DeliveryFamily for Cooperative {
    type Arg<E: Numeric, V: Size> = TileArg<'static, E, V>;

    fn tile<E: Numeric, V: Size>(
        arg: &Self::Arg<E, V>,
        #[comptime] space: Partitioning,
    ) -> Tile<E> {
        comptime!(match arg.spec.storage {
            Storage::Strided | Storage::Tiled(_) => {}
            Storage::Contiguous =>
                panic!("Cooperative: a launched spec is never inside a storage tile"),
        });
        arg.tile(space)
    }
}

#[cube]
impl DeliveryFamily for Tma {
    type Arg<E: Numeric, V: Size> = TmaTileArg<E>;

    fn tile<E: Numeric, V: Size>(
        arg: &Self::Arg<E, V>,
        #[comptime] space: Partitioning,
    ) -> Tile<E> {
        arg.tile(space)
    }
}
