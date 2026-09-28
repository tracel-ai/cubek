//! Who moves an operand's bytes: the [`Delivery`] (the cube's own units, or the TMA engine), its
//! type-level twin [`DeliveryFamily`], which lets one kernel body serve every argument type, and
//! [`DeliveryLaunch`], the host side that builds each family's argument.
//!
//! How the operand is *stored* is a separate fact, riding the spec's [`Storage`](crate::Storage):
//! a storage-tiled operand states the level its tile is the tile of; every mover here serves it.
//! Orthogonal on purpose, so a weight tiled to the stage can move under TMA as well as the units.

use cubecl::prelude::*;

use crate::{
    AccumulateArg, AccumulateArgLaunch, Bound, Partitioning, Refusal, Rendezvous, Space, Storage,
    Tile, TileArg, TileArgLaunch, TmaOperand, TmaTileArg, TmaTileArgLaunch,
};

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
/// stage edges are the box dims ([`Delivery::moves`]).
const TMA_MAX_BOX_DIM: usize = 256;

impl Delivery {
    /// Whether this delivery moves `stage`, one transfer a stage, and why not where it cannot.
    ///
    /// Only [`Tma`](Delivery::Tma) is bounded: its bulk copy fills a stage per box, so every stage
    /// edge is a box edge, and the descriptor caps each. The cube's own units, copying or
    /// computing, move any stage. A selector sizing a stage asks this of each candidate rather than
    /// holding the limit itself.
    pub fn moves(self, stage: &Space) -> Result<(), Refusal> {
        match self {
            Delivery::Copy | Delivery::Procedural => Ok(()),
            Delivery::Tma => match stage
                .extents()
                .into_iter()
                .find(|&(_, edge)| edge > TMA_MAX_BOX_DIM)
            {
                Some((axis, edge)) => Err(Refusal::BoxPastDescriptor {
                    axis,
                    edge,
                    most: TMA_MAX_BOX_DIM,
                }),
                None => Ok(()),
            },
        }
    }

    pub fn is_tma(&self) -> bool {
        matches!(self, Delivery::Tma)
    }

    /// The synchronization required to materialize this source in a staging slot.
    pub(crate) fn rendezvous(&self) -> Rendezvous {
        match self {
            Delivery::Copy | Delivery::Procedural => Rendezvous::Cube,
            Delivery::Tma => Rendezvous::Barrier,
        }
    }
}

/// [`Delivery`]'s type-level twin: which launchable argument carries an operand and how a kernel
/// serves it as a [`Tile`]. Each argument bundles its comptime [`TileSpec`](crate::TileSpec); only
/// the kernel's one [`Partitioning`] crosses the seam.
///
/// A kernel body written over `D: DeliveryFamily` runs strided, storage-tiled, TMA or accumulating
/// unchanged; the launch entry picks the family. One family covers both operands, since
/// `Rendezvous::for_deliveries` rejects a mixed pair anyway.
#[cube]
pub trait DeliveryFamily: Send + core::marker::Sync + 'static {
    /// The launchable argument carrying one operand and its spec.
    type Arg<E: Numeric, V: Size>: LaunchArg + CubeType;

    /// Serve the argument as a [`Tile`] under the kernel's one `partitioning`.
    fn tile<E: Numeric, V: Size>(
        arg: &Self::Arg<E, V>,
        #[comptime] partitioning: Partitioning,
    ) -> Tile<E>;
}

/// [`DeliveryFamily`]'s host side: how a built operand becomes this family's launch argument.
/// Complete over every family, so a launch entry written over `D` names no argument type.
pub trait DeliveryLaunch: DeliveryFamily {
    /// What this family launches from: a [`Bound`] operand for the tensor families, a tensor map
    /// with its box for TMA.
    type Operand;

    fn arg<E: Numeric, V: Size>(
        operand: Self::Operand,
    ) -> <Self::Arg<E, V> as LaunchArg>::RuntimeArg;
}

/// [`Delivery::Copy`]'s family: a tensor + spec ([`TileArg`]), the cube's units moving it, tiled
/// in-kernel by [`GlobalOperand::tile`](crate::GlobalOperand::tile). Serves plain and storage-tiled
/// operands alike: the spec's [`Storage`] says which; a storage tile only makes each stage one
/// contiguous run instead of a row at a time.
pub struct Cooperative;

/// [`Delivery::Tma`]'s family: a tensor map ([`TmaTileArg`]), hardware bulk-copied.
pub struct Tma;

/// An output several instances add into ([`AccumulateArg`]): the family a product leaves by when
/// the cubes that share a cell each hold a slice of it.
pub struct Accumulated;

#[cube]
impl DeliveryFamily for Cooperative {
    type Arg<E: Numeric, V: Size> = TileArg<'static, E, V>;

    fn tile<E: Numeric, V: Size>(
        arg: &Self::Arg<E, V>,
        #[comptime] partitioning: Partitioning,
    ) -> Tile<E> {
        comptime!(match arg.spec.storage {
            Storage::Strided | Storage::Tiled(_) => {}
            Storage::Contiguous =>
                panic!("Cooperative: a launched spec is never inside a storage tile"),
        });
        arg.tile(partitioning)
    }
}

#[cube]
impl DeliveryFamily for Tma {
    type Arg<E: Numeric, V: Size> = TmaTileArg<E>;

    fn tile<E: Numeric, V: Size>(
        arg: &Self::Arg<E, V>,
        #[comptime] partitioning: Partitioning,
    ) -> Tile<E> {
        arg.tile(partitioning)
    }
}

#[cube]
impl DeliveryFamily for Accumulated {
    type Arg<E: Numeric, V: Size> = AccumulateArg<'static, E>;

    fn tile<E: Numeric, V: Size>(
        arg: &Self::Arg<E, V>,
        #[comptime] partitioning: Partitioning,
    ) -> Tile<E> {
        arg.tile::<V>(partitioning)
    }
}

impl DeliveryLaunch for Cooperative {
    type Operand = Bound;

    fn arg<E: Numeric, V: Size>(operand: Bound) -> TileArgLaunch<'static, E, V> {
        operand.arg()
    }
}

impl DeliveryLaunch for Tma {
    type Operand = TmaOperand;

    fn arg<E: Numeric, V: Size>(operand: TmaOperand) -> TmaTileArgLaunch<E> {
        TmaTileArgLaunch::tensor_map(operand.map, &operand.axes, operand.shape)
    }
}

impl DeliveryLaunch for Accumulated {
    type Operand = Bound;

    fn arg<E: Numeric, V: Size>(operand: Bound) -> AccumulateArgLaunch<'static, E> {
        let spec = operand.spec.clone();
        AccumulateArgLaunch::new(operand.tensor(), spec)
    }
}
