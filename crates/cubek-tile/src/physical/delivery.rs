//! How an operand's bytes move: the [`Delivery`] (cooperative buffer copy or TMA bulk
//! copy) and its type-level twin [`DeliveryFamily`], which lets one kernel body serve
//! both argument types.

use cubecl::prelude::*;
use cubecl::zspace::SmallVec;

use crate::{Leaf, MAX_LEVELS, Space, Sync, Tile, TileArg, TmaTileArg};

/// How an operand reaches a stage: a buffered cooperative copy, coordinate-backed cooperative
/// materialization, or a TMA hardware bulk copy. Read off a tile via
/// [`delivery`](crate::Tile::delivery); the staging sync comes from it.
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
    pub fn rendezvous(&self) -> Sync {
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

/// How a derived smem stage lays out its buffer: storage-tiled at the final tile (one
/// contiguous block per fragment) or plain strided rows (legacy `sync_full_strided`).
/// A per-operand comptime plan config ([`storage`](crate::StridedTileSource::storage)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum StageStorage {
    Tiled,
    Strided,
}

/// Where an operand's cells physically sit while the level below reads or writes them. One
/// vocabulary for both directions: an input is *filled* from its source into its residence, an
/// output *drains* from its residence into its sink ([`promote`](crate::Tile::promote)).
///
/// Stated per level, coarse to fine, by the operand itself ([`StagePlan`]). A level says only how
/// deeply its walk is buffered ([`Buffering`](crate::Buffering)); where each of its operands lives
/// is the operand's own business, so one level can stage `lhs` into shared memory while `rhs`
/// streams straight from where it already is.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Residence {
    /// Read where the operand already is: a global window, a recipe evaluated at the leaf, or
    /// registers a level above already holds. The level's walk still runs its ring, but its slots
    /// allocate nothing and fill nothing: each holds the operand whole, and the read selects the
    /// region's own window (or block of fragments) out of it.
    InPlace,
    /// A cooperatively filled shared-memory buffer the leaf reads windows from. How many physical
    /// buffers back it is the ring's business, not the operand's: one per slot while the walk moves
    /// its window, one for the whole ring once the walk leaves it fixed (see
    /// [`WindowMode`](crate::WindowMode)).
    Smem,
    /// Plane-private register fragments, selected by comptime coordinate (so the level's walk
    /// unrolls).
    Plane,
}

impl StageStorage {
    /// The safe default for an operand that becomes `leaf`: a cmma fragment load reads a whole
    /// transaction, so tile its stages. Anything else keeps plain strided rows, the manual-mma leaf
    /// included: it addresses each element by computed offset, so contiguity buys it nothing.
    pub fn for_leaf(leaf: Leaf) -> Self {
        match leaf {
            Leaf::Cmma => StageStorage::Tiled,
            Leaf::Memory { .. } | Leaf::Mma { .. } => StageStorage::Strided,
        }
    }
}

/// Where an operand lives at each level of its space, plus the two facts a materialized level
/// needs to lay a buffer out: the `storage` layout and the launch's `units` (cube size). One
/// comptime value threaded from the operand's [`TileSpec`](crate::TileSpec) through every stage
/// derived from it, so a fill never re-derives either.
///
/// The residences are a stream, not an indexed table: [`head`](StagePlan::head) is the current
/// level's, and [`descend`](StagePlan::descend) pops it wherever a space is
/// [`divide`](crate::Space::divide)d. Plan and partitioner therefore stay in step with no depth
/// arithmetic, and a plan that runs out answers [`InPlace`](Residence::InPlace) forever: below the
/// last level there is only the leaf, which reads its operands where they are.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct StagePlan {
    residence: SmallVec<[Residence; MAX_LEVELS]>,
    /// How a buffer built from this plan lays itself out, whether the staging walk builds it or a
    /// caller does. A level whose residence is [`InPlace`](Residence::InPlace) builds none, and
    /// leaves this unread.
    pub storage: StageStorage,
    /// The launch's cube size (units per cube), `0` when unknown. A comptime worker count
    /// lets a fill emit straight-line tasks instead of a rolled loop whose runtime
    /// `CUBE_DIM` stride blocks unrolling; `0` falls back to the rolled loop.
    pub units: usize,
}

impl StagePlan {
    /// A plan staging nothing: every level [`InPlace`](Residence::InPlace). The default, so an
    /// operand that states no residence is read where it already lives.
    pub fn in_place() -> Self {
        StagePlan::new(&[], StageStorage::Strided, 0)
    }

    /// The default for an operand that becomes `leaf` (tiled for cmma, else strided), staging
    /// nothing and with an unknown worker count. A [`Launcher`](crate::Launcher) stamps `units` and
    /// the caller its `residence` on top.
    pub fn for_leaf(leaf: Leaf) -> Self {
        StagePlan::new(&[], StageStorage::for_leaf(leaf), 0)
    }

    /// A plan over `residence`, one entry per level of the operand's space, coarse to fine.
    pub fn new(residence: &[Residence], storage: StageStorage, units: usize) -> Self {
        StagePlan {
            residence: SmallVec::from_slice(residence),
            storage,
            units,
        }
    }

    /// This level's residence. An exhausted plan answers [`InPlace`](Residence::InPlace).
    pub fn head(&self) -> Residence {
        self.residence
            .first()
            .copied()
            .unwrap_or(Residence::InPlace)
    }

    /// The plan one level down, this level's residence consumed. Called wherever a space is
    /// divided, so the two descend together.
    pub fn descend(&self) -> Self {
        StagePlan {
            residence: self.residence.iter().skip(1).copied().collect(),
            storage: self.storage,
            units: self.units,
        }
    }
}

impl Default for StagePlan {
    fn default() -> Self {
        StagePlan::in_place()
    }
}

/// [`Delivery`]'s type-level twin: which launchable argument carries an operand and how a
/// kernel serves that argument as a [`Tile`]. Each argument bundles its own comptime
/// [`TileSpec`] ([`TileArg`] strided, [`TmaTileArg`] tensor map), so a tensor can never
/// pair with another operand's spec; only the kernel's one [`Space`] crosses the seam. A
/// kernel body written over `D: DeliveryFamily` runs strided or TMA unchanged; the launch
/// entry picks the family. One family covers both operands, since
/// [`Sync::for_deliveries`](crate::Sync::for_deliveries) rejects a mixed pair anyway.
#[cube]
pub trait DeliveryFamily: Send + core::marker::Sync + 'static {
    /// The launchable argument carrying one operand and its spec.
    type Arg<E: Numeric, V: Size>: LaunchArg + CubeType;

    /// Serve the argument as a [`Tile`]: the kernel's one `space` projected onto the
    /// argument's own spec axes.
    fn tile<E: Numeric, V: Size>(arg: &Self::Arg<E, V>, #[comptime] space: Space) -> Tile<E>;
}

/// [`Delivery::Copy`]'s family: a plain tensor + spec ([`TileArg`]), tiled in-kernel
/// by [`Tile::of`].
pub struct Strided;

/// [`Delivery::Tma`]'s family: a tensor map ([`TmaTileArg`]), hardware bulk-copied.
pub struct Tma;

#[cube]
impl DeliveryFamily for Strided {
    type Arg<E: Numeric, V: Size> = TileArg<'static, E, V>;

    fn tile<E: Numeric, V: Size>(arg: &Self::Arg<E, V>, #[comptime] space: Space) -> Tile<E> {
        arg.tile(space)
    }
}

#[cube]
impl DeliveryFamily for Tma {
    type Arg<E: Numeric, V: Size> = TmaTileArg<E>;

    fn tile<E: Numeric, V: Size>(arg: &Self::Arg<E, V>, #[comptime] space: Space) -> Tile<E> {
        arg.tile(space)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The plan is consumed in lockstep with the level chain, so what a tile asks for is always
    /// the head, and descending is what keeps the two aligned.
    #[test]
    fn a_plan_hands_out_one_residence_per_level() {
        let plan = StagePlan::new(
            &[Residence::Smem, Residence::InPlace, Residence::Plane],
            StageStorage::Strided,
            0,
        );
        assert_eq!(plan.head(), Residence::Smem);
        assert_eq!(plan.descend().head(), Residence::InPlace);
        assert_eq!(plan.descend().descend().head(), Residence::Plane);
    }

    /// Below the last level there is only the leaf, which reads its operands where they are, so an
    /// exhausted plan keeps answering rather than running out of entries.
    #[test]
    fn an_exhausted_plan_stays_in_place() {
        let plan = StagePlan::new(&[Residence::Smem], StageStorage::Strided, 0);
        assert_eq!(plan.descend().head(), Residence::InPlace);
        assert_eq!(plan.descend().descend().head(), Residence::InPlace);
        assert_eq!(StagePlan::in_place().head(), Residence::InPlace);
    }

    /// The layout and worker count are facts about the operand, not about one level, so they
    /// survive the descent that consumes the residences.
    #[test]
    fn descending_keeps_the_storage_facts() {
        let plan = StagePlan::new(&[Residence::Smem], StageStorage::Tiled, 128);
        let below = plan.descend();
        assert_eq!(below.storage, StageStorage::Tiled);
        assert_eq!(below.units, 128);
    }
}
