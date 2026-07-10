//! [`Staging`]: a matmul-agnostic staging slot, a payload `T` plus a [`Pipeline`]
//! sequencing its fill against its read. The [`Barrier`](Sync::Barrier) strategy mirrors
//! cubek-matmul's `specialized/matmul.rs`; [`Cube`](Sync::Cube) and [`Solo`](Sync::Solo)
//! are degenerate cases.
//!
//! `fill`/`consume` are hand-written expand methods because a `Drop` guard can't emit a barrier
//! op in cubecl and `#[cube]` rejects `impl Trait` args.

use cubecl::prelude::barrier::Barrier;
use cubecl::prelude::*;
use cubecl::unexpanded;

use crate::{
    CmmaPartition, Delivery, MemData, Space, Tile, TileExpand, TileKind, TileKindExpand,
    partition_level,
};

/// How a slot rendezvouses its fill against its read; fixed comptime at construction
/// from the operands' delivery.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sync {
    /// One unit fills and reads its own slot: no collective (single-plane / CPU).
    Solo,
    /// Cooperative element copy rendezvoused on one cube-wide `sync_cube` per phase. The sync sits
    /// in `write` and covers both this slot's fill→read and the sibling's read→refill.
    Cube,
    /// Hardware async bulk copy (TMA): `full`/`empty` mbarrier pair with a `phase` parity, producer
    /// and consumer decoupled so the copy overlaps compute.
    Barrier,
}

impl Sync {
    /// Deduce the strategy from the operands' [`Delivery`]: both async (TMA) → `Barrier`,
    /// both strided → `Cube`. A mix is rejected.
    pub fn of(lhs: Delivery, rhs: Delivery) -> Sync {
        match (lhs, rhs) {
            (Delivery::Tma, Delivery::Tma) => Sync::Barrier,
            (Delivery::Strided, Delivery::Strided) => Sync::Cube,
            _ => panic!("Staging: mixed delivery — both operands must be TMA sources or neither"),
        }
    }
}

/// The rendezvous for one slot, and every barrier it owns. The acquire/release operations live
/// on [`Staging`]; [`fill`](Pipeline::fill) is the one op a `write` body reaches for directly.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub enum Pipeline {
    /// Synchronous element copy. `collective` → rendezvous on one `sync_cube` per phase; otherwise a
    /// single unit fills its own slot with no collective at all.
    Cube { collective: bool },
    /// Async producer/consumer decoupled over a `full`/`empty` mbarrier pair with a `phase`
    /// parity, so the fill overlaps compute. TMA motivates it, but the barrier itself is
    /// delivery-agnostic; see [`Pipeline::fill`].
    Barrier {
        /// Producer→consumer (one producer arrival): flips once the fill's transaction bytes land.
        full: Shared<Barrier>,
        /// Consumer→producer (one arrival per unit): flips once every unit has read and freed the slot.
        empty: Shared<Barrier>,
        /// mbarrier parity for `wait_parity`; flipped once per read.
        phase: u32,
    },
}

#[cube]
impl Pipeline {
    /// Allocate the pipeline for `sync`: the `full`/`empty` mbarrier pair, sealed by a proxy fence
    /// before any bulk copy, for [`Barrier`](Sync::Barrier); nothing to allocate otherwise.
    fn new(#[comptime] sync: Sync) -> Pipeline {
        match sync {
            Sync::Solo => Pipeline::new_Cube(false),
            Sync::Cube => Pipeline::new_Cube(true),
            Sync::Barrier => {
                // full: one producer arrival; empty: one arrival per unit.
                let full = Barrier::shared(1, UNIT_POS == 0);
                let empty = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
                sync_async_proxy_shared();
                sync_cube();
                Pipeline::new_Barrier(full, empty, 0)
            }
        }
    }

    /// Fill staged `dst` from `src`, the one operation a `fill` body performs. A `Barrier` slot
    /// stages under its `full` mbarrier; a `Cube` slot is a plain blocking
    /// [`copy_from`](Tile::copy_from).
    pub fn fill<E: Numeric>(&self, dst: &mut Tile<E>, src: &Tile<E>) {
        match self {
            Pipeline::Barrier { full, .. } => match (&mut dst.tile_kind, &src.tile_kind) {
                (TileKind::Smem(d), TileKind::TmaGmem(s)) => {
                    if UNIT_POS == 0 {
                        full.expect_tx(d.size_bytes());
                    }
                    s.stage_into(d, full);
                }
                // A strided source under a barrier is a plain synchronous copy.
                (TileKind::Smem(d), TileKind::Gmem(s) | TileKind::Smem(s)) => d.fill_from(s),
                _ => panic!("Pipeline::fill: unsupported kind pairing"),
            },
            Pipeline::Cube { .. } => dst.copy_from(src),
        }
    }
}

/// One slot of the staged `mma` pipeline: its payload `T` and the [`Pipeline`] sequencing fill vs
/// read. Generic over `T`, so the slot is matmul-agnostic — it just hands out a synchronized `&mut T`
/// to fill (`write`) and a synchronized `&T` to consume (`read`).
#[derive(CubeType)]
pub struct Staging<T: CubeType> {
    data: T,
    pipeline: Pipeline,
}

#[cube]
impl<T: CubeType> Staging<T> {
    /// Wrap an already-built payload and pipeline. Private: the public entry is the operand-deducing
    /// [`new`](Staging::new). (Split out so the tuple `T` never sits in a struct-literal turbofish,
    /// which `#[cube]` can't parse; `Staging::<T>` can.)
    fn wrap(data: T, pipeline: Pipeline) -> Staging<T> {
        Staging::<T> { data, pipeline }
    }

    /// Producer acquire: wait the slot is free (`empty`, WAR) for `Barrier`; a `collective` `Cube`
    /// slot rendezvouses on `sync_cube`; a lone-unit one does nothing.
    fn acquire_write(&self) {
        match &self.pipeline {
            Pipeline::Barrier { empty, phase, .. } => empty.wait_parity(*phase ^ 1),
            Pipeline::Cube { collective } => {
                if *collective {
                    sync_cube();
                }
            }
        }
    }

    /// Producer release: arrive `full` (elected unit) so the consumer's `full` wait can pass. The
    /// bytes were declared per tile by [`Pipeline::fill`], so this is a bare arrival. No-op for `Cube`.
    fn release_write(&self) {
        match &self.pipeline {
            Pipeline::Barrier { full, .. } => {
                if UNIT_POS == 0 {
                    full.arrive();
                }
            }
            Pipeline::Cube { .. } => {}
        }
    }

    /// Consumer acquire: wait the slot's fill (`full`, RAW) for `Barrier`; nothing for `Cube` (already
    /// rendezvoused in `write`).
    fn acquire_read(&self) {
        match &self.pipeline {
            Pipeline::Barrier { full, phase, .. } => full.wait_parity(*phase),
            Pipeline::Cube { .. } => {}
        }
    }

    /// Consumer release: arrive `empty` (free the slot) and flip the phase for `Barrier`; nothing for
    /// `Cube`.
    fn release_read(&mut self) {
        match &mut self.pipeline {
            Pipeline::Barrier { empty, phase, .. } => {
                empty.arrive();
                *phase ^= 1;
            }
            Pipeline::Cube { .. } => {}
        }
    }

    /// Publish this slot's last fill when no successor fill's rendezvous will (the walk's final
    /// regions). Only a collective `Cube` slot needs it; reached only through
    /// [`consume_final`](Staging::consume_final).
    fn publish(&self) {
        match &self.pipeline {
            Pipeline::Cube { collective } => {
                if *collective {
                    sync_cube();
                }
            }
            Pipeline::Barrier { .. } => {}
        }
    }
}

#[cube]
impl<Lhs: Numeric, Rhs: Numeric> Staging<(Tile<Lhs>, Tile<Rhs>)> {
    /// Build a slot staging one region of the operands `lhs`/`rhs`. When the level below
    /// `out` is the fragment grid (cmma leaf), the operands stage into plane-private
    /// register partitions ([`Solo`](Sync::Solo)); otherwise fresh shared memory, with
    /// [`Sync`] deduced from the operands' delivery.
    pub fn new(
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] out: Space,
    ) -> Staging<(Tile<Lhs>, Tile<Rhs>)> {
        let lhs_delivery = lhs.delivery();
        let rhs_delivery = rhs.delivery();
        let register = comptime!(
            out.partitioner().leaf().is_cmma() && partition_level(&out.divide()).is_some()
        );
        if register {
            comptime!(assert!(
                !lhs_delivery.is_tma() && !rhs_delivery.is_tma(),
                "Staging: a TMA source cannot stage into registers"
            ));
            let a = CmmaPartition::store(comptime!(lhs.space.divide()), comptime!(out.clone()));
            let b = CmmaPartition::store(comptime!(rhs.space.divide()), comptime!(out.clone()));
            Staging::wrap((a, b), Pipeline::new(Sync::Solo))
        } else {
            let sync = comptime!(Sync::of(lhs_delivery, rhs_delivery));
            Staging::wrap(
                (MemData::smem_like(lhs), MemData::smem_like(rhs)),
                Pipeline::new(sync),
            )
        }
    }
}

// `fill`/`consume` take closures so the body stays caller-defined (fill each buffer however, run the
// mma). They're provided for the `(Tile<Lhs>, Tile<Rhs>)` payload (not generic `T`): closure-parameter
// inference can't resolve the projection `&mut T::ExpandType` through a generic `T`, but resolves the
// spelled-out tiles fine.
impl<Lhs: Numeric, Rhs: Numeric> Staging<(Tile<Lhs>, Tile<Rhs>)> {
    /// Producer: wait the slot is free, run `fill` over the staged buffers and the slot's
    /// [`Pipeline`], then publish. See [`StagingExpand::__expand_fill_method`].
    pub fn fill(&mut self, _fill: impl FnOnce(&mut (Tile<Lhs>, Tile<Rhs>), &Pipeline)) {
        unexpanded!()
    }

    /// Consumer: wait the slot's fill, hand the two staged tiles to `compute`, then free the slot.
    /// See [`StagingExpand::__expand_consume_method`].
    pub fn consume(&mut self, _compute: impl FnOnce(&Tile<Lhs>, &Tile<Rhs>)) {
        unexpanded!()
    }

    /// Consumer for a fill no later fill will publish (the walk's final regions): publish
    /// the slot first, then consume. See [`StagingExpand::__expand_consume_final_method`].
    pub fn consume_final(&mut self, _compute: impl FnOnce(&Tile<Lhs>, &Tile<Rhs>)) {
        unexpanded!()
    }
}

impl<Lhs: Numeric, Rhs: Numeric> StagingExpand<(Tile<Lhs>, Tile<Rhs>)> {
    pub fn __expand_fill_method<F>(&mut self, scope: &Scope, fill: F)
    where
        F: FnOnce(&Scope, &mut (TileExpand<Lhs>, TileExpand<Rhs>), &PipelineExpand),
    {
        self.__expand_acquire_write_method(scope);
        fill(scope, &mut self.data, &self.pipeline);
        self.__expand_release_write_method(scope);
    }

    pub fn __expand_consume_method<F>(&mut self, scope: &Scope, compute: F)
    where
        F: FnOnce(&Scope, &TileExpand<Lhs>, &TileExpand<Rhs>),
    {
        self.__expand_acquire_read_method(scope);
        compute(scope, &self.data.0, &self.data.1);
        self.__expand_release_read_method(scope);
    }

    pub fn __expand_consume_final_method<F>(&mut self, scope: &Scope, compute: F)
    where
        F: FnOnce(&Scope, &TileExpand<Lhs>, &TileExpand<Rhs>),
    {
        self.__expand_publish_method(scope);
        self.__expand_consume_method(scope, compute);
    }
}
