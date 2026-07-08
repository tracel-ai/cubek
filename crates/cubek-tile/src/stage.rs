//! [`Staging`]: a matmul-agnostic double-buffer slot. It owns a payload `T` (for matmul, a
//! `(Tile<Lhs>, Tile<Rhs>)` tuple) and a [`Pipeline`] sequencing the payload's fill against its read.
//! `fill`/`consume` each acquire the slot, run a closure over the payload, then release: the caller
//! never makes the rendezvous call, and no raw barrier escapes.
//!
//! The [`Barrier`](Sync::Barrier) strategy is the reference design (mirrors cubek-matmul's
//! `specialized/matmul.rs`): a `full`/`empty` mbarrier pair with a `phase` parity, producer and
//! consumer decoupled so a hardware bulk copy overlaps compute. `write` waits `empty` (WAR), the
//! fill pushes an async copy onto `full`, `write` arrives `full`; `read` waits `full` (RAW), the
//! body reads, `read` arrives `empty` and flips the phase. [`Cube`](Sync::Cube) (strided, one
//! `sync_cube`) and [`Solo`](Sync::Solo) (single unit, no collective) are degenerate cases.
//!
//! A `Guard` with `Drop` would be the natural spelling, but a `Drop` can't emit a barrier op in
//! cubecl (it never receives a `Scope`), so the release is emitted by the wrapper right after the
//! closure body. And because `#[cube]` rejects `impl Trait` kernel args, `fill`/`consume` are
//! hand-written expand methods (mirroring `ComptimeOption::map`) delegating to the [`Pipeline`].

use cubecl::prelude::barrier::Barrier;
use cubecl::prelude::*;
use cubecl::unexpanded;

use crate::{Tile, TileExpand};

/// How a slot rendezvouses its fill against its read. Comptime — fixed at construction from the
/// operands' delivery, never inferred at the call site.
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
    /// Deduce the strategy from the operands' delivery: both async (TMA) → `Barrier`, both strided →
    /// `Cube`. A mix is rejected.
    pub fn of(lhs_tma: bool, rhs_tma: bool) -> Sync {
        assert!(
            lhs_tma == rhs_tma,
            "Staging: mixed delivery — both operands must be TMA sources or neither"
        );
        if lhs_tma { Sync::Barrier } else { Sync::Cube }
    }
}

/// The rendezvous for one slot, and every barrier it owns. Only [`Barrier`](Pipeline::Barrier) carries
/// an mbarrier or a phase — [`Cube`](Pipeline::Cube) carries just the comptime flag distinguishing a
/// `sync_cube` fill from a lone-unit one. The acquire/release operations live on [`Staging`] (matched
/// off `&self.pipeline`, the way `Tile` operations match off its `tile_kind`); [`fill`](Pipeline::fill)
/// is the one op a `write` body reaches for directly.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub enum Pipeline {
    /// Synchronous element copy. `collective` → rendezvous on one `sync_cube` per phase; otherwise a
    /// single unit fills its own slot with no collective at all.
    Cube { collective: bool },
    /// Async producer/consumer decoupled over a `full`/`empty` mbarrier pair with a `phase` parity, so
    /// the fill overlaps compute. TMA is the fill that motivates it (the bulk copy lands the `full`
    /// transaction), but the barrier itself is delivery-agnostic — see [`Tile::stage_from`].
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

    /// Fill staged `dst` from `src`, the one operation a `fill` body performs. A `Barrier` slot stages
    /// under its `full` mbarrier ([`stage_from`](Tile::stage_from), which itself picks TMA vs a plain
    /// copy off the source); a `Cube` slot is a plain element [`copy_from`](Tile::copy_from).
    pub fn fill<E: Numeric>(&self, dst: &mut Tile<E>, src: &Tile<E>) {
        match self {
            Pipeline::Barrier { full, .. } => dst.stage_from(src, full),
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
}

#[cube]
impl<Lhs: Numeric, Rhs: Numeric> Staging<(Tile<Lhs>, Tile<Rhs>)> {
    /// Build a double-buffer slot staging the gmem operands `lhs`/`rhs` into fresh shared memory.
    /// The [`Sync`] strategy is deduced from the operands' delivery — both TMA sources → async
    /// `Barrier`, otherwise strided `Cube`; mixed delivery is rejected at comptime. Each smem buffer
    /// is sized from its operand (row-major over the level's divide, at the operand's physical width,
    /// so the scalar slice holds `tile_size * width`).
    pub fn new(lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) -> Staging<(Tile<Lhs>, Tile<Rhs>)> {
        let lhs_tma = lhs.is_tma();
        let rhs_tma = rhs.is_tma();
        let sync = comptime!(Sync::of(lhs_tma, rhs_tma));

        Staging::wrap((lhs.smem_like(), rhs.smem_like()), Pipeline::new(sync))
    }
}

// `fill`/`consume` take closures so the body stays caller-defined (fill each buffer however, run the
// mma). They're provided for the `(Tile<Lhs>, Tile<Rhs>)` payload (not generic `T`): closure-parameter
// inference can't resolve the projection `&mut T::ExpandType` through a generic `T`, but resolves the
// spelled-out tiles fine.
impl<Lhs: Numeric, Rhs: Numeric> Staging<(Tile<Lhs>, Tile<Rhs>)> {
    /// Producer: wait the slot is free, run `fill` over its two staged buffers, then publish. `fill`
    /// gets the buffers and the slot's [`Pipeline`]; call [`Pipeline::fill`] per operand and one body
    /// serves every [`Sync`]. See [`StagingExpand::__expand_fill_method`].
    pub fn fill(&mut self, _fill: impl FnOnce(&mut (Tile<Lhs>, Tile<Rhs>), &Pipeline)) {
        unexpanded!()
    }

    /// Consumer: wait the slot's fill, hand the two staged tiles to `compute`, then free the slot.
    /// See [`StagingExpand::__expand_consume_method`].
    pub fn consume(&mut self, _compute: impl FnOnce(&Tile<Lhs>, &Tile<Rhs>)) {
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
}
