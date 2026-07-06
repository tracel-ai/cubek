//! A stream is one lane of the double-buffered `mma` pipeline: it owns its slot's staged lhs/rhs
//! tiles and the barrier(s) that sequence its fill against its read. The schedule drives two
//! streams, calling `load` then `mma` on each in turn — the wait is not a call the schedule makes,
//! it lives inside `load`/`mma`.
//!
//! Two fill kinds, picked by the barriers the stream holds (nothing is inferred from the operands):
//!
//! - **strided** (no barriers): a cooperative element copy rendezvoused on the one cube-wide
//!   `sync_cube`. The sync sits in `load` (before the refill), where a single collective per phase
//!   covers both hazards — the fill→read (RAW) of this lane and the read→refill (WAR) of the other.
//!   This is the portable path (e.g. Metal, which has no mbarrier).
//!
//! - **TMA** (a `full`/`empty` mbarrier pair, after CUTLASS): a hardware bulk copy, producer and
//!   consumer decoupled so the copy overlaps compute. `full` (producer→consumer, arrival count 1)
//!   flips when the elected unit has issued the loads and declared their transaction bytes; `empty`
//!   (consumer→producer, arrival count `CUBE_DIM`) flips when every unit has finished reading and
//!   released the slot. `load` waits `empty` (WAR) then issues the copy and arrives `full`; `mma`
//!   waits `full` (RAW) then reads and arrives `empty`. A per-lane `phase` bit tracks the parity.
//!
//! TMA is dormant — no launch path builds a tma source yet — so in practice both lanes run strided.

use cubecl::prelude::barrier::Barrier;
use cubecl::prelude::*;

use super::*;

/// One lane of the double buffer: its staged operands, a phase bit, and the barrier(s) sequencing
/// fill vs read. `barriers` is empty for a strided lane or holds `[full, empty]` for a TMA lane —
/// its presence *is* the fill kind, so nothing is inferred from the source operands.
#[derive(CubeType)]
pub struct Stream<Lhs: Numeric, Rhs: Numeric> {
    a: Tile<Lhs>,
    b: Tile<Rhs>,
    barriers: Sequence<Shared<Barrier>>,
    /// mbarrier parity for this lane's `wait_parity`; flips once per `mma`. Unused on the strided
    /// path.
    phase: u32,
}

#[cube]
impl<Lhs: Numeric, Rhs: Numeric> Stream<Lhs, Rhs> {
    /// Build one lane over its staged `a`/`b` slot buffers and its `barriers` (empty = strided,
    /// `[full, empty]` = TMA). The caller owns the fill-kind decision; the stream just carries it.
    pub fn new(
        a: Tile<Lhs>,
        b: Tile<Rhs>,
        barriers: Sequence<Shared<Barrier>>,
    ) -> Stream<Lhs, Rhs> {
        Stream::<Lhs, Rhs> {
            a,
            b,
            barriers,
            phase: 0,
        }
    }

    /// Fill this lane from `lhs`/`rhs` at `region`. Strided: rendezvous first (frees the lane after
    /// its prior read and publishes the sibling's fill), then a cooperative element copy. TMA: wait
    /// the lane is free (`empty`, WAR), issue the async bulk copy onto `full`, then arrive `full`
    /// with the whole-stage transaction bytes (elected unit only). Never blocks on the fill itself —
    /// that wait is `mma`'s.
    pub fn load(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>, region: Region) {
        if comptime!(self.barriers.len() > 0) {
            let full = self.barriers.index(0);
            let empty = self.barriers.index(1);
            empty.wait_parity(self.phase ^ 1);
            self.a.stage_tma(&lhs.at(&region), full);
            self.b.stage_tma(&rhs.at(&region), full);
            let bytes = self.a.buffer_bytes() + self.b.buffer_bytes();
            if UNIT_POS == 0 {
                full.arrive_and_expect_tx(1, bytes);
            }
        } else {
            sync_cube();
            self.a.stage(&lhs.at(&region));
            self.b.stage(&rhs.at(&region));
        }
    }

    /// Contract this lane into `out` at `region`. TMA waits the lane's fill (`full`, RAW), reads,
    /// then releases the slot (`empty`) and flips the phase; the strided lane already rendezvoused
    /// in `load`, so it reads straight through.
    pub fn mma<Acc: Numeric>(&mut self, out: &mut Tile<Acc>, region: Region) {
        if comptime!(self.barriers.len() > 0) {
            let full = self.barriers.index(0);
            let empty = self.barriers.index(1);
            full.wait_parity(self.phase);
            out.at(&region).mma(&self.a, &self.b);
            empty.arrive();
            self.phase = self.phase ^ 1;
        } else {
            out.at(&region).mma(&self.a, &self.b);
        }
    }
}
