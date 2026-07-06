//! The double-buffered `mma` pipeline: two slots, each pairing both operands' staged tiles under
//! one barrier, so a slot costs a single collective rather than one per operand.
//!
//! The synchronization is not a call the schedule makes: [`produce`](Stream::produce) pushes a
//! slot's fill, and [`consume`](Stream::consume) is where its tiles cross from the fill stream to
//! the compute — that crossing is the one place the wait appears. A cube rendezvous both waits the
//! fill (RAW) and fences the previous reader so the next `produce` can overwrite (WAR); one per
//! slot keeps a single collective per region, matching the old `sync_cube` count.
//!
//! Two fills, decided once from the operands' [`Payload`] kind (which is comptime — so the match
//! folds and the mbarrier type is emitted only on the TMA build, never on the strided one or on
//! backends without mbarrier support such as Metal):
//! - strided — a cooperative element copy; the rendezvous is a portable [`sync_cube`].
//! - TMA — a hardware bulk copy onto a slot mbarrier; the rendezvous also waits the transaction
//!   count, so the DMA overlaps the caller's compute.
//!
//! The kind lives only in the payload; a populated `barriers` sequence (length per slot) is what
//! [`produce`](Stream::produce)/[`consume`](Stream::consume) read to pick the fill — no separate
//! flag shadows it.

use cubecl::prelude::barrier::Barrier;
use cubecl::prelude::*;

use super::*;

/// The two-slot double buffer. Each slot pairs both operands' staged tiles under one barrier.
/// Buffers live in `Sequence`s (like [`Ring`]) so their lifetime-erased shared-memory handles
/// survive being stored. `barriers` holds one per slot for a TMA fill, and is empty for a strided
/// fill — its presence *is* the fill kind.
#[derive(CubeType)]
pub struct Stream<Lhs: CubePrimitive, Rhs: CubePrimitive> {
    a: Sequence<Tile<Lhs>>,
    b: Sequence<Tile<Rhs>>,
    barriers: Sequence<Shared<Barrier>>,
}

#[cube]
impl<Lhs: CubePrimitive, Rhs: CubePrimitive> Stream<Lhs, Rhs> {
    /// Build the double buffer over `a`/`b`, reading the fill kind off the operands `lhs`/`rhs`.
    /// The payload matches fold (comptime-variant enum), so the mbarrier is created only when both
    /// operands are TMA sources.
    pub fn new(
        a: Sequence<Tile<Lhs>>,
        b: Sequence<Tile<Rhs>>,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
    ) -> Stream<Lhs, Rhs> {
        #[allow(clippy::match_like_matches_macro)]
        let lhs_tma = match &lhs.payload {
            Payload::TmaGmem(_) => true,
            _ => false,
        };
        #[allow(clippy::match_like_matches_macro)]
        let rhs_tma = match &rhs.payload {
            Payload::TmaGmem(_) => true,
            _ => false,
        };
        let mut barriers = Sequence::<Shared<Barrier>>::new();
        // Uniform delivery, as in cubek-matmul; a mixed pair takes the strided rendezvous.
        if lhs_tma && rhs_tma {
            barriers.push(Barrier::shared(CUBE_DIM, UNIT_POS == 0));
            barriers.push(Barrier::shared(CUBE_DIM, UNIT_POS == 0));
            // Bulk copy writes through the async proxy; fence the barrier inits before any load.
            sync_async_proxy_shared();
        }
        Stream::<Lhs, Rhs> { a, b, barriers }
    }

    /// Push slot `slot`'s fill: stage both operands. TMA issues the bulk copies (async, returns
    /// immediately, overlaps whatever the caller computes next); the strided path copies
    /// element-wise. Never waits — that is [`consume`](Stream::consume)'s job.
    pub fn produce(&mut self, #[comptime] slot: usize, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) {
        if comptime!(self.barriers.len() > 0) {
            let barrier = self.barriers.index(slot);
            self.a.index_mut(slot).stage_tma(lhs, barrier);
            self.b.index_mut(slot).stage_tma(rhs, barrier);
        } else {
            self.a.index_mut(slot).stage(lhs);
            self.b.index_mut(slot).stage(rhs);
        }
    }

    /// Wait until slot `slot`'s fill has landed and fence the previous reader, then its tiles are
    /// readable. TMA waits the transaction count (the DMA in flight during the caller's last
    /// compute); the strided path is a portable cube rendezvous.
    pub fn consume(&self, #[comptime] slot: usize) {
        if comptime!(self.barriers.len() > 0) {
            let barrier = self.barriers.index(slot);
            let bytes = self.a.index(slot).buffer_bytes() + self.b.index(slot).buffer_bytes();
            let expected = select(UNIT_POS == 0, bytes, 0);
            let token = barrier.arrive_and_expect_tx(1, expected);
            barrier.wait(token);
        } else {
            sync_cube();
        }
    }

    pub fn a(&self, #[comptime] slot: usize) -> &Tile<Lhs> {
        self.a.index(slot)
    }

    pub fn b(&self, #[comptime] slot: usize) -> &Tile<Rhs> {
        self.b.index(slot)
    }
}
