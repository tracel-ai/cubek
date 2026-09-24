//! What every slot of one stages decides at comptime ([`StagePlan`]): how it rendezvouses, and
//! per operand when it is filled.
//!
//! Deduced once from the operands, so the stages' slots agree by construction rather than by each
//! re-deriving the same answers off the same tiles.

use super::payload::StageOperand;
use crate::*;

/// When a slot's buffer is brought to its region across the walk. The walk moves each operand's
/// window or it does not, and a window that never moves need be neither refilled nor duplicated
/// per slot; those two savings are the same fact, so one mode carries both.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Refill {
    /// The walk moves this operand's window, so every region refills it.
    EveryRegion,
    /// The window does not move across the walk: filled once, above the loop.
    Once,
    /// Filled once, and this slot reads the first slot's buffer rather than one of its own.
    /// Nobody fills it here.
    Shared,
}

impl Refill {
    /// This operand's refill in a later slot, given its refill in the first: a window that does
    /// not move is read from the first slot's buffer, one that moves is rebuilt per slot.
    fn in_later_slot(self) -> Refill {
        match self {
            Refill::Once => Refill::Shared,
            Refill::EveryRegion => Refill::EveryRegion,
            Refill::Shared => unreachable!("Shared is produced only in later slots"),
        }
    }
}

/// What every slot of one stages decides at comptime: how it rendezvouses, and per operand when
/// it is filled.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct StagePlan {
    refills: Vec<Refill>,
    sync: Rendezvous,
    collective_full: bool,
    fillers: usize,
}

impl StagePlan {
    /// Read the operands a walk of `level` over `op_space` stages, or refuse a walk whose fill
    /// cannot reach every cell it must.
    ///
    /// # Panics
    ///
    /// A walk that sets planes aside to fill stages one of its operands fills cooperatively: the
    /// cooperative fill deals its elements out over every unit position of the cube, so planes
    /// that are not there leave their share of the stage unwritten, and quietly — the slot
    /// publishes on schedule and the wrong bytes are read.
    pub(crate) fn new(operands: &[StageOperand], op_space: &Space, level: &Level) -> StagePlan {
        let deliveries: Vec<_> = operands.iter().map(|op| op.delivery).collect();
        let fillers = level.fillers();
        let sync = Rendezvous::for_deliveries(&deliveries, fillers);
        let collective_full = Rendezvous::collective_full(&deliveries);
        assert!(
            fillers == 0 || !collective_full,
            "Slot: a slot that mixes a cooperative fill with a bulk copy cannot be filled by \
             a subset of the cube, and this walk sets {fillers} plane(s) aside to fill it"
        );
        // Fix an operand only when its window is invariant across the walk. A barrier slot arrives
        // `full` once per fill, so lifting one operand out of the joint fill leaves its parity
        // counting fills that no longer happen; that and a dynamic level fall back to streaming.
        let can_fix_invariants = op_space.is_static() && sync != Rendezvous::Barrier;
        let refills = operands
            .iter()
            .map(
                |op| match can_fix_invariants && walk_invariant(level, op_space, &op.space) {
                    true => Refill::Once,
                    false => Refill::EveryRegion,
                },
            )
            .collect();
        StagePlan {
            refills,
            sync,
            collective_full,
            fillers,
        }
    }

    /// When each of slot `slot`'s operands is filled, in the order the payload holds them. The
    /// first slot owns every buffer; a later slot reuses the first slot's for whatever the walk
    /// never rewrites.
    pub(crate) fn refills(&self, slot: usize) -> Vec<Refill> {
        match slot {
            FIRST_SLOT => self.refills.clone(),
            _ => self.refills.iter().map(|r| r.in_later_slot()).collect(),
        }
    }

    /// How this walk's slots rendezvous with their readers.
    pub(crate) fn sync(&self) -> Rendezvous {
        self.sync
    }

    /// Whether every unit of the cube arrives at a slot's fill.
    pub(crate) fn collective_full(&self) -> bool {
        self.collective_full
    }

    /// Planes of the cube that fill this walk's stages and take no tile ([`Level::filled_by`]).
    pub(crate) fn fillers(&self) -> usize {
        self.fillers
    }
}

/// Whether a walk of `level` over `space` leaves `operand`'s window unchanged: every axis the
/// walk steps (more than one tile) is absent from the operand, as in broadcast omission. A staged
/// walk fills such an operand once, above the loop. Host-side, static extents.
fn walk_invariant(level: &Level, space: &Space, operand: &Space) -> bool {
    space
        .axes()
        .all(|axis| level.tiles(space, axis) == 1 || !operand.contains(axis))
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// A space over `M`/`N`/`K` cut once by `level`, plus a projection per operand so a slot can
    /// be planned against it. `lhs` spans `M`/`K`, `rhs` spans `K`/`N`, so a `K` walk moves both.
    fn spaces() -> (Space, Space, Space) {
        let space = Space::new(&[(M, 8), (N, 8), (K, 8)]);
        let lhs = space.subspace(&[M, K]);
        let rhs = space.subspace(&[K, N]);
        (space, lhs, rhs)
    }

    fn operand(delivery: Delivery, space: &Space) -> StageOperand {
        StageOperand {
            delivery,
            space: space.clone(),
        }
    }

    #[test]
    fn a_streamed_operand_is_rebuilt_in_every_slot() {
        let (space, lhs, rhs) = spaces();
        let level = Level::every(&[(M, 8), (N, 8), (K, 4)]);
        let plan = StagePlan::new(
            &[operand(Delivery::Copy, &lhs), operand(Delivery::Copy, &rhs)],
            &space,
            &level,
        );
        for slot in 0..2 {
            assert_eq!(plan.refills(slot)[0], Refill::EveryRegion);
        }
    }

    /// An operand whose window the walk never moves is filled once and shares its buffer.
    #[test]
    fn a_fixed_operand_reuses_the_first_slots_buffer() {
        let (space, lhs, rhs) = spaces();
        let level = Level::every(&[(M, 8), (N, 4), (K, 8)]);
        let plan = StagePlan::new(
            &[operand(Delivery::Copy, &lhs), operand(Delivery::Copy, &rhs)],
            &space,
            &level,
        );
        assert_eq!(plan.refills(0)[0], Refill::Once);
        assert_eq!(plan.refills(1)[0], Refill::Shared);
        assert_eq!(plan.refills(1)[1], Refill::EveryRegion);
    }

    /// The count a walk states rides down onto every slot it plans.
    #[test]
    fn a_slot_of_a_filled_walk_carries_the_count() {
        let (space, lhs, rhs) = spaces();
        let level = Levels::leaf(&[(M, 8), (N, 8), (K, 4)])
            .walk_every(&[M, N, K])
            .filled_by(2)
            .level();
        let plan = StagePlan::new(
            &[operand(Delivery::Tma, &lhs), operand(Delivery::Tma, &rhs)],
            &space,
            &level,
        );
        assert_eq!(plan.fillers(), 2);
    }

    /// A cooperative fill is spread over every unit position of the cube, so planes that are not
    /// there leave their share unwritten. Refused by name rather than read back as wrong bytes.
    #[test]
    #[should_panic(expected = "cannot be filled by a subset of the cube")]
    fn a_walk_cannot_set_planes_aside_to_fill_a_slot_it_also_fills_cooperatively() {
        let (space, lhs, rhs) = spaces();
        let level = Levels::leaf(&[(M, 8), (N, 8), (K, 4)])
            .walk_every(&[M, N, K])
            .filled_by(1)
            .level();
        StagePlan::new(
            &[operand(Delivery::Tma, &lhs), operand(Delivery::Copy, &rhs)],
            &space,
            &level,
        );
    }

    /// A barrier pipeline arrives once per fill, so a TMA operand streams even when fixed.
    #[test]
    fn a_tma_operand_is_never_fixed() {
        let (space, lhs, rhs) = spaces();
        let level = Level::every(&[(M, 8), (N, 4), (K, 8)]);
        let plan = StagePlan::new(
            &[operand(Delivery::Tma, &lhs), operand(Delivery::Tma, &rhs)],
            &space,
            &level,
        );
        assert_eq!(plan.refills(0)[0], Refill::EveryRegion);
    }
}
