//! The [`Staging`] slot: a matmul-agnostic payload `T` plus the [`Pipeline`] sequencing its fill
//! against its read. Generic slot mechanics only: the producer/consumer acquire/release and the
//! final publish; the operand-specific construction and fill live in [`fill`](crate::fill).

use core::option::Option;
use cubecl::prelude::*;

use crate::*;

pub(crate) const FIRST_SLOT: usize = 0;

pub(crate) const LHS: usize = 0;
pub(crate) const RHS: usize = 1;

/// What one operand's slot payload *is*, which decides what consuming the slot has to do with it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SlotPayload {
    /// A buffer this slot filled, already the current region's when the slot is consumed:
    /// [`WindowMode`] says how it got there.
    Windowed(WindowMode),
    /// No buffer: the payload is the operand where it already lies, so the slot holds it whole and
    /// each read selects the region out of it. Nothing is filled and nothing is stored, which is
    /// what makes an [`InPlace`](Residence::InPlace) level cost exactly the reads it would have
    /// made without a ring.
    AtRegion,
}

/// Whether a slot can transport an operand from its current backing, or merely retains fragments
/// already resident in registers.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum StageSource {
    Transport(Delivery),
    ResidentFragment,
}

impl StageSource {
    pub(crate) fn delivery(self) -> Option<Delivery> {
        match self {
            StageSource::Transport(delivery) => Some(delivery),
            StageSource::ResidentFragment => None,
        }
    }
}

/// When a windowed payload is brought to its region across the walk. The walk moves each operand's
/// window or it does not, and a window that never moves need be neither refilled nor duplicated per
/// slot; those two savings are the same fact, so one mode carries both.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WindowMode {
    /// The walk moves this operand's window, so every region refills it.
    Streamed,
    /// The window is fixed across the walk: filled once, above the loop.
    Fixed,
    /// Fixed, with this slot reusing the first slot's buffer, which is already filled. Nobody fills
    /// it here. Only a shared ([`Smem`](Residence::Smem)) operand reaches this: plane tiles are
    /// private to their slot, and an in-place operand has no buffer to reuse.
    Reused,
}

impl SlotPayload {
    /// Whether [`fill_fixed`](Staging::fill_fixed) fills this operand: it owns a fixed window.
    pub(crate) fn is_fixed(self) -> bool {
        matches!(self, SlotPayload::Windowed(WindowMode::Fixed))
    }

    /// Whether [`fill_streamed`](Staging::fill_streamed) fills this operand.
    pub(crate) fn is_streamed(self) -> bool {
        matches!(self, SlotPayload::Windowed(WindowMode::Streamed))
    }

    /// Whether this slot takes the first slot's buffer for this operand instead of allocating one.
    pub(crate) fn reuses_first_buffer(self) -> bool {
        matches!(self, SlotPayload::Windowed(WindowMode::Reused))
    }

    /// This operand's payload in a later ring slot, given its payload in the first. A fixed shared
    /// operand reuses the first slot's buffer; anything else is rebuilt per slot.
    pub(crate) fn in_later_slot(self, residence: Residence) -> SlotPayload {
        match self {
            SlotPayload::Windowed(WindowMode::Fixed) => match residence {
                Residence::Smem => SlotPayload::Windowed(WindowMode::Reused),
                Residence::Register => self,
                Residence::InPlace => unreachable!("an in-place operand is never windowed"),
            },
            // Nothing was allocated for the first slot either: every slot names the same operand.
            SlotPayload::Windowed(WindowMode::Streamed) | SlotPayload::AtRegion => self,
            SlotPayload::Windowed(WindowMode::Reused) => {
                unreachable!("Reused is produced only in later slots")
            }
        }
    }
}

/// Read one operand out of its slot for `region`. A buffer the slot filled is already this
/// region's; an [`AtRegion`](SlotPayload::AtRegion) payload is the whole operand, out of which the
/// region selects its own window (or, for plane-private cells, its own block of fragments).
///
/// A free function, not a `SlotPayload` method: the payload is comptime-only (never a
/// [`CubeType`]), so the runtime read it drives lives in an ordinary `#[cube]` function.
#[cube]
pub(crate) fn read_operand<T: Numeric>(
    staged: &Tile<T>,
    region: &Region,
    #[comptime] payload: SlotPayload,
) -> Tile<T> {
    match comptime!(payload) {
        SlotPayload::AtRegion => staged.at(region),
        SlotPayload::Windowed(_) => staged.clone(),
    }
}

/// What one operand's payload is within a staging slot, where its cells live, and what a slot can
/// obtain it from.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct OperandPlan {
    pub payload: SlotPayload,
    pub residence: Residence,
    pub source: StageSource,
}

impl OperandPlan {
    pub(crate) const fn new(
        payload: SlotPayload,
        residence: Residence,
        source: StageSource,
    ) -> Self {
        Self {
            payload,
            residence,
            source,
        }
    }

    /// Whether reading this operand selects fragments by comptime coordinate: the slot staged it
    /// into plane tiles, or a level above left it resident in registers. The two arrive at the
    /// selection from opposite ends, so neither alone answers for the walk.
    pub(crate) fn reads_fragments(self) -> bool {
        match self.source {
            StageSource::ResidentFragment => true,
            StageSource::Transport(_) => matches!(self.residence, Residence::Register),
        }
    }
}

/// One slot of the staged `mma` pipeline: its payload `T` and the [`Pipeline`] sequencing fill vs
/// read. Generic over `T`, so the slot is matmul-agnostic; it just hands out a synchronized `&mut T`
/// to fill (`write`) and a synchronized `&T` to consume (`read`).
#[derive(CubeType)]
pub struct Staging<T: CubeType> {
    pub(crate) data: T,
    pub(crate) pipeline: Pipeline,
    /// What each operand's payload is, and where it lives at this level; both resolved when the
    /// slot was built. The slot's payload `T` fixes its arity, so a unary slot's right-hand entry
    /// is `None` and asking for it is a bug, not a default.
    #[cube(comptime)]
    pub(crate) lhs: OperandPlan,
    #[cube(comptime)]
    pub(crate) rhs: Option<OperandPlan>,
}

#[cube]
impl<T: CubeType> Staging<T> {
    /// Wrap an already-built payload and pipeline. Private: the public entry is the operand-deducing
    /// [`new`](Staging::new). (Split out so the tuple `T` never sits in a struct-literal turbofish,
    /// which `#[cube]` can't parse; `Staging::<T>` can.)
    pub(crate) fn wrap(
        data: T,
        pipeline: Pipeline,
        #[comptime] lhs: OperandPlan,
        #[comptime] rhs: Option<OperandPlan>,
    ) -> Staging<T> {
        Staging::<T> {
            data,
            pipeline,
            lhs,
            rhs,
        }
    }

    /// The resolved plan for `operand`. Unary slots have only [`LHS`].
    pub(crate) fn plan(&self, #[comptime] operand: usize) -> comptime_type!(OperandPlan) {
        comptime!(match operand {
            LHS => self.lhs,
            RHS => self.rhs.expect("Staging: unary slot has no rhs"),
            _ => panic!("Staging: invalid operand index"),
        })
    }

    /// Whether this slot has any fixed operand.
    pub(crate) fn has_fixed(&self) -> comptime_type!(bool) {
        comptime!(
            self.lhs.payload.is_fixed()
                || matches!(self.rhs, Option::Some(p) if p.payload.is_fixed())
        )
    }

    /// Whether either operand is read by selecting fragments, requiring an unrolled walk.
    pub(crate) fn has_fragment_read(&self) -> comptime_type!(bool) {
        comptime!(
            self.lhs.reads_fragments()
                || matches!(self.rhs, Option::Some(p) if p.reads_fragments())
        )
    }

    /// Producer acquire: wait the slot is free (`empty`, WAR) for `Barrier`; a `collective` `Cube`
    /// slot rendezvouses on `sync_cube`; a lone-unit one does nothing.
    pub(crate) fn acquire_write(&self) {
        match &self.pipeline {
            Pipeline::Barrier { empty, phase, .. } => empty.wait_parity(*phase ^ 1),
            Pipeline::Cube => sync_cube(),
            Pipeline::Solo => {}
        }
    }

    /// Producer release publishes a barrier slot after its required arrivals and any TMA bytes
    /// declared by [`Pipeline::fill`] land. Mixed slots need every unit; pure TMA uses unit 0.
    pub(crate) fn release_write(&self) {
        match &self.pipeline {
            Pipeline::Barrier {
                full,
                collective_full,
                ..
            } => {
                if comptime!(*collective_full) || UNIT_POS == 0 {
                    full.arrive();
                }
            }
            Pipeline::Cube | Pipeline::Solo => {}
        }
    }

    /// Consumer acquire: wait the slot's fill (`full`, RAW) for `Barrier`; nothing for `Cube` (already
    /// rendezvoused in `write`).
    pub(crate) fn acquire_read(&self) {
        match &self.pipeline {
            Pipeline::Barrier { full, phase, .. } => full.wait_parity(*phase),
            Pipeline::Cube | Pipeline::Solo => {}
        }
    }

    /// Consumer release: arrive `empty` (free the slot) and flip the phase for `Barrier`; nothing for
    /// `Cube`.
    pub(crate) fn release_read(&mut self) {
        match &mut self.pipeline {
            Pipeline::Barrier { empty, phase, .. } => {
                empty.arrive();
                *phase ^= 1;
            }
            Pipeline::Cube | Pipeline::Solo => {}
        }
    }

    /// Publish this slot's last fill when no successor fill's rendezvous will (the walk's final
    /// regions). Only a collective `Cube` slot needs it; callers invoke this immediately before
    /// [`consume`](Staging::consume).
    pub fn publish(&self) {
        match &self.pipeline {
            Pipeline::Cube => sync_cube(),
            Pipeline::Solo | Pipeline::Barrier { .. } => {}
        }
    }
}
