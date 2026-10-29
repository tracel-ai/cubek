use crate::Hardware;
use crate::microkernel::{StageFill, StagedOperand};
use core::num::NonZeroUsize;
use cubecl::tile::stage::Prefetch;

/// How a walk through shared memory holds its stages: a staged cube's, or each plane's own,
/// whatever operands it stages.
///
/// One stage in flight fills a stage and contracts it, two cube barriers a step. Two fill the next
/// stage before contracting this one, at twice the shared memory. Where the next stage waits while
/// this one is contracted is [`prefetch`](Self::prefetch). The tile DSL drives every depth through
/// one schedule each, so neither is a kernel of its own. Who fills the stages is the microkernel's
/// own question.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StageRing {
    /// Stages the ring holds at once.
    pub stages_in_flight: NonZeroUsize,
    /// Where the next stage's loads wait while this one is contracted: in its own slot, the
    /// whole fill running before the contraction of the stage before it (the tile DSL's
    /// `Stages::pipelined`), or in registers, each unit reading its share of the next stage before
    /// the contraction and writing it into its slot after (the tile DSL's `Stages::prefetched`).
    pub prefetch: Prefetch,
}

impl StageRing {
    /// A stage filled and then contracted.
    pub const SINGLE: Self = Self {
        stages_in_flight: NonZeroUsize::new(1).unwrap(),
        prefetch: Prefetch::InSlots,
    };

    /// The next stage filled while this one is contracted.
    pub const DOUBLE: Self = Self {
        stages_in_flight: NonZeroUsize::new(2).unwrap(),
        prefetch: Prefetch::InSlots,
    };

    /// The next two stages filled while this one is contracted: what a fill that runs beside the
    /// contraction, the TMA engine's, keeps ahead of it.
    pub const TRIPLE: Self = Self {
        stages_in_flight: NonZeroUsize::new(3).unwrap(),
        prefetch: Prefetch::InSlots,
    };

    /// One stage, the next held in registers while it is contracted.
    pub const SINGLE_THROUGH_REGISTERS: Self = Self {
        stages_in_flight: NonZeroUsize::new(1).unwrap(),
        prefetch: Prefetch::InRegisters,
    };

    /// Two stages, the next held in registers while this one is contracted.
    pub const DOUBLE_THROUGH_REGISTERS: Self = Self {
        stages_in_flight: NonZeroUsize::new(2).unwrap(),
        prefetch: Prefetch::InRegisters,
    };

    /// Bytes of shared memory the ring holds, one stage holding `stage_bytes`.
    pub fn bytes(&self, stage_bytes: usize) -> usize {
        self.stages_in_flight.get() * stage_bytes
    }

    /// The ring that fills stages of `operands` on a cube of `units` units: this one, prefetching
    /// into its slots where the tile DSL would refuse to hold the next stage in registers. A
    /// refusal is made at expansion, where it is a launch that writes nothing; so every setting
    /// runs on every problem, and where it resolves to another's ring it compiles that one's
    /// launch.
    ///
    /// Asked before the operands' widths are settled, so each operand's share is its most over
    /// every width it could bind at: the powers of two up to the widest line `hw` reads.
    ///
    /// An operand whose fill converts it ([`StageFill::Converted`]) is not carried by the register
    /// prefetch, which reads a plain source at its own width: its ring prefetches into its slots.
    ///
    /// The tile DSL holds one stage ahead in registers, so a ring of more than two stages would
    /// leave the slots past the second idle; and it holds at most [`Prefetch::fits`] scalars a unit
    /// of every operand's stage ([`StagedOperand::most_prefetched`]).
    pub fn resolve(self, hw: &Hardware, operands: &[StagedOperand], units: usize) -> Self {
        if operands
            .iter()
            .any(|operand| operand.fill == StageFill::Converted)
        {
            return Self {
                prefetch: Prefetch::InSlots,
                ..self
            };
        }
        let held = operands
            .iter()
            .map(|operand| operand.most_prefetched(hw, units))
            .sum();
        let admitted = self.stages_in_flight.get() <= 2 && Prefetch::fits(held);
        match (self.prefetch, admitted) {
            (Prefetch::InRegisters, false) => Self {
                prefetch: Prefetch::InSlots,
                ..self
            },
            (Prefetch::InRegisters, true) | (Prefetch::InSlots, _) => self,
        }
    }
}

/// Nothing for a single stage filled in its slot, so the settings that hold one keep their names.
impl core::fmt::Display for StageRing {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.stages_in_flight.get() {
            1 => {}
            stages => write!(f, " stages_in_flight={stages}")?,
        }
        match self.prefetch {
            Prefetch::InSlots => Ok(()),
            Prefetch::InRegisters => write!(f, " prefetch=registers"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices;
    use crate::matmul_axes::{K, M, N};
    use cubecl::ir::{ElemType, FloatKind};
    use cubecl::tile::{Axis, Space};

    /// A small stage of both operands of a product in `f16`, which fits in registers.
    fn small_stage() -> [StagedOperand; 2] {
        let operand = |axes: [(Axis, usize); 2]| StagedOperand {
            element: ElemType::Float(FloatKind::F16),
            space: Space::new(&axes),
            innermost: axes[1].0,
            fill: StageFill::AsStored,
        };
        [operand([(M, 16), (K, 16)]), operand([(K, 16), (N, 16)])]
    }

    /// The register prefetch does not carry an operand its fill converts, so its ring keeps the
    /// slots.
    #[test]
    fn an_operand_converted_by_its_fill_prefetches_into_its_slots() {
        let [lhs, rhs] = small_stage();
        let converted = [lhs.converted_by_fill(), rhs];
        assert_eq!(
            StageRing::DOUBLE_THROUGH_REGISTERS.resolve(&devices::m2_pro(), &converted, 128),
            StageRing::DOUBLE
        );
    }

    /// The tile DSL holds one stage ahead in registers and refuses a deeper ring at expansion,
    /// where the refusal is a launch that writes nothing: a deeper ring keeps its depth and
    /// prefetches into its slots instead.
    #[test]
    fn a_ring_deeper_than_two_prefetches_into_its_slots() {
        let deep = |prefetch| StageRing {
            stages_in_flight: NonZeroUsize::new(3).unwrap(),
            prefetch,
        };
        let resolved = deep(Prefetch::InRegisters).resolve(&devices::m2_pro(), &small_stage(), 128);
        assert_eq!(resolved, deep(Prefetch::InSlots));
        assert_eq!(
            StageRing::DOUBLE_THROUGH_REGISTERS.resolve(&devices::m2_pro(), &small_stage(), 128),
            StageRing::DOUBLE_THROUGH_REGISTERS
        );
    }
}
