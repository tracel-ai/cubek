//! What runs once the levels are exhausted: the [`Instruction`] a space states at its last
//! level, and the [`Residence`](crate::Residence) an operand must reach for it.

use crate::{MmaIOConfig, RegisterBlock, Residence};

/// The encoding of a register-resident tile: the software instruction's register array (whose
/// execution `config` rides along, being the one instruction fact no stage placement implies),
/// or a matrix fragment in one of the two hardware forms. `io` rides the manual form because it
/// comes from a device query, which cannot run in-kernel.
///
/// Also the instruction vocabulary: an operand *is* its operand.s finest register stage at the
/// instruction ([`register_stage`](Self::register_stage)), and an accumulator no stages answers
/// for takes the blueprint's statement at the kernel top
/// ([`Tile::instruction`](crate::Tile::instruction)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Instruction {
    Registers { config: RegisterBlock },
    Cmma,
    Mma { io: MmaIOConfig },
}

impl Instruction {
    /// The compiler-emitted contraction under a scalar register budget, with neither edge
    /// specialization nor lane fan-out. The shape most callers want; state a
    /// [`RegisterBlock`] directly to turn either on.
    pub const fn registers(budget: usize) -> Self {
        Instruction::Registers {
            config: RegisterBlock::new(budget),
        }
    }

    /// Whether any level stages this operand into registers. Which instruction those registers
    /// are for is the space's statement, never the operand's, so this answers presence only.
    pub fn stages_to_registers(stages: &[Residence]) -> bool {
        stages.iter().any(|residence| match residence {
            Residence::Register => true,
            Residence::InPlace | Residence::Smem => false,
        })
    }
}
