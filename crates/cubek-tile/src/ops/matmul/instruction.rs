//! What one plane contracts through ([`Instruction`]).

use crate::{MmaIo, RegisterBlock};

/// What one plane contracts through: a software register block or a hardware fragment form.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Instruction {
    /// A register block, run under `config`. Asks nothing of the hardware.
    Registers { config: RegisterBlock },
    /// The vendor's fragment API: `wmma` on CUDA, rocWMMA on HIP.
    Cmma,
    /// cubek's own fragment transport over the matrix intrinsics, with per-role `io`, over
    /// `operands` of one form.
    Mma { io: MmaIo, operands: MmaOperands },
    /// Hopper's warpgroup MMA: a plane group of four contracts its `64 × n` tile together,
    /// reading both operands out of shared memory, asynchronously.
    Wgmma,
}

/// What a manual-mma instruction's operands are: the blueprint's choice, which a leaf reads
/// rather than guessing it from the operands it is handed.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum MmaOperands {
    /// Values as the instruction takes them. A scaled or packed operand is decoded before it
    /// reaches one, or read through lane runs where its fragment is deeper than the instruction.
    #[default]
    Plain,
    /// `e2m1` values under `e4m3` block scales, each read as stored by the device's block-scaled
    /// instruction.
    BlockScaled,
}
