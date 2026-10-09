//! What one plane contracts through ([`Instruction`]).

use crate::{MmaIo, RegisterBlock};

/// What one plane contracts through: a software register block or a hardware fragment form.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Instruction {
    /// A register block, run under `config`. Asks nothing of the hardware.
    Registers { config: RegisterBlock },
    /// The vendor's fragment API: `wmma` on CUDA, rocWMMA on HIP.
    Cmma,
    /// cubek's own fragment transport over the matrix intrinsics, with per-role `io`.
    Mma { io: MmaIo },
    /// Hopper's warpgroup MMA: a plane group of four contracts its `64 × n` tile together,
    /// reading both operands out of shared memory, asynchronously.
    Wgmma,
}
