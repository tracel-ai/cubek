//! What one plane contracts through ([`Instruction`]).

use crate::{MmaIOConfig, RegisterBlock};

/// What one plane contracts through: the software instruction's register array (whose execution
/// `config` rides along), or a matrix fragment in one of the two hardware forms. `io` rides the
/// manual form because it comes from a device query, which cannot run in-kernel.
///
/// **A kernel states this once and hands it to both sides**, rather than picking a constructor
/// per form: [`Tile::accumulator`](crate::Tile::accumulator) opens what a plane sums into and
/// [`PlanePartition::operand`](crate::PlanePartition::operand) loads what it sums, and neither
/// asks which form it was given. A kernel that knows its form statically may still name it by
/// the constructor that spells it ([`Tile::cmma_accumulator`](crate::Tile::cmma_accumulator) and
/// its siblings), which is the same call with the form written out.
///
/// The point of the pair is a kernel whose form is *data* — elected by a derivation from what the
/// device offers — which would otherwise match on it at every site that touches a fragment, and
/// drift between them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Instruction {
    /// A register block, run under `config`. Asks nothing of the hardware.
    Registers { config: RegisterBlock },
    /// The vendor's fragment API: `wmma` on CUDA, rocWMMA on HIP.
    Cmma,
    /// cubek's own fragment transport over the matrix intrinsics, carrying the per-role
    /// load and store `io` the device supports.
    Mma { io: MmaIOConfig },
}
