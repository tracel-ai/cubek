//! How wide a launch alone binds its operands.

use cubecl::tile::launch::Vectorization;

/// How wide a launch alone binds its operands: one value a line, the least that runs a
/// microkernel, or as wide as a routine binds them, which reaches the paths a microkernel takes
/// for whole lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lines {
    /// One value a line.
    One,
    /// Each operand at the widest line the device loads its element in that every access the
    /// microkernel states ([`Microkernel::vectorize`](cubek_std::microkernel::Microkernel)), and
    /// the launch walking it, admits.
    Widest,
}

impl Lines {
    pub const EVERY: [Lines; 2] = [Lines::One, Lines::Widest];

    /// The width `operand` binds at, `vectorization` holding what the microkernel and the launch
    /// require of the operands and `ceiling` the widest line the device loads each one in.
    pub fn width<O: Copy + Eq>(
        self,
        operand: O,
        vectorization: &Vectorization<O>,
        ceiling: impl Fn(O) -> usize,
    ) -> usize {
        match self {
            Lines::One => 1,
            Lines::Widest => vectorization.width(operand, ceiling),
        }
    }
}
