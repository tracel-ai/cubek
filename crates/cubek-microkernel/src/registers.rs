//! What a unit may hold in registers.

/// The most vector lines of a row or a state a unit holds in registers: past it the registers, not
/// the data, bound the kernel, and its unrolled walk takes longer to compile than the launch it
/// serves takes to run.
pub const MAX_HELD_LINES_PER_UNIT: usize = 16;
