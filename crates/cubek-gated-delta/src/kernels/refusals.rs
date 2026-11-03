//! What the gated delta kernels decline with, by name.

use cubecl::tile::launch::Refusal;

/// The buffers a walk needs a device to bind: one for each of its 8 operands, and one for the
/// metadata buffer some runtimes bind beside them.
pub const GATED_DELTA_WALK_BINDING_BUFFERS: u32 = 8 + 1;

/// What admission declines a problem with when it describes no head: a walk needs a key row and a
/// value column to work on.
pub const GATED_DELTA_WALK_HEADS_EMPTY: Refusal = Refusal::Declined {
    why: "the gated delta walk needs a key head, a value head, a key feature and a value feature",
};

/// What admission declines a device with that binds fewer buffers to a kernel than the walk's
/// launch does.
pub const GATED_DELTA_WALK_BINDINGS_PAST_DEVICE: Refusal = Refusal::Declined {
    why: "the gated delta walk binds more buffers than the device binds to one kernel",
};

/// What admission declines an operand with that is not a plain float, a packed activation
/// included: the walk reads floats, so a caller falls back to unpacking above it.
pub const GATED_DELTA_WALK_ELEMENT_NOT_FLOAT: Refusal = Refusal::Declined {
    why: "the gated delta walk reads plain float operands, not packed or integer ones",
};

/// What admission declines a float with that the walk is not verified at: an input that is not
/// `f32`, or an output that is not `f32`, `f16` or `bf16`.
pub const GATED_DELTA_WALK_ELEMENT_UNVERIFIED: Refusal = Refusal::Declined {
    why: "the gated delta walk reads f32 inputs and writes f32, f16 or bf16 outputs",
};
