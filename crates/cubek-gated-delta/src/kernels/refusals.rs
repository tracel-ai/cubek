//! What the gated delta kernels decline with, by name.

use cubecl::tile::launch::Refusal;

/// The buffers a walk needs a device to bind: one for each of its 8 operands, and one for the
/// metadata buffer some runtimes bind beside them.
pub const GATED_DELTA_WALK_BINDING_BUFFERS: u32 = 8 + 1;

/// The buffers a walk in a store's slots needs a device to bind: the seven operands it shares with
/// the owned walk, the slot table and the position table, and the metadata buffer.
pub const GATED_DELTA_WALK_SLOTS_BINDING_BUFFERS: u32 = 9 + 1;

/// What admission declines a problem with when it describes no head: a walk needs a key row and a
/// value column to work on.
pub const GATED_DELTA_WALK_HEADS_EMPTY: Refusal = Refusal::Declined {
    why: "the gated delta walk needs a key head, a value head, a key feature and a value feature",
};

/// What admission declines a walk in a store's slots with that keeps more than the one state each
/// slot holds: a ring of states is walked over states it owns.
pub const GATED_DELTA_WALK_SLOTS_KEEP_RING: Refusal = Refusal::Declined {
    why: "the gated delta walk in a store's slots keeps one state, not a ring of them",
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
