//! What the linear recurrence kernels decline with, by name.

use cubecl::features::TypeUsage;
use cubecl::ir::ElemType;
use cubecl::tile::launch::Refusal;
use cubek_matmul::kernels::tensor::ELEMENT_NOT_ON_DEVICE;
use cubek_std::Hardware;

/// Whether `hw` computes in and binds buffers of every one of `elements`.
///
/// # Errors
///
/// [`ELEMENT_NOT_ON_DEVICE`] for the first it does not.
pub(crate) fn admit_on_device(hw: &Hardware, elements: [ElemType; 2]) -> Result<(), Refusal> {
    for element in elements {
        let usage = hw.properties.features.type_usage(element);
        if !usage.is_superset(TypeUsage::Arithmetic | TypeUsage::Buffer) {
            return Err(ELEMENT_NOT_ON_DEVICE);
        }
    }
    Ok(())
}

/// What a routine declines a problem with that states the update rule of the other routine.
pub const RECURRENCE_WALK_RULE_NOT_SERVED: Refusal = Refusal::Declined {
    why: "the routine walks the other update rule",
};

/// What admission declines a problem with when it describes no head: a walk needs a key row and a
/// value column to work on.
pub const RECURRENCE_WALK_HEADS_EMPTY: Refusal = Refusal::Declined {
    why: "the walk needs a key head, a value head, a key feature and a value feature",
};

/// What admission declines a device with that binds fewer buffers to a kernel than the walk's
/// launch does.
pub const RECURRENCE_WALK_BINDINGS_PAST_DEVICE: Refusal = Refusal::Declined {
    why: "the walk binds more buffers than the device binds to one kernel",
};

/// What admission declines an operand with that is not a plain float, a packed activation
/// included: the walk reads floats, so a caller falls back to unpacking above it.
pub const RECURRENCE_WALK_ELEMENT_NOT_FLOAT: Refusal = Refusal::Declined {
    why: "the walk reads plain float operands, not packed or integer ones",
};

/// What admission declines a float with that the walk is not verified at: an input that is not
/// `f32`, or an output that is not `f32`, `f16` or `bf16`.
pub const RECURRENCE_WALK_ELEMENT_UNVERIFIED: Refusal = Refusal::Declined {
    why: "the walk reads f32 inputs and writes f32, f16 or bf16 outputs",
};

/// The buffers a conv over a window needs a device to bind: one for each of its 7 operands, the
/// slot table among them, and one for the metadata buffer some runtimes bind beside them.
pub const QK_NORMED_CONV_WINDOW_BINDING_BUFFERS: u32 = 8 + 1;

/// The buffers a conv over a stream needs a device to bind: one for each of its 5 operands, and one
/// for the metadata buffer some runtimes bind beside them.
pub const QK_NORMED_CONV_STREAM_BINDING_BUFFERS: u32 = 5 + 1;

/// What admission declines a problem with when it describes no head: a conv needs a key head, a
/// value head, a key feature and a value feature.
pub const QK_NORMED_CONV_HEADS_EMPTY: Refusal = Refusal::Declined {
    why: "the q|k|v conv needs a key head, a value head, a key feature and a value feature",
};

/// What admission declines a problem with that holds no row, or no position to convolve.
pub const QK_NORMED_CONV_NOTHING_TO_CONVOLVE: Refusal = Refusal::Declined {
    why: "the q|k|v conv needs a row and a position to convolve",
};

/// What admission declines a filter or a window with that cannot be read back: fewer than two
/// taps, or a window of fewer columns than the taps read back.
pub const QK_NORMED_CONV_TAPS_PAST_WINDOW: Refusal = Refusal::Declined {
    why: "the q|k|v conv needs two taps or more and a window of a column fewer than its taps or more",
};

/// What admission declines a device with that binds fewer buffers to a kernel than a conv's
/// launch does.
pub const QK_NORMED_CONV_BINDINGS_PAST_DEVICE: Refusal = Refusal::Declined {
    why: "the q|k|v conv binds more buffers than the device binds to one kernel",
};

/// What admission declines an element with that is not `f16` or `f32`: a packed or integer element
/// is not a float the conv reads, and no other float is verified.
pub const QK_NORMED_CONV_ELEMENT_UNSUPPORTED: Refusal = Refusal::Declined {
    why: "the q|k|v conv reads f16 or f32 activations",
};
