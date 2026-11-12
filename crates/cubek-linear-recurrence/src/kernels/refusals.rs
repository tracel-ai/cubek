//! What the linear recurrence kernels decline with, by name.

use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::launch::Refusal;
use cubek_matmul::kernels::tensor::ELEMENT_NOT_ON_DEVICE;
use cubek_std::Hardware;

/// Whether `hw` computes in and binds buffers of every one of `elements`.
///
/// # Errors
///
/// [`ELEMENT_NOT_ON_DEVICE`] for the first it does not.
pub(crate) fn admit_on_device(hw: &Hardware, elements: [ElemType; 2]) -> Result<(), Refusal> {
    match elements.into_iter().all(|element| hw.runs(element)) {
        true => Ok(()),
        false => Err(ELEMENT_NOT_ON_DEVICE),
    }
}

/// The activation elements the convs are verified at.
const ELEMENTS: [ElemType; 3] = [
    ElemType::Float(FloatKind::F16),
    ElemType::Float(FloatKind::BF16),
    ElemType::Float(FloatKind::F32),
];

/// Whether the convs read and write activations of `element`.
///
/// # Errors
///
/// [`CONV_ELEMENT_UNSUPPORTED`] for any but `f16`, `bf16` and `f32`.
pub(crate) fn admit_element(element: ElemType) -> Result<(), Refusal> {
    if ELEMENTS.contains(&element) {
        Ok(())
    } else {
        Err(CONV_ELEMENT_UNSUPPORTED)
    }
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

/// The buffers a q,k,v conv needs a device to bind: one for each of its 7 operands, one for the
/// slot table and one for the metadata buffer some runtimes bind beside them.
pub const QK_NORMED_CONV_BINDING_BUFFERS: u32 = 8 + 1;

/// The buffers a causal conv needs a device to bind: one for each of its 6 operands, one for the
/// slot table and one for the metadata buffer some runtimes bind beside them.
pub const CAUSAL_CONV_BINDING_BUFFERS: u32 = 7 + 1;

/// What admission declines a problem with when it describes no channel to convolve.
pub const CAUSAL_CONV_CHANNELS_EMPTY: Refusal = Refusal::Declined {
    why: "the causal conv needs a channel",
};

/// What admission declines an output element with that is neither `f32` nor the activations'.
pub const CAUSAL_CONV_OUTPUT_UNSUPPORTED: Refusal = Refusal::Declined {
    why: "the causal conv writes f32 or the activations' element",
};

/// What admission declines a problem with when it describes no head: a conv needs a key head, a
/// value head, a key feature and a value feature.
pub const QK_NORMED_CONV_HEADS_EMPTY: Refusal = Refusal::Declined {
    why: "the q|k|v conv needs a key head, a value head, a key feature and a value feature",
};

/// What admission declines a problem with that holds no row, or no position to convolve.
pub const CONV_NOTHING_TO_CONVOLVE: Refusal = Refusal::Declined {
    why: "the conv needs a row and a position to convolve",
};

/// What admission declines a filter or a ring with that cannot be read back: fewer than two
/// taps, or a ring of fewer columns than the taps read back.
pub const CONV_TAPS_PAST_RING: Refusal = Refusal::Declined {
    why: "the conv needs at least two taps and a ring of at least `taps - 1` columns",
};

/// What admission declines a device with that binds fewer buffers to a kernel than a conv's
/// launch does.
pub const CONV_BINDINGS_PAST_DEVICE: Refusal = Refusal::Declined {
    why: "the conv binds more buffers than the device binds to one kernel",
};

/// What admission declines an element with that is not `f16`, `bf16` or `f32`: a packed or integer
/// element is not a float the conv reads, and no other float is verified.
pub const CONV_ELEMENT_UNSUPPORTED: Refusal = Refusal::Declined {
    why: "the conv reads f16, bf16 or f32 activations",
};
