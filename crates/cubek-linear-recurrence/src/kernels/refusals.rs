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

/// The buffers a causal conv needs a device to bind: one for each of its 6 operands, one for the
/// slot table and one for the metadata buffer some runtimes bind beside them.
pub const CAUSAL_CONV_BINDING_BUFFERS: u32 = 7 + 1;

/// What admission declines a problem with when it describes no channel to convolve.
pub const CAUSAL_CONV_CHANNELS_EMPTY: Refusal = Refusal::Declined {
    why: "the causal conv needs a channel",
};

/// What admission declines an output element with that is neither `f32` nor the activations'.
pub const CAUSAL_CONV_OUTPUT_UNVERIFIED: Refusal = Refusal::Declined {
    why: "the causal conv writes f32 or the activations' element",
};

/// What admission declines an element with that is not `f16`, `bf16` or `f32`: a packed or integer
/// element is not a float the conv reads, and no other float is verified.
pub const CAUSAL_CONV_ELEMENT_UNVERIFIED: Refusal = Refusal::Declined {
    why: "the causal conv reads f16, bf16 or f32 activations",
};

/// What admission declines a problem with that holds no row, or no position to convolve.
pub const CAUSAL_CONV_NOTHING_TO_CONVOLVE: Refusal = Refusal::Declined {
    why: "the causal conv needs a row and a position to convolve",
};

/// What admission declines a filter with fewer than two taps: the conv reads at least one tap
/// behind the newest.
pub const CAUSAL_CONV_TAPS_TOO_FEW: Refusal = Refusal::Declined {
    why: "the causal conv needs at least two taps",
};

/// What admission declines a ring with that holds fewer columns than the taps read back.
pub const CAUSAL_CONV_RING_SHORTER_THAN_TAPS: Refusal = Refusal::Declined {
    why: "the causal conv needs a ring of at least `taps - 1` columns",
};

/// What admission declines a device with that binds fewer buffers to a kernel than the conv's
/// launch does.
pub const CAUSAL_CONV_BINDINGS_PAST_DEVICE: Refusal = Refusal::Declined {
    why: "the causal conv binds more buffers than the device binds to one kernel",
};
