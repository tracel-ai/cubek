//! The entry point a caller runs a backward pass through.

use crate::tune::launch_tuned;
use crate::{AttentionBackwardGradients, AttentionBackwardTensors};
use cubecl::Device;

/// The gradients of the attention `tensors` state, on `device`: the fastest routine autotune
/// measured for the problem, or the unit routine's lead where none other runs.
pub fn attention_backward(
    device: &Device,
    tensors: AttentionBackwardTensors,
) -> AttentionBackwardGradients {
    launch_tuned(device, tensors)
}
