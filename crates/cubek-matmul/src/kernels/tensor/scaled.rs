//! A float rhs read under a scale tensor: each value multiplied, as the stage fill reads it, by
//! the scale at the position it shares with it.

use super::PlainTensor;
use crate::kernels::MatmulInputs;
use cubecl::client::Client;
use cubecl::tile::launch::Refusal;

/// What [`ScaledTensor::new`] declines: a scale that is not a float the routines are verified at,
/// or not the rhs's batch dims followed by its extent along `K`.
pub const SCALE_NOT_OVER_OPERAND: Refusal = Refusal::Declined {
    why: "a scale is an f32, f16 or bf16 tensor read as [batches.., k]: the rhs's own batch dims, \
          then its extent along k",
};

/// A float rhs multiplied by a scale over the axes they share: its batch axes and `K`, never its
/// own `N`, which a stage's fill could not apply without reading the scale per column.
///
/// The scale is laid `[batches.., k]`; a dim it does not vary along has stride zero. It weights
/// each position of the contracted axis, as decay weights do in a chunked linear recurrence.
#[derive(Debug, Clone)]
pub struct ScaledTensor {
    values: PlainTensor,
    scale: PlainTensor,
}

impl ScaledTensor {
    /// The rhs `values`, `[.., k, n]`, under `scale`.
    ///
    /// # Errors
    ///
    /// [`SCALE_NOT_OVER_OPERAND`].
    pub fn new(values: PlainTensor, scale: PlainTensor) -> Result<Self, Refusal> {
        let shape = values.shape();
        let rank = shape.rank();
        if rank < 2 || !MatmulInputs::FLOATS.contains(&scale.elem) {
            return Err(SCALE_NOT_OVER_OPERAND);
        }
        let expected = shape.as_slice()[..rank - 2]
            .iter()
            .copied()
            .chain([shape[rank - 2]]);
        if !scale.shape().as_slice().iter().copied().eq(expected) {
            return Err(SCALE_NOT_OVER_OPERAND);
        }
        Ok(Self { values, scale })
    }

    /// The tensor the scale multiplies.
    pub fn values(&self) -> &PlainTensor {
        &self.values
    }

    /// The scale.
    pub fn scale(&self) -> &PlainTensor {
        &self.scale
    }

    /// This operand with its values replaced by `values`, which the launch read from them, under
    /// the same scale, read where it lies unless it aliases itself.
    pub(crate) fn over(self, client: &Client, values: PlainTensor) -> Self {
        Self {
            values,
            scale: self.scale.addressable(client),
        }
    }
}
