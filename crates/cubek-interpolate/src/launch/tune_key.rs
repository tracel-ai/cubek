use cubecl::{AutotuneKey, ir::ElemType};
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of interpolation versions
pub struct InterpolateAutotuneKey {
    elem_input: ElemType,
    elem_output: ElemType,
    elem_acc: ElemType,

    /// Number of channels
    ///
    /// # Notes
    ///
    /// Max is 4096, so 12 values are possible.
    #[autotune(anchor(exp(max = 4096, base = 2)))]
    pub channels: usize,

    /// Height of the interpolated output.
    ///
    /// # Notes
    ///
    /// Max is 8192, so 14 values are possible.
    #[autotune(anchor(exp(max = 8192, base = 2)))]
    pub out_height: usize,

    /// Width of the interpolated output.
    ///
    /// # Notes
    ///
    /// Max is 8192, so 14 values are possible.
    #[autotune(anchor(exp(max = 8192, base = 2)))]
    pub out_width: usize,
}

impl InterpolateAutotuneKey {
    pub fn generate(
        elem_input: ElemType,
        elem_output: ElemType,
        input_shape: &[usize],
        output_shape: &[usize],
    ) -> Self {
        let channels = input_shape[3];

        let output_height = output_shape[1];
        let output_width = output_shape[2];

        InterpolateAutotuneKey::new(
            elem_input,
            elem_output,
            channels,
            output_height,
            output_width,
        )
    }
}
