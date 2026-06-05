use cubecl::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Placement {
    /// Continuous slide mapping: source_x = out_x * scale + offset.
    Continuous { scale: f32, offset: f32 },

    /// Integer-aligned sliding windows.
    Windowed {
        step: usize,
        pad: usize,
        dilation: usize,
    },

    /// Positions come from an external index tensor.
    Explicit { index: TensorBinding<R> },
}
