use crate::definition::InterpolateMode;
use cubecl::{AutotuneKey, ir::ElemType, tune::anchor};
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of interpolation kernel shapes.
pub struct InterpolateAutotuneKey {
    elem_input: ElemType,
    elem_output: ElemType,
    mode: InterpolateMode,
    align_corners: bool,

    /// Input height, anchored with the same bucket as the output height.
    #[autotune(anchor(exp(max = 8192, base = 2)))]
    pub input_height: usize,
    /// Input width, anchored with the same bucket as the output width.
    #[autotune(anchor(exp(max = 8192, base = 2)))]
    pub input_width: usize,
    /// Output height.
    #[autotune(anchor(exp(max = 8192, base = 2)))]
    pub output_height: usize,
    /// Output width.
    #[autotune(anchor(exp(max = 8192, base = 2)))]
    pub output_width: usize,

    /// Number of channels.
    #[autotune(anchor(exp(max = 4096, base = 2)))]
    pub channels: usize,
    /// Alignment of the contiguous channel axis, which bounds vectorization.
    pub channels_pow2_factor: u8,
    /// Alignment in bytes of the contiguous row stride, which also bounds vectorization.
    pub channels_stride_factor: u8,
}

impl InterpolateAutotuneKey {
    /// Creates a key from the extents of the NHWC tensors passed to the interpolation kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        elem_input: ElemType,
        elem_output: ElemType,
        mode: InterpolateMode,
        align_corners: bool,
        input_height: usize,
        input_width: usize,
        channels: usize,
        output_height: usize,
        output_width: usize,
    ) -> Self {
        let channels_anchored = anchor(channels, Some(4096), None, Some(2));

        Self::new(
            elem_input,
            elem_output,
            mode,
            align_corners,
            input_height,
            input_width,
            output_height,
            output_width,
            channels,
            pow2_factor(channels_anchored),
            stride_factor(channels_anchored, elem_input),
        )
    }
}

/// The largest vector alignment relevant to CubeCL's optimized vector sizes.
fn pow2_factor(extent: usize) -> u8 {
    extent.trailing_zeros().min(4) as u8
}

/// Maximum factor relevant to strides: CubeCL's largest swizzle repeat is 128 bytes.
const MAX_STRIDE_FACTOR: u32 = 10;

/// Alignment in powers of two of the contiguous NHWC row stride.
fn stride_factor(channels: usize, elem: ElemType) -> u8 {
    let bytes = (channels * elem.size_bits()) / 8;
    bytes.trailing_zeros().min(MAX_STRIDE_FACTOR) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};

    const F32: ElemType = ElemType::Float(FloatKind::F32);

    // Keep this next to the key fields. A kernel comptime input added to either interpolation
    // architecture must be represented here and in `InterpolateAutotuneKey` before it can share
    // autotune results safely.
    const KERNEL_COMPTIME_INPUTS: &[&str] = &[
        "mode",
        "align_corners",
        "height transform (input and output height)",
        "width transform (input and output width)",
        "vector size (channel and row-stride alignment)",
        "input dtype",
        "accumulator dtype",
        "tiled bounds check and boundary",
    ];

    fn key(
        input_height: usize,
        input_width: usize,
        channels: usize,
        output_height: usize,
        output_width: usize,
        align_corners: bool,
    ) -> InterpolateAutotuneKey {
        InterpolateAutotuneKey::generate(
            F32,
            F32,
            InterpolateMode::Bilinear,
            align_corners,
            input_height,
            input_width,
            channels,
            output_height,
            output_width,
        )
    }

    fn key_with_mode(mode: InterpolateMode, elem: ElemType) -> InterpolateAutotuneKey {
        InterpolateAutotuneKey::generate(elem, elem, mode, true, 64, 64, 64, 128, 128)
    }

    #[test]
    fn raw_extents_within_anchored_buckets_share_a_key() {
        let reference = key(69, 70, 69, 130, 129, true);
        for (input_height, input_width, channels, output_height, output_width) in [
            (70, 76, 76, 140, 140),
            (88, 96, 96, 192, 192),
            (128, 128, 128, 256, 256),
        ] {
            assert_eq!(
                reference,
                key(
                    input_height,
                    input_width,
                    channels,
                    output_height,
                    output_width,
                    true
                )
            );
        }
    }

    #[test]
    fn anchored_kernel_shape_inputs_change_the_key() {
        let reference = key(64, 64, 64, 128, 128, true);
        assert_ne!(reference, key(128, 64, 64, 128, 128, true));
        assert_ne!(reference, key(64, 128, 64, 128, 128, true));
        assert_ne!(reference, key(64, 64, 64, 64, 128, true));
        assert_ne!(reference, key(64, 64, 64, 128, 64, true));
        assert_ne!(reference, key(64, 64, 64, 128, 128, false));
        assert_ne!(reference, key(64, 64, 128, 128, 128, true));
        assert_ne!(
            key_with_mode(InterpolateMode::Bilinear, F32),
            key_with_mode(InterpolateMode::Bicubic, F32)
        );
        assert_ne!(
            key_with_mode(InterpolateMode::Bilinear, F32),
            key_with_mode(InterpolateMode::Bilinear, ElemType::Float(FloatKind::F16))
        );
    }

    #[test]
    fn comptime_inputs_are_documented_with_the_key() {
        assert_eq!(KERNEL_COMPTIME_INPUTS.len(), 8);
        assert!(KERNEL_COMPTIME_INPUTS.contains(&"align_corners"));
        assert!(KERNEL_COMPTIME_INPUTS.contains(&"vector size (channel and row-stride alignment)"));
    }
}
