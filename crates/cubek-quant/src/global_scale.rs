//! The global scale of a two-level scheme, the one factor normalizing every block scale.
//!
//! Called global rather than per-tensor throughout, matching `QuantLevel::global_param` and the
//! `global` binding: per-tensor is already taken by [`QuantLevel::Tensor`](crate::scheme::QuantLevel),
//! a one-level scheme with a single scale and no block scales under it.
//!
//! Reading is split from applying so a kernel scaling several blocks per unit can hoist the load
//! out of its loop. `FG` is the scale's own type, not the compute type: reading an f32 scale as
//! f16 because that is what the kernel computes in returns garbage.
//!
//! A two-level scheme forms its effective scale in f32 and narrows only the result. The scale is a
//! tensor magnitude times a block spread, so it can be subnormal in a narrow compute type even
//! when every value it scales is representable there, and narrowing it first rounds it to zero.
//!
//! A one-level scheme has nothing to fold in, so it multiplies in the compute type as before.
//! Widening there would cost the packed two-lane multiply a narrow type gets for free, which
//! measures as a few percent on f16 output.
//!
//! Both directions are the symmetric reconstruction, `x = s * x_q`, which is why applying the
//! scale is all there is to them. A mode carrying a zero point would have to offset around them.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::linear::{LinearView, LinearViewMut};

use crate::dequantize::dequantize_symmetric;

#[cube]
pub(crate) fn read_global<FG: Numeric>(
    global: ComptimeOption<LinearView<'_, FG>>,
) -> ComptimeOption<FG> {
    #[comptime]
    match global {
        ComptimeOption::Some(global) => ComptimeOption::new_Some(global.read(0)),
        ComptimeOption::None => ComptimeOption::new_None(),
    }
}

/// Multiply dequantized values by their block scale, folding in the global scale if there is one.
///
/// Both arms are [`dequantize_symmetric`]; what the global buys is only the type the multiply
/// happens in.
#[cube]
pub(crate) fn dequantize_symmetric_scaled<F: Float, FG: Numeric, FS: CubePrimitive, N: Size>(
    values: Vector<F, N>,
    block: FS,
    global: ComptimeOption<FG>,
) -> Vector<F, N> {
    #[comptime]
    match global {
        ComptimeOption::Some(global) => {
            let scale = f32::cast_from(global) * f32::cast_from(block);
            Vector::cast_from(dequantize_symmetric::<f32, f32, N>(
                Vector::<f32, N>::cast_from(values),
                scale,
            ))
        }
        ComptimeOption::None => dequantize_symmetric::<F, FS, N>(values, block),
    }
}

/// Divide values by the same effective scale, the quantize direction of
/// [`dequantize_symmetric_scaled`]. The quotient is within the quantization range, so narrowing it
/// back to `F` is safe. There is no divide counterpart to [`dequantize_symmetric`] to call: the
/// crate's `quantize_symmetric` is this plus the round and clamp, so this is where the divide is
/// defined.
#[cube]
pub(crate) fn quantize_symmetric_scaled<F: Float, FG: Numeric, FS: CubePrimitive, N: Size>(
    values: Vector<F, N>,
    block: FS,
    global: ComptimeOption<FG>,
) -> Vector<F, N> {
    #[comptime]
    match global {
        ComptimeOption::Some(global) => {
            let scale = f32::cast_from(global) * f32::cast_from(block);
            Vector::cast_from(Vector::<f32, N>::cast_from(values) / Vector::new(scale))
        }
        ComptimeOption::None => values / Vector::cast_from(block),
    }
}

/// Copy the global scale into the quantized tensor's own scale region, where dequantize reads it
/// back from.
///
/// The caller cannot hand its input buffer through instead: a quantized tensor's scales live in
/// one allocation the tensor owns, so the scale has to land inside that allocation, and only the
/// kernel writing it gets there without a second copy.
#[cube]
pub(crate) fn write_global<FG: Numeric>(
    global: ComptimeOption<FG>,
    out_global: ComptimeOption<LinearViewMut<'_, FG>>,
) {
    #[comptime]
    match out_global {
        ComptimeOption::Some(mut out) =>
        {
            #[comptime]
            match global {
                ComptimeOption::Some(global) => {
                    if ABSOLUTE_POS == 0 {
                        out.write(0, global);
                    }
                }
                ComptimeOption::None => {}
            }
        }
        ComptimeOption::None => {}
    }
}
