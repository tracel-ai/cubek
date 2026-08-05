//! The per-tensor scale of a two-level scheme.
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

use cubecl::prelude::*;
use cubecl::std::tensor::layout::linear::LinearView;

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

/// Multiply dequantized values by their block scale, folding in the per-tensor scale if there is
/// one.
#[cube]
pub(crate) fn dequantize_scaled<F: Float, FG: Numeric, FS: CubePrimitive, N: Size>(
    values: Vector<F, N>,
    block: FS,
    global: ComptimeOption<FG>,
) -> Vector<F, N> {
    #[comptime]
    match global {
        ComptimeOption::Some(global) => {
            let scale = f32::cast_from(global) * f32::cast_from(block);
            Vector::cast_from(Vector::<f32, N>::cast_from(values) * Vector::new(scale))
        }
        ComptimeOption::None => Vector::cast_from(block) * values,
    }
}

/// Divide values by the same effective scale, the quantize direction of [`dequantize_scaled`]. The
/// quotient is within the quantization range, so narrowing it back to `F` is safe.
#[cube]
pub(crate) fn quantize_scaled<F: Float, FG: Numeric, FS: CubePrimitive, N: Size>(
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
