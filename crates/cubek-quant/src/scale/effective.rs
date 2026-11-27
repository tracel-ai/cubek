//! The effective scale: the factor a quantized value is reconstructed with, however many levels
//! the scheme spreads it across.
//!
//! Symmetric only, like the rest of the crate. A mode carrying a zero point offsets around the
//! scale rather than replacing it, so it would add methods here, not change these.

use cubecl::prelude::*;

/// The scale the finest level holds for this value's block, times the coarser level's under a
/// two-level scheme.
///
/// A two-level scheme forms the product in f32 and narrows only the result. The inner scale is a
/// tensor magnitude divided out by a normalizing factor, so it can be subnormal in a narrow
/// compute type even when every value it scales is representable there, and narrowing it first
/// rounds it to zero.
///
/// A one-level scheme has nothing to fold in, so it stays in the compute type. Widening there
/// would cost the packed two-value multiply a narrow type gets for free, which measures as a few
/// percent on f16 output.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Scale<FS: CubePrimitive> {
    pub(super) inner: FS,
    pub(super) coarser: ComptimeOption<f32>,
}

#[cube]
impl<FS: CubePrimitive> Scale<FS> {
    /// Reconstruct quantized values: `x = s * x_q`.
    pub fn dequantize_symmetric<F: Float, N: Size>(&self, values: Vector<F, N>) -> Vector<F, N> {
        #[comptime]
        match self.coarser {
            ComptimeOption::Some(coarser) => Vector::cast_from(
                Vector::<f32, N>::cast_from(values) * Vector::new(self.effective(coarser)),
            ),
            ComptimeOption::None => values * Vector::cast_from(self.inner),
        }
    }

    /// Divide values by the same factor, leaving the round and the clamp to the caller. The
    /// quotient is within the quantization range, so narrowing it back to `F` is safe.
    pub fn quantize_symmetric<F: Float, N: Size>(&self, values: Vector<F, N>) -> Vector<F, N> {
        #[comptime]
        match self.coarser {
            ComptimeOption::Some(coarser) => Vector::cast_from(
                Vector::<f32, N>::cast_from(values) / Vector::new(self.effective(coarser)),
            ),
            ComptimeOption::None => values / Vector::cast_from(self.inner),
        }
    }

    fn effective(&self, coarser: f32) -> f32 {
        coarser * f32::cast_from(self.inner)
    }
}
