use cubecl::prelude::*;

/// Converts a `u32` into a `f32` in the unit interval `[0.0, 1.0)`.
#[cube]
pub fn to_unit_interval_closed_open<N: Size>(int_random: Vector<u32, N>) -> Vector<f32, N> {
    // Use upper 24 bits for f32 precision
    // https://lemire.me/blog/2017/02/28/how-many-floating-point-numbers-are-in-the-interval-01/
    let shifted = int_random >> Vector::new(8u32);
    Vector::cast_from(shifted) / Vector::new(16777216.0f32) // 2^24
}

/// Converts a `u32` into a `f32` in the unit interval `(0.0, 1.0)`.
#[cube]
pub fn to_unit_interval_open<N: Size>(int_random: Vector<u32, N>) -> Vector<f32, N> {
    // Use upper 23 bits to leave room for the offset
    let shifted = int_random >> Vector::new(9u32);
    (Vector::cast_from(shifted) + Vector::new(1.0f32)) / Vector::new(8388609.0f32) // 2^23 + 1
}
