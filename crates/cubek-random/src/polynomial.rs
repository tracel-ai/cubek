//! Transcendentals evaluated as polynomials over the domain a draw hands them.
//!
//! A GPU issues `ln`, `cos`, and `sin` as instructions. The CPU JIT has no vector math
//! library behind them, so each one scalarizes into a libm call per lane and costs more
//! than every other step of a draw together.

use cubecl::prelude::*;
use std::f32::consts::{LN_2, SQRT_2};

// Taylor coefficients of `sin(pi * offset / 2)`, named by the power of `offset` each
// multiplies.
const SIN_1: f32 = 1.5707964;
const SIN_3: f32 = -0.6459641;
const SIN_5: f32 = 0.079692625;
const SIN_7: f32 = -0.004681754;
const SIN_9: f32 = 0.00016044118;

// Taylor coefficients of `cos(pi * offset / 2)`; the zeroth power is one.
const COS_2: f32 = -1.2337005;
const COS_4: f32 = 0.2536695;
const COS_6: f32 = -0.02086348;
const COS_8: f32 = 0.00091926026;

// Coefficients of `2 atanh(ratio) = 2 (ratio + ratio^3/3 + ratio^5/5 + ratio^7/7)`.
const ATANH_1: f32 = 2.0;
const ATANH_3: f32 = 0.6666667;
const ATANH_5: f32 = 0.4;
const ATANH_7: f32 = 0.2857143;

/// Cosine and sine of `turns` of a full turn, for `turns` in `[0, 1)`.
///
/// A caller holding its angle in turns needs no range reduction: the quadrant is one
/// truncating conversion, and the two series then run on the eighth of a turn around
/// zero, where five terms reach what an `f32` can hold. A negative `turns` would
/// truncate the wrong way and land in the quadrant below the one it belongs to.
#[cube]
pub fn cos_sin_turns<N: Size>(turns: Vector<f32, N>) -> (Vector<f32, N>, Vector<f32, N>) {
    let quarters = turns * Vector::new(4.0f32);
    let quadrant = Vector::<i32, N>::cast_from(quarters + Vector::new(0.5f32));
    let offset = quarters - Vector::<f32, N>::cast_from(quadrant);
    let square = offset * offset;

    let sine = ((((Vector::new(SIN_9) * square + Vector::new(SIN_7)) * square
        + Vector::new(SIN_5))
        * square
        + Vector::new(SIN_3))
        * square
        + Vector::new(SIN_1))
        * offset;
    let cosine = (((Vector::new(COS_8) * square + Vector::new(COS_6)) * square
        + Vector::new(COS_4))
        * square
        + Vector::new(COS_2))
        * square
        + Vector::new(1.0f32);

    let quadrant = quadrant & Vector::new(3i32);
    let swapped = (quadrant & Vector::new(1i32)).equal(&Vector::new(1i32));
    let cosine_magnitude = select_many(swapped, sine, cosine);
    let sine_magnitude = select_many(swapped, cosine, sine);

    let cosine_negative =
        ((quadrant + Vector::new(1i32)) & Vector::new(2i32)).equal(&Vector::new(2i32));
    let sine_negative = (quadrant & Vector::new(2i32)).equal(&Vector::new(2i32));

    (
        select_many(cosine_negative, -cosine_magnitude, cosine_magnitude),
        select_many(sine_negative, -sine_magnitude, sine_magnitude),
    )
}

/// Natural logarithm of a positive normal `x`.
///
/// The exponent carries the whole range, so only the mantissa is approximated, by the
/// odd series `ln(mantissa) = 2 atanh(ratio)`. A subnormal `x` holds no exponent to
/// extract, and comes back as nonsense rather than as a large negative number.
#[cube]
pub fn ln<N: Size>(x: Vector<f32, N>) -> Vector<f32, N> {
    let bits = Vector::<u32, N>::reinterpret(x);
    let exponent = Vector::<i32, N>::cast_from(bits >> Vector::new(23u32)) - Vector::new(127i32);
    let mantissa = Vector::<f32, N>::reinterpret(
        (bits & Vector::new(0x007f_ffffu32)) | Vector::new(0x3f80_0000u32),
    );

    // Dropping the upper half of `[1, 2)` by an octave centres the series argument on
    // zero. Left where it is, the top of the range would need twice the terms.
    let halved = mantissa.greater_than(&Vector::new(SQRT_2));
    let mantissa = select_many(halved, mantissa * Vector::new(0.5f32), mantissa);
    let exponent = select_many(halved, exponent + Vector::new(1i32), exponent);

    let ratio = (mantissa - Vector::new(1.0f32)) / (mantissa + Vector::new(1.0f32));
    let square = ratio * ratio;
    let series = ((Vector::new(ATANH_7) * square + Vector::new(ATANH_5)) * square
        + Vector::new(ATANH_3))
        * square
        + Vector::new(ATANH_1);

    Vector::<f32, N>::cast_from(exponent) * Vector::new(LN_2) + ratio * series
}
