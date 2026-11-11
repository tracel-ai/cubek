//! The activations, in `f32`. A line is walked element by element, so only scalar float intrinsics
//! are needed.

use cubecl::{self as cubecl, cube, prelude::*};

/// The logistic function.
#[cube]
pub fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}

/// `x · sigmoid(x)`.
#[cube]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `x · silu(gate)` element by element of a line.
#[cube]
pub fn gated<N: Size>(x: Vector<f32, N>, gate: Vector<f32, N>) -> Vector<f32, N> {
    let mut out = Vector::<f32, N>::empty();
    #[unroll]
    for element in 0..N::value() {
        out.insert(element, x.extract(element) * silu(gate.extract(element)));
    }
    out
}
