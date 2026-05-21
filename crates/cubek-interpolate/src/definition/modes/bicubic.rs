use crate::definition::Interpolate;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Bicubic {}

#[cube]
impl Interpolate for Bicubic {
    const HALO: usize = 4;

    fn compute_weights<F: Float, N: Size>(frac: F) -> Array<Vector<F, N>> {
        let mut weights = Array::<Vector<F, N>>::new(Self::HALO);

        let a = float(-0.75);

        let f = Vector::new(frac);

        let inv_f = Vector::new(F::one() - frac);

        weights[0] = cubic_convolution_2(f + float(1.0), a);
        weights[1] = cubic_convolution_1(f, a);
        weights[2] = cubic_convolution_1(inv_f, a);
        weights[3] = cubic_convolution_2(inv_f + float(1.0), a);

        weights
    }
}

#[cube]
fn cubic_convolution_1<F: Float, N: Size>(x: Vector<F, N>, a: Vector<F, N>) -> Vector<F, N> {
    let conv = (a + float(2.0)) * x;
    let tmp = a + float(3.0);
    (conv - tmp) * x * x + float(1.0)
}

#[cube]
fn cubic_convolution_2<F: Float, N: Size>(x: Vector<F, N>, a: Vector<F, N>) -> Vector<F, N> {
    let conv = a * x;
    let conv = (conv - float(5.0) * a) * x;
    let tmp = float(8.0) * a;
    let conv = (conv + tmp) * x;

    conv - float(4.0) * a
}

#[cube]
fn float<F: Float, N: Size>(#[comptime] v: f32) -> Vector<F, N> {
    Vector::new(F::new(v))
}
