// use super::Interpolate;
// use cubecl::prelude::*;

// pub struct Bilinear {}

// #[cube]
// impl Interpolate for Bilinear {
//     fn halo() -> usize {
//         2 as usize
//     }

//     fn compute_weights<F: Float, N: Size>(
//         x_fraction: Vector<F, N>,
//         weights: &mut Array<Vector<F, N>>,
//     ) {
//         let one = Vector::new(F::cast_from(1.0));
//         let w0 = one - x_fraction;
//         let w1 = x_fraction;
//         weights[0] = w0;
//         weights[1] = w1;
//     }
// }
