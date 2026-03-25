use std::f32::consts::PI;

use cubecl::{
    prelude::*,
    std::tensor::{View, layout::Coords1d},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Whether doing RFFT or IRFFT.
pub enum FftMode {
    Forward,
    Inverse,
}

impl FftMode {
    pub fn sign(&self) -> f32 {
        match self {
            FftMode::Forward => -1.,
            FftMode::Inverse => 1.,
        }
    }
}

#[cube]
/// In-place FFT of a 1D complex signal.
/// Reorders input with bit-reversal and applies butterfly stages
pub(crate) fn fft_inner_compute<F: Float>(
    spectrum_re: &mut View<F, Coords1d, ReadWrite>,
    spectrum_im: &mut View<F, Coords1d, ReadWrite>,
    #[comptime] fft_mode: FftMode,
) {
    let num_samples = spectrum_re.shape();

    bit_reverse_permutation(spectrum_re, spectrum_im, num_samples);

    fft_butterfly_stages(spectrum_re, spectrum_im, fft_mode);
}

#[cube]
/// In-place bit-reversal permutation.
///
/// Reorders elements so index `i` maps to the index formed by
/// reversing the `log2(n)` bits of `i`.
fn bit_reverse_permutation<F: Float>(
    view_re: &mut View<F, Coords1d, ReadWrite>,
    view_im: &mut View<F, Coords1d, ReadWrite>,
    n: usize,
) {
    let mut j = 0;
    for i in 0..n {
        if i < j {
            swap(view_re, i, j);
            swap(view_im, i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

#[cube]
/// Swap two elements of a 1D array.
fn swap<F: Float>(view_1d: &mut View<F, Coords1d, ReadWrite>, i: usize, j: usize) {
    let tmp = view_1d[i];
    view_1d[i] = view_1d[j];
    view_1d[j] = tmp;
}

#[cube]
/// Iterative radix-2 FFT butterfly computation.
/// Combines pairs of elements using twiddle factors to compute higher-level FFT outputs.
fn fft_butterfly_stages<F: Float>(
    spectrum_re: &mut View<F, Coords1d, ReadWrite>,
    spectrum_im: &mut View<F, Coords1d, ReadWrite>,
    #[comptime] fft_mode: FftMode,
) {
    let n = spectrum_re.shape();
    let mut m = 2;

    while m <= n {
        let half_m = m >> 1;

        // twiddle base: exp(-2πi / m)
        let theta = F::new(fft_mode.sign() * 2.0 * PI) / F::cast_from(m);

        let wm_re = theta.cos();
        let wm_in = theta.sin();

        let mut k = 0;
        while k < n {
            let mut w_re = F::new(1.0);
            let mut w_im = F::new(0.0);

            let mut j = 0;
            while j < half_m {
                let i0 = k + j;
                let i1 = i0 + half_m;

                let a = (spectrum_re[i0], spectrum_im[i0]);
                let b = (spectrum_re[i1], spectrum_im[i1]);

                let t = complex_mul::<F>((w_re, w_im), b);
                let out0 = complex_add::<F>(a, t);
                let out1 = complex_sub::<F>(a, t);

                spectrum_re[i0] = out0.0;
                spectrum_im[i0] = out0.1;
                spectrum_re[i1] = out1.0;
                spectrum_im[i1] = out1.1;

                let new_w = complex_mul::<F>((w_re, w_im), (wm_re, wm_in));
                w_re = new_w.0;
                w_im = new_w.1;

                j += 1;
            }

            k += m;
        }

        m <<= 1;
    }
}

#[cube]
/// Addition on a complex number encoded as a pair of floats
fn complex_add<F: Float>(a: (F, F), b: (F, F)) -> (F, F) {
    (a.0 + b.0, a.1 + b.1)
}

#[cube]
/// Subtraction on a complex number encoded as a pair of floats
fn complex_sub<F: Float>(a: (F, F), b: (F, F)) -> (F, F) {
    (a.0 - b.0, a.1 - b.1)
}

#[cube]
/// Multiplication on a complex number encoded as a pair of floats
fn complex_mul<F: Float>(a: (F, F), b: (F, F)) -> (F, F) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
