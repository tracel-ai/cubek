#![allow(clippy::needless_range_loop)]

use std::f32::consts::PI;

use cubecl::zspace::{Shape, Strides};
use cubek_test_utils::{HostData, HostDataVec, StrideSpec};
use num_complex::Complex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Whether doing RFFT or IRFFT.
enum FftMode {
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

/// Recursive Cooley-Tukey IFFT for complex inputs (length must be power of 2)
fn fft_recursive(x: &mut [Complex<f32>], fft_mode: FftMode) {
    let n = x.len();
    if n <= 1 {
        return;
    }

    let mut even: Vec<_> = x.iter().step_by(2).cloned().collect();
    let mut odd: Vec<_> = x.iter().skip(1).step_by(2).cloned().collect();

    fft_recursive(&mut even, fft_mode);
    fft_recursive(&mut odd, fft_mode);

    for k in 0..n / 2 {
        let t = Complex::from_polar(1.0, fft_mode.sign() * 2.0 * PI * k as f32 / n as f32) * odd[k];
        x[k] = even[k] + t;
        x[k + n / 2] = even[k] - t;
    }
}
/// Reference IRFFT: reconstruct real signal from first n/2 + 1 complex bins
pub fn irfft_ref(re: &HostData, im: &HostData, dim: usize) -> HostData {
    let in_shape = re.shape.as_slice();
    let num_freq_bins = in_shape[dim];
    let sample_window = (num_freq_bins - 1) * 2; // reconstruct original window length
    assert!(
        sample_window.is_power_of_two(),
        "Requires power-of-2 sample_window length"
    );

    let mut out_shape_vec = in_shape.to_vec();
    out_shape_vec[dim] = sample_window;
    let out_shape = Shape::from(out_shape_vec);
    let num_windows = re.shape.num_elements() / num_freq_bins;
    let out_strides = StrideSpec::RowMajor.compute_strides(&out_shape);

    let mut flattened = vec![0.0; out_shape.num_elements()];

    for l in 0..num_windows {
        // Reconstruct full complex spectrum
        let mut coords = get_coords(l, in_shape, dim);
        let mut spectrum = vec![Complex::new(0.0, 0.0); sample_window];

        for k in 0..num_freq_bins {
            coords[dim] = k;
            let r = re.get_f32(&coords);
            let i = im.get_f32(&coords);
            spectrum[k] = Complex::new(r, i);
        }

        // Fill mirrored bins for Hermitian symmetry
        for k in 1..num_freq_bins - 1 {
            spectrum[sample_window - k] = spectrum[k].conj();
        }

        // Inverse FFT
        fft_recursive(&mut spectrum, FftMode::Inverse);

        for i in 0..sample_window {
            coords[dim] = i;
            let flat_idx = compute_index(&out_strides, coords.as_slice());

            // normalize amplitude
            flattened[flat_idx] = spectrum[i].re / sample_window as f32;
        }
    }

    HostData {
        data: HostDataVec::F32(flattened),
        shape: out_shape,
        strides: out_strides,
    }
}

/// Reference RFFT: input real slice, output first n/2 + 1 complex numbers
pub fn rfft_ref(signal: &HostData, dim: usize) -> (HostData, HostData) {
    let in_shape = signal.shape.as_slice();
    let sample_window = in_shape[dim];
    let num_freq_bins = sample_window / 2 + 1;
    assert!(
        sample_window.is_power_of_two(),
        "Requires power-of-2 sample_window length"
    );

    let mut out_shape_vec = in_shape.to_vec();
    out_shape_vec[dim] = num_freq_bins;
    let out_shape = Shape::from(out_shape_vec);
    let num_windows = signal.shape.num_elements() / sample_window;
    let out_strides = StrideSpec::RowMajor.compute_strides(&out_shape);

    let mut re_data = vec![0.0; out_shape.num_elements()];
    let mut im_data = vec![0.0; out_shape.num_elements()];
    for l in 0..num_windows {
        let mut coords = get_coords(l, in_shape, dim);
        let mut spectrum = Vec::with_capacity(sample_window);
        for i in 0..sample_window {
            coords[dim] = i;
            let v = signal.get_f32(&coords);
            let complex = Complex::new(v, 0.);
            spectrum.push(complex);
        }

        fft_recursive(&mut spectrum, FftMode::Forward);
        for k in 0..num_freq_bins {
            coords[dim] = k;
            let flat_idx = compute_index(&out_strides, coords.as_slice());
            re_data[flat_idx] = spectrum[k].re;
            im_data[flat_idx] = spectrum[k].im;
        }
    }

    (
        HostData {
            data: HostDataVec::F32(re_data),
            shape: out_shape.clone(),
            strides: out_strides.clone(),
        },
        HostData {
            data: HostDataVec::F32(im_data),
            shape: out_shape,
            strides: out_strides,
        },
    )
}

fn get_coords(lane_idx: usize, shape: &[usize], dim: usize) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    let mut temp = lane_idx;
    for i in (0..shape.len()).rev() {
        if i == dim {
            continue;
        }
        coords[i] = temp % shape[i];
        temp /= shape[i];
    }
    coords
}

pub fn compute_index(strides: &Strides, coords: &[usize]) -> usize {
    assert_eq!(
        coords.len(),
        strides.rank(),
        "Coordinate rank must match stride rank",
    );

    coords
        .iter()
        .zip(strides.iter())
        .map(|(&c, &s)| c * s)
        .sum()
}
