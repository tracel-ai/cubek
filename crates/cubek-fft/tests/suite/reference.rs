use std::f32::consts::PI;

use cubecl::zspace::Shape;
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
pub fn irfft_ref(re: &HostData, im: &HostData) -> HostData {
    // Expect shape: [num_windows, num_channels, num_freq_bins]
    let [num_windows, num_channels, num_freq_bins] = re
        .shape
        .as_slice()
        .try_into()
        .expect("Spectrum shape should be [num_windows, num_channels, num_freq_bins]");

    let sample_window = (num_freq_bins - 1) * 2; // reconstruct original window length
    assert!(
        sample_window.is_power_of_two(),
        "Requires power-of-2 sample_window length"
    );

    let out_shape = Shape::new([num_windows, num_channels, sample_window]);
    let out_strides = StrideSpec::RowMajor.compute_strides(&out_shape);

    let mut windows: Vec<Vec<Complex<f32>>> = Vec::with_capacity(num_windows * num_channels);

    for window in 0..num_windows {
        for channel in 0..num_channels {
            // Reconstruct full complex spectrum
            let mut spectrum = vec![Complex::new(0.0, 0.0); sample_window];

            for k in 0..num_freq_bins {
                let r = re.get_f32(&[window, channel, k]);
                let i = im.get_f32(&[window, channel, k]);
                spectrum[k] = Complex::new(r, i);
            }

            // Fill mirrored bins for Hermitian symmetry
            for k in 1..num_freq_bins - 1 {
                spectrum[sample_window - k] = spectrum[k].conj();
            }

            // Inverse FFT
            fft_recursive(&mut spectrum, FftMode::Inverse);

            // normalize amplitude
            for v in spectrum.iter_mut() {
                *v /= sample_window as f32;
            }

            windows.push(spectrum);
        }
    }

    // Flatten all windows
    let flattened: Vec<f32> = windows
        .into_iter()
        .flat_map(|v| v.into_iter().map(|c| c.re))
        .collect();

    HostData {
        data: HostDataVec::F32(flattened),
        shape: out_shape,
        strides: out_strides,
    }
}

/// Reference RFFT: input real slice, output first n/2 + 1 complex numbers
pub fn rfft_ref(signal: &HostData) -> (HostData, HostData) {
    let [num_windows, num_channels, sample_window] = signal
        .shape
        .as_slice()
        .try_into()
        .expect("Signal shape should be [num_windows, num_channels, sample_window]");
    assert!(
        sample_window.is_power_of_two(),
        "Requires power-of-2 sample_window length"
    );

    // We keep only first n/2 + 1 elements (Hermitian symmetry)
    let num_freq_bins = sample_window / 2 + 1;
    let out_shape = Shape::new([num_windows, num_channels, num_freq_bins]);
    let out_strides = StrideSpec::RowMajor.compute_strides(&out_shape);

    let mut spectrums = Vec::with_capacity(num_windows * num_channels);

    for window in 0..num_windows {
        for channel in 0..num_channels {
            let mut spectrum = Vec::with_capacity(sample_window);
            for i in 0..sample_window {
                let v = signal.get_f32(&[window, channel, i]);
                let complex = Complex::new(v, 0.);
                spectrum.push(complex);
            }

            fft_recursive(&mut spectrum, FftMode::Forward);

            spectrums.push(spectrum[..num_freq_bins].to_vec());
        }
    }

    let batched_spectrums: Vec<Complex<f32>> = spectrums.into_iter().flatten().collect();
    let (re, im): (Vec<f32>, Vec<f32>) =
        batched_spectrums.into_iter().map(|c| (c.re, c.im)).unzip();

    (
        HostData {
            data: HostDataVec::F32(re),
            shape: out_shape.clone(),
            strides: out_strides.clone(),
        },
        HostData {
            data: HostDataVec::F32(im),
            shape: out_shape,
            strides: out_strides,
        },
    )
}
