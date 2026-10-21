use core::f32::consts::PI;

use cubecl::{CubeElement, client::Client, prelude::Scalar, std::tensor::TensorHandle};
use cubek_fft::{
    ComplexTensorHandle, FftNormalization, irfft_interleaved_launch_padded,
    rfft_interleaved_launch_padded,
};
use num_complex::Complex32;

fn real_input(client: &Client, values: &[f32]) -> TensorHandle {
    TensorHandle::new_contiguous(
        vec![1, values.len()],
        client.create_from_slice(f32::as_bytes(values)),
        f32::elem_type_native(),
    )
}

fn rfft_reference(signal: &[f32], n_fft: usize, norm: FftNormalization) -> Vec<Complex32> {
    let scale = norm.scale_f32(n_fft).unwrap();
    (0..=n_fft / 2)
        .map(|k| {
            let mut sum = Complex32::new(0.0, 0.0);
            for (t, &value) in signal.iter().enumerate() {
                let angle = -2.0 * PI * (k * t) as f32 / n_fft as f32;
                sum += value * Complex32::new(angle.cos(), angle.sin());
            }
            sum * scale
        })
        .collect()
}

fn run_rfft(n_fft: usize, signal: &[f32], active: usize, tolerance: f32) {
    let client = cubecl::test_device().client();
    let input = real_input(&client, signal);
    for norm in [
        FftNormalization::None,
        FftNormalization::ByN,
        FftNormalization::Ortho,
    ] {
        let spectrum = ComplexTensorHandle::empty(&client, vec![1, n_fft / 2 + 1]).unwrap();
        rfft_interleaved_launch_padded(&client, &input, spectrum.binding(), 1, active, norm)
            .unwrap();
        let bytes = client.read_one(spectrum.into_raw_parts().handle).unwrap();
        let scalars = f32::from_bytes(&bytes);
        let expected = rfft_reference(&signal[..active], n_fft, norm);
        for (k, expected) in expected.iter().enumerate() {
            let (re, im) = (scalars[2 * k], scalars[2 * k + 1]);
            assert!(
                (re - expected.re).abs() < tolerance,
                "N={n_fft}, k={k}, {norm:?}: {re} != {}",
                expected.re
            );
            assert!(
                (im - expected.im).abs() < tolerance,
                "N={n_fft}, k={k}, {norm:?}: {im} != {}",
                expected.im
            );
        }
    }
}

#[test]
fn rfft_interleaved_n2_and_zero_padded_n8_match_dft() {
    run_rfft(2, &[1.0, -2.0], 2, 1e-5);
    run_rfft(8, &[1.0, 2.0, -0.5, 3.0, -1.0], 5, 1e-4);
}

#[test]
fn rfft_interleaved_packed_shared_and_four_step_match_sparse_reference() {
    let client = cubecl::test_device().client();
    let max_elems =
        client.properties().hardware.max_shared_memory_size / (2 * f32::elem_type_native().size());
    let shared = if max_elems.is_power_of_two() {
        max_elems
    } else {
        max_elems.next_power_of_two() / 2
    };
    for n_fft in [shared * 2, shared * 4] {
        let mut input = vec![0.0f32; n_fft];
        input[0] = 0.75;
        input[1] = -1.25;
        // The DFT helper would be quadratic for a large input. The first
        // two nonzero samples have X[k] = x[0] + x[1] exp(-2πik/N).
        let real = real_input(&client, &input);
        for norm in [
            FftNormalization::None,
            FftNormalization::ByN,
            FftNormalization::Ortho,
        ] {
            let output = ComplexTensorHandle::empty(&client, vec![1, n_fft / 2 + 1]).unwrap();
            rfft_interleaved_launch_padded(&client, &real, output.binding(), 1, n_fft, norm)
                .unwrap();
            let bytes = client.read_one(output.into_raw_parts().handle).unwrap();
            let actual = f32::from_bytes(&bytes);
            let scale = norm.scale_f32(n_fft).unwrap();
            for k in 0..=n_fft / 2 {
                let angle = -2.0 * PI * k as f32 / n_fft as f32;
                let expected = scale
                    * (Complex32::new(0.75, 0.0) - 1.25 * Complex32::new(angle.cos(), angle.sin()));
                assert!(
                    (actual[2 * k] - expected.re).abs() < 1e-3,
                    "N={n_fft} k={k} {norm:?} real"
                );
                assert!(
                    (actual[2 * k + 1] - expected.im).abs() < 1e-3,
                    "N={n_fft} k={k} {norm:?} imag"
                );
            }
        }
    }
}

fn complex_input(client: &Client, bins: &[Complex32]) -> ComplexTensorHandle {
    let scalars: Vec<f32> = bins.iter().flat_map(|bin| [bin.re, bin.im]).collect();
    ComplexTensorHandle::new_contiguous(
        vec![1, bins.len()],
        client.create_from_slice(f32::as_bytes(&scalars)),
        f32::elem_type_native(),
    )
    .unwrap()
}

fn irfft_reference(bins: &[Complex32], n_fft: usize, norm: FftNormalization) -> Vec<f32> {
    let scale = norm.scale_f32(n_fft).unwrap();
    (0..n_fft)
        .map(|t| {
            let mut value = bins[0].re;
            for (k, &bin) in bins.iter().enumerate().skip(1) {
                if k == n_fft / 2 {
                    value += bin.re * if t % 2 == 0 { 1.0 } else { -1.0 };
                } else {
                    let angle = 2.0 * PI * (k * t) as f32 / n_fft as f32;
                    value += 2.0 * (bin.re * angle.cos() - bin.im * angle.sin());
                }
            }
            value * scale
        })
        .collect()
}

fn run_irfft(n_fft: usize, bins: &[Complex32], active: usize, tolerance: f32) {
    let client = cubecl::test_device().client();
    let spectrum = complex_input(&client, bins);
    for norm in [
        FftNormalization::None,
        FftNormalization::ByN,
        FftNormalization::Ortho,
    ] {
        let output = TensorHandle::new_contiguous(
            vec![1, n_fft],
            client.empty(n_fft * 4),
            f32::elem_type_native(),
        );
        irfft_interleaved_launch_padded(&client, spectrum.binding(), &output, 1, active, norm)
            .unwrap();
        let bytes = client.read_one(output.handle).unwrap();
        let actual = f32::from_bytes(&bytes);
        for (i, expected) in irfft_reference(&bins[..active], n_fft, norm)
            .iter()
            .enumerate()
        {
            assert!(
                (actual[i] - expected).abs() < tolerance,
                "N={n_fft}, t={i}, {norm:?}: {} != {expected}",
                actual[i]
            );
        }
    }
}

#[test]
fn irfft_interleaved_n2_and_short_bins_n8_match_inverse_dft() {
    run_irfft(
        2,
        &[Complex32::new(1.0, 0.0), Complex32::new(-2.0, 0.0)],
        2,
        1e-5,
    );
    run_irfft(
        8,
        &[
            Complex32::new(1.0, 0.0),
            Complex32::new(-0.5, 2.0),
            Complex32::new(0.75, -1.0),
            Complex32::new(8.0, -3.0),
        ],
        3,
        1e-4,
    );
}

#[test]
fn irfft_interleaved_packed_shared_and_four_step_match_sparse_reference() {
    let client = cubecl::test_device().client();
    let max_elems =
        client.properties().hardware.max_shared_memory_size / (2 * f32::elem_type_native().size());
    let shared = if max_elems.is_power_of_two() {
        max_elems
    } else {
        max_elems.next_power_of_two() / 2
    };
    for n_fft in [shared * 2, shared * 4] {
        let bins = [Complex32::new(0.75, 0.0), Complex32::new(-0.5, 1.25)];
        let spectrum = complex_input(&client, &bins);
        for norm in [
            FftNormalization::None,
            FftNormalization::ByN,
            FftNormalization::Ortho,
        ] {
            let output = TensorHandle::new_contiguous(
                vec![1, n_fft],
                client.empty(n_fft * 4),
                f32::elem_type_native(),
            );
            irfft_interleaved_launch_padded(&client, spectrum.binding(), &output, 1, 2, norm)
                .unwrap();
            let bytes = client.read_one(output.handle).unwrap();
            let actual = f32::from_bytes(&bytes);
            let scale = norm.scale_f32(n_fft).unwrap();
            for (t, &value) in actual.iter().take(n_fft).enumerate() {
                let angle = 2.0 * PI * t as f32 / n_fft as f32;
                let expected = scale * (0.75 + 2.0 * (-0.5 * angle.cos() - 1.25 * angle.sin()));
                assert!(
                    (value - expected).abs() < 2e-3,
                    "N={n_fft} t={t} {norm:?}: {value} != {expected}",
                );
            }
        }
    }
}
