//! CPU reference and seeded "produce a HostData" primitives for FFT.
//!
//! Both `kernel_result` and `cpu_reference_result` build the same input bits
//! from `(seed_lhs, seed_rhs)`, so the two `HostData`s they return are
//! directly comparable.
//!
//! For the forward (RFFT) mode, the output is the imaginary and real parts
//! concatenated along a fresh leading dim of size 2: `[2, ...freq_shape]` with
//! index `0` = real, `1` = imag. The inverse (IRFFT) mode just returns the
//! reconstructed signal.

#![allow(clippy::needless_range_loop)]

use std::f32::consts::PI;

use cubecl::{
    TestRuntime,
    client::ComputeClient,
    frontend::CubePrimitive,
    zspace::{Shape, Strides},
};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, HostDataVec, StrideSpec, TestInput,
    launch_and_capture_outcome,
};
use num_complex::Complex;

use crate::fft::{FftMode, irfft_launch, rfft_launch};

/// Run the FFT kernel for `mode` against the given problem with seeded inputs
/// and return its output as a [`HostData`].
pub fn kernel_result(
    client: ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    dim: usize,
    mode: FftMode,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    let dtype = f32::as_type_native_unchecked().storage_type();

    match mode {
        FftMode::Forward => {
            let (signal, _) = TestInput::builder(client.clone(), shape.clone())
                .dtype(dtype)
                .uniform(seed_lhs, -1., 1.)
                .generate_with_f32_host_data();

            let mut spectrum_shape = shape.clone();
            spectrum_shape[dim] = shape[dim] / 2 + 1;

            let re = TestInput::builder(client.clone(), spectrum_shape.clone())
                .dtype(dtype)
                .zeros()
                .generate_without_host_data();
            let im = TestInput::builder(client.clone(), spectrum_shape.clone())
                .dtype(dtype)
                .zeros()
                .generate_without_host_data();

            let outcome = launch_and_capture_outcome(&client, |c| {
                rfft_launch::<TestRuntime>(
                    c,
                    signal.clone().binding(),
                    re.clone().binding(),
                    im.clone().binding(),
                    dim,
                    dtype,
                )
                .into()
            });

            match outcome {
                ExecutionOutcome::CompileError(e) => Err(format!("compile error: {e}")),
                ExecutionOutcome::Executed => {
                    let re_host = HostData::from_tensor_handle(&client, re, HostDataType::F32);
                    let im_host = HostData::from_tensor_handle(&client, im, HostDataType::F32);
                    Ok(stack_re_im(re_host, im_host))
                }
            }
        }
        FftMode::Inverse => {
            let mut spectrum_shape = shape.clone();
            spectrum_shape[dim] = shape[dim] / 2 + 1;

            let (re, _) = TestInput::builder(client.clone(), spectrum_shape.clone())
                .dtype(dtype)
                .uniform(seed_lhs, -1., 1.)
                .generate_with_f32_host_data();
            let (im, _) = TestInput::builder(client.clone(), spectrum_shape.clone())
                .dtype(dtype)
                .uniform(seed_rhs, -1., 1.)
                .generate_with_f32_host_data();

            let signal = TestInput::builder(client.clone(), shape.clone())
                .dtype(dtype)
                .zeros()
                .generate_without_host_data();

            let outcome = launch_and_capture_outcome(&client, |c| {
                irfft_launch::<TestRuntime>(
                    c,
                    re.binding(),
                    im.binding(),
                    signal.clone().binding(),
                    dim,
                    dtype,
                )
                .into()
            });

            match outcome {
                ExecutionOutcome::CompileError(e) => Err(format!("compile error: {e}")),
                ExecutionOutcome::Executed => Ok(HostData::from_tensor_handle(
                    &client,
                    signal,
                    HostDataType::F32,
                )),
            }
        }
    }
}

/// CPU-only counterpart to [`kernel_result`]: generate the same seeded inputs
/// and run the recursive Cooley-Tukey reference. Returns the stacked re/im
/// pair for [`FftMode::Forward`] and the reconstructed signal for
/// [`FftMode::Inverse`].
pub fn cpu_reference_result(
    client: ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    dim: usize,
    mode: FftMode,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    let dtype = f32::as_type_native_unchecked().storage_type();

    match mode {
        FftMode::Forward => {
            let (_, signal) = TestInput::builder(client.clone(), shape.clone())
                .dtype(dtype)
                .uniform(seed_lhs, -1., 1.)
                .generate_with_f32_host_data();
            let (re, im) = rfft_ref(&signal, dim);
            Ok(stack_re_im(re, im))
        }
        FftMode::Inverse => {
            let mut spectrum_shape = shape.clone();
            spectrum_shape[dim] = shape[dim] / 2 + 1;

            let (_, re) = TestInput::builder(client.clone(), spectrum_shape.clone())
                .dtype(dtype)
                .uniform(seed_lhs, -1., 1.)
                .generate_with_f32_host_data();
            let (_, im) = TestInput::builder(client.clone(), spectrum_shape.clone())
                .dtype(dtype)
                .uniform(seed_rhs, -1., 1.)
                .generate_with_f32_host_data();

            Ok(irfft_ref(&re, &im, dim))
        }
    }
}

/// Stack two equal-shape `HostData` blobs along a fresh leading dim of size 2.
/// Index `0` along that dim is `re`, index `1` is `im`. Used so the forward
/// mode can produce a single comparable [`HostData`] from a (re, im) pair.
fn stack_re_im(re: HostData, im: HostData) -> HostData {
    assert_eq!(re.shape, im.shape, "re/im shape mismatch");
    let inner_shape = re.shape.as_slice().to_vec();
    let inner_numel: usize = inner_shape.iter().product();

    let HostDataVec::F32(re_vec) = re.data else {
        panic!("re must be F32");
    };
    let HostDataVec::F32(im_vec) = im.data else {
        panic!("im must be F32");
    };

    let re_strides_slice: &[usize] = &re.strides;
    let im_strides_slice: &[usize] = &im.strides;
    let mut packed = Vec::with_capacity(inner_numel * 2);
    pack_contiguous(&mut packed, &re_vec, re_strides_slice, &inner_shape);
    pack_contiguous(&mut packed, &im_vec, im_strides_slice, &inner_shape);

    let mut out_shape_vec = vec![2];
    out_shape_vec.extend(inner_shape);
    let out_shape = Shape::from(out_shape_vec);
    let strides = StrideSpec::RowMajor.compute_strides(&out_shape);

    HostData {
        data: HostDataVec::F32(packed),
        shape: out_shape,
        strides,
    }
}

fn pack_contiguous(out: &mut Vec<f32>, data: &[f32], strides: &[usize], shape: &[usize]) {
    let mut idx = vec![0usize; shape.len()];
    let total: usize = shape.iter().product();
    for _ in 0..total {
        let mut linear = 0;
        for (s, c) in strides.iter().zip(idx.iter()) {
            linear += s * c;
        }
        out.push(data[linear]);

        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

/// Recursive Cooley-Tukey FFT for complex inputs (length must be power of 2).
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

/// Reference IRFFT: reconstruct real signal from first n/2 + 1 complex bins.
pub fn irfft_ref(re: &HostData, im: &HostData, dim: usize) -> HostData {
    let in_shape = re.shape.as_slice();
    let num_freq_bins = in_shape[dim];
    let sample_window = (num_freq_bins - 1) * 2;
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
        let mut coords = get_coords(l, in_shape, dim);
        let mut spectrum = vec![Complex::new(0.0, 0.0); sample_window];

        for k in 0..num_freq_bins {
            coords[dim] = k;
            let r = re.get_f32(&coords);
            let i = im.get_f32(&coords);
            spectrum[k] = Complex::new(r, i);
        }

        for k in 1..num_freq_bins - 1 {
            spectrum[sample_window - k] = spectrum[k].conj();
        }

        fft_recursive(&mut spectrum, FftMode::Inverse);

        for i in 0..sample_window {
            coords[dim] = i;
            let flat_idx = compute_index(&out_strides, coords.as_slice());

            flattened[flat_idx] = spectrum[i].re / sample_window as f32;
        }
    }

    HostData {
        data: HostDataVec::F32(flattened),
        shape: out_shape,
        strides: out_strides,
    }
}

/// Reference RFFT: input real slice, output first n/2 + 1 complex numbers.
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
            spectrum.push(Complex::new(v, 0.));
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

fn compute_index(strides: &Strides, coords: &[usize]) -> usize {
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
