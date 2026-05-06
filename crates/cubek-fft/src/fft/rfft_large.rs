//! Large-`n_fft` RFFT / IRFFT via the packed-real trick.
//!
//! Packed-real FFT turns an N-point real FFT into an M=N/2 point complex
//! FFT plus two elementwise passes. With the current shared-memory cutoff,
//! the large paths are:
//!
//! * N = 8192 → M = 4096 → single-pass shared-memory cfft (fast path).
//! * N = 16384 → M = 8192 → four-step cfft (still fast, no global
//!   ping-pong per stage).
//!
//! Compared to running a complex FFT with a zeroed imaginary input, the
//! packed form halves both FLOPs and memory traffic in the inner FFT. That
//! lets N = 8192 use a single shared-memory CFFT of size 4096 instead of the
//! four-step path.
//!
//! Forward `rfft` pipeline:
//!   1. `rfft_pack_kernel` — pack real `x[0..N]` into complex
//!      `y[k] = x[2k] + i*x[2k+1]`, length M.
//!   2. `cfft_launch_any_size(FORWARD)` — complex FFT of `y`.
//!   3. `rfft_post_kernel` — recover the half-spectrum
//!      `X[0..N/2+1]` from `Y` using the Z_even / Z_odd split.
//!
//! Inverse `irfft` pipeline:
//!   1. `irfft_pre_kernel` — rebuild the packed `Y` of length M from the
//!      half-spectrum `X[0..N/2+1]` (inverse of step 3 above).
//!   2. `cfft_launch_any_size(INVERSE)` — complex IFFT of `Y` into `y`.
//!      Note: butterfly output is unnormalised (sum, not mean); we fold
//!      the `1/M` factor into the unpack step.
//!   3. `irfft_unpack_kernel` — write `x[2k] = Re(y[k])`,
//!      `x[2k+1] = Im(y[k])`, applying the `1/M` normalisation.
//!
//! Invariants:
//! * `n_fft` power of two, `n_fft >= 4` (M >= 2 for the packed FFT).

use core::f32::consts::PI;

use cubecl::prelude::*;
use cubecl::std::tensor::{
    AsView as _, AsViewExpand, AsViewMut as _, AsViewMutExpand, TensorHandle,
};

use crate::{
    fft::{
        FftMode,
        cfft::{CfftBindings, cfft_launch_any_size},
    },
    layout::BatchSignalLayout,
};

/// Forward large-`n_fft` RFFT. Shapes:
/// * `signal`: (..., <= n_fft) real.
/// * `spectrum_re`, `spectrum_im`: (..., n_fft/2 + 1) complex.
pub(crate) fn rfft_large_launch<R: Runtime>(
    client: &ComputeClient<R>,
    signal: TensorBinding<R>,
    spectrum_re: TensorBinding<R>,
    spectrum_im: TensorBinding<R>,
    dim: usize,
    signal_len: usize,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let n_fft = (spectrum_re.shape[dim] - 1) * 2;
    let m = n_fft / 2;
    let count: usize = signal
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != dim)
        .map(|(_, e)| *e)
        .product();

    // Packed buffers of length M.
    let packed_shape: Vec<usize> = signal
        .shape
        .iter()
        .enumerate()
        .map(|(i, &s)| if i == dim { m } else { s })
        .collect();
    let packed_elems: usize = packed_shape.iter().product();
    let packed_re = TensorHandle::<R>::new_contiguous(
        packed_shape.clone(),
        client.empty(packed_elems * dtype.size()),
        dtype,
    );
    let packed_im = TensorHandle::<R>::new_contiguous(
        packed_shape.clone(),
        client.empty(packed_elems * dtype.size()),
        dtype,
    );

    // Step 1: pack x → y.
    {
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = cubecl::calculate_cube_count_elemwise(client, count * m, cube_dim);

        rfft_pack_kernel::launch::<f32, R>(
            client,
            cube_count,
            cube_dim,
            signal.into_tensor_arg(),
            packed_re.clone().binding().into_tensor_arg(),
            packed_im.clone().binding().into_tensor_arg(),
            (count * m) as u32,
            signal_len as u32,
            m,
            dim,
        );
    }

    // Step 2: Y = FFT_M(y), in-place on the packed buffers.
    cfft_launch_any_size::<R>(
        client,
        CfftBindings {
            input_re: packed_re.clone().binding(),
            input_im: packed_im.clone().binding(),
            output_re: packed_re.clone().binding(),
            output_im: packed_im.clone().binding(),
        },
        dim,
        dtype,
        FftMode::Forward,
    )?;

    // Step 3: recover half-spectrum X from Y.
    {
        let n_freq = m + 1;
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = cubecl::calculate_cube_count_elemwise(client, count * n_freq, cube_dim);

        rfft_post_kernel::launch::<f32, R>(
            client,
            cube_count,
            cube_dim,
            packed_re.binding().into_tensor_arg(),
            packed_im.binding().into_tensor_arg(),
            spectrum_re.into_tensor_arg(),
            spectrum_im.into_tensor_arg(),
            (count * n_freq) as u32,
            n_fft,
            m,
            dim,
        );
    }

    Ok(())
}

/// Inverse large-`n_fft` IRFFT. Shapes:
/// * `spectrum_re`, `spectrum_im`: (..., n_fft/2 + 1) complex.
/// * `signal`: (..., n_fft) real.
pub(crate) fn irfft_large_launch<R: Runtime>(
    client: &ComputeClient<R>,
    spectrum_re: TensorBinding<R>,
    spectrum_im: TensorBinding<R>,
    signal: TensorBinding<R>,
    dim: usize,
    spec_bins: usize,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let n_fft = signal.shape[dim];
    let m = n_fft / 2;
    let count: usize = signal
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != dim)
        .map(|(_, e)| *e)
        .product();

    let packed_shape: Vec<usize> = signal
        .shape
        .iter()
        .enumerate()
        .map(|(i, &s)| if i == dim { m } else { s })
        .collect();
    let packed_elems: usize = packed_shape.iter().product();
    let packed_in_re = TensorHandle::<R>::new_contiguous(
        packed_shape.clone(),
        client.empty(packed_elems * dtype.size()),
        dtype,
    );
    let packed_in_im = TensorHandle::<R>::new_contiguous(
        packed_shape.clone(),
        client.empty(packed_elems * dtype.size()),
        dtype,
    );
    let packed_out_re = TensorHandle::<R>::new_contiguous(
        packed_shape.clone(),
        client.empty(packed_elems * dtype.size()),
        dtype,
    );
    let packed_out_im = TensorHandle::<R>::new_contiguous(
        packed_shape.clone(),
        client.empty(packed_elems * dtype.size()),
        dtype,
    );

    // Step 1: build packed Y from half-spectrum X.
    {
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = cubecl::calculate_cube_count_elemwise(client, count * m, cube_dim);

        irfft_pre_kernel::launch::<f32, R>(
            client,
            cube_count,
            cube_dim,
            spectrum_re.into_tensor_arg(),
            spectrum_im.into_tensor_arg(),
            packed_in_re.clone().binding().into_tensor_arg(),
            packed_in_im.clone().binding().into_tensor_arg(),
            (count * m) as u32,
            spec_bins as u32,
            n_fft,
            m,
            dim,
        );
    }

    // Step 2: y = IFFT_M(Y). Need a separate destination because
    // cfft_four_step_launch ping-pongs internally and may not support
    // aliasing. (Small-path cfft aliases fine but we keep one code path.)
    cfft_launch_any_size::<R>(
        client,
        CfftBindings {
            input_re: packed_in_re.binding(),
            input_im: packed_in_im.binding(),
            output_re: packed_out_re.clone().binding(),
            output_im: packed_out_im.clone().binding(),
        },
        dim,
        dtype,
        FftMode::Inverse,
    )?;

    // Step 3: unpack y into real x with the 1/M normalisation.
    {
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = cubecl::calculate_cube_count_elemwise(client, count * m, cube_dim);

        irfft_unpack_kernel::launch::<f32, R>(
            client,
            cube_count,
            cube_dim,
            packed_out_re.binding().into_tensor_arg(),
            packed_out_im.binding().into_tensor_arg(),
            signal.into_tensor_arg(),
            (count * m) as u32,
            m,
            dim,
        );
    }

    Ok(())
}

// --- pack / post / pre / unpack kernels --------------------------------

/// `y[k] = x[2k] + i * x[2k+1]`, one thread per `k`.
#[cube(launch)]
fn rfft_pack_kernel<F: Float>(
    signal: &Tensor<F>,
    packed_re: &mut Tensor<F>,
    packed_im: &mut Tensor<F>,
    total: u32,
    signal_len: u32,
    #[comptime] m: usize,
    #[comptime] dim: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize {
        terminate!();
    }
    let k = pos % m;
    let window = pos / m;
    let signal_view = signal.view(BatchSignalLayout::new(signal, window, dim));
    let mut packed_re_view = packed_re.view_mut(BatchSignalLayout::new(packed_re, window, dim));
    let mut packed_im_view = packed_im.view_mut(BatchSignalLayout::new(packed_im, window, dim));
    let even = 2 * k;
    let odd = even + 1;
    let even_active = even < signal_len as usize;
    let odd_active = odd < signal_len as usize;
    let even = select(even_active, even, 0);
    let odd = select(odd_active, odd, 0);
    packed_re_view[k] = select(even_active, signal_view[even], F::new(0.0));
    packed_im_view[k] = select(odd_active, signal_view[odd], F::new(0.0));
}

/// Recover `X[0..N/2+1]` from `Y[0..M]` for the packed-real forward path.
///
/// Let `A = Y[k]`, `B = conj(Y[M-k])` for 0 < k < M.
///   `Z_e[k] = (A + B) / 2`
///   `Z_o[k] = -i * (A - B) / 2`
///   `X[k] = Z_e[k] + W_N^k * Z_o[k]`
/// Edge cases:
///   `X[0] = Re(Y[0]) + Im(Y[0])`, `X[M] = Re(Y[0]) - Im(Y[0])` (both real).
///
/// One thread per output bin `k in [0, M+1)`.
#[cube(launch)]
fn rfft_post_kernel<F: Float>(
    packed_re: &Tensor<F>,
    packed_im: &Tensor<F>,
    spectrum_re: &mut Tensor<F>,
    spectrum_im: &mut Tensor<F>,
    total: u32,
    #[comptime] n_fft: usize,
    #[comptime] m: usize,
    #[comptime] dim: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize {
        terminate!();
    }
    let n_freq = comptime![m + 1];
    let k = pos % n_freq;
    let window = pos / n_freq;
    let packed_re_view = packed_re.view(BatchSignalLayout::new(packed_re, window, dim));
    let packed_im_view = packed_im.view(BatchSignalLayout::new(packed_im, window, dim));
    let mut spectrum_re_view =
        spectrum_re.view_mut(BatchSignalLayout::new(spectrum_re, window, dim));
    let mut spectrum_im_view =
        spectrum_im.view_mut(BatchSignalLayout::new(spectrum_im, window, dim));

    if k == 0 {
        let y0_re = packed_re_view[0];
        let y0_im = packed_im_view[0];
        spectrum_re_view[k] = y0_re + y0_im;
        spectrum_im_view[k] = F::new(0.0);
    } else if k == m {
        let y0_re = packed_re_view[0];
        let y0_im = packed_im_view[0];
        spectrum_re_view[k] = y0_re - y0_im;
        spectrum_im_view[k] = F::new(0.0);
    } else {
        let a_re = packed_re_view[k];
        let a_im = packed_im_view[k];
        let b_re = packed_re_view[m - k];
        let b_im_raw = packed_im_view[m - k];
        let b_im = -b_im_raw; // conj(Y[M-k])

        // Forward twiddle W_N^k = cos(-2π k / N) + i sin(-2π k / N).
        let two_pi = F::new(2.0 * PI);
        let theta = -two_pi * F::cast_from(k) / F::cast_from(n_fft);
        let c = theta.cos();
        let s = theta.sin();

        // Precompute reused sums. Derivation:
        //   1 - i*W = (1 + s) - i*c
        //   1 + i*W = (1 - s) + i*c
        //   2 X[k]  = A*(1 - i*W) + B*(1 + i*W)
        let one_plus_s = F::new(1.0) + s;
        let one_minus_s = F::new(1.0) - s;
        let x_re = F::new(0.5) * (a_re * one_plus_s + a_im * c + b_re * one_minus_s - b_im * c);
        let x_im = F::new(0.5) * (a_im * one_plus_s - a_re * c + b_re * c + b_im * one_minus_s);
        spectrum_re_view[k] = x_re;
        spectrum_im_view[k] = x_im;
    }
}

/// Build packed `Y[0..M]` from half-spectrum `X[0..N/2+1]` for the
/// packed-real inverse path. Inverse of `rfft_post_kernel`.
///
/// For 0 < k < M:
///   `Z_e[k] = (X[k] + conj(X[M-k])) / 2`
///   `Z_o[k] = W_N^{-k} * (X[k] - conj(X[M-k])) / 2`
///   `Y[k] = Z_e[k] + i * Z_o[k]`
/// Edge case `k = 0`:
///   `Y[0] = (X[0] + X[M]) / 2  +  i * (X[0] - X[M]) / 2`.
///
/// One thread per packed bin `k in [0, M)`.
#[cube(launch)]
fn irfft_pre_kernel<F: Float>(
    spectrum_re: &Tensor<F>,
    spectrum_im: &Tensor<F>,
    packed_re: &mut Tensor<F>,
    packed_im: &mut Tensor<F>,
    total: u32,
    spec_bins: u32,
    #[comptime] n_fft: usize,
    #[comptime] m: usize,
    #[comptime] dim: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize {
        terminate!();
    }
    let k = pos % m;
    let window = pos / m;
    let spectrum_re_view = spectrum_re.view(BatchSignalLayout::new(spectrum_re, window, dim));
    let spectrum_im_view = spectrum_im.view(BatchSignalLayout::new(spectrum_im, window, dim));
    let mut packed_re_view = packed_re.view_mut(BatchSignalLayout::new(packed_re, window, dim));
    let mut packed_im_view = packed_im.view_mut(BatchSignalLayout::new(packed_im, window, dim));

    if k == 0 {
        let has_nyquist = m < spec_bins as usize;
        let x0_re = spectrum_re_view[0];
        let xm = select(has_nyquist, m, 0);
        let xm_re = select(has_nyquist, spectrum_re_view[xm], F::new(0.0));
        packed_re_view[k] = F::new(0.5) * (x0_re + xm_re);
        packed_im_view[k] = F::new(0.5) * (x0_re - xm_re);
    } else {
        let active = k < spec_bins as usize;
        let src = select(active, k, 0);
        let x_re = select(active, spectrum_re_view[src], F::new(0.0));
        let x_im = select(active, spectrum_im_view[src], F::new(0.0));
        let mirror = m - k;
        let mirror_active = mirror < spec_bins as usize;
        let mirror = select(mirror_active, mirror, 0);
        let xm_re = select(mirror_active, spectrum_re_view[mirror], F::new(0.0));
        let xm_im_raw = select(mirror_active, spectrum_im_view[mirror], F::new(0.0));
        let xm_im = -xm_im_raw; // conj(X[M-k])

        // Inverse twiddle W_N^{-k} = cos(2π k / N) + i sin(2π k / N).
        let two_pi = F::new(2.0 * PI);
        let theta = two_pi * F::cast_from(k) / F::cast_from(n_fft);
        let c = theta.cos();
        let s = theta.sin();

        // Derivation (inverse post). Let W = W_N^{-k} = c + i*s.
        //   1 + i*W = (1 - s) + i*c        → A * (1 + i*W):
        //     Re = x_re*(1-s) - x_im*c
        //     Im = x_re*c    + x_im*(1-s)
        //   1 - i*W = (1 + s) - i*c        → B * (1 - i*W), B = conj(X[M-k]):
        //     Re = xm_re*(1+s) + xm_im*c   (xm_im is already negated here)
        //     Im = -xm_re*c   + xm_im*(1+s)
        //   2 Y[k] = A*(1 + i*W) + B*(1 - i*W).
        let one_plus_s = F::new(1.0) + s;
        let one_minus_s = F::new(1.0) - s;
        let y_re = F::new(0.5) * (x_re * one_minus_s - x_im * c + xm_re * one_plus_s + xm_im * c);
        let y_im = F::new(0.5) * (x_im * one_minus_s + x_re * c - xm_re * c + xm_im * one_plus_s);
        packed_re_view[k] = y_re;
        packed_im_view[k] = y_im;
    }
}

/// Unpack `y[k]` into real `x` with the 1/M normalisation folded in.
/// `x[2k] = Re(y[k]) / M`, `x[2k+1] = Im(y[k]) / M`. One thread per `k`.
#[cube(launch)]
fn irfft_unpack_kernel<F: Float>(
    packed_re: &Tensor<F>,
    packed_im: &Tensor<F>,
    signal: &mut Tensor<F>,
    total: u32,
    #[comptime] m: usize,
    #[comptime] dim: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize {
        terminate!();
    }
    let k = pos % m;
    let window = pos / m;
    let packed_re_view = packed_re.view(BatchSignalLayout::new(packed_re, window, dim));
    let packed_im_view = packed_im.view(BatchSignalLayout::new(packed_im, window, dim));
    let mut signal_view = signal.view_mut(BatchSignalLayout::new(signal, window, dim));
    let scale = F::new(1.0) / F::cast_from(m);
    signal_view[2 * k] = packed_re_view[k] * scale;
    signal_view[2 * k + 1] = packed_im_view[k] * scale;
}
