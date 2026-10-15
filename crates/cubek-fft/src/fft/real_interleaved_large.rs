//! Packed-real large FFTs with interleaved spectrum boundaries.
//!
//! Modified from tensor4all/cubek `02efeba9` (MIT OR Apache-2.0): the
//! boundary kernels use the current upstream packed CFFT and an in-place
//! inverse while preserving explicit public normalization.

use core::f64::consts::PI;

use cubecl::prelude::*;
use cubecl::std::tensor::{
    AsView as _, AsViewExpand, AsViewMut as _, AsViewMutExpand, TensorHandle,
};

use crate::{
    ComplexTensorBinding, FftError, FftNormalization,
    fft::{
        FftMode,
        cfft::{CfftBindings, cfft_launch_any_size},
        rfft_large::rfft_pack_kernel,
    },
    interleaved_layout::InterleavedBatchSignalLayout,
    layout::BatchSignalLayout,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn rfft_interleaved_large_launch(
    client: &Client,
    signal: &TensorHandle,
    spectrum: ComplexTensorBinding<'_>,
    dim: usize,
    signal_len: u32,
    normalization: FftNormalization,
    n_fft: usize,
    packed: usize,
    post: usize,
) -> Result<(), FftError> {
    let m = n_fft / 2;
    let packed_shape = signal
        .shape()
        .iter()
        .enumerate()
        .map(|(axis, &extent)| if axis == dim { m } else { extent })
        .collect::<Vec<_>>();
    let bytes = packed
        .checked_mul(f32::elem_type_native().size())
        .ok_or(FftError::SizeOverflow)?;
    let packed_re = TensorHandle::new_contiguous(
        packed_shape.clone(),
        client.empty(bytes),
        f32::elem_type_native(),
    );
    let packed_im =
        TensorHandle::new_contiguous(packed_shape, client.empty(bytes), f32::elem_type_native());
    let cube_dim = CubeDim::new_1d(256);
    let cube_count = cubecl::calculate_cube_count_elemwise(client, packed, cube_dim);
    rfft_pack_kernel::launch(
        client,
        cube_count,
        cube_dim,
        signal.clone().binding().into_tensor_arg(),
        packed_re.clone().binding().into_tensor_arg(),
        packed_im.clone().binding().into_tensor_arg(),
        packed as u32,
        signal_len,
        m,
        dim,
        f32::elem_type_native(),
    );
    cfft_launch_any_size(
        client,
        CfftBindings {
            input_re: packed_re.clone().binding(),
            input_im: packed_im.clone().binding(),
            output_re: packed_re.clone().binding(),
            output_im: packed_im.clone().binding(),
        },
        dim,
        f32::elem_type_native(),
        FftMode::Forward,
    )?;
    let cube_count = cubecl::calculate_cube_count_elemwise(client, post, cube_dim);
    rfft_post_interleaved_kernel::launch(
        client,
        cube_count,
        cube_dim,
        packed_re.binding().into_tensor_arg(),
        packed_im.binding().into_tensor_arg(),
        spectrum.tensor().into_tensor_arg(),
        post as u32,
        n_fft,
        m,
        dim,
        normalization,
        f32::elem_type_native(),
    );
    Ok(())
}

/// Recover `X[0..N/2+1]` directly into adjacent real/imaginary scalars.
#[cube(launch)]
fn rfft_post_interleaved_kernel<F: Float>(
    packed_re: &Tensor<F>,
    packed_im: &Tensor<F>,
    spectrum: &mut Tensor<F>,
    total: u32,
    #[comptime] n_fft: usize,
    #[comptime] m: usize,
    #[comptime] dim: usize,
    #[comptime] normalization: FftNormalization,
    #[define(F)] _dtype: ElemType,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize {
        terminate!();
    }
    let n_freq = comptime![m + 1];
    let k = pos % n_freq;
    let window = pos / n_freq;
    let re = packed_re.view(BatchSignalLayout::new(packed_re, window, dim));
    let im = packed_im.view(BatchSignalLayout::new(packed_im, window, dim));
    let scale = match normalization {
        FftNormalization::None => F::new(1.0),
        FftNormalization::ByN => F::new(1.0) / F::cast_from(n_fft),
        FftNormalization::Ortho => F::new(1.0) / F::cast_from(n_fft).sqrt(),
    };
    let (x_re, x_im) = if k == 0 {
        (re.read_checked(0) + im.read_checked(0), F::new(0.0))
    } else if k == m {
        (re.read_checked(0) - im.read_checked(0), F::new(0.0))
    } else {
        let a_re = re.read_checked(k);
        let a_im = im.read_checked(k);
        let b_re = re.read_checked(m - k);
        let b_im = -im.read_checked(m - k);
        let theta = -F::new(2.0 * PI) * F::cast_from(k) / F::cast_from(n_fft);
        let c = theta.cos();
        let s = theta.sin();
        (
            F::new(0.5)
                * (a_re * (F::new(1.0) + s) + a_im * c + b_re * (F::new(1.0) - s) - b_im * c),
            F::new(0.5)
                * (a_im * (F::new(1.0) + s) - a_re * c + b_re * c + b_im * (F::new(1.0) - s)),
        )
    };
    {
        let mut out_re = spectrum.view_mut(InterleavedBatchSignalLayout::new(
            &*spectrum, window, dim, 0usize,
        ));
        out_re.write_checked(k, x_re * scale);
    }
    let mut out_im = spectrum.view_mut(InterleavedBatchSignalLayout::new(
        &*spectrum, window, dim, 1usize,
    ));
    out_im.write_checked(k, x_im * scale);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn irfft_interleaved_large_launch(
    client: &Client,
    spectrum: ComplexTensorBinding<'_>,
    signal: &TensorHandle,
    dim: usize,
    spec_bins: u32,
    normalization: FftNormalization,
    n_fft: usize,
    packed: usize,
) -> Result<(), FftError> {
    let m = n_fft / 2;
    let packed_shape = signal
        .shape()
        .iter()
        .enumerate()
        .map(|(axis, &extent)| if axis == dim { m } else { extent })
        .collect::<Vec<_>>();
    let bytes = packed
        .checked_mul(f32::elem_type_native().size())
        .ok_or(FftError::SizeOverflow)?;
    let packed_re = TensorHandle::new_contiguous(
        packed_shape.clone(),
        client.empty(bytes),
        f32::elem_type_native(),
    );
    let packed_im =
        TensorHandle::new_contiguous(packed_shape, client.empty(bytes), f32::elem_type_native());
    let cube_dim = CubeDim::new_1d(256);
    let cube_count = cubecl::calculate_cube_count_elemwise(client, packed, cube_dim);
    irfft_pre_interleaved_kernel::launch(
        client,
        cube_count,
        cube_dim,
        spectrum.tensor().into_tensor_arg(),
        packed_re.clone().binding().into_tensor_arg(),
        packed_im.clone().binding().into_tensor_arg(),
        packed as u32,
        spec_bins,
        n_fft,
        m,
        dim,
        f32::elem_type_native(),
    );
    // The upstream CFFT supports in-place input/output, including its four-step path.
    cfft_launch_any_size(
        client,
        CfftBindings {
            input_re: packed_re.clone().binding(),
            input_im: packed_im.clone().binding(),
            output_re: packed_re.clone().binding(),
            output_im: packed_im.clone().binding(),
        },
        dim,
        f32::elem_type_native(),
        FftMode::Inverse,
    )?;
    let cube_count = cubecl::calculate_cube_count_elemwise(client, packed, cube_dim);
    irfft_unpack_interleaved_kernel::launch(
        client,
        cube_count,
        cube_dim,
        packed_re.binding().into_tensor_arg(),
        packed_im.binding().into_tensor_arg(),
        signal.clone().binding().into_tensor_arg(),
        packed as u32,
        m,
        dim,
        normalization,
        f32::elem_type_native(),
    );
    Ok(())
}

/// Convert the one-sided interleaved input into a packed complex spectrum.
#[cube(launch)]
fn irfft_pre_interleaved_kernel<F: Float>(
    spectrum: &Tensor<F>,
    packed_re: &mut Tensor<F>,
    packed_im: &mut Tensor<F>,
    total: u32,
    spec_bins: u32,
    #[comptime] n_fft: usize,
    #[comptime] m: usize,
    #[comptime] dim: usize,
    #[define(F)] _dtype: ElemType,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize {
        terminate!();
    }
    let k = pos % m;
    let window = pos / m;
    let spectrum_re = spectrum.view(InterleavedBatchSignalLayout::new(
        spectrum, window, dim, 0usize,
    ));
    let spectrum_im = spectrum.view(InterleavedBatchSignalLayout::new(
        spectrum, window, dim, 1usize,
    ));
    let mut output_re = packed_re.view_mut(BatchSignalLayout::new(&*packed_re, window, dim));
    let mut output_im = packed_im.view_mut(BatchSignalLayout::new(&*packed_im, window, dim));
    if k == 0 {
        let nyquist = m < spec_bins as usize;
        let x0 = spectrum_re.read_checked(0);
        let xm = select(nyquist, m, 0);
        let xm_re = select(nyquist, spectrum_re.read_checked(xm), F::new(0.0));
        output_re.write_checked(k, F::new(0.5) * (x0 + xm_re));
        output_im.write_checked(k, F::new(0.5) * (x0 - xm_re));
    } else {
        let active = k < spec_bins as usize;
        let src = select(active, k, 0);
        let x_re = select(active, spectrum_re.read_checked(src), F::new(0.0));
        let x_im = select(active, spectrum_im.read_checked(src), F::new(0.0));
        let mirrored = m - k;
        let mirror_active = mirrored < spec_bins as usize;
        let mirror = select(mirror_active, mirrored, 0);
        let xm_re = select(mirror_active, spectrum_re.read_checked(mirror), F::new(0.0));
        let xm_im = -select(mirror_active, spectrum_im.read_checked(mirror), F::new(0.0));
        let theta = F::new(2.0 * PI) * F::cast_from(k) / F::cast_from(n_fft);
        let c = theta.cos();
        let s = theta.sin();
        let y_re = F::new(0.5)
            * (x_re * (F::new(1.0) - s) - x_im * c + xm_re * (F::new(1.0) + s) + xm_im * c);
        let y_im = F::new(0.5)
            * (x_im * (F::new(1.0) - s) + x_re * c - xm_re * c + xm_im * (F::new(1.0) + s));
        output_re.write_checked(k, y_re);
        output_im.write_checked(k, y_im);
    }
}

/// The packed M-point inverse is unnormalized. The half factors above mean
/// multiplying by N/M (not M/N) yields the unnormalized N-point real inverse.
#[cube(launch)]
fn irfft_unpack_interleaved_kernel<F: Float>(
    packed_re: &Tensor<F>,
    packed_im: &Tensor<F>,
    signal: &mut Tensor<F>,
    total: u32,
    #[comptime] m: usize,
    #[comptime] dim: usize,
    #[comptime] normalization: FftNormalization,
    #[define(F)] _dtype: ElemType,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize {
        terminate!();
    }
    let k = pos % m;
    let window = pos / m;
    let re = packed_re.view(BatchSignalLayout::new(packed_re, window, dim));
    let im = packed_im.view(BatchSignalLayout::new(packed_im, window, dim));
    let mut output = signal.view_mut(BatchSignalLayout::new(&*signal, window, dim));
    let n_fft = comptime![2 * m];
    let adjustment = match normalization {
        FftNormalization::None => F::cast_from(n_fft),
        FftNormalization::ByN => F::new(1.0),
        FftNormalization::Ortho => F::cast_from(n_fft).sqrt(),
    };
    let scale = adjustment / F::cast_from(m);
    output.write_checked(2 * k, re.read_checked(k) * scale);
    output.write_checked(2 * k + 1, im.read_checked(k) * scale);
}
