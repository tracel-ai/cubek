//! Inverse real F32 FFT from an interleaved C32 half-spectrum.
//!
//! Modified from tensor4all/cubek `02efeba9` (MIT OR Apache-2.0): adapted
//! the launch ABI, output ownership checks, and current upstream FFT kernels.

use cubecl::prelude::*;
use cubecl::std::tensor::{
    AsView as _, AsViewExpand, AsViewMut as _, AsViewMutExpand, TensorHandle,
};

use crate::{
    ComplexTensorBinding, FftError, FftNormalization,
    complex::ensure_unique_output,
    fft::{
        FftMode,
        cfft_interleaved::ensure_non_overlapping_output_layout,
        fft_parallel::{bit_reverse, fft_butterfly_parallel},
        limits::{max_shared_fft_n, max_units_per_cube},
        real_interleaved_large::irfft_interleaved_large_launch,
    },
    interleaved_layout::InterleavedBatchSignalLayout,
    layout::BatchSignalLayout,
};

/// Launch an inverse real FFT, treating spectrum bins at `spec_bins..n_fft/2+1`
/// as zero. The output axis contains `n_fft` real samples. `None` leaves the
/// inverse unnormalized, while `ByN` and `Ortho` scale by `1/N` and `1/sqrt(N)`.
pub fn irfft_interleaved_launch_padded(
    client: &Client,
    spectrum: ComplexTensorBinding<'_>,
    signal: &TensorHandle,
    dim: usize,
    spec_bins: usize,
    normalization: FftNormalization,
) -> Result<(), FftError> {
    let shape = signal.shape();
    if signal.dtype != f32::elem_type_native() {
        return Err(FftError::UnsupportedDtype {
            actual: signal.dtype,
        });
    }
    let &n_fft = shape.get(dim).ok_or(FftError::AxisOutOfBounds {
        dim,
        rank: shape.len(),
    })?;
    normalization.scale_f32(n_fft)?;
    if spectrum.shape().len() != shape.len()
        || spectrum
            .shape()
            .iter()
            .enumerate()
            .any(|(axis, &extent)| axis != dim && extent != shape[axis])
    {
        let mut expected = shape.to_vec();
        if dim < spectrum.shape().len() {
            expected[dim] = spectrum.shape()[dim];
        }
        return Err(FftError::ShapeMismatch {
            name: "spectrum",
            actual: spectrum.shape().to_vec(),
            expected,
        });
    }
    let n_freq = n_fft / 2 + 1;
    let available = spectrum.shape()[dim].min(n_freq);
    if spec_bins == 0 || spec_bins > available {
        return Err(FftError::InvalidLength {
            name: "spec_bins",
            value: spec_bins,
            min: 1,
            max: available,
        });
    }
    ensure_non_overlapping_output_layout(shape, signal.strides())?;
    let max_shared = max_shared_fft_n(client, f32::elem_type_native());
    let max_n_fft = max_shared.saturating_mul(max_shared).saturating_mul(2);
    if n_fft > max_n_fft {
        return Err(FftError::FftLengthExceedsDeviceLimit { n_fft, max_n_fft });
    }
    let count = shape
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != dim)
        .try_fold(1usize, |count, (_, extent)| {
            count.checked_mul(*extent).ok_or(FftError::SizeOverflow)
        })?;
    let packed = count.checked_mul(n_fft / 2).ok_or(FftError::SizeOverflow)?;
    let count_u32 = u32::try_from(count).map_err(|_| FftError::SizeOverflow)?;
    let spec_bins_u32 = u32::try_from(spec_bins).map_err(|_| FftError::SizeOverflow)?;
    u32::try_from(packed).map_err(|_| FftError::SizeOverflow)?;
    ensure_unique_output(signal)?;
    if count == 0 {
        return Ok(());
    }
    if n_fft > max_shared {
        return irfft_interleaved_large_launch(
            client,
            spectrum,
            signal,
            dim,
            spec_bins_u32,
            normalization,
            n_fft,
            packed,
        );
    }
    let threads = (n_fft / 2).clamp(1, max_units_per_cube(client));
    let cube_dim = CubeDim::new_1d(threads as u32);
    let cube_count = cubecl::calculate_cube_count_elemwise(client, count, CubeDim::new_single());
    irfft_interleaved_kernel::launch(
        client,
        cube_count,
        cube_dim,
        spectrum.tensor().into_tensor_arg(),
        signal.clone().binding().into_tensor_arg(),
        count_u32,
        spec_bins_u32,
        n_fft,
        n_fft.trailing_zeros() as usize,
        threads,
        dim,
        normalization,
        f32::elem_type_native(),
    );
    Ok(())
}

#[cube(launch)]
fn irfft_interleaved_kernel<F: Float>(
    spectrum: &Tensor<F>,
    signal: &mut Tensor<F>,
    num_windows: u32,
    spec_bins: u32,
    #[comptime] n_fft: usize,
    #[comptime] log2_n: usize,
    #[comptime] threads_per_cube: usize,
    #[comptime] dim: usize,
    #[comptime] normalization: FftNormalization,
    #[define(F)] _dtype: ElemType,
) {
    let window = CUBE_POS;
    if (window as u32) >= num_windows {
        terminate!();
    }
    let spectrum_re = spectrum.view(InterleavedBatchSignalLayout::new(
        spectrum, window, dim, 0usize,
    ));
    let spectrum_im = spectrum.view(InterleavedBatchSignalLayout::new(
        spectrum, window, dim, 1usize,
    ));
    let mut signal_view = signal.view_mut(BatchSignalLayout::new(&*signal, window, dim));
    let mut shared_re = Shared::new_slice(n_fft);
    let mut shared_im = Shared::new_slice(n_fft);
    let n_freq = comptime![n_fft / 2 + 1];
    let mut k = UNIT_POS as usize;
    while k < n_fft {
        let dst = bit_reverse(k, log2_n);
        let src = select(k < n_freq, k, n_fft - k);
        let active = src < spec_bins as usize;
        let src = select(active, src, 0);
        let sign = select(k < n_freq, F::new(1.0), F::new(-1.0));
        shared_re[dst] = select(active, spectrum_re.read_checked(src), F::new(0.0));
        shared_im[dst] = select(active, spectrum_im.read_checked(src) * sign, F::new(0.0));
        k += threads_per_cube;
    }
    sync_cube();
    fft_butterfly_parallel::<F>(
        &mut shared_re,
        &mut shared_im,
        n_fft,
        log2_n,
        threads_per_cube,
        FftMode::Inverse,
    );
    let scale = match normalization {
        FftNormalization::None => F::new(1.0),
        FftNormalization::ByN => F::new(1.0) / F::cast_from(n_fft),
        FftNormalization::Ortho => F::new(1.0) / F::cast_from(n_fft).sqrt(),
    };
    let mut i = UNIT_POS as usize;
    while i < n_fft {
        signal_view.write_checked(i, shared_re[i] * scale);
        i += threads_per_cube;
    }
    sync_cube();
}
