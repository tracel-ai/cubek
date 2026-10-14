//! Real F32 FFT into an interleaved C32 half-spectrum.
//!
//! Modified from tensor4all/cubek `02efeba9` (MIT OR Apache-2.0): adapted
//! the launch ABI, checked bounds, and current upstream FFT kernels.

use cubecl::prelude::*;
use cubecl::std::tensor::{
    AsView as _, AsViewExpand, AsViewMut as _, AsViewMutExpand, TensorHandle,
};

use crate::{
    ComplexTensorBinding, FftError, FftNormalization,
    fft::{
        FftMode,
        cfft_interleaved::ensure_non_overlapping_output_layout,
        fft_parallel::{bit_reverse, fft_butterfly_parallel},
        limits::{max_shared_fft_n, max_units_per_cube},
        real_interleaved_large::rfft_interleaved_large_launch,
    },
    interleaved_layout::InterleavedBatchSignalLayout,
    layout::BatchSignalLayout,
};

/// Launch a real F32 FFT, treating samples at `signal_len..n_fft` as zero.
/// The output axis contains `n_fft/2 + 1` complex bins; normalization is
/// applied in both the shared and packed-real paths.
pub fn rfft_interleaved_launch_padded(
    client: &Client,
    signal: &TensorHandle,
    spectrum: ComplexTensorBinding<'_>,
    dim: usize,
    signal_len: usize,
    normalization: FftNormalization,
) -> Result<(), FftError> {
    let shape = signal.shape();
    if signal.dtype != f32::elem_type_native() {
        return Err(FftError::UnsupportedDtype {
            actual: signal.dtype,
        });
    }
    if dim >= shape.len() {
        return Err(FftError::AxisOutOfBounds {
            dim,
            rank: shape.len(),
        });
    }
    let n_freq = *spectrum.shape().get(dim).ok_or(FftError::AxisOutOfBounds {
        dim,
        rank: spectrum.shape().len(),
    })?;
    let n_fft = n_freq
        .checked_sub(1)
        .and_then(|n| n.checked_mul(2))
        .ok_or(FftError::InvalidFftLength { n_fft: 0 })?;
    normalization.scale_f32(n_fft)?;
    let mut expected = shape.to_vec();
    expected[dim] = n_freq;
    if spectrum.shape() != expected {
        return Err(FftError::ShapeMismatch {
            name: "spectrum",
            actual: spectrum.shape().to_vec(),
            expected,
        });
    }
    ensure_non_overlapping_output_layout(spectrum.shape(), spectrum.strides())?;
    if signal_len > shape[dim] || signal_len > n_fft {
        return Err(FftError::InvalidLength {
            name: "signal_len",
            value: signal_len,
            min: 0,
            max: shape[dim].min(n_fft),
        });
    }
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
    let post = count.checked_mul(n_freq).ok_or(FftError::SizeOverflow)?;
    let count_u32 = u32::try_from(count).map_err(|_| FftError::SizeOverflow)?;
    let signal_len_u32 = u32::try_from(signal_len).map_err(|_| FftError::SizeOverflow)?;
    u32::try_from(packed).map_err(|_| FftError::SizeOverflow)?;
    u32::try_from(post).map_err(|_| FftError::SizeOverflow)?;
    spectrum.ensure_unique_output()?;
    if count == 0 {
        return Ok(());
    }
    if n_fft > max_shared {
        return rfft_interleaved_large_launch(
            client,
            signal,
            spectrum,
            dim,
            signal_len_u32,
            normalization,
            n_fft,
            packed,
            post,
        );
    }
    let threads = (n_fft / 2).clamp(1, max_units_per_cube(client));
    let cube_dim = CubeDim::new_1d(threads as u32);
    let cube_count = cubecl::calculate_cube_count_elemwise(client, count, CubeDim::new_single());
    rfft_interleaved_kernel::launch(
        client,
        cube_count,
        cube_dim,
        signal.clone().binding().into_tensor_arg(),
        spectrum.tensor().into_tensor_arg(),
        count_u32,
        signal_len_u32,
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
fn rfft_interleaved_kernel<F: Float>(
    signal: &Tensor<F>,
    spectrum: &mut Tensor<F>,
    num_windows: u32,
    signal_len: u32,
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
    let signal_view = signal.view(BatchSignalLayout::new(signal, window, dim));
    let mut shared_re = Shared::new_slice(n_fft);
    let mut shared_im = Shared::new_slice(n_fft);
    let mut i = UNIT_POS as usize;
    while i < n_fft {
        let j = bit_reverse(i, log2_n);
        let active = i < signal_len as usize;
        shared_re[j] = select(
            active,
            signal_view.read_checked(select(active, i, 0)),
            F::new(0.0),
        );
        shared_im[j] = F::new(0.0);
        i += threads_per_cube;
    }
    sync_cube();
    fft_butterfly_parallel::<F>(
        &mut shared_re,
        &mut shared_im,
        n_fft,
        log2_n,
        threads_per_cube,
        FftMode::Forward,
    );
    let scale = match normalization {
        FftNormalization::None => F::new(1.0),
        FftNormalization::ByN => F::new(1.0) / F::cast_from(n_fft),
        FftNormalization::Ortho => F::new(1.0) / F::cast_from(n_fft).sqrt(),
    };
    let n_freq = comptime![n_fft / 2 + 1];
    {
        let mut real = spectrum.view_mut(InterleavedBatchSignalLayout::new(
            &*spectrum, window, dim, 0usize,
        ));
        let mut k = UNIT_POS as usize;
        while k < n_freq {
            real.write_checked(k, shared_re[k] * scale);
            k += threads_per_cube;
        }
    }
    let mut imag = spectrum.view_mut(InterleavedBatchSignalLayout::new(
        &*spectrum, window, dim, 1usize,
    ));
    let mut k = UNIT_POS as usize;
    while k < n_freq {
        imag.write_checked(k, shared_im[k] * scale);
        k += threads_per_cube;
    }
    sync_cube();
}
