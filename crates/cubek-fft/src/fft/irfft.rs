//! Inverse real-valued FFT with an intra-cube-parallel radix-2 kernel.

use cubecl::prelude::*;
use cubecl::std::tensor::{
    AsView as _, AsViewExpand, AsViewMut as _, AsViewMutExpand, TensorHandle,
};

use crate::{
    fft::{
        FftMode,
        fft_parallel::{bit_reverse, fft_butterfly_parallel},
        limits::{max_shared_fft_n, max_units_per_cube},
        rfft_large::irfft_large_launch,
    },
    layout::BatchSignalLayout,
};

/// Inverse Real-valued Fast Fourier Transform.
pub fn irfft<R: Runtime>(
    spectrum_re: TensorHandle<R>,
    spectrum_im: TensorHandle<R>,
    dim: usize,
    dtype: StorageType,
) -> TensorHandle<R> {
    assert!(
        spectrum_re.shape() == spectrum_im.shape(),
        "Spectrum's real and imaginary parts should be the same shape, got {:?} and {:?}",
        spectrum_re.shape(),
        spectrum_im.shape()
    );

    let client = <R as Runtime>::client(&Default::default());

    let mut signal_shape = spectrum_re.shape().clone();
    signal_shape[dim] = (spectrum_re.shape()[dim] - 1) * 2;
    let num_elems = signal_shape.iter().product::<usize>();
    let signal = TensorHandle::new_contiguous(
        signal_shape.clone(),
        client.empty(num_elems * dtype.size()),
        dtype,
    );

    irfft_launch::<R>(
        &client,
        spectrum_re.binding(),
        spectrum_im.binding(),
        signal.clone().binding(),
        dim,
        dtype,
    )
    .unwrap();

    signal
}

/// Launches the IRFFT kernel.
pub fn irfft_launch<R: Runtime>(
    client: &ComputeClient<R>,
    spectrum_re: TensorBinding<R>,
    spectrum_im: TensorBinding<R>,
    signal: TensorBinding<R>,
    dim: usize,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let spec_bins = spectrum_re.shape[dim];
    irfft_launch_padded::<R>(
        client,
        spectrum_re,
        spectrum_im,
        signal,
        dim,
        spec_bins,
        dtype,
    )
}

/// Launches the IRFFT kernel while treating bins at `spec_bins..n_freq` as zero.
pub fn irfft_launch_padded<R: Runtime>(
    client: &ComputeClient<R>,
    spectrum_re: TensorBinding<R>,
    spectrum_im: TensorBinding<R>,
    signal: TensorBinding<R>,
    dim: usize,
    spec_bins: usize,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    assert!(
        spectrum_re.shape == spectrum_im.shape,
        "spectrum real and imaginary shapes must match"
    );
    assert!(dim < signal.shape.len(), "dim must be in bounds");

    let n_fft = signal.shape[dim];
    assert!(n_fft.is_power_of_two(), "IRFFT requires power-of-2 length");
    assert!(n_fft >= 2, "IRFFT requires n_fft >= 2");
    let n_freq = n_fft / 2 + 1;
    assert!(
        spec_bins <= spectrum_re.shape[dim],
        "spec_bins ({spec_bins}) must be <= spectrum dimension ({})",
        spectrum_re.shape[dim]
    );
    assert!(spec_bins >= 1, "spec_bins must be >= 1");
    assert!(
        spec_bins <= n_freq,
        "spec_bins ({spec_bins}) must be <= n_fft / 2 + 1 ({n_freq})"
    );

    let count: usize = signal
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != dim)
        .map(|(_, e)| *e)
        .product();
    if count == 0 {
        return Ok(());
    }

    if n_fft > max_shared_fft_n(client) {
        return irfft_large_launch::<R>(
            client,
            spectrum_re,
            spectrum_im,
            signal,
            dim,
            spec_bins,
            dtype,
        );
    }

    let log2_n = n_fft.trailing_zeros() as usize;
    let threads_per_cube = (n_fft / 2).clamp(1, max_units_per_cube(client));

    let cube_dim = CubeDim::new_1d(threads_per_cube as u32);
    let cube_count = cubecl::calculate_cube_count_elemwise(client, count, CubeDim::new_single());

    irfft_kernel::launch::<f32, R>(
        client,
        cube_count,
        cube_dim,
        spectrum_re.into_tensor_arg(),
        spectrum_im.into_tensor_arg(),
        signal.into_tensor_arg(),
        count as u32,
        spec_bins as u32,
        n_fft,
        log2_n,
        threads_per_cube,
        dim,
    );
    Ok(())
}

#[cube(launch)]
fn irfft_kernel<F: Float>(
    spectrum_re: &Tensor<F>,
    spectrum_im: &Tensor<F>,
    signal: &mut Tensor<F>,
    num_windows: u32,
    spec_bins: u32,
    #[comptime] n_fft: usize,
    #[comptime] log2_n: usize,
    #[comptime] threads_per_cube: usize,
    #[comptime] dim: usize,
) {
    let window_index = CUBE_POS;
    if (window_index as u32) >= num_windows {
        terminate!();
    }

    let spectrum_re_view = spectrum_re.view(BatchSignalLayout::new(spectrum_re, window_index, dim));
    let spectrum_im_view = spectrum_im.view(BatchSignalLayout::new(spectrum_im, window_index, dim));
    let signal_view = signal.view_mut(BatchSignalLayout::new(&*signal, window_index, dim));

    let mut shared_re = SharedMemory::<F>::new(n_fft);
    let mut shared_im = SharedMemory::<F>::new(n_fft);

    let n_freq = comptime![n_fft / 2 + 1];

    let mut k = UNIT_POS as usize;
    while k < n_fft {
        let dst = bit_reverse(k, log2_n);
        let src_bin = select(k < n_freq, k, n_fft - k);
        let active = src_bin < spec_bins as usize;
        let src_bin = select(active, src_bin, 0);
        let im_sign = select(k < n_freq, F::new(1.0), F::new(-1.0));
        shared_re[dst] = select(active, spectrum_re_view.read_checked(src_bin), F::new(0.0));
        shared_im[dst] = select(
            active,
            spectrum_im_view.read_checked(src_bin) * im_sign,
            F::new(0.0),
        );
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

    let scale = F::new(1.0) / F::cast_from(n_fft);
    let mut i = UNIT_POS as usize;
    while i < n_fft {
        signal_view.write_checked(i, shared_re[i] * scale);
        i += threads_per_cube;
    }
    sync_cube();
}
