use cubecl::prelude::*;
use cubecl::std::tensor::layout::plain::PlainLayout;
use cubecl::std::tensor::{
    AsView as _, AsViewExpand, AsViewMut as _, AsViewMutExpand, TensorHandle,
};

use crate::fft::{FftMode, fft_inner_compute};
use crate::layout::BatchSignalLayout;

/// Inverse Real-valued Fast Fourier Transform kernel.
///
/// Creates signal tensor
/// then launches the IRFFT kernel to fill it with the right values
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

/// Launches the IRFFT with the specified Cube Count, Cube Dim and vectorization (line size)
pub fn irfft_launch<R: Runtime>(
    client: &ComputeClient<R>,
    spectrum_re: TensorBinding<R>,
    spectrum_im: TensorBinding<R>,
    signal: TensorBinding<R>,
    dim: usize,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let count: usize = signal
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != dim)
        .map(|(_, e)| *e)
        .product();

    let cube_count = CubeCount::new_1d(count as u32);
    let cube_dim = CubeDim::new_single();
    let vectorization = 1;
    let shape = signal.shape[dim];

    irfft_kernel::launch::<R>(
        client,
        cube_count,
        cube_dim,
        spectrum_re.into_tensor_arg(),
        spectrum_im.into_tensor_arg(),
        signal.into_tensor_arg(),
        shape,
        dim,
        dtype,
        vectorization,
    );
    Ok(())
}

#[cube(launch)]
/// Kernel that loops over each window and applies the IRFFT on each
pub(crate) fn irfft_kernel<F: Float, N: Size>(
    spectrums_re: &Tensor<Vector<F, N>>,
    spectrums_im: &Tensor<Vector<F, N>>,
    signal: &mut Tensor<Vector<F, N>>,
    #[comptime] num_samples: usize,
    #[comptime] dim: usize,
    #[define(F)] _dtype: StorageType,
    #[define(N)] _vector_size: usize,
) {
    let batch_index = CUBE_POS;
    irfft_kernel_one_batch(
        spectrums_re,
        spectrums_im,
        signal,
        batch_index,
        num_samples,
        dim,
    );
}

#[cube]
/// Applies the IRFFT on one window.
/// Starts by putting all the window in shared memory, where the compute will occur
/// Then stores back the content of the shared memory
/// There are a few extra steps for normalization compared to forward RFFT
pub(crate) fn irfft_kernel_one_batch<F: Float, N: Size>(
    spectrums_re: &Tensor<Vector<F, N>>,
    spectrums_im: &Tensor<Vector<F, N>>,
    signal: &mut Tensor<Vector<F, N>>,
    window_index: usize,
    #[comptime] num_samples: usize,
    #[comptime] dim: usize,
) {
    // The following code allow to ignore the batch index and assume only one window
    // - spectrums have shape: [num_freq_bins]
    // - signal has shape: [num_samples]
    let spectrums_re_layout = BatchSignalLayout::new(spectrums_re, window_index, dim);
    let spectrums_im_layout = BatchSignalLayout::new(spectrums_im, window_index, dim);

    let signal_layout = BatchSignalLayout::new(signal, window_index, dim);
    let spectrums_re_view = spectrums_re.view(spectrums_re_layout);
    let spectrums_im_view = spectrums_im.view(spectrums_im_layout);
    let signal_view = signal.view_mut(signal_layout);

    let num_freq_bins = spectrums_re_view.shape();

    // The shared memories are not vectorized because the inner FFT compute will need to work independently on each element
    let mut spectrum_re =
        SharedMemory::<F>::new(num_samples).view_mut(PlainLayout::new(num_samples));
    let mut spectrum_im =
        SharedMemory::<F>::new(num_samples).view_mut(PlainLayout::new(num_samples));

    // Load all the frequency bins to shared memory
    for i in 0..num_freq_bins {
        // Warning: this assumes that spectrum views have lines of 1 element
        // For larger lines, iterate over the line's content
        // You can get the line_size of a tensor/view with .line_size()
        spectrum_re[i] = spectrums_re_view.read(i)[0];
        spectrum_im[i] = spectrums_im_view.read(i)[0];
    }

    // Fill the Hermitian-conjugate mirrored bins
    for k in 1..num_freq_bins - 1 {
        spectrum_re[num_samples - k] = spectrum_re[k];
        spectrum_im[num_samples - k] = -spectrum_im[k]; // conjugate
    }

    // Run inverse FFT
    fft_inner_compute(&mut spectrum_re, &mut spectrum_im, FftMode::Inverse);

    // Normalize by number of samples
    for i in 0..num_samples {
        spectrum_re[i] = spectrum_re[i] / F::cast_from(num_samples);
        spectrum_im[i] = spectrum_im[i] / F::cast_from(num_samples);
    }

    // Write full real output
    for i in 0..num_samples {
        // Warning: this assumes that output_view have lines of 1 element
        // If lines had more elements, the ith element would be duplicated as it is
        signal_view.write(i, Vector::cast_from(spectrum_re[i]));
    }
}
