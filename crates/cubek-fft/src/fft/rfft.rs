use cubecl::{prelude::*, std::tensor::TensorHandle};

use cubecl::std::tensor::{
    AsView as _, AsViewExpand, AsViewMut as _, AsViewMutExpand, layout::plain::PlainLayout,
};

use crate::{
    fft::{FftMode, fft_inner_compute},
    layout::BatchSignalLayout,
};

/// Real-valued Fast Fourier Transform kernel.
///
/// Creates spectrum (real and imaginary) tensors
/// then launches the RFFT kernel to fill them with the right values
pub fn rfft<R: Runtime>(
    signal: TensorHandle<R>,
    dim: usize,
    dtype: StorageType,
) -> (TensorHandle<R>, TensorHandle<R>) {
    assert!(
        dim < signal.shape().len(),
        "dim must be between 0 and {}",
        signal.shape().len()
    );
    assert!(
        signal.shape()[dim].is_power_of_two(),
        "RFFT requires power-of-2 length"
    );
    let client = <R as Runtime>::client(&Default::default());

    let mut spectrum_shape = signal.shape().clone();
    spectrum_shape[dim] = signal.shape()[dim] / 2 + 1;

    let spectrum_re = TensorHandle::new_contiguous(
        spectrum_shape.clone(),
        client.empty(spectrum_shape.iter().product::<usize>() * dtype.size()),
        dtype,
    );

    let spectrum_im = TensorHandle::new_contiguous(
        spectrum_shape.clone(),
        client.empty(spectrum_shape.iter().product::<usize>() * dtype.size()),
        dtype,
    );

    rfft_launch::<R>(
        &client,
        signal.binding(),
        spectrum_re.clone().binding(),
        spectrum_im.clone().binding(),
        dim,
        dtype,
    )
    .unwrap();

    (spectrum_re, spectrum_im)
}

/// Launches the RFFT with the specified Cube Count, Cube Dim and vectorization (line size)
pub fn rfft_launch<R: Runtime>(
    client: &ComputeClient<R>,
    signal: TensorBinding<R>,
    spectrum_re: TensorBinding<R>,
    spectrum_im: TensorBinding<R>,
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
    let shape = signal.shape.as_slice()[dim];

    rfft_kernel::launch::<R>(
        client,
        cube_count,
        cube_dim,
        signal.into_tensor_arg(),
        spectrum_re.into_tensor_arg(),
        spectrum_im.into_tensor_arg(),
        shape,
        dim,
        dtype,
        vectorization,
    );
    Ok(())
}

#[cube(launch)]
/// Kernel that loops over each window and applies the RFFT on each
pub(crate) fn rfft_kernel<F: Float, N: Size>(
    signal: &Tensor<Vector<F, N>>,
    spectrums_re: &mut Tensor<Vector<F, N>>,
    spectrums_im: &mut Tensor<Vector<F, N>>,
    #[comptime] num_samples: usize,
    #[comptime] dim: usize,
    #[define(F)] _dtype: StorageType,
    #[define(N)] _vector_size: usize,
) {
    let window_index = CUBE_POS;
    rfft_kernel_one_window(
        signal,
        spectrums_re,
        spectrums_im,
        window_index,
        num_samples,
        dim,
    );
}

#[cube]
/// Applies the RFFT on one window.
/// Starts by putting all the window in shared memory, where the compute will occur
/// Then stores back the content of the shared memory
pub(crate) fn rfft_kernel_one_window<F: Float, N: Size>(
    signal: &Tensor<Vector<F, N>>,
    spectrums_re: &mut Tensor<Vector<F, N>>,
    spectrums_im: &mut Tensor<Vector<F, N>>,
    window_index: usize,
    #[comptime] num_samples: usize,
    #[comptime] dim: usize,
) {
    let signal_layout = BatchSignalLayout::new(signal, window_index, dim);
    let spectrums_re_layout = BatchSignalLayout::new(spectrums_re, window_index, dim);
    let spectrums_im_layout = BatchSignalLayout::new(spectrums_im, window_index, dim);
    let signal_view = signal.view(signal_layout);
    let spectrums_re_view = spectrums_re.view_mut(spectrums_re_layout);
    let spectrums_im_view = spectrums_im.view_mut(spectrums_im_layout);

    // The shared memories are not vectorized because the inner FFT compute will need to work independently on each element
    let mut spectrum_re =
        SharedMemory::<F>::new(num_samples).view_mut(PlainLayout::new(num_samples));
    let mut spectrum_im =
        SharedMemory::<F>::new(num_samples).view_mut(PlainLayout::new(num_samples));

    // Load all samples of the window to shared memory
    for i in 0..num_samples {
        // Warning: this assumes that signal_view has lines of 1 element
        // For larger lines, iterate over the line's content
        // You can get the line_size of a tensor/view with .line_size()
        spectrum_re[i] = signal_view.read(i)[0];
        spectrum_im[i] = F::cast_from(0);
    }

    fft_inner_compute(&mut spectrum_re, &mut spectrum_im, FftMode::Forward);

    for i in 0..spectrums_re_view.shape() {
        // Warning: this assumes that spectrum views have lines of 1 element
        // If lines had more elements, the ith element would be duplicated as it is
        spectrums_re_view.write(i, Vector::cast_from(spectrum_re[i]));
        spectrums_im_view.write(i, Vector::cast_from(spectrum_im[i]));
    }
}
