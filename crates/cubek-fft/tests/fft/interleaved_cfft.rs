//! Direct known-value tests for the interleaved C32 CFFT.
//!
//! Modified from tensor4all/cubek `02efeba9` (MIT OR Apache-2.0):
//! test the current nongeneric CubeCL launch API and direct DFT values.
//!
//! These compare forward and inverse launches against an independent naive
//! DFT, not against a round trip, so a matching sign/permutation error cannot
//! cancel out. Every case uses nonzero imaginary input and an explicit
//! [`FftNormalization`].

use cubecl::{CubeElement, client::Client, prelude::Scalar, server::Handle};
use cubek_fft::{
    ComplexTensorHandle, FftError, FftMode, FftNormalization, cfft_interleaved,
    cfft_interleaved_launch,
};
use num_complex::Complex32;

fn dtype() -> cubecl::ir::ElemType {
    f32::elem_type_native()
}

/// Row-major logical values with nonzero real and imaginary parts.
fn values_for(shape: &[usize]) -> Vec<Complex32> {
    (0..shape.iter().product::<usize>())
        .map(|i| {
            let f = i as f32;
            Complex32::new(f * 0.5 - 1.0, 0.25 - f * 0.75)
        })
        .collect()
}

fn complex_handle(client: &Client, shape: Vec<usize>, values: &[Complex32]) -> ComplexTensorHandle {
    let scalars: Vec<f32> = values.iter().flat_map(|c| [c.re, c.im]).collect();
    ComplexTensorHandle::new_contiguous(
        shape,
        client.create_from_slice(f32::as_bytes(&scalars)),
        dtype(),
    )
    .unwrap()
}

fn read_scalars(client: &Client, handle: Handle) -> Vec<f32> {
    f32::from_bytes(&client.read_one(handle).unwrap()).to_vec()
}

/// Gather logical complex elements through the handle's scalar strides.
fn read_logical(client: &Client, tensor: ComplexTensorHandle) -> Vec<Complex32> {
    let shape = tensor.shape().to_vec();
    let scalar_strides = tensor.scalar_strides().to_vec();
    let scalars = read_scalars(client, tensor.into_raw_parts().handle);
    let total = shape.iter().product::<usize>();
    (0..total)
        .map(|logical| {
            let mut remaining = logical;
            let mut scalar_index = 0;
            for axis in (0..shape.len()).rev() {
                let coord = remaining % shape[axis];
                remaining /= shape[axis];
                scalar_index += coord * scalar_strides[axis];
            }
            Complex32::new(scalars[scalar_index], scalars[scalar_index + 1])
        })
        .collect()
}

fn unravel(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coords[axis] = index % shape[axis];
        index /= shape[axis];
    }
    coords
}

fn ravel(coords: &[usize], shape: &[usize]) -> usize {
    let mut index = 0;
    for axis in 0..shape.len() {
        index = index * shape[axis] + coords[axis];
    }
    index
}

fn scale_of(n: usize, normalization: FftNormalization) -> f32 {
    match normalization {
        FftNormalization::None => 1.0,
        FftNormalization::ByN => 1.0 / n as f32,
        FftNormalization::Ortho => 1.0 / (n as f32).sqrt(),
    }
}

fn sign_of(inverse: bool) -> f32 {
    if inverse { 1.0 } else { -1.0 }
}

/// Independent naive DFT: `sum_j x[window, j] * exp(sign * 2*pi*i*k*j/N)`,
/// then the explicit normalization factor.
fn direct_dft(
    shape: &[usize],
    dim: usize,
    input: &[Complex32],
    inverse: bool,
    normalization: FftNormalization,
) -> Vec<Complex32> {
    let n = shape[dim];
    let sign = sign_of(inverse);
    let scale = scale_of(n, normalization);
    let total = shape.iter().product::<usize>();
    (0..total)
        .map(|out_index| {
            let coords = unravel(out_index, shape);
            let k = coords[dim];
            let mut acc = Complex32::new(0.0, 0.0);
            for j in 0..n {
                let mut src = coords.clone();
                src[dim] = j;
                let angle = sign * 2.0 * core::f32::consts::PI * k as f32 * j as f32 / n as f32;
                acc += input[ravel(&src, shape)] * Complex32::new(angle.cos(), angle.sin());
            }
            acc * scale
        })
        .collect()
}

fn assert_close(actual: &[Complex32], expected: &[Complex32], epsilon: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).norm();
        // Relative to the element magnitude so large outputs (larger windows)
        // are not held to an absolute tolerance below f32 resolution.
        let tolerance = epsilon * (1.0 + expected.norm());
        assert!(
            error <= tolerance,
            "element {index}: got {actual}, expected {expected} (error {error} > {tolerance})"
        );
    }
}

fn forward_and_inverse_match_direct_dft(shape: Vec<usize>, dim: usize) {
    let client = cubecl::test_device().client();
    let values = values_for(&shape);
    for normalization in [
        FftNormalization::None,
        FftNormalization::ByN,
        FftNormalization::Ortho,
    ] {
        let input = complex_handle(&client, shape.clone(), &values);
        let spectrum =
            cfft_interleaved(&client, input, dim, FftMode::Forward, normalization).unwrap();
        let actual = read_logical(&client, spectrum);
        assert_close(
            &actual,
            &direct_dft(&shape, dim, &values, false, normalization),
            1e-4,
        );

        let input = complex_handle(&client, shape.clone(), &values);
        let recovered =
            cfft_interleaved(&client, input, dim, FftMode::Inverse, normalization).unwrap();
        let actual = read_logical(&client, recovered);
        assert_close(
            &actual,
            &direct_dft(&shape, dim, &values, true, normalization),
            1e-4,
        );
    }
}

#[test]
fn forward_and_inverse_match_direct_dft_n8() {
    forward_and_inverse_match_direct_dft(vec![2, 8], 1);
}

#[test]
fn forward_and_inverse_match_direct_dft_minimum_n_fft() {
    forward_and_inverse_match_direct_dft(vec![1, 2], 1);
}

#[test]
fn forward_and_inverse_match_direct_dft_axis_zero() {
    forward_and_inverse_match_direct_dft(vec![8, 3], 0);
}

#[test]
fn forward_and_inverse_match_direct_dft_middle_axis() {
    forward_and_inverse_match_direct_dft(vec![2, 8, 3], 1);
}

/// Interleaved transform of a single-window sparse signal has a closed form,
/// so large sizes can be checked directly without an O(N^2) reference.
fn sparse_closed_form(n: usize, inverse: bool, normalization: FftNormalization) -> Vec<Complex32> {
    let x0 = Complex32::new(0.75, -1.25);
    let x1 = Complex32::new(-0.5, 2.0);
    let sign = sign_of(inverse);
    let scale = scale_of(n, normalization);
    (0..n)
        .map(|k| {
            let angle = sign * 2.0 * core::f32::consts::PI * k as f32 / n as f32;
            (x0 + x1 * Complex32::new(angle.cos(), angle.sin())) * scale
        })
        .collect()
}

fn sparse_input(n: usize) -> Vec<Complex32> {
    let mut values = vec![Complex32::new(0.0, 0.0); n];
    values[0] = Complex32::new(0.75, -1.25);
    values[1] = Complex32::new(-0.5, 2.0);
    values
}

fn check_sparse(n: usize, epsilon: f32) {
    let client = cubecl::test_device().client();
    let values = sparse_input(n);
    for normalization in [
        FftNormalization::None,
        FftNormalization::ByN,
        FftNormalization::Ortho,
    ] {
        for (mode, inverse) in [(FftMode::Forward, false), (FftMode::Inverse, true)] {
            let input = complex_handle(&client, vec![1, n], &values);
            let spectrum = cfft_interleaved(&client, input, 1, mode, normalization).unwrap();
            let actual = read_logical(&client, spectrum);
            assert_close(
                &actual,
                &sparse_closed_form(n, inverse, normalization),
                epsilon,
            );
        }
    }
}

fn floor_power_of_two(n: usize) -> usize {
    if n.is_power_of_two() {
        n
    } else {
        n.next_power_of_two() >> 1
    }
}

fn max_shared_n(client: &Client) -> usize {
    let max_elems = client.properties().hardware.max_shared_memory_size / (2 * dtype().size());
    floor_power_of_two(max_elems)
}

#[test]
fn shared_memory_boundary_matches_direct_closed_form() {
    let client = cubecl::test_device().client();
    let n = max_shared_n(&client);
    check_sparse(n, 1e-3);
}

#[test]
fn first_four_step_size_matches_direct_closed_form() {
    let client = cubecl::test_device().client();
    let n = 2 * max_shared_n(&client);
    check_sparse(n, 2e-3);
}

/// Build a C32 handle whose logical values are laid out with the given logical
/// strides; scalars outside the used positions keep `fill`. Returns the handle
/// and a mask of the used scalar positions.
fn strided_handle(
    client: &Client,
    shape: &[usize],
    strides: &[usize],
    values: &[Complex32],
    fill: f32,
) -> (ComplexTensorHandle, Vec<bool>) {
    let max_complex_offset = shape
        .iter()
        .zip(strides)
        .map(|(extent, stride)| (extent - 1) * stride)
        .sum::<usize>();
    let physical_len = 2 * (max_complex_offset + 1);
    let mut physical = vec![fill; physical_len];
    let mut used_scalar = vec![false; physical_len];
    for (logical, value) in values.iter().enumerate() {
        let coords = unravel(logical, shape);
        let offset: usize = coords.iter().zip(strides).map(|(c, s)| c * s).sum();
        physical[2 * offset] = value.re;
        physical[2 * offset + 1] = value.im;
        used_scalar[2 * offset] = true;
        used_scalar[2 * offset + 1] = true;
    }
    let handle = ComplexTensorHandle::new_strided(
        shape.to_vec(),
        strides.to_vec(),
        client.create_from_slice(f32::as_bytes(&physical)),
        dtype(),
    )
    .unwrap();
    (handle, used_scalar)
}

#[test]
fn strided_logical_layout_matches_direct_dft_and_keeps_padding() {
    let client = cubecl::test_device().client();
    let shape = vec![2, 8, 3];
    let strides = vec![30, 3, 1];
    let values = values_for(&shape);

    let (input, _) = strided_handle(&client, &shape, &strides, &values, 0.0);
    // Pre-filled output so untouched padding is observable.
    let (output, used_scalar) = strided_handle(
        &client,
        &shape,
        &strides,
        &vec![Complex32::new(0.0, 0.0); values.len()],
        7.0,
    );

    cfft_interleaved_launch(
        &client,
        input.binding(),
        output.binding(),
        1,
        FftMode::Forward,
        FftNormalization::None,
    )
    .unwrap();

    assert_close(
        &read_logical(&client, output.clone()),
        &direct_dft(&shape, 1, &values, false, FftNormalization::None),
        1e-4,
    );

    let scalars = read_scalars(&client, output.into_raw_parts().handle);
    for (offset, used) in used_scalar.iter().enumerate() {
        if !used {
            assert_eq!(scalars[offset], 7.0, "padding scalar {offset} was written");
        }
    }
}

/// Column-major (Fortran-order) strides, as supplied by the Tenferro interop.
#[test]
fn column_major_layout_matches_direct_dft() {
    let client = cubecl::test_device().client();
    let shape = vec![3, 8];
    let strides = vec![1, 3];
    let values = values_for(&shape);

    let (input, _) = strided_handle(&client, &shape, &strides, &values, 0.0);
    let spectrum =
        cfft_interleaved(&client, input, 1, FftMode::Forward, FftNormalization::None).unwrap();

    assert_close(
        &read_logical(&client, spectrum),
        &direct_dft(&shape, 1, &values, false, FftNormalization::None),
        1e-4,
    );
}

#[test]
fn rejects_invalid_axis() {
    let client = cubecl::test_device().client();
    let input = complex_handle(&client, vec![8], &values_for(&[8]));
    assert!(matches!(
        cfft_interleaved(&client, input, 1, FftMode::Forward, FftNormalization::None),
        Err(FftError::AxisOutOfBounds { dim: 1, rank: 1 })
    ));
}

#[test]
fn rejects_invalid_length() {
    let client = cubecl::test_device().client();
    let input = complex_handle(&client, vec![3], &values_for(&[3]));
    assert!(matches!(
        cfft_interleaved(&client, input, 0, FftMode::Forward, FftNormalization::None),
        Err(FftError::InvalidFftLength { n_fft: 3 })
    ));
}

/// Sizes above `max_shared_fft_n^2` are unsupported and must be reported, not
/// silently accepted. A zero-extent batch axis makes the metadata cheap to
/// build without allocating the impossible signal.
#[test]
fn rejects_unsupported_size_instead_of_silently_accepting_it() {
    let client = cubecl::test_device().client();
    let too_large = max_shared_n(&client).saturating_mul(max_shared_n(&client)) * 2;
    let shape = vec![too_large, 0];

    let input =
        ComplexTensorHandle::new_contiguous(shape.clone(), client.empty(0), dtype()).unwrap();
    let output =
        ComplexTensorHandle::new_contiguous(shape.clone(), client.empty(0), dtype()).unwrap();

    assert!(matches!(
        cfft_interleaved_launch(
            &client,
            input.binding(),
            output.binding(),
            0,
            FftMode::Forward,
            FftNormalization::None,
        ),
        Err(FftError::InvalidFftLength { n_fft }) if n_fft == too_large
    ));
}

#[test]
fn offset_binding_preserves_prefix_and_suffix() {
    let client = cubecl::test_device().client();
    // WebGPU storage bindings require an aligned offset (256 bytes is portable).
    const PAD: usize = 64;
    let mut input_scalars = vec![42.0; PAD];
    input_scalars.extend([1.0, 2.0, 3.0, -4.0]);
    input_scalars.extend(vec![-42.0; PAD]);
    let mut output_scalars = vec![42.0; PAD];
    output_scalars.extend([0.0; 4]);
    output_scalars.extend(vec![-42.0; PAD]);
    let input = ComplexTensorHandle::new_contiguous(
        vec![1, 2],
        client
            .create_from_slice(f32::as_bytes(&input_scalars))
            .offset_start((PAD * 4) as u64)
            .offset_end((PAD * 4) as u64),
        dtype(),
    )
    .unwrap();
    let output = ComplexTensorHandle::new_contiguous(
        vec![1, 2],
        client
            .create_from_slice(f32::as_bytes(&output_scalars))
            .offset_start((PAD * 4) as u64)
            .offset_end((PAD * 4) as u64),
        dtype(),
    )
    .unwrap();
    cfft_interleaved_launch(
        &client,
        input.binding(),
        output.binding(),
        1,
        FftMode::Forward,
        FftNormalization::None,
    )
    .unwrap();
    let mut full = output.into_raw_parts().handle;
    full.offset_start = None;
    full.offset_end = None;
    output_scalars[PAD..PAD + 4].copy_from_slice(&[4.0, -2.0, -2.0, 6.0]);
    let actual = read_scalars(&client, full);
    assert_eq!(&actual[..output_scalars.len()], output_scalars.as_slice());
}

#[test]
fn launch_rejects_same_tensor_as_input_and_output() {
    let client = cubecl::test_device().client();
    let input = complex_handle(&client, vec![8], &values_for(&[8]));
    assert!(matches!(
        cfft_interleaved_launch(
            &client,
            input.binding(),
            input.binding(),
            0,
            FftMode::Forward,
            FftNormalization::None,
        ),
        Err(FftError::OverlappingBindings)
    ));
}
