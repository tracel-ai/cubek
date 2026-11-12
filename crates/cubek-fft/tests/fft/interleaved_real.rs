//! The interleaved real transforms against a direct DFT, under every normalization: at two points
//! and eight, with the signal or spectrum shorter than the transform and with more in its buffer
//! than the transform reads, along the first and a middle axis through layouts with gaps, and past
//! a cube's shared memory. Then the launches they must refuse.
//!
//! Modified from tensor4all/cubek `02efeba9` (MIT OR Apache-2.0).

use cubecl::prelude::Scalar;
use cubecl::std::tensor::TensorHandle;
use cubek_fft::{
    ComplexTensorHandle, FftError, FftNormalization, irfft_interleaved_launch_padded,
    rfft_interleaved_launch_padded,
};
use cubek_test_utils::ScopeUnderTest;

use crate::fft::{
    InterleavedIrfft, InterleavedRfft, IrfftInterleavedLaunch, Placement, RfftInterleavedLaunch,
    Values, longest_shared,
};

/// Two points; eight from a signal of five, which the kernel zero-pads, and from the first five
/// of eight samples, the rest of which it must not read; along the first axis of a column-major
/// signal into a spectrum with a gap after each column; and along a middle axis.
#[test]
fn rfft_interleaved_matches_a_direct_dft() {
    let cases = [
        (
            Placement::contiguous(&[1, 2]),
            Placement::contiguous(&[1, 2]),
            1,
            2,
        ),
        (
            Placement::contiguous(&[1, 5]),
            Placement::contiguous(&[1, 5]),
            1,
            5,
        ),
        (
            Placement::contiguous(&[1, 8]),
            Placement::contiguous(&[1, 5]),
            1,
            5,
        ),
        (
            Placement::strided(&[4, 2], &[1, 4]),
            Placement::strided(&[3, 2], &[1, 4]),
            0,
            4,
        ),
        (
            Placement::contiguous(&[2, 4, 2]),
            Placement::contiguous(&[2, 3, 2]),
            1,
            4,
        ),
    ];
    RfftInterleavedLaunch.check(cases.into_iter().flat_map(
        |(signal, spectrum, dim, signal_len)| {
            InterleavedRfft::every_normalization(signal, spectrum, dim, signal_len, Values::Ramp)
        },
    ));
}

/// Twice and four times the longest complex transform one cube's shared memory holds: the packed
/// real transform at its shared path's end and past it.
#[test]
fn rfft_interleaved_past_shared_memory_matches_a_direct_dft() {
    let shared = longest_shared(&cubecl::test_device().client());
    RfftInterleavedLaunch.check([2 * shared, 4 * shared].into_iter().flat_map(|n| {
        InterleavedRfft::every_normalization(
            Placement::contiguous(&[1, n]),
            Placement::contiguous(&[1, n / 2 + 1]),
            1,
            n,
            Values::Pair,
        )
    }));
}

/// Two points; eight from three of four bins, the rest zero-padded; along the first axis of a
/// column-major spectrum into a signal with a gap after each column; and along a middle axis.
#[test]
fn irfft_interleaved_matches_a_direct_dft() {
    let cases = [
        (
            Placement::contiguous(&[1, 2]),
            Placement::contiguous(&[1, 2]),
            1,
            2,
        ),
        (
            Placement::contiguous(&[1, 4]),
            Placement::contiguous(&[1, 8]),
            1,
            3,
        ),
        (
            Placement::strided(&[3, 2], &[1, 3]),
            Placement::strided(&[4, 2], &[1, 5]),
            0,
            3,
        ),
        (
            Placement::contiguous(&[2, 3, 2]),
            Placement::contiguous(&[2, 4, 2]),
            1,
            3,
        ),
    ];
    IrfftInterleavedLaunch.check(cases.into_iter().flat_map(|(spectrum, signal, dim, bins)| {
        InterleavedIrfft::every_normalization(spectrum, signal, dim, bins, Values::Ramp)
    }));
}

/// Two bins into twice and four times the longest complex transform one cube's shared memory
/// holds.
#[test]
fn irfft_interleaved_past_shared_memory_matches_a_direct_dft() {
    let shared = longest_shared(&cubecl::test_device().client());
    IrfftInterleavedLaunch.check([2 * shared, 4 * shared].into_iter().flat_map(|n| {
        InterleavedIrfft::every_normalization(
            Placement::contiguous(&[1, 2]),
            Placement::contiguous(&[1, n]),
            1,
            2,
            Values::Pair,
        )
    }));
}

#[test]
fn outputs_reject_aliased_storage_and_overlapping_strides() {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let overlapping =
        ComplexTensorHandle::new_strided(vec![3, 2], vec![0, 1], client.empty(16), dtype).unwrap();
    let batched_signal = TensorHandle::new_contiguous(vec![4, 2], client.empty(8 * 4), dtype);
    assert!(matches!(
        rfft_interleaved_launch_padded(
            &client,
            &batched_signal,
            overlapping.binding(),
            0,
            4,
            FftNormalization::None
        ),
        Err(FftError::OverlappingBindings),
    ));
    let bins = ComplexTensorHandle::empty(&client, vec![3]).unwrap();
    let alias = client.empty(4 * 4);
    let retained = alias.clone();
    let output = TensorHandle::new_contiguous(vec![4], alias, dtype);
    assert!(matches!(
        irfft_interleaved_launch_padded(
            &client,
            bins.binding(),
            &output,
            0,
            3,
            FftNormalization::None
        ),
        Err(FftError::OverlappingBindings),
    ));
    drop(retained);
}

#[test]
fn unsupported_packed_length_rejected_before_allocating() {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let shared = longest_shared(&client);
    let n_fft = shared * shared * 4;
    let signal = TensorHandle::new_contiguous(vec![0, 1], client.empty(0), dtype);
    let spectrum = ComplexTensorHandle::new_strided(
        vec![0, n_fft / 2 + 1],
        vec![0, 1],
        client.empty(0),
        dtype,
    )
    .unwrap();
    assert!(matches!(
        rfft_interleaved_launch_padded(
            &client,
            &signal,
            spectrum.binding(),
            1,
            0,
            FftNormalization::None
        ),
        Err(FftError::FftLengthExceedsDeviceLimit { .. }),
    ));
    let output = TensorHandle::new_contiguous(vec![0, n_fft], client.empty(0), dtype);
    assert!(matches!(
        irfft_interleaved_launch_padded(
            &client,
            spectrum.binding(),
            &output,
            1,
            1,
            FftNormalization::None
        ),
        Err(FftError::FftLengthExceedsDeviceLimit { .. }),
    ));
}
