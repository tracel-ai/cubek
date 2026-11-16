//! The interleaved complex transform against a direct DFT, forward and inverse under every
//! normalization: along every axis, past a cube's shared memory, and through layouts with gaps
//! and margins it must leave as they were. Then the launches it must refuse.
//!
//! Modified from tensor4all/cubek `02efeba9` (MIT OR Apache-2.0).

use cubecl::prelude::Scalar;
use cubek_fft::{
    ComplexTensorHandle, FftError, FftMode, FftNormalization, cfft_interleaved,
    cfft_interleaved_launch,
};
use cubek_test_utils::ScopeUnderTest;

use crate::fft::{CfftInterleavedLaunch, InterleavedCfft, Placement, Values, longest_shared};

/// Along the last axis, the first and a middle one, at two points and at eight. A direct DFT, not
/// a round trip, so a sign or a permutation the inverse undoes cannot hide.
#[test]
fn cfft_interleaved_matches_a_direct_dft() {
    let axes: [(&[usize], usize); 4] = [(&[2, 8], 1), (&[1, 2], 1), (&[8, 3], 0), (&[2, 8, 3], 1)];
    CfftInterleavedLaunch.check(axes.into_iter().flat_map(|(shape, dim)| {
        InterleavedCfft::every_way(Placement::contiguous(shape), dim, Values::Ramp)
    }));
}

/// The longest transform one cube's shared memory holds, and twice it, the first the four-step
/// split takes.
#[test]
fn cfft_interleaved_past_shared_memory_matches_a_direct_dft() {
    let shared = longest_shared(&cubecl::test_device().client());
    CfftInterleavedLaunch.check(
        [shared, 2 * shared].into_iter().flat_map(|n| {
            InterleavedCfft::every_way(Placement::contiguous(&[1, n]), 1, Values::Pair)
        }),
    );
}

/// Rows with gaps between them, column-major as an interop hands over, and a tensor bound 256
/// bytes into a larger buffer (an offset wgpu accepts): the transform reads and writes only where
/// the layout puts values.
#[test]
fn cfft_interleaved_keeps_what_lies_outside_its_layout() {
    let layouts = [
        Placement::strided(&[2, 8, 3], &[30, 3, 1]),
        Placement::strided(&[3, 8], &[1, 3]),
        Placement::contiguous(&[1, 2]).with_margin(64),
    ];
    CfftInterleavedLaunch.check(
        layouts
            .into_iter()
            .flat_map(|layout| InterleavedCfft::every_way(layout, 1, Values::Ramp)),
    );
}

#[test]
fn rejects_invalid_axis() {
    let client = cubecl::test_device().client();
    let input = Placement::contiguous(&[8]).complex(&client, &[0.0; 16]);
    assert!(matches!(
        cfft_interleaved(&client, input, 1, FftMode::Forward, FftNormalization::None),
        Err(FftError::AxisOutOfBounds { dim: 1, rank: 1 })
    ));
}

#[test]
fn rejects_invalid_length() {
    let client = cubecl::test_device().client();
    let input = Placement::contiguous(&[3]).complex(&client, &[0.0; 6]);
    assert!(matches!(
        cfft_interleaved(&client, input, 0, FftMode::Forward, FftNormalization::None),
        Err(FftError::InvalidFftLength { n_fft: 3 })
    ));
}

/// Sizes above `longest_shared²` are unsupported and must be reported, not silently accepted. A
/// zero-extent batch axis makes the metadata cheap to build without allocating the impossible
/// signal.
#[test]
fn rejects_unsupported_size_instead_of_silently_accepting_it() {
    let client = cubecl::test_device().client();
    let shared = longest_shared(&client);
    let too_large = shared.saturating_mul(shared) * 2;
    let shape = vec![too_large, 0];
    let dtype = f32::elem_type_native();
    let input = ComplexTensorHandle::new_contiguous(shape.clone(), client.empty(0), dtype).unwrap();
    let output = ComplexTensorHandle::new_contiguous(shape, client.empty(0), dtype).unwrap();

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
fn launch_rejects_same_tensor_as_input_and_output() {
    let client = cubecl::test_device().client();
    let input = Placement::contiguous(&[8]).complex(&client, &[0.0; 16]);
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
