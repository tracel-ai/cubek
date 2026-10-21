use cubecl::{CubeElement, prelude::Scalar, std::tensor::TensorHandle};
use cubek_fft::{
    ComplexTensorHandle, FftError, FftNormalization, irfft_interleaved_launch_padded,
    rfft_interleaved_launch_padded,
};

#[test]
fn axis_zero_column_major_output_keeps_padding_in_both_directions() {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let signal = TensorHandle::new(
        client.create_from_slice(f32::as_bytes(&[1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0])),
        vec![4, 2],
        vec![1, 4],
        dtype,
    );
    let spectrum = ComplexTensorHandle::new_strided(
        vec![3, 2],
        vec![1, 4],
        client.create_from_slice(f32::as_bytes(&[99.0f32; 14])),
        dtype,
    )
    .unwrap();
    rfft_interleaved_launch_padded(
        &client,
        &signal,
        spectrum.binding(),
        0,
        4,
        FftNormalization::None,
    )
    .unwrap();
    let bytes = client.read_one(spectrum.into_raw_parts().handle).unwrap();
    let actual = f32::from_bytes(&bytes);
    let expected = [
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 99.0, 99.0, 2.0, 0.0, 0.0, -2.0, -2.0, 0.0,
    ];
    for (got, wanted) in actual.iter().zip(expected) {
        assert!((got - wanted).abs() < 1e-5, "{got} != {wanted}");
    }

    let bins = [
        1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, -2.0, -2.0, 0.0,
    ];
    let spectrum = ComplexTensorHandle::new_strided(
        vec![3, 2],
        vec![1, 3],
        client.create_from_slice(f32::as_bytes(&bins)),
        dtype,
    )
    .unwrap();
    let output = TensorHandle::new(
        client.create_from_slice(f32::as_bytes(&[99.0f32; 9])),
        vec![4, 2],
        vec![1, 5],
        dtype,
    );
    irfft_interleaved_launch_padded(
        &client,
        spectrum.binding(),
        &output,
        0,
        3,
        FftNormalization::None,
    )
    .unwrap();
    let bytes = client.read_one(output.handle).unwrap();
    let actual = f32::from_bytes(&bytes);
    let expected = [4.0, 0.0, 0.0, 0.0, 99.0, 0.0, 8.0, 0.0, 0.0];
    for (got, wanted) in actual.iter().zip(expected) {
        assert!((got - wanted).abs() < 1e-5, "{got} != {wanted}");
    }
}

#[test]
fn middle_axis_real_fft_preserves_each_batch_and_channel() {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let mut input_values = [0.0f32; 16];
    let impulses = [(0usize, 1.0f32), (1, 2.0), (2, 3.0), (3, 4.0)];
    for (window, &(t, amplitude)) in impulses.iter().enumerate() {
        input_values[(window / 2) * 8 + t * 2 + window % 2] = amplitude;
    }
    let input = TensorHandle::new_contiguous(
        vec![2, 4, 2],
        client.create_from_slice(f32::as_bytes(&input_values)),
        dtype,
    );
    let spectrum = ComplexTensorHandle::empty(&client, vec![2, 3, 2]).unwrap();
    rfft_interleaved_launch_padded(
        &client,
        &input,
        spectrum.binding(),
        1,
        4,
        FftNormalization::None,
    )
    .unwrap();
    let handle = spectrum.into_raw_parts().handle;
    let bytes = client.read_one(handle.clone()).unwrap();
    let bins = f32::from_bytes(&bytes);
    for (window, &(t, amplitude)) in impulses.iter().enumerate() {
        for k in 0..3 {
            let angle = -2.0 * core::f32::consts::PI * (k * t) as f32 / 4.0;
            let index = 2 * ((window / 2) * 6 + k * 2 + window % 2);
            assert!((bins[index] - amplitude * angle.cos()).abs() < 1e-5);
            assert!((bins[index + 1] - amplitude * angle.sin()).abs() < 1e-5);
        }
    }
    let spectrum = ComplexTensorHandle::new_contiguous(vec![2, 3, 2], handle, dtype).unwrap();
    let recovered = TensorHandle::new_contiguous(vec![2, 4, 2], client.empty(16 * 4), dtype);
    irfft_interleaved_launch_padded(
        &client,
        spectrum.binding(),
        &recovered,
        1,
        3,
        FftNormalization::ByN,
    )
    .unwrap();
    let bytes = client.read_one(recovered.handle).unwrap();
    let actual = f32::from_bytes(&bytes);
    for (got, wanted) in actual.iter().zip(input_values) {
        assert!((got - wanted).abs() < 1e-5, "{got} != {wanted}");
    }
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
    let shared = client.properties().hardware.max_shared_memory_size / (2 * dtype.size());
    let shared = if shared.is_power_of_two() {
        shared
    } else {
        shared.next_power_of_two() / 2
    };
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
