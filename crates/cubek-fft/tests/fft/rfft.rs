//! The real forward transform against the host's: along the last axis and strided across others,
//! batched, at sizes past a cube's shared memory and past the four-step split, in `f64`, and
//! virtually padded past a signal's end.

#[cfg(feature = "heavy")]
use cubecl::CubeElement;
use cubecl::prelude::Scalar;
#[cfg(feature = "heavy")]
use cubecl::std::tensor::TensorHandle;
#[cfg(feature = "heavy")]
use cubek_fft::rfft_launch;
use cubek_test_utils::ScopeUnderTest;
#[cfg(feature = "heavy")]
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, HostDataVec, TestInput, TestOutcome,
    ValidationResult, launch_and_capture_outcome,
};

use crate::fft::{Rfft, RfftLaunch, RfftPadding};

#[cfg(feature = "heavy")]
fn to_f32(host: HostData) -> Vec<f32> {
    match host.data {
        HostDataVec::F32(v) => v,
        _ => panic!("expected f32 host data"),
    }
}

#[test]
fn rfft_light_axis_last() {
    RfftLaunch.check([Rfft::seeded(&[1, 8], 1)]);
}

#[test]
fn rfft_light_f64() {
    RfftLaunch.check([Rfft::seeded(&[2, 8], 1).in_element(f64::elem_type_native())]);
}

#[test]
fn rfft_light_axis_1_strided() {
    RfftLaunch.check([Rfft::seeded(&[2, 8, 1], 1)]);
}

#[test]
fn rfft_light_axis_1_strided_trailing_batch() {
    RfftLaunch.check([Rfft::seeded(&[3, 8, 2], 1)]);
}

#[test]
fn rfft_light_axis_0_strided() {
    RfftLaunch.check([Rfft::seeded(&[8, 2], 0)]);
}

#[test]
fn rfft_light_axis_last_n16() {
    RfftLaunch.check([Rfft::seeded(&[1, 16], 1)]);
}

#[test]
fn rfft_virtual_padding_axis_1_matches_materialized_zero_padding() {
    RfftLaunch.check([Rfft::padded(
        &[2, 5, 3],
        1,
        RfftPadding {
            signal_len: 5,
            n_fft: 8,
        },
    )]);
}

#[test]
fn rfft_virtual_padding_ignores_tail_after_signal_len() {
    RfftLaunch.check([Rfft::padded(
        &[2, 7, 3],
        1,
        RfftPadding {
            signal_len: 5,
            n_fft: 8,
        },
    )]);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_3d_axis_last() {
    RfftLaunch.check([Rfft::seeded(&[5, 2, 2048], 2)]);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_3d_axis_1_strided() {
    RfftLaunch.check([Rfft::seeded(&[5, 64, 1000], 1)]);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_3d_axis_0_strided() {
    RfftLaunch.check([Rfft::seeded(&[128, 6, 1000], 0)]);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_4d_axis_1_strided() {
    RfftLaunch.check([Rfft::seeded(&[5, 256, 6, 42], 1)]);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_shared_memory_cap_axis_1_strided() {
    RfftLaunch.check([Rfft::seeded(&[1, 4096, 1], 1)]);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_large_axis_1_strided() {
    RfftLaunch.check([Rfft::seeded(&[1, 8192, 1], 1)]);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_four_step_axis_1_strided() {
    RfftLaunch.check([Rfft::seeded(&[1, 16384, 1], 1)]);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_batched_large_axis_last() {
    RfftLaunch.check([Rfft::seeded(&[3, 8192], 1)]);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_large_virtual_padding_matches_materialized_zero_padding() {
    RfftLaunch.check([Rfft::padded(
        &[1, 5000],
        1,
        RfftPadding {
            signal_len: 5000,
            n_fft: 8192,
        },
    )]);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_3d_batch_singleton_dim() {
    RfftLaunch.check([Rfft::seeded(&[22, 1, 2048], 2)]);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_dispatch_more_than_wgpu_x_axis_limit() {
    RfftLaunch.check([Rfft::seeded(&[65_536, 2], 1)]);
}

/// A signal alternating `±1` is all Nyquist: `n_fft` in the last bin, nothing in the others,
/// checked exactly at sizes the four-step split serves.
#[test]
#[cfg(feature = "heavy")]
fn rfft_nyquist_bin_large_sizes() {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();

    for &n_fft in &[8192usize, 16384] {
        let batch = 2;
        let n_freq = n_fft / 2 + 1;
        let signal_shape = [batch, n_fft].to_vec();
        let spectrum_shape = [batch, n_freq].to_vec();

        let signal_data: Vec<f32> = (0..batch)
            .flat_map(|_| (0..n_fft).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }))
            .collect();
        let signal_handle = client.create_from_slice(f32::as_bytes(&signal_data));
        let signal = TensorHandle::new_contiguous(signal_shape, signal_handle, dtype);
        let spectrum_re = TestInput::builder(client.clone(), spectrum_shape.clone())
            .dtype(dtype)
            .zeros()
            .generate_without_host_data();
        let spectrum_im = TestInput::builder(client.clone(), spectrum_shape)
            .dtype(dtype)
            .zeros()
            .generate_without_host_data();

        let signal_binding = signal.binding();
        let re_binding = spectrum_re.clone().binding();
        let im_binding = spectrum_im.clone().binding();

        let outcome =
            launch_and_capture_outcome(&client, &[&spectrum_re.handle, &spectrum_im.handle], |c| {
                rfft_launch(c, signal_binding, re_binding, im_binding, 1, dtype).into()
            });

        let outcome = match outcome {
            ExecutionOutcome::Executed => {
                let re = to_f32(HostData::from_tensor_handle(
                    &client,
                    spectrum_re,
                    HostDataType::F32,
                ));
                let im = to_f32(HostData::from_tensor_handle(
                    &client,
                    spectrum_im,
                    HostDataType::F32,
                ));
                let mut result = ValidationResult::Pass;
                'check: for b in 0..batch {
                    let base = b * n_freq;
                    for k in 0..n_freq {
                        let expected = if k == n_fft / 2 { n_fft as f32 } else { 0.0 };
                        if (re[base + k] - expected).abs() >= 1.0 {
                            result = ValidationResult::Fail(format!(
                                "n_fft={n_fft}, batch={b}, bin={k}: real={}, want {expected}",
                                re[base + k]
                            ));
                            break 'check;
                        }
                        if im[base + k].abs() >= 1.0 {
                            result = ValidationResult::Fail(format!(
                                "n_fft={n_fft}, batch={b}, bin={k}: imag={}",
                                im[base + k]
                            ));
                            break 'check;
                        }
                    }
                }
                result.as_test_outcome()
            }
            ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
        };
        outcome.enforce();
    }
}
