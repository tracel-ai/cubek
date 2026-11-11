//! The real inverse transform against the host's: along the last axis and strided across others,
//! batched, at sizes past a cube's shared memory and past the four-step split, and virtually
//! padded past a spectrum's last bin.

use cubek_test_utils::ScopeUnderTest;

use crate::fft::{Irfft, IrfftLaunch};

#[test]
fn irfft_light_axis_last() {
    IrfftLaunch.check([Irfft::seeded(&[1, 5], 1)]);
}

#[test]
fn irfft_light_axis_1_strided() {
    IrfftLaunch.check([Irfft::seeded(&[2, 5, 1], 1)]);
}

#[test]
fn irfft_light_axis_1_strided_trailing_batch() {
    IrfftLaunch.check([Irfft::seeded(&[3, 5, 2], 1)]);
}

#[test]
fn irfft_light_axis_0_strided() {
    IrfftLaunch.check([Irfft::seeded(&[5, 2], 0)]);
}

#[test]
fn irfft_light_axis_last_n16() {
    IrfftLaunch.check([Irfft::seeded(&[1, 9], 1)]);
}

#[test]
fn irfft_virtual_padding_axis_1_matches_materialized_zero_padding() {
    IrfftLaunch.check([Irfft::padded(&[2, 3, 3], 1, 8)]);
}

#[test]
fn irfft_virtual_padding_dc_only_matches_materialized_zero_padding() {
    IrfftLaunch.check([Irfft::padded(&[2, 1, 3], 1, 8)]);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_3d_last_axis() {
    IrfftLaunch.check([Irfft::seeded(&[5, 2, 1025], 2)]);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_3d_axis_0() {
    IrfftLaunch.check([Irfft::seeded(&[33, 2, 1024], 0)]);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_3d_axis_1() {
    IrfftLaunch.check([Irfft::seeded(&[33, 5, 1024], 1)]);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_4d_axis_2() {
    IrfftLaunch.check([Irfft::seeded(&[12, 8, 513, 4], 2)]);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_shared_memory_cap_axis_1_strided() {
    IrfftLaunch.check([Irfft::seeded(&[1, 2049, 1], 1)]);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_large_axis_1_strided() {
    IrfftLaunch.check([Irfft::seeded(&[1, 4097, 1], 1)]);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_four_step_axis_1_strided() {
    IrfftLaunch.check([Irfft::seeded(&[1, 8193, 1], 1)]);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_batched_large_axis_last() {
    IrfftLaunch.check([Irfft::seeded(&[3, 4097], 1)]);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_large_virtual_padding_matches_materialized_zero_padding() {
    IrfftLaunch.check([Irfft::padded(&[1, 3000], 1, 8192)]);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_3d_batch_singleton_dim() {
    IrfftLaunch.check([Irfft::seeded(&[22, 1, 1025], 2)]);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_dispatch_more_than_wgpu_x_axis_limit() {
    IrfftLaunch.check([Irfft::seeded(&[65_536, 2], 1)]);
}
