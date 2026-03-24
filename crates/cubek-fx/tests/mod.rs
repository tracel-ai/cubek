//! These tests run on wgpu runtime only
//! For other backends, prefer backend_tests

mod irfft;
mod reference;
mod rfft;
mod fft_round_trip;

pub(crate) use reference::*;
