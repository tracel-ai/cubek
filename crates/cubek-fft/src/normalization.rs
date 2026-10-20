//! Adapted from tensor4all/cubek `02efeba9` (MIT OR Apache-2.0)
//! for the current CubeCL FFT API.

use crate::FftError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FftNormalization {
    None,
    ByN,
    Ortho,
}

impl FftNormalization {
    pub fn scale_f32(self, n_fft: usize) -> Result<f32, FftError> {
        if n_fft < 2 || !n_fft.is_power_of_two() {
            return Err(FftError::InvalidFftLength { n_fft });
        }
        Ok(match self {
            Self::None => 1.0,
            Self::ByN => 1.0 / n_fft as f32,
            Self::Ortho => 1.0 / (n_fft as f32).sqrt(),
        })
    }
}
