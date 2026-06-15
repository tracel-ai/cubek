use cubecl::prelude::*;

/// Convolution window parameters.
#[derive(Debug, Clone, PartialEq, CubeType)]
pub struct Window {
    /// Number of taps.
    size: usize,
    /// Tap spacing.
    dilation: usize,
}
