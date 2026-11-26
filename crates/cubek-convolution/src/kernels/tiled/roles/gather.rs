use crate::kernels::tiled::SpatialDim;
use crate::launch::SpatialWindow;
use cubecl::tile::Dimension;
use cubecl::tile::layout::PhysicalAxisMap;

/// How the gathered input's positions are addressed from an output dimension and a tap dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvGather {
    /// The window of a convolution: `out * stride + tap * dilation - padding`.
    Window,
    /// The window read backward, at stride 1: `out + padding - tap * dilation`, which falls before
    /// the input's start at the first positions whatever the padding.
    Gradient,
}

impl ConvGather {
    /// The map of `window` over the dimensions `out_dimension` and `tap_dimension`.
    pub fn map(
        self,
        window: &SpatialWindow,
        out_dimension: Dimension,
        tap_dimension: Dimension,
    ) -> PhysicalAxisMap {
        match self {
            Self::Window => window.position_map(out_dimension, tap_dimension),
            Self::Gradient => window.gradient_position_map(out_dimension, tap_dimension),
        }
    }

    /// The window's input and output extents along `dim`.
    pub fn window_ends(self, spatial_dim: &SpatialDim) -> (usize, usize) {
        match self {
            Self::Window => (spatial_dim.gathered, spatial_dim.out),
            Self::Gradient => (spatial_dim.out, spatial_dim.gathered),
        }
    }
}
