//! Modified from tensor4all/cubek `02efeba9` (MIT OR Apache-2.0):
//! express interleaved scalar-pair indexing in the current CubeCL layout API.

use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords1d, Layout, LayoutExpand},
};

/// A one-dimensional component view over an interleaved C32 tensor window.
///
/// Coordinates are logical complex elements along `dim`; the returned scalar
/// index adds the component offset (`0` for real, `1` for imaginary) so both
/// components address the same physical allocation at scalar resolution.
#[derive(CubeType, Clone, Copy)]
pub(crate) struct InterleavedBatchSignalLayout {
    num_samples: usize,
    stride_samples: usize,
    batch_offset: usize,
    component: usize,
}

#[cube]
impl InterleavedBatchSignalLayout {
    pub fn new<F: Numeric>(
        tensor: &Tensor<F>,
        batch_index: usize,
        #[comptime] dim: usize,
        #[comptime] component: usize,
    ) -> Self {
        let rank = tensor.rank();
        let mut batch_offset = 0;
        let mut temp_idx = batch_index;

        for axis in 0..rank {
            if axis != dim {
                let size = tensor.shape(axis);
                let stride = tensor.stride(axis);
                let coord = temp_idx % size;
                batch_offset += coord * stride;
                temp_idx /= size;
            }
        }

        InterleavedBatchSignalLayout {
            num_samples: tensor.shape(dim),
            stride_samples: tensor.stride(dim),
            batch_offset,
            component,
        }
    }
}

#[cube]
impl Layout for InterleavedBatchSignalLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> usize {
        self.batch_offset + coords * self.stride_samples + self.component
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (usize, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        self.num_samples
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos < self.num_samples
    }
}
