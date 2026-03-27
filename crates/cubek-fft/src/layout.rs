use cubecl::prelude::*;
use cubecl::{
    self,
    std::tensor::layout::{Coords1d, Layout, LayoutExpand},
};

#[derive(CubeType, Clone, Copy)]
/// Allows to work on the last dimension of the signal/spectrum (one window),
/// abstracting batches
pub struct BatchSignalLayout {
    num_samples: usize,
    stride_samples: usize,
    batch_offset: usize,
    #[cube(comptime)]
    vector_size: usize,
}

#[cube]
impl BatchSignalLayout {
    pub fn new<F: Float, N: Size>(
        tensor: &Tensor<Vector<F, N>>,
        batch_index: usize,
        #[comptime] dim: usize,
    ) -> Self {
        let rank = tensor.rank();
        let mut batch_offset = 0;
        let mut temp_idx = batch_index;

        for i in 0..rank {
            if i != dim {
                let size = tensor.shape(i);
                let stride = tensor.stride(i);

                let coord = temp_idx % size;
                batch_offset += coord * stride;
                temp_idx /= size;
            }
        }

        BatchSignalLayout {
            num_samples: tensor.shape(dim),
            stride_samples: tensor.stride(dim),
            batch_offset,
            vector_size: tensor.vector_size(),
        }
    }
}

#[cube]
impl Layout for BatchSignalLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> usize {
        (self.batch_offset + coords * self.stride_samples) / self.vector_size
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
