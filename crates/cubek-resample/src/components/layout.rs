use cubecl::prelude::*;
pub use cubecl::std::tensor::layout::{Coords1d, CoordsDyn, Layout, LayoutExpand};

#[derive(CubeType, CubeLaunch)]
pub struct NdLayout {
    pub shape: CoordsDyn,
    pub strides: Sequence<usize>,
}

impl<R: Runtime> NdLayoutLaunch<R> {
    pub fn from_tensor(tensor: &TensorBinding<R>) -> Self {
        let mut coords = SequenceArg::new();
        for i in 0..tensor.shape.len() {
            coords.push(tensor.shape[i] as u32);
        }

        let mut strides_seq = SequenceArg::new();
        for i in 0..tensor.strides.len() {
            strides_seq.push(tensor.strides[i]);
        }

        Self::new(coords, strides_seq)
    }
}

#[cube]
impl Layout for NdLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = usize;

    fn to_source_pos(&self, coords: Self::Coordinates) -> Self::SourceCoordinates {
        let mut idx = 0;
        #[unroll]
        for i in 0..self.strides.len() {
            idx += coords[i] as usize * self.strides[i];
        }
        idx
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(coords.clone());
        (self.to_source_pos(coords), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape.clone()
    }

    fn is_in_bounds(&self, coords: Self::Coordinates) -> bool {
        let shape = self.shape();
        let mut in_bounds = true;
        for i in 0..coords.len() {
            if coords[i] >= shape[i] {
                in_bounds = false;
            }
        }
        in_bounds
    }
}

#[cube]
impl NdLayout {
    pub fn from_linear(&self, index: usize) -> CoordsDyn {
        let mut coords = CoordsDyn::new();
        let mut remaining = index;

        #[unroll]
        for i in 0..self.shape.len() {
            let mut logical_stride = 1;
            #[unroll]
            for j in 0..self.shape.len() {
                if j > i {
                    logical_stride *= self.shape[j] as usize;
                }
            }
            let coord = remaining / logical_stride;
            coords.push(coord as u32);
            remaining -= coord * logical_stride;
        }
        coords
    }
}
