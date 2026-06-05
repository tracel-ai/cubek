use cubecl::prelude::*;
use cubecl::std::FastDivmod;

#[derive(CubeType, CubeLaunch, Clone)]
#[expand(derive(Clone))]
pub struct NdLayout {
    /// FastDivmod for each dimension used to decompose a linear index into N-D coordinates.
    /// Order: Inner-most dimension first (e.g., for NHWC, C then W then H then N).
    pub divmods: Sequence<FastDivmod<usize>>,

    /// Strides for each dimension to compute the linear source index from N-D coordinates.
    /// Order: Inner-most dimension first (matching divmods).
    pub strides: Sequence<usize>,
}

#[cube]
impl NdLayout {
    /// Decomposes a linear index into N-Dimensional coordinates.
    /// The returned sequence has the inner-most dimension first.
    pub fn from_linear(&self, mut linear_idx: usize) -> Sequence<u32> {
        let mut coords = Sequence::<u32>::new();
        let rank = self.divmods.len();

        #[unroll]
        for i in 0..rank {
            let (rem, c) = self.divmods[i].div_mod(linear_idx);
            coords.push(c as u32);
            linear_idx = rem;
        }
        coords
    }

    /// Computes the linear source index from N-Dimensional coordinates.
    pub fn to_source_pos(&self, coords: Sequence<u32>) -> usize {
        let mut idx = 0;
        let rank = self.strides.len();

        #[unroll]
        for i in 0..rank {
            idx += coords[i] as usize * self.strides[i];
        }
        idx
    }
}
