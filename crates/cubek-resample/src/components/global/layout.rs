use cubecl::prelude::*;
use cubecl::std::FastDivmod;
use cubecl::std::tensor::layout::CoordsDyn;

#[cube]
pub trait Layout {
    fn from_linear(&self, linear_idx: usize) -> CoordsDyn;

    fn to_source_pos(&self, coords: &CoordsDyn) -> usize;
}

#[derive(CubeType, CubeLaunch)]
pub struct NdLayout {
    pub divmods: Sequence<FastDivmod<usize>>,
    pub strides: Sequence<usize>,
}

#[cube]
impl Layout for NdLayout {
    fn from_linear(&self, linear_idx: usize) -> CoordsDyn {
        let mut coords = CoordsDyn::new();
        let rank = self.divmods.len();

        let mut idx = linear_idx;

        #[unroll]
        for i in 0..rank {
            let (rem, c) = self.divmods.index(i).div_mod(idx);
            coords.push(c as u32);
            idx = rem;
        }
        coords
    }

    fn to_source_pos(&self, coords: &CoordsDyn) -> usize {
        let mut idx = 0;
        let rank = self.strides.len();

        #[unroll]
        for i in 0..rank {
            idx += coords[i] as usize * self.strides[i];
        }
        idx
    }
}
