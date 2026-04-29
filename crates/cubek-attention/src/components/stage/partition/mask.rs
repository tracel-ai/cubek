use cubecl;
use cubecl::prelude::*;

use cubecl::std::tensor::layout::Coords2d;

use crate::components::tile::{MaskConfig, MaskTile};

#[derive(CubeType)]
/// Holds a single live mask tile (it is applied immediately to the score tile,
/// so we never need more than one in flight).
pub struct MaskPartition<F: Float> {
    sequence: Sequence<MaskTile<F>>,
}

#[cube]
impl<F: Float> MaskPartition<F> {
    pub fn new(
        out_of_bounds: ComptimeOption<Coords2d>,
        #[comptime] mask_config: MaskConfig,
    ) -> MaskPartition<F> {
        let mut sequence = Sequence::new();
        sequence.push(MaskTile::<F>::new(out_of_bounds, mask_config));
        MaskPartition::<F> { sequence }
    }

    pub fn get(&self) -> &MaskTile<F> {
        &self.sequence[0]
    }

    pub fn get_mut(&mut self) -> &mut MaskTile<F> {
        self.sequence.index_mut(0usize)
    }
}
