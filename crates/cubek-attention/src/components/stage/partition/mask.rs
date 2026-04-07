use cubecl;
use cubecl::prelude::*;

use cubecl::std::tensor::layout::Coords2d;

use crate::{components::tile::MaskTile, components::tile::softmax::Softmax};

#[derive(CubeType)]
/// We can keep only one mask tile at a time because it is directly applied to softmax tile
pub struct MaskPartition<F: Float, SMX: Softmax<F>> {
    sequence: Sequence<MaskTile<F, SMX>>,
}

#[cube]
impl<F: Float, SMX: Softmax<F>> MaskPartition<F, SMX> {
    pub fn new(
        out_of_bounds: ComptimeOption<Coords2d>,
        #[comptime] config: SMX::Config,
    ) -> MaskPartition<F, SMX> {
        let mut sequence = Sequence::new();

        sequence.push(MaskTile::new(out_of_bounds, config));

        MaskPartition::<F, SMX> { sequence }
    }

    pub fn get(&self) -> &MaskTile<F, SMX> {
        &self.sequence[0]
    }

    pub fn get_mut(&mut self) -> &mut MaskTile<F, SMX> {
        self.sequence.index_mut(0usize)
    }
}
