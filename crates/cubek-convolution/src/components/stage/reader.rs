use cubecl;
use cubecl::{prelude::*, std::tensor::layout::Coords2d};
use cubek_matmul::components::stage::TilingValidation;
use cubek_std::{
    stage::StageMemoryConfig,
    tile::StridedTile,
    {InvalidConfigError, MatrixLayout},
};

use crate::components::stage::bias_stage::BiasStageMemory;

#[derive(Clone, Copy)]
/// Tiling layout specific for bias, which is one-dimensional with stride 0
pub struct BiasTilingLayout {}

#[cube]
impl BiasTilingLayout {
    pub fn get_tile<ES: Numeric, NS: Size>(
        stage: &BiasStageMemory<ES, NS>,
        tile: Coords2d,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedTile<ES, NS> {
        if config.num_stages > 1 {
            unimplemented!()
        }

        let (_, col) = tile;

        let stage_vector_size = config.vector_size;
        let tile_size_col = config.elements_per_tile_along_col / stage_vector_size;

        let length = tile_size_col;
        let start = col * tile_size_col;

        StridedTile::new_strided(
            stage.as_slice(),
            start,
            start + length,
            0,
            stage.swizzle,
            MatrixLayout::RowMajor,
        )
    }
}

impl TilingValidation for BiasTilingLayout {
    fn check(config: StageMemoryConfig) -> Result<(), InvalidConfigError> {
        let stage_width = config.elements_per_stage_along_col();
        if config.vector_size > stage_width {
            return Err(Box::new(format!(
                "Invalid vector size. Got {:?} which should not be >{:?}",
                config.vector_size, stage_width,
            )));
        }
        Ok(())
    }
}
