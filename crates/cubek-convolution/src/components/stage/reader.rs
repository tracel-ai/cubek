use cubecl;
use cubecl::prelude::*;
use cubecl::std::tensor::layout::Coords2d;
use cubek_matmul::{
    components::{
        stage::{StageMemoryConfig, TilingValidation},
        tile::StridedTile,
    },
    definition::{InvalidConfigError, MatrixLayout},
};

use crate::components::stage::bias_stage::BiasStageMemory;

#[derive(Clone, Copy)]
/// Tiling layout specific for bias, which is one-dimensional with stride 0
pub struct BiasTilingLayout {}

#[cube]
impl BiasTilingLayout {
    pub fn get_tile<ES: Numeric>(
        stage: &BiasStageMemory<ES>,
        tile: Coords2d,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedTile<ES> {
        if config.num_stages > 1 {
            unimplemented!()
        }

        let (_, col) = tile;

        let stage_line_size = config.line_size;
        let tile_size_col = config.elements_per_tile_along_col / stage_line_size;

        let length = tile_size_col;
        let start = col * tile_size_col;

        StridedTile::new_strided(
            stage.as_slice(stage_line_size as usize),
            start,
            start + length,
            0,
            stage.swizzle,
            MatrixLayout::RowMajor,
            stage_line_size,
        )
    }
}

impl TilingValidation for BiasTilingLayout {
    fn check(config: StageMemoryConfig) -> Result<(), InvalidConfigError> {
        let stage_width = config.elements_per_stage_along_col();
        if config.line_size > stage_width {
            return Err(Box::new(format!(
                "Invalid line size. Got {:?} which should not be >{:?}",
                config.line_size, stage_width,
            )));
        }
        Ok(())
    }
}
