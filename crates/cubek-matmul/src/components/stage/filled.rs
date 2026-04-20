use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::components::{
    stage::{Stage, StageFamily, TilingLayout},
    tile_matmul::{Scope, Tile, Value},
};

pub struct FilledStageFamily;

impl StageFamily for FilledStageFamily {
    type Stage<ES: Numeric, NS: Size, T: TilingLayout> = FilledStage<ES>;
}

#[derive(CubeType, Clone)]
pub struct FilledStage<ES: Numeric> {
    value: ES,
}

#[cube]
impl<ES: Numeric> FilledStage<ES> {
    pub fn new(value: ES) -> Self {
        FilledStage::<ES> { value }
    }
}

#[cube]
impl<ES: Numeric, NS: Size> Stage<ES, NS, ReadOnly> for FilledStage<ES> {
    fn tile<Sc: Scope>(this: &Self, _tile: Coords2d) -> Tile<ES, NS, Sc, ReadOnly> {
        Tile::new_Broadcasted(Value::<ES> { val: this.value })
    }
}
