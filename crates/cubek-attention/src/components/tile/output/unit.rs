use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;
use cubek_std::{
    MatrixLayout, StageIdent, SwizzleModes,
    tile::{
        Plane, ProductType, RegisterMatmul, RowWise, RowwiseTileKind, RowwiseTileWorkspace, Tile,
        register_allocate_acc,
    },
};

use crate::{components::tile::output::AttentionOutput, definition::AttentionTileSize};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnitOutputConfig {
    pub tile_size: AttentionTileSize,
}

impl UnitOutputConfig {
    fn register(&self) -> RegisterMatmul {
        RegisterMatmul {
            tile_size: self.tile_size.to_value_matmul_tile_size(),
            plane_dim: 1,
            swizzle_modes: SwizzleModes::default(),
            product_type: ProductType::Inner,
        }
    }
}

#[derive(CubeType)]
/// Accumulator tile for Tile Attention
pub struct UnitAttentionOutput<SM: Float, Acc: Float> {
    #[cube(comptime)]
    _phantom: PhantomData<(SM, Acc)>,
}

#[cube]
impl<SM: Float, Acc: Float, VA: Size> AttentionOutput<Acc, VA> for UnitAttentionOutput<SM, Acc> {
    type Config = UnitOutputConfig;
    type ScaleColumn = RowWise<SM>;
    type RunningState = (RowWise<SM>, RowWise<SM>);
    type Workspace = RowwiseTileWorkspace<Acc>;

    fn scale_mul(
        tile: &mut Tile<Acc, VA, Plane, ReadWrite>,
        scale: &Self::ScaleColumn,
        workspace: &mut Self::Workspace,
        #[comptime] _config: Self::Config,
    ) {
        let scale_acc = RowWise::<SM>::cast_from::<Acc>(scale);
        tile.rowwise_scale(&scale_acc, workspace);
    }

    fn scale_div(
        tile: &mut Tile<Acc, VA, Plane, ReadWrite>,
        running_state: &Self::RunningState,
        workspace: &mut Self::Workspace,
        #[comptime] _config: Self::Config,
    ) {
        let mut scale = RowWise::<SM>::cast_from::<Acc>(&running_state.1);
        scale.recip_inplace();
        tile.rowwise_scale(&scale, workspace);
    }

    fn init_workspace(#[comptime] _config: Self::Config) -> Self::Workspace {
        RowwiseTileWorkspace::new(RowwiseTileKind::Direct)
    }

    fn init_tile(#[comptime] config: Self::Config) -> Tile<Acc, VA, Plane, ReadWrite> {
        let mut tile =
            register_allocate_acc::<Acc, VA, Plane>(MatrixLayout::RowMajor, config.register());
        tile.fill_zero();
        tile
    }

    fn write_results<E: Float, ES: Size>(
        source: &Tile<Acc, VA, Plane, ReadWrite>,
        dest: &mut Tile<E, ES, Plane, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        dest.copy_from::<Acc, VA, Acc, Acc, Acc, ReadWrite>(source, StageIdent::Out);
    }
}
