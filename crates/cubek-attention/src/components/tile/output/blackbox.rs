use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;
use cubek_std::{
    MatrixLayout, StageIdent,
    tile::{
        BounceConfig, Plane, RowWise, RowwiseTileKind, RowwiseTileWorkspace, Tile,
        cmma_allocate_acc,
    },
};

use crate::{
    components::tile::output::AttentionOutput, components::tile::pipeline::InnerLayout,
    definition::AttentionTileSize,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct BlackboxOutputConfig {
    pub tile_size: AttentionTileSize,
    pub num_planes: u32,
    pub plane_dim: u32,
    pub inner_layout: InnerLayout,
}

#[derive(CubeType)]
/// Accumulator tile for Tile Attention
pub struct BlackboxAttentionOutput<SM: Float, Acc: Float> {
    #[cube(comptime)]
    _phantom: PhantomData<(SM, Acc)>,
}

#[cube]
impl<SM: Float, Acc: Float, VA: Size> AttentionOutput<Acc, VA>
    for BlackboxAttentionOutput<SM, Acc>
{
    type Config = BlackboxOutputConfig;
    type ScaleColumn = RowWise<SM>;
    type RunningState = (RowWise<SM>, RowWise<SM>);
    type Workspace = RowwiseTileWorkspace<Acc>;

    fn scale_mul(
        tile: &mut Tile<Acc, VA, Plane, ReadWrite>,
        scale: &Self::ScaleColumn,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) {
        let scale_acc = RowWise::<SM>::cast_from::<Acc>(scale);
        let stride = config.tile_size.val_dim;
        tile.bounce_in(workspace, stride);
        tile.rowwise_scale(&scale_acc, workspace);
        tile.bounce_out(workspace, stride);
    }

    fn scale_div(
        tile: &mut Tile<Acc, VA, Plane, ReadWrite>,
        running_state: &Self::RunningState,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) {
        let mut scale = RowWise::<SM>::cast_from::<Acc>(&running_state.1);
        scale.recip_inplace();
        let stride = config.tile_size.val_dim;
        tile.bounce_in(workspace, stride);
        tile.rowwise_scale(&scale, workspace);
        tile.bounce_out(workspace, stride);
    }

    fn init_workspace(#[comptime] config: Self::Config) -> Self::Workspace {
        let kind = comptime! {
            RowwiseTileKind::Bounce(BounceConfig {
                tile_shape: (config.tile_size.seq_q, config.tile_size.val_dim),
                num_planes: config.num_planes,
                plane_dim: config.plane_dim,
                inner_layout: config.inner_layout,
            })
        };
        RowwiseTileWorkspace::new(kind)
    }

    fn init_tile(#[comptime] config: Self::Config) -> Tile<Acc, VA, Plane, ReadWrite> {
        let mut tile = cmma_allocate_acc::<Acc, VA, Plane>(
            MatrixLayout::RowMajor,
            config.tile_size.to_value_matmul_tile_size(),
        );
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
