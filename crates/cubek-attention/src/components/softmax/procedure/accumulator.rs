use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;

use crate::components::softmax::{Accumulator, AccumulatorProcedureConfig, TileAttention};
use crate::components::tile::RowWise;
use crate::components::tile::accelerated_blackbox::LocalTile;
use crate::components::tile::{AccumulatorPipeline, AccumulatorPipelineExpand};
use crate::components::tile::{AccumulatorRowwise, AccumulatorRowwiseExpand};
use crate::definition::AttentionPrecision;
use crate::definition::attention_types::SM;

#[derive(CubeType)]
/// Accumulator tile for Tile Attention
pub struct AccumulatorProcedure<Acc: Float> {
    #[cube(comptime)]
    _phantom: PhantomData<Acc>,
}

// #[cube]
// impl<Acc: Float> AccumulatorProcedure<Acc> {
//     pub fn new(
//         shared: &mut TA::AccumulatorTransit,
//         #[comptime] config: TA::Config,
//     ) -> AccumulatorProcedure<AP, TA> {
//         let mut fragment = TA::allocate_accumulator(shared, config);
//         fragment.zero();

//         AccumulatorProcedure::<AP, TA> { fragment }
//     }
// }

#[derive(CubeType)]
pub struct AccumulatorProcedureWorkspace<Acc: Float> {
    acc_smem_slice: SliceMut<Acc>,
    local_tile: LocalTile<Acc>,
}

#[cube]
impl<Acc: Float> Accumulator for AccumulatorProcedure<Acc> {
    type Config = AccumulatorProcedureConfig;
    type ScaleColumn = RowWise<Acc>;
    type RunningState = (RowWise<Acc>, RowWise<Acc>);
    type Tile = cmma::Matrix<Acc>;
    type Workspace = AccumulatorProcedureWorkspace<Acc>;

    fn scale_mul(
        tile: &mut Self::Tile,
        scale: &Self::ScaleColumn,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) {
        cmma::store(
            &mut workspace.acc_smem_slice,
            &tile,
            config.tile_size.val_dim,
            cmma::MatrixLayout::RowMajor,
        );

        sync_cube();

        workspace
            .local_tile
            .load_from_slice(&workspace.acc_smem_slice.to_slice());

        sync_cube();

        workspace
            .local_tile
            .rowwise_scale(&RowWise::<Acc>::cast_from(&scale));

        workspace.local_tile.store_to(&mut workspace.acc_smem_slice);

        sync_cube();

        cmma::load_with_layout(
            &tile,
            &workspace.acc_smem_slice.to_slice(),
            config.tile_size.val_dim,
            cmma::MatrixLayout::RowMajor,
        )
    }

    fn scale_div(
        tile: &mut Self::Tile,
        running_state: &Self::RunningState,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) {
        let mut scale = RowWise::<Acc>::cast_from(&running_state.1);
        scale.recip_inplace();

        cmma::store(
            &mut workspace.acc_smem_slice,
            &tile,
            config.tile_size.val_dim,
            cmma::MatrixLayout::RowMajor,
        );

        sync_cube();

        workspace
            .local_tile
            .load_from_slice(&workspace.acc_smem_slice.to_slice());

        sync_cube();

        workspace.local_tile.rowwise_scale(&scale);

        workspace.local_tile.store_to(&mut workspace.acc_smem_slice);

        sync_cube();

        cmma::load_with_layout(
            &tile,
            &workspace.acc_smem_slice.to_slice(),
            config.tile_size.val_dim,
            cmma::MatrixLayout::RowMajor,
        )
    }

    fn init_workspace(#[comptime] config: Self::Config) -> Self::Workspace {
        todo!()
    }

    fn init_tile(workspace: &mut Self::Workspace, #[comptime] config: Self::Config) -> Self::Tile {
        todo!()
    }

    fn write_results<E: Float>(
        tile: &Self::Tile,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
        todo!()
    }
}
