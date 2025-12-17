use cubecl::{Runtime, client::ComputeClient};

use std::marker::PhantomData;

use crate::{
    components::{
        batch::{BatchMatmulFamily, PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            UnitWriterFamily,
            read::{FullLoadingStrategy, sync_full_cyclic::SyncFullCyclicLoading},
            single_stage::simple::SimpleMatmulFamily,
        },
        stage::{
            ColMajorTilingOrder, FilledStageFamily, RowMajorTilingOrder, StridedStageFamily,
            UnitMatmulFamily,
        },
        tile::{TileMatmulFamily, io::Filled, register::RegisterMatmul},
    },
    definition::{MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSetupError, TilingBlueprint},
    routines::selector::{
        PartitionScaling, StageScaling, TileSizeSelection, UnitTilingBlueprintOptions,
        infer_blueprint_unit,
    },
};

use super::Routine;

/// Unit single stage matmul with configurable readers (default to cyclic)
pub struct SimpleUnitAlgorithm<
    LL = SyncFullCyclicLoading<ColMajorTilingOrder>,
    RL = SyncFullCyclicLoading<RowMajorTilingOrder>,
> {
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
}

#[derive(Default, Clone, Debug)]
pub struct SimpleUnitSelectionArgs {
    pub tile_size: TileSizeSelection,
}

impl<LL, RL> Routine for SimpleUnitAlgorithm<LL, RL>
where
    LL: FullLoadingStrategy,
    RL: FullLoadingStrategy<SyncStrategy = LL::SyncStrategy>,
{
    type Strategy = SimpleUnitSelectionArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        SimpleMatmulFamily<
            UnitMatmulFamily<RegisterMatmul<Filled>, StridedStageFamily, FilledStageFamily>,
            LL,
            RL,
            UnitWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >;
    type Blueprint = TilingBlueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily>::Config;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        args: &Self::Strategy,
        dtypes: &mut MatmulElems,
    ) -> Result<TilingBlueprint, MatmulSetupError> {
        Ok(infer_blueprint_unit(
            client,
            problem,
            plane_dim,
            false,
            line_sizes,
            UnitTilingBlueprintOptions {
                tile: args.tile_size,
                stage: match args.tile_size {
                    TileSizeSelection::MinTileSize => StageScaling::Enabled(2),
                    TileSizeSelection::MaxTileSize => StageScaling::Disabled,
                },
                partition: match args.tile_size {
                    TileSizeSelection::MinTileSize => PartitionScaling::Disabled,
                    TileSizeSelection::MaxTileSize => PartitionScaling::Enabled,
                },
                swizzle: <RegisterMatmul as TileMatmulFamily>::should_swizzle(client),
            },
            dtypes,
        ))
    }

    fn select_plane_dim<R: Runtime>(client: &ComputeClient<R>) -> u32 {
        client.properties().hardware.plane_size_min
    }

    fn can_cast_stage_element() -> bool {
        RegisterMatmul::<Filled>::can_cast_stage_element()
    }
}
