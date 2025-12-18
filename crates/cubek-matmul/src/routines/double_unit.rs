use std::fmt::Display;

use cubecl::{Runtime, client::ComputeClient};

use crate::{
    components::{
        batch::{BatchMatmulFamily, PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            UnitWriterFamily, multi_stage::double_buffering::DoubleBufferingMatmulFamily,
            read::sync_partial_cyclic::SyncPartialCyclicLoading,
        },
        stage::{FilledStageFamily, RowMajorTilingOrder, StridedStageFamily, UnitMatmulFamily},
        tile::{TileMatmulFamily, io::Filled, register::RegisterMatmul},
    },
    definition::{MatmulElems, MatmulProblem, MatmulSetupError, TilingBlueprint},
    routines::{
        BlueprintStrategy, DeviceSettings, LaunchInfo, Routine,
        selector::{TileSizeSelection, UnitTilingBlueprintOptions, infer_blueprint_unit},
    },
};

/// Unit double buffered matmul with cyclic readers
pub struct DoubleUnitAlgorithm {}

#[derive(Default, Clone, Debug)]
pub struct DoubleUnitSelectionArgs {
    pub tile_size: TileSizeSelection,
}

impl Display for DoubleUnitSelectionArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_{}", self.tile_size)
    }
}

impl Routine for DoubleUnitAlgorithm {
    type Strategy = DoubleUnitSelectionArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        DoubleBufferingMatmulFamily<
            UnitMatmulFamily<RegisterMatmul<Filled>, StridedStageFamily, FilledStageFamily>,
            SyncPartialCyclicLoading<RowMajorTilingOrder>,
            SyncPartialCyclicLoading<RowMajorTilingOrder>,
            UnitWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >;
    type Blueprint = TilingBlueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily>::Config;

    fn prepare<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<Self>,
    ) -> Result<LaunchInfo<TilingBlueprint>, MatmulSetupError> {
        match strategy {
            BlueprintStrategy::Forced(blueprint) => Ok(LaunchInfo {
                blueprint: blueprint.clone(),
                dtypes: MatmulElems::from_globals(&problem.global_dtypes),
            }),
            BlueprintStrategy::Inferred(strategy) => Ok(infer_blueprint_unit(
                &device_settings.client,
                problem,
                device_settings.plane_dim,
                true,
                &device_settings.line_sizes,
                UnitTilingBlueprintOptions {
                    tile: strategy.tile_size,
                    ..Default::default()
                },
                &problem.global_dtypes,
            )),
        }
    }

    fn select_plane_dim<R: Runtime>(client: &ComputeClient<R>) -> u32 {
        client.properties().hardware.plane_size_min
    }

    fn can_cast_stage_element() -> bool {
        RegisterMatmul::<Filled>::can_cast_stage_element()
    }
}
