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
    definition::{MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSelection, MatmulSetupError},
    routines::{
        Routine,
        selector::{TileSizeSelection, UnitMatmulSelectionOptions, unit_matmul_selection},
    },
};

/// Unit double buffered matmul with cyclic readers
pub struct DoubleUnitAlgorithm {}

#[derive(Default, Clone, Debug)]
pub struct DoubleUnitSelectionArgs {
    pub tile_size: TileSizeSelection,
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
    type Blueprint = MatmulSelection;
    type Config = <Self::BatchMatmul as BatchMatmulFamily>::Config;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        args: &Self::Strategy,
        dtypes: &mut MatmulElems,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        Ok(unit_matmul_selection(
            client,
            problem,
            plane_dim,
            true,
            line_sizes,
            UnitMatmulSelectionOptions {
                tile: args.tile_size,
                ..Default::default()
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
