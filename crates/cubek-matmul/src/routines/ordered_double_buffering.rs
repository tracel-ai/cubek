use std::marker::PhantomData;

use cubecl::Runtime;
use cubecl::client::ComputeClient;

use crate::components::batch::BatchMatmulFamily;
use crate::components::global::PlaneWriterFamily;
use crate::components::stage::{PlaneMatmulFamily, RowMajorTilingOrder};
use crate::components::tile;
use crate::components::{
    batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
    stage::{FilledStageFamily, StridedStageFamily},
};
use crate::components::{
    global::multi_stage::ordered::OrderedDoubleBufferingMatmulFamily, tile::io::Filled,
};
use crate::components::{
    global::read::sync_partial_cyclic::SyncPartialCyclicLoading, tile::io::Strided,
};
use crate::definition::{
    MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSetupError, MultiRowStrategy,
    TilingBlueprint,
};
use crate::routines::Routine;
use crate::routines::selector::{PlaneTilingBlueprintOptions, infer_blueprint_plane};

/// Plane accelerated double buffered matmul ordered on Lhs with cyclic reader on Rhs
pub struct OrderedDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

#[derive(Debug, Clone, Default)]
pub struct OrderedSelectionArgs {
    pub partition_k: Option<u32>,
    pub row_count: Option<u32>,
    pub rows_per_plane: Option<u32>,
}

impl<TMM> Routine for OrderedDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = Filled,
            OutTile = Strided,
        >,
{
    type Strategy = OrderedSelectionArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        OrderedDoubleBufferingMatmulFamily<
            PlaneMatmulFamily<TMM, StridedStageFamily, StridedStageFamily, FilledStageFamily>,
            SyncPartialCyclicLoading<RowMajorTilingOrder>,
            PlaneWriterFamily,
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
        infer_blueprint_plane::<TMM, R>(
            client,
            problem,
            plane_dim,
            dtypes,
            line_sizes,
            PlaneTilingBlueprintOptions {
                partition_k: args.partition_k,
                row_count: args.row_count,
                multi_row_strategy: args
                    .rows_per_plane
                    .map(MultiRowStrategy::Always)
                    .unwrap_or_else(|| MultiRowStrategy::Adaptive {
                        minimum_stage_count: 8,
                    }),
                swizzled: TMM::should_swizzle(client),
                ..Default::default()
            },
        )
    }

    fn can_cast_stage_element() -> bool {
        TMM::can_cast_stage_element()
    }
}
