use std::marker::PhantomData;

use cubecl::Runtime;
use cubecl::client::ComputeClient;

use crate::components::batch::BatchMatmulFamily;
use crate::components::global::read::{
    async_partial_cyclic::AsyncPartialCyclicLoading,
    async_partial_strided::AsyncPartialStridedLoading, async_partial_tma::AsyncPartialTmaLoading,
    sync_partial_cyclic::SyncPartialCyclicLoading,
};
use crate::components::global::{
    PlaneWriterFamily, read::sync_partial_tilewise::SyncPartialTilewiseLoading,
};
use crate::components::stage::{ColMajorTilingOrder, PlaneMatmulFamily, RowMajorTilingOrder};
use crate::components::tile;
use crate::components::{
    batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
    tile::io::{Filled, Strided},
};
use crate::components::{
    global::multi_stage::double_buffering::DoubleBufferingMatmulFamily,
    stage::{FilledStageFamily, StridedStageFamily},
};
use crate::definition::{
    MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSetupError, MultiRowStrategy,
    TilingBlueprint,
};
use crate::routines::Routine;
use crate::routines::base;
use crate::routines::selector::{PlaneTilingBlueprintOptions, infer_blueprint_plane};

/// Plane accelerated double buffered matmul with cyclic readers
pub struct CyclicDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

/// Plane accelerated double buffered matmul with cyclic readers
pub struct AsyncCyclicDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

/// Plane accelerated double buffered matmul with tilewise readers
pub struct TilewiseDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

/// Plane accelerated double buffered matmul with tilewise reader on Lhs and cyclic on Rhs
pub struct HybridDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

/// Plane accelerated double buffered matmul with TMA readers
pub struct TmaDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

/// Plane accelerated double buffered matmul with cyclic readers
pub struct AsyncStridedDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct DoubleBufferingArgs {
    pub specialized: bool,
}

impl<TMM> base::Routine for CyclicDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = Filled,
            OutTile = Strided,
        >,
{
    type Strategy = DoubleBufferingArgs;

    type BatchMatmul = PartitionedBatchMatmulFamily<
        DoubleBufferingMatmulFamily<
            PlaneMatmulFamily<TMM, StridedStageFamily, StridedStageFamily, FilledStageFamily>,
            SyncPartialCyclicLoading<RowMajorTilingOrder>,
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
                specialized: args.specialized,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                swizzled: TMM::should_swizzle(client),
                ..Default::default()
            },
        )
    }

    fn can_cast_stage_element() -> bool {
        TMM::can_cast_stage_element()
    }
}

impl<TMM> base::Routine for AsyncCyclicDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = Filled,
            OutTile = Strided,
        >,
{
    type Strategy = DoubleBufferingArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        DoubleBufferingMatmulFamily<
            PlaneMatmulFamily<TMM, StridedStageFamily, StridedStageFamily, FilledStageFamily>,
            AsyncPartialCyclicLoading<RowMajorTilingOrder>,
            AsyncPartialCyclicLoading<RowMajorTilingOrder>,
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
                specialized: args.specialized,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                swizzled: TMM::should_swizzle(client),
                ..Default::default()
            },
        )
    }

    fn can_cast_stage_element() -> bool {
        TMM::can_cast_stage_element()
    }
}

impl<TMM> Routine for TilewiseDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = Filled,
            OutTile = Strided,
        >,
{
    type Strategy = DoubleBufferingArgs;

    type BatchMatmul = PartitionedBatchMatmulFamily<
        DoubleBufferingMatmulFamily<
            PlaneMatmulFamily<TMM, StridedStageFamily, StridedStageFamily, FilledStageFamily>,
            // Other tiling orders are not supported
            SyncPartialTilewiseLoading<RowMajorTilingOrder>,
            SyncPartialTilewiseLoading<ColMajorTilingOrder>,
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
                specialized: args.specialized,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                swizzled: TMM::should_swizzle(client),
                ..Default::default()
            },
        )
    }

    fn can_cast_stage_element() -> bool {
        TMM::can_cast_stage_element()
    }
}

impl<TMM> base::Routine for HybridDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = Filled,
            OutTile = Strided,
        >,
{
    type Strategy = DoubleBufferingArgs;

    type BatchMatmul = PartitionedBatchMatmulFamily<
        DoubleBufferingMatmulFamily<
            PlaneMatmulFamily<TMM, StridedStageFamily, StridedStageFamily, FilledStageFamily>,
            SyncPartialTilewiseLoading<RowMajorTilingOrder>,
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
                specialized: args.specialized,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                swizzled: TMM::should_swizzle(client),
                ..Default::default()
            },
        )
    }

    fn can_cast_stage_element() -> bool {
        TMM::can_cast_stage_element()
    }
}

impl<TMM> base::Routine for TmaDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = Filled,
            OutTile = Strided,
        >,
{
    type Strategy = DoubleBufferingArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        DoubleBufferingMatmulFamily<
            PlaneMatmulFamily<TMM, StridedStageFamily, StridedStageFamily, FilledStageFamily>,
            AsyncPartialTmaLoading,
            AsyncPartialTmaLoading,
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
                specialized: args.specialized,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                swizzled: TMM::should_swizzle(client),
                ..Default::default()
            },
        )
    }

    fn can_cast_stage_element() -> bool {
        TMM::can_cast_stage_element()
    }
}

impl<TMM> base::Routine for AsyncStridedDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = Filled,
            OutTile = Strided,
        >,
{
    type Strategy = DoubleBufferingArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        DoubleBufferingMatmulFamily<
            PlaneMatmulFamily<TMM, StridedStageFamily, StridedStageFamily, FilledStageFamily>,
            AsyncPartialStridedLoading,
            AsyncPartialStridedLoading,
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
                specialized: args.specialized,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                swizzled: TMM::should_swizzle(client),
                ..Default::default()
            },
        )
    }

    fn can_cast_stage_element() -> bool {
        TMM::can_cast_stage_element()
    }
}
