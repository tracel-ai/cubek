use crate::components::CubeDimResource;
use crate::components::global::memory::{GlobalMemoryConfig, ViewDirection};
use crate::components::global::multi_stage::EventLoadingMode;
use crate::components::global::read::LoadingValidation as _;
use crate::components::global::{
    GlobalReaderConfig, GlobalWriterConfig, PlaneFlowConfig, SharedGlobalMatmulConfig,
};
use crate::components::global::{
    GlobalWriterFamily,
    read::{FullLoadingStrategy, PartialLoadingStrategy, sync::Synchronous},
};
use crate::components::global::{
    WriteTiling,
    multi_stage::ordered::{LL, OrderedDoubleBufferingMatmul},
};
use crate::components::stage::StridedStageFamily;
use crate::components::stage::{self, StageConfig};
use crate::components::{global::GlobalMatmulFamily, stage::FilledStageFamily};
use crate::components::{global::MaxGlobalReaderPlanes, stage::NoTilingLayout};
use crate::definition::TilingBlueprint;
use crate::definition::{MatmulElems, MatmulPrecision, MatmulProblem, MatmulSetupError};
use crate::definition::{MatmulLineSizes, MatrixLayout, StageIdent};
use cubecl::prelude::*;
use std::marker::PhantomData;

/// Ordered double buffering matmul family for any precision
pub struct OrderedDoubleBufferingMatmulFamily<
    SMM: stage::StageMatmulFamily,
    RL: PartialLoadingStrategy,
    GW: GlobalWriterFamily,
> {
    _stage_matmul: PhantomData<SMM>,
    _rhs_loading: PhantomData<RL>,
    _writer: PhantomData<GW>,
}

impl<SMM, RL, GW> GlobalMatmulFamily for OrderedDoubleBufferingMatmulFamily<SMM, RL, GW>
where
    SMM: stage::StageMatmulFamily<
            LhsStage = StridedStageFamily,
            RhsStage = StridedStageFamily,
            AccStage = FilledStageFamily,
            OutStage = GW::Stage,
        >,
    RL: PartialLoadingStrategy<Stage = StridedStageFamily, SyncStrategy = Synchronous>,
    GW: GlobalWriterFamily,
{
    type Matmul<MP: MatmulPrecision> = OrderedDoubleBufferingMatmul<
        MP,
        SMM::Matmul<
            MP,
            <LL as FullLoadingStrategy>::TilingLayout,
            RL::TilingLayout,
            NoTilingLayout,
            WriteTiling,
        >,
        RL,
        GW::Writer<MP::Acc>,
    >;
    type Config = SharedGlobalMatmulConfig<SMM::Config>;

    fn expand_config(
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let plane_dim = blueprint.plane_dim;
        let plane_flow_config = Self::cubedim_resource(blueprint, dtypes, line_sizes)?
            .as_plane_flow_config(plane_dim)?;

        let stage_config = SMM::expand_config(
            blueprint,
            plane_flow_config,
            (1, 2).into(),
            dtypes,
            line_sizes,
        )?;

        let precompute_job = blueprint.loading_precompute_strategy.into();
        let reader_mode = blueprint.reader_mode;

        let lhs_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.lhs as u32,
            check_row_bounds: blueprint.check_m_bounds,
            check_col_bounds: blueprint.check_k_bounds,
            matrix_layout: blueprint.lhs_layout,
            view_direction: ViewDirection::Col,
            dtype: dtypes.lhs_global,
        };

        let rhs_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.rhs as u32,
            check_row_bounds: blueprint.check_k_bounds,
            check_col_bounds: blueprint.check_n_bounds,
            matrix_layout: blueprint.rhs_layout,
            view_direction: ViewDirection::Row,
            dtype: dtypes.rhs_global,
        };

        let out_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.out as u32,
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: blueprint.check_m_bounds,
            check_col_bounds: blueprint.check_n_bounds,
            view_direction: ViewDirection::None,
            dtype: dtypes.acc_global,
        };

        let lhs_reader_config = GlobalReaderConfig {
            gmem_config: lhs_gmem_config,
            smem_config: stage_config.lhs_smem_config(),
            precompute_job,
            plane_dim,
            plane_flow_config,
            reader_mode,
            stage_ident: StageIdent::Lhs,
            event_loading_mode: EventLoadingMode::Ordered,
            input_load_flow: blueprint.load_flows.lhs,
        };

        let rhs_reader_config = GlobalReaderConfig {
            gmem_config: rhs_gmem_config,
            smem_config: stage_config.rhs_smem_config(),
            precompute_job,
            plane_dim,
            plane_flow_config,
            reader_mode,
            stage_ident: StageIdent::Rhs,
            event_loading_mode: EventLoadingMode::Relaxed,
            input_load_flow: blueprint.load_flows.rhs,
        };

        let writer_config = GlobalWriterConfig {
            gmem_config: out_gmem_config,
            smem_config: stage_config.out_smem_config(),
            plane_flow_partition_rule: plane_flow_config.partition_rule,
            plane_dim: blueprint.plane_dim,
        };

        Ok(SharedGlobalMatmulConfig {
            stage_config,
            num_planes: plane_flow_config.counts.total_count(),
            lhs_reader_config,
            rhs_reader_config,
            writer_config,
            must_sync_plane_after_execution: true,
        })
    }

    fn cubedim_resource(
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<CubeDimResource, MatmulSetupError> {
        let max_global_readers = blueprint.load_flows.has_specialization().then(|| {
            MaxGlobalReaderPlanes::new::<LL, RL>(
                &blueprint.tiling_scheme,
                line_sizes,
                blueprint.plane_dim,
                dtypes,
            )
        });

        let plane_dim = blueprint.plane_dim;
        let plane_flow_config = PlaneFlowConfig::new(
            blueprint.load_flows,
            max_global_readers,
            SMM::cubedim_resource(blueprint)?.num_planes(plane_dim)?,
        )?;

        Ok(CubeDimResource::Specialized(plane_flow_config))
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<(), MatmulSetupError> {
        LL::validate_with_problem(problem, dtypes, StageIdent::Lhs)?;
        RL::validate_with_problem(problem, dtypes, StageIdent::Rhs)?;

        if blueprint.tiling_scheme.partitions_per_stage_along_n() > 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Ordered does not support number of stage partitions > 1 in n",
            )));
        }

        SMM::validate_blueprint(client, blueprint, (1, 2).into(), dtypes, line_sizes)
    }
}
