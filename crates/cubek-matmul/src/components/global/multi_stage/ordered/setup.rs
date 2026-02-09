use crate::components::global::MaxGlobalReaderPlanes;
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
use crate::components::{global::GlobalMatmulFamily, stage::NumStages};
use crate::components::{
    global::memory::{GlobalMemoryConfig, ViewDirection},
    tile::io::Strided,
};
use crate::definition::TilingBlueprint;
use crate::definition::{MatmulElems, MatmulPrecision, MatmulProblem, MatmulSetupError};
use crate::definition::{MatmulLineSizes, MatrixLayout, StageIdent};
use crate::{components::CubeDimResource, launch::RuntimeConfig};
use cubecl::{ir::DeviceProperties, prelude::*};
use std::marker::PhantomData;

/// Ordered double buffering matmul family for any precision
pub struct OrderedDoubleBufferingMatmulFamily<
    SMM: stage::StageMatmulFamily,
    RC: RuntimeConfig,
    RL: PartialLoadingStrategy<RC>,
    AL: FullLoadingStrategy<RC>,
    GW: GlobalWriterFamily,
> {
    _stage_matmul: PhantomData<SMM>,
    _rc: PhantomData<RC>,
    _rhs_loading: PhantomData<RL>,
    _acc_loading: PhantomData<AL>,
    _writer: PhantomData<GW>,
}

impl<SMM, RC, RL, AL, GW> GlobalMatmulFamily<RC>
    for OrderedDoubleBufferingMatmulFamily<SMM, RC, RL, AL, GW>
where
    SMM: stage::StageMatmulFamily<
            LhsStage = StridedStageFamily,
            RhsStage = RL::Stage,
            AccStage = Option<AL::Stage>,
            OutStage = GW::Stage,
        >,
    RC: RuntimeConfig,
    RL: PartialLoadingStrategy<RC, TileKind = Strided, SyncStrategy = Synchronous>,
    AL: FullLoadingStrategy<RC, TileKind = Strided, SyncStrategy = Synchronous>,
    GW: GlobalWriterFamily,
{
    type Matmul<MP: MatmulPrecision> = OrderedDoubleBufferingMatmul<
        MP,
        SMM::Matmul<
            MP,
            <LL as FullLoadingStrategy<RC>>::TilingLayout,
            RL::TilingLayout,
            AL::TilingLayout,
            WriteTiling,
        >,
        RC,
        RL,
        AL,
        GW::Writer<MP::Acc>,
    >;
    type Config = SharedGlobalMatmulConfig<SMM::Config>;

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let plane_dim = blueprint.plane_dim;
        let plane_flow_config = Self::cubedim_resource(blueprint, dtypes, line_sizes)?
            .as_plane_flow_config(plane_dim)?;

        let stage_config = SMM::expand_config(
            device_props,
            blueprint,
            plane_flow_config,
            Self::num_stages(),
            dtypes,
            line_sizes,
        )?;

        let precompute_job = blueprint.loading_precompute_strategy.into();
        let reader_mode = blueprint.reader_mode;

        let lhs_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.lhs,
            check_row_bounds: blueprint.check_m_bounds,
            check_col_bounds: blueprint.check_k_bounds,
            matrix_layout: blueprint.lhs_layout,
            view_direction: ViewDirection::Col,
            dtype: dtypes.lhs_global,
        };

        let rhs_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.rhs,
            check_row_bounds: blueprint.check_k_bounds,
            check_col_bounds: blueprint.check_n_bounds,
            matrix_layout: blueprint.rhs_layout,
            view_direction: ViewDirection::Row,
            dtype: dtypes.rhs_global,
        };

        let out_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.out,
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

        let acc_reader_config = GlobalReaderConfig {
            gmem_config: out_gmem_config,
            smem_config: stage_config.acc_smem_config(),
            precompute_job,
            plane_dim,
            plane_flow_config,
            reader_mode,
            stage_ident: StageIdent::Acc,
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
            acc_reader_config,
            writer_config,
            must_sync_plane_after_execution: true,
        })
    }

    fn num_stages() -> NumStages {
        (1, 2).into()
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

        SMM::validate_blueprint(client, blueprint, dtypes, line_sizes)
    }
}
