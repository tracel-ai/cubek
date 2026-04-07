use crate::components::global::{
    GlobalReaderConfig, GlobalWriterConfig, PlaneFlowConfig, SharedGlobalMatmulConfig,
};
use crate::components::global::{
    GlobalWriterFamily, multi_stage::double_buffering::DoubleBufferingMatmul,
};
use crate::definition::{
    MatmulElems, MatmulProblem, MatmulSetupError, MatmulTypes, MatmulVectorSizes, StageIdent,
};
use crate::launch::RuntimeConfig;
use crate::{
    components::global::memory::{GlobalMemoryConfig, ViewDirection},
    components::global::multi_stage::EventLoadingMode,
};
use crate::{
    components::global::{WriteTiling, read::PartialLoadingStrategy},
    components::stage::StageConfig,
    components::{CubeDimResource, global::read::FullLoadingStrategy},
    components::{global::GlobalMatmulFamily, stage},
    components::{global::MaxGlobalReaderPlanes, stage::NumStages},
    definition::TilingBlueprint,
};
use cubecl::{ir::DeviceProperties, prelude::*};
use cubek_std::{MatrixLayout, tile::Strided};
use std::marker::PhantomData;

/// Double buffering matmul family for any precision
pub struct DoubleBufferingMatmulFamily<
    SMM: stage::StageMatmulFamily,
    RC: RuntimeConfig,
    LL: PartialLoadingStrategy<RC>,
    RL: PartialLoadingStrategy<RC>,
    AL: FullLoadingStrategy<RC>,
    GW: GlobalWriterFamily,
> {
    _stage_matmul: PhantomData<SMM>,
    _rc: PhantomData<RC>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
    _acc_loading: PhantomData<AL>,
    _writer: PhantomData<GW>,
}

impl<SMM, RC: RuntimeConfig, LL, RL, AL, GW> GlobalMatmulFamily<RC>
    for DoubleBufferingMatmulFamily<SMM, RC, LL, RL, AL, GW>
where
    SMM: stage::StageMatmulFamily<
            LhsStage = LL::Stage,
            RhsStage = RL::Stage,
            AccStage = Option<AL::Stage>,
            OutStage = GW::Stage,
        >,
    LL: PartialLoadingStrategy<RC, TileKind = Strided>,
    RL: PartialLoadingStrategy<RC, TileKind = Strided, SyncStrategy = LL::SyncStrategy>,
    AL: FullLoadingStrategy<RC, TileKind = Strided, SyncStrategy = LL::SyncStrategy>,
    GW: GlobalWriterFamily,
{
    type Matmul<MP: MatmulTypes> = DoubleBufferingMatmul<
        MP,
        SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout, AL::TilingLayout, WriteTiling>,
        RC,
        LL,
        RL,
        AL,
        GW::Writer<MP::Acc>,
    >;
    type Config = SharedGlobalMatmulConfig<SMM::Config>;

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let plane_dim = blueprint.plane_dim;
        let plane_flow_config = Self::cubedim_resource(blueprint, dtypes, vector_sizes)?
            .as_plane_flow_config(plane_dim)?;

        let stage_config = SMM::expand_config(
            device_props,
            blueprint,
            plane_flow_config,
            Self::num_stages(),
            dtypes,
            vector_sizes,
        )?;

        let precompute_job = blueprint.loading_precompute_strategy.into();
        let plane_dim = blueprint.plane_dim;
        let event_loading_mode = EventLoadingMode::Relaxed;
        let reader_mode = blueprint.reader_mode;

        let lhs_gmem_config = GlobalMemoryConfig {
            vector_size: vector_sizes.lhs,
            check_row_bounds: blueprint.check_m_bounds,
            check_col_bounds: blueprint.check_k_bounds,
            matrix_layout: blueprint.lhs_layout,
            view_direction: ViewDirection::Col,
            dtype: dtypes.lhs_global,
        };

        let rhs_gmem_config = GlobalMemoryConfig {
            vector_size: vector_sizes.rhs,
            check_row_bounds: blueprint.check_k_bounds,
            check_col_bounds: blueprint.check_n_bounds,
            matrix_layout: blueprint.rhs_layout,
            view_direction: ViewDirection::Row,
            dtype: dtypes.rhs_global,
        };

        let out_gmem_config = GlobalMemoryConfig {
            vector_size: vector_sizes.out,
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
            event_loading_mode,
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
            event_loading_mode,
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
            event_loading_mode,
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
            must_sync_plane_after_execution: false,
        })
    }

    fn num_stages() -> NumStages {
        (2, 2).into()
    }

    fn cubedim_resource(
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<CubeDimResource, MatmulSetupError> {
        let max_global_readers = blueprint.load_flows.has_specialization().then(|| {
            MaxGlobalReaderPlanes::new::<LL, RL>(
                &blueprint.tiling_scheme,
                vector_sizes,
                blueprint.plane_dim,
                dtypes,
            )
        });

        let plane_flow_config = PlaneFlowConfig::new(
            blueprint.load_flows,
            max_global_readers,
            SMM::cubedim_resource(blueprint)?.num_planes(blueprint.plane_dim)?,
        )?;

        Ok(CubeDimResource::Specialized(plane_flow_config))
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        LL::validate_with_problem(problem, dtypes, StageIdent::Lhs)?;
        RL::validate_with_problem(problem, dtypes, StageIdent::Rhs)?;

        SMM::validate_blueprint(client, blueprint, dtypes, vector_sizes)
    }
}
