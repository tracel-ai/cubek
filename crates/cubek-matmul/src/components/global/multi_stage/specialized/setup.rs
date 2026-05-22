use crate::components::CubeDimResource;
use crate::components::global::{
    GlobalReaderConfig, GlobalWriterConfig, SharedGlobalMatmulConfig, make_plane_flow_config,
};
use crate::components::global::{
    memory::{GlobalMemoryConfig, ViewDirection},
    read::AsyncPartialLoadingStrategy,
};
use crate::{
    components::global::GlobalMatmulFamily,
    components::global::{multi_stage::EventLoadingMode, read::FullLoadingStrategy},
    components::stage::StagePartitioner,
    components::{global::MaxGlobalReaderPlanes, stage::NumStages},
    definition::MatmulVectorSizes,
    definition::StageIdent,
    definition::TilingBlueprint,
    definition::{MatmulElems, MatmulSetupError},
    definition::{MatmulProblem, MatmulTypes},
    launch::RuntimeConfig,
};
use crate::{
    components::global::{GlobalWriterFamily, multi_stage::specialized::SpecializedMatmul},
    components::global::{InputLoadFlow, LoadFlows},
};
use cubecl::{ir::DeviceProperties, prelude::*};
use cubek_std::MatrixLayout;
use std::marker::PhantomData;

/// Double buffering matmul family for any precision
pub struct SpecializedMatmulFamily<
    SP: StagePartitioner,
    RC: RuntimeConfig,
    L: AsyncPartialLoadingStrategy<RC>,
    AL: FullLoadingStrategy<RC>,
    GW: GlobalWriterFamily,
> {
    _sp: PhantomData<SP>,
    _rc: PhantomData<RC>,
    _loading: PhantomData<L>,
    _acc_loading: PhantomData<AL>,
    _writer: PhantomData<GW>,
}

impl<SP, RC, L, AL, GW> GlobalMatmulFamily<RC> for SpecializedMatmulFamily<SP, RC, L, AL, GW>
where
    SP: StagePartitioner,
    RC: RuntimeConfig,
    L: AsyncPartialLoadingStrategy<RC>,
    AL: FullLoadingStrategy<RC>,
    GW: GlobalWriterFamily,
{
    type Matmul<MP: MatmulTypes> = SpecializedMatmul<MP, SP, RC, L, AL, GW::Writer<MP::Acc>>;
    type Config = SharedGlobalMatmulConfig;

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let plane_dim = blueprint.plane_dim;
        let plane_flow_config =
            Self::cubedim_resource(blueprint, dtypes, vector_sizes)?.as_specialized(plane_dim)?;

        let stage_config = SP::KIND.expand_stage_matmul(
            device_props,
            blueprint,
            plane_flow_config,
            Self::num_stages(),
            dtypes,
            vector_sizes,
        )?;

        let precompute_job = blueprint.loading_precompute_strategy.into();
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
        let mut blueprint = blueprint.clone();
        blueprint.load_flows = LoadFlows {
            lhs: InputLoadFlow::LoadOnly,
            rhs: InputLoadFlow::LoadOnly,
        };

        let max_global_readers = MaxGlobalReaderPlanes::new::<L, L>(
            &blueprint.tiling_scheme,
            vector_sizes,
            blueprint.plane_dim,
            dtypes,
        );

        let plane_dim = blueprint.plane_dim;
        let plane_flow_config = make_plane_flow_config(
            blueprint.load_flows,
            Some(max_global_readers),
            SP::KIND
                .cubedim_resource(&blueprint)?
                .num_planes(plane_dim)?,
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
        L::validate_with_problem(problem, dtypes, StageIdent::Lhs)?;
        L::validate_with_problem(problem, dtypes, StageIdent::Rhs)?;
        SP::KIND.validate_blueprint(client, blueprint, dtypes, vector_sizes)
    }
}
