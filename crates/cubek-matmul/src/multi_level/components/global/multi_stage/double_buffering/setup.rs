use crate::{
    definition::{MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes, StageIdent},
    multi_level::{
        args::RuntimeConfig,
        components::{
            CubeDimResource,
            global::{
                GlobalMatmulFamily, GlobalReaderConfig, GlobalWriterConfig, GlobalWriterFamily,
                MaxGlobalReaderPlanes, SharedGlobalMatmulConfig, make_plane_flow_config,
                memory::{GlobalMemoryConfig, ViewDirection},
                multi_stage::{EventLoadingMode, double_buffering::DoubleBufferingMatmul},
                read::{FullLoadingStrategy, PartialLoadingStrategy},
            },
            stage::{NumStages, StagePartitioner},
        },
        definition::{BatchMatmulBlueprint, MatmulTypes},
    },
};
use cubecl::{ir::DeviceProperties, prelude::*};
use cubek_std::MatrixLayout;
use std::marker::PhantomData;

/// Double buffering matmul family for any precision
pub struct DoubleBufferingMatmulFamily<
    SP: StagePartitioner,
    RC: RuntimeConfig,
    LL: PartialLoadingStrategy<RC>,
    RL: PartialLoadingStrategy<RC>,
    AL: FullLoadingStrategy<RC>,
    GW: GlobalWriterFamily,
> {
    _sp: PhantomData<SP>,
    _rc: PhantomData<RC>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
    _acc_loading: PhantomData<AL>,
    _writer: PhantomData<GW>,
}

impl<SP, RC: RuntimeConfig, LL, RL, AL, GW> GlobalMatmulFamily<RC>
    for DoubleBufferingMatmulFamily<SP, RC, LL, RL, AL, GW>
where
    SP: StagePartitioner,
    LL: PartialLoadingStrategy<RC>,
    RL: PartialLoadingStrategy<RC, SyncStrategy = LL::SyncStrategy>,
    AL: FullLoadingStrategy<RC, SyncStrategy = LL::SyncStrategy>,
    GW: GlobalWriterFamily,
{
    type Matmul<MP: MatmulTypes> = DoubleBufferingMatmul<MP, SP, RC, LL, RL, AL, GW>;
    type Config = SharedGlobalMatmulConfig;

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &BatchMatmulBlueprint,
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

        // Checked here — where a violation is a recoverable setup error —
        // rather than only at the kernel's comptime re-check, whose failure
        // surfaces as an asynchronous compile error no caller can recover
        // from. See the same block in the ordered family's `expand_config`.
        LL::validate_with_config(device_props, &lhs_reader_config)?;
        RL::validate_with_config(device_props, &rhs_reader_config)?;

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
        blueprint: &BatchMatmulBlueprint,
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

        let plane_flow_config = make_plane_flow_config(
            blueprint.load_flows,
            max_global_readers,
            SP::KIND
                .cubedim_resource(blueprint)?
                .num_planes(blueprint.plane_dim)?,
        )?;

        Ok(CubeDimResource::Specialized(plane_flow_config))
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &BatchMatmulBlueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        LL::validate_with_problem(problem, dtypes, StageIdent::Lhs)?;
        RL::validate_with_problem(problem, dtypes, StageIdent::Rhs)?;

        SP::KIND.validate_blueprint(client, blueprint, dtypes, vector_sizes)
    }
}
