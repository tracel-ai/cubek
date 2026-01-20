use cubecl::{Runtime, client::ComputeClient, ir::DeviceProperties};
use cubek_matmul::components::{
    global::{
        GlobalConfig, GlobalReaderConfig, GlobalWriterConfig, MatmulPlaneCounts,
        PartitionedStageFamily, PlaneFlowConfig, SharedGlobalMatmulConfig, WriteTiling,
        cube_dim_validation,
        memory::{GlobalMemoryConfig, ViewDirection},
        multi_stage::EventLoadingMode,
        read::sync_full_cyclic::SyncFullCyclicLoading,
    },
    stage::{
        ColMajorTilingOrder, ContiguousTilingLayout, RowMajorTilingOrder, StageConfig,
        StageMatmulFamily, StridedStageFamily,
    },
};
use cubek_matmul::definition::{
    AvailableLineSizes, MatmulElems, MatmulLineSizes, MatmulPrecision, MatmulSetupError,
    MatrixLayout, StageIdent, TilingBlueprint,
};
use std::marker::PhantomData;

use crate::components::{
    ConvolutionConfig, ConvolutionOperation, ConvolutionProblem,
    global::{
        GlobalConvolutionFamily, read::full_reader::FullLoadingStrategy,
        single_stage::simple::SimpleConvolution,
    },
    stage::reader::BiasTilingLayout,
};

pub type ConvTilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;

pub struct SimpleConvolutionFamily<
    SMM: StageMatmulFamily,
    LL: FullLoadingStrategy = SyncFullCyclicLoading<RowMajorTilingOrder>,
    LR: FullLoadingStrategy = SyncFullCyclicLoading<ColMajorTilingOrder>,
> {
    _smm: PhantomData<SMM>,
    _loaders: PhantomData<(LL, LR)>,
}

impl<SMM, LL, LR> GlobalConvolutionFamily for SimpleConvolutionFamily<SMM, LL, LR>
where
    SMM: StageMatmulFamily<
            LhsStage = StridedStageFamily,
            RhsStage = StridedStageFamily,
            AccStage = Option<StridedStageFamily>,
            OutStage = PartitionedStageFamily,
        >,
    LL: FullLoadingStrategy,
    LR: FullLoadingStrategy<SyncStrategy = LL::SyncStrategy>,
{
    type Convolution<MP: MatmulPrecision> = SimpleConvolution<
        MP,
        SMM::Matmul<MP, LL::TilingLayout, LR::TilingLayout, BiasTilingLayout, WriteTiling>,
        LL,
        LR,
    >;
    type Config = ConvolutionConfig<SharedGlobalMatmulConfig<SMM::Config>>;

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }

    fn expand_config(
        device_props: &DeviceProperties,
        problem: &ConvolutionProblem,
        blueprint: &TilingBlueprint,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError> {
        let num_main_flow_planes =
            SMM::cubedim_resource(blueprint)?.num_planes(blueprint.plane_dim)?;
        let plane_flow_config =
            PlaneFlowConfig::new(blueprint.load_flows, None, num_main_flow_planes)?;
        let plane_counts = MatmulPlaneCounts::new(blueprint.load_flows, plane_flow_config.counts);

        let stage_config = SMM::expand_config(
            device_props,
            blueprint,
            plane_flow_config,
            (1, 1).into(),
            dtypes,
            line_sizes,
        )?;

        let stage_size_m = stage_config.elements_in_stage_m() as usize;
        let stage_size_n = stage_config.elements_in_stage_n() as usize;
        let stage_size_k = stage_config.elements_in_stage_k() as usize;

        // k is tricky and is handled specially by different loaders so always check for now.
        // m and n don't have padding so checks work as normal.
        let check_m_bounds = !problem.m.is_multiple_of(stage_size_m);
        let check_n_bounds = match problem.operation {
            ConvolutionOperation::Forward
            | ConvolutionOperation::ForwardTransposed
            | ConvolutionOperation::BackwardData => !problem.n.is_multiple_of(stage_size_n),
            ConvolutionOperation::BackwardWeight => true,
        };
        let check_k_bounds = match problem.operation {
            ConvolutionOperation::BackwardWeight => !problem.k.is_multiple_of(stage_size_k),
            ConvolutionOperation::Forward
            | ConvolutionOperation::ForwardTransposed
            | ConvolutionOperation::BackwardData => true,
        };

        let precompute_job = blueprint.loading_precompute_strategy.into();
        let plane_dim = blueprint.plane_dim;
        let event_loading_mode = EventLoadingMode::Relaxed;
        let reader_mode = blueprint.reader_mode;

        let lhs_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.lhs,
            check_row_bounds: check_m_bounds,
            check_col_bounds: check_k_bounds,
            matrix_layout: problem.lhs_layout,
            view_direction: ViewDirection::Col,
            dtype: dtypes.lhs_global,
        };

        let rhs_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.rhs,
            check_row_bounds: check_k_bounds,
            check_col_bounds: check_n_bounds,
            matrix_layout: problem.rhs_layout,
            view_direction: ViewDirection::Row,
            dtype: dtypes.rhs_global,
        };

        let out_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.out,
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: check_m_bounds,
            check_col_bounds: check_n_bounds,
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

        let writer_config = GlobalWriterConfig {
            gmem_config: out_gmem_config,
            smem_config: stage_config.out_smem_config(),
            plane_dim: blueprint.plane_dim,
            plane_flow_partition_rule: plane_flow_config.partition_rule,
        };

        let matmul_config = SharedGlobalMatmulConfig {
            stage_config,
            num_planes: plane_counts.total,
            lhs_reader_config,
            rhs_reader_config,
            writer_config,
            must_sync_plane_after_execution: false,
        };

        cube_dim_validation(matmul_config.cube_dim())?;

        ConvolutionConfig::new(
            matmul_config,
            &problem.kernel_size,
            &problem.stride,
            &problem.dilation,
            &problem.padding,
            problem.dimensionality,
            problem.operation,
        )
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        problem: &ConvolutionProblem,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<(), MatmulSetupError> {
        let problem = problem.as_matmul_problem();
        LL::validate_with_problem(&problem, dtypes, StageIdent::Lhs)?;
        LR::validate_with_problem(&problem, dtypes, StageIdent::Rhs)?;

        if blueprint.tiling_scheme.partitions_per_stage_along_n() > 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Ordered does not support number of stage partitions > 1 in n",
            )));
        }

        SMM::validate_blueprint(client, blueprint, (1, 2).into(), dtypes, line_sizes)
    }
}
