use cubecl::{Runtime, client::ComputeClient};
use cubek_matmul::components::{
    global::{
        GlobalReaderConfig, GlobalWriterConfig, MatmulPlaneCounts, PartitionedStageFamily,
        SharedGlobalMatmulConfig, WriteTiling, cube_dim_validation,
        memory::{GlobalMemoryConfig, ViewDirection},
        multi_stage::EventLoadingMode,
        read::{LoadingValidation, sync_full_cyclic::SyncFullCyclicLoading},
    },
    stage::{
        ColMajorTilingOrder, ContiguousTilingLayout, RowMajorTilingOrder, StageConfig,
        StageMatmulFamily, StridedStageFamily,
    },
};
use cubek_matmul::definition::{
    AvailableLineSizes, MatmulElems, MatmulLineSizes, MatmulPrecision, MatmulProblem,
    MatmulSetupError, MatrixLayout, StageIdent, TilingBlueprint,
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

    fn expand_config<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        selection: &TilingBlueprint,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError> {
        let stage_config = SMM::expand_config(
            client,
            &problem.as_matmul_problem(),
            selection,
            line_sizes,
            (1, 1).into(),
            None,
            dtypes,
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

        let plane_role_config = stage_config.plane_role_config();
        let plane_counts = MatmulPlaneCounts::new(
            selection.load_specialization_config,
            plane_role_config.plane_roles,
        );

        let precompute_job = selection.loading_precompute_strategy.into();
        let plane_dim = selection.plane_dim;
        let event_loading_mode = EventLoadingMode::Relaxed;
        let reader_mode = selection.reader_mode;

        let lhs_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.lhs as u32,
            check_row_bounds: check_m_bounds,
            check_col_bounds: check_k_bounds,
            matrix_layout: problem.lhs_layout,
            view_direction: ViewDirection::Col,
        };

        let rhs_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.rhs as u32,
            check_row_bounds: check_k_bounds,
            check_col_bounds: check_n_bounds,
            matrix_layout: problem.rhs_layout,
            view_direction: ViewDirection::Row,
        };

        let out_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.out as u32,
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: check_m_bounds,
            check_col_bounds: check_n_bounds,
            view_direction: ViewDirection::None,
        };

        let lhs_reader_config = GlobalReaderConfig {
            gmem_config: lhs_gmem_config,
            smem_config: stage_config.lhs_smem_config(),
            precompute_job,
            plane_dim,
            plane_role_config,
            reader_mode,
            stage_ident: StageIdent::Lhs,
            event_loading_mode,
            specialization_tensor_config: selection.load_specialization_config.lhs,
        };

        let rhs_reader_config = GlobalReaderConfig {
            gmem_config: rhs_gmem_config,
            smem_config: stage_config.rhs_smem_config(),
            precompute_job,
            plane_dim,
            plane_role_config,
            reader_mode,
            stage_ident: StageIdent::Rhs,
            event_loading_mode,
            specialization_tensor_config: selection.load_specialization_config.rhs,
        };

        let writer_config = GlobalWriterConfig {
            gmem_config: out_gmem_config,
            smem_config: stage_config.out_smem_config(),
            role_rule_config: plane_role_config.rule,
            plane_dim: selection.plane_dim,
        };

        let matmul_config = SharedGlobalMatmulConfig {
            stage_config,
            num_planes: plane_counts.total,
            lhs_reader_config,
            rhs_reader_config,
            writer_config,
            must_sync_plane_after_execution: false,
        };

        validate::<LL, LR, _, _>(matmul_config, client, &problem.as_matmul_problem(), dtypes)?;

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
}

fn validate<LL: LoadingValidation, RL: LoadingValidation, S: StageConfig, R: Runtime>(
    config: SharedGlobalMatmulConfig<S>,
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    dtypes: &MatmulElems,
) -> Result<SharedGlobalMatmulConfig<S>, MatmulSetupError> {
    LL::check(client, problem, &config.lhs_reader_config, dtypes)?;
    RL::check(client, problem, &config.rhs_reader_config, dtypes)?;
    cube_dim_validation(config)?;
    Ok(config)
}
