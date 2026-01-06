use crate::components::CubeDimResource;
use crate::definition::{
    InvalidConfigError, MatmulElems, MatmulLineSizes, MatmulPrecision, MatmulProblem,
    MatmulSetupError, MatrixLayout, StageIdent,
};
use crate::{
    components::{
        global::{
            GlobalReaderConfig, GlobalWriterConfig, GlobalWriterFamily, SharedGlobalMatmulConfig,
            SpecializationTensorConfig, WriteTiling,
            memory::{GlobalMemoryConfig, ViewDirection},
            multi_stage::EventLoadingMode,
            read::FullLoadingStrategy,
            single_stage::simple::matmul::SimpleMatmul,
        },
        stage::{FilledStageFamily, NoTilingLayout, StageConfig, StridedStageFamily},
    },
    definition::TilingBlueprint,
};
use cubecl::prelude::*;
use std::marker::PhantomData;

use crate::components::{global::GlobalMatmulFamily, stage};

/// Simple matmul family for any precision
pub struct SimpleMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: FullLoadingStrategy,
    RL: FullLoadingStrategy,
    GW: GlobalWriterFamily,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
    _writer: PhantomData<GW>,
}

impl<SMM, LL, RL, GW> GlobalMatmulFamily for SimpleMatmulFamily<SMM, LL, RL, GW>
where
    SMM: stage::StageMatmulFamily<
            LhsStage = StridedStageFamily,
            RhsStage = StridedStageFamily,
            AccStage = FilledStageFamily,
            OutStage = GW::Stage,
        >,
    LL: FullLoadingStrategy,
    RL: FullLoadingStrategy<SyncStrategy = LL::SyncStrategy>,
    GW: GlobalWriterFamily,
{
    type Matmul<MP: MatmulPrecision> = SimpleMatmul<
        MP,
        SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout, NoTilingLayout, WriteTiling>,
        LL,
        RL,
        GW::Writer<MP::Acc>,
    >;
    type Config = SharedGlobalMatmulConfig<SMM::Config>;

    fn expand_config(
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let stage_config = SMM::expand_config(&blueprint, None, (1, 1).into(), dtypes, line_sizes)?;

        let plane_role_config = stage_config.plane_role_config();
        let precompute_job = blueprint.loading_precompute_strategy.into();
        let reader_mode = blueprint.reader_mode;
        let plane_dim = blueprint.plane_dim;
        let specialization_tensor_config = SpecializationTensorConfig::MainFlowOnly;

        // Not used in simple
        let event_loading_mode = EventLoadingMode::Relaxed;

        let lhs_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.lhs as u32,
            check_row_bounds: blueprint.check_m_bounds,
            check_col_bounds: blueprint.check_k_bounds,
            matrix_layout: blueprint.lhs_layout,
            view_direction: ViewDirection::Col,
        };

        let rhs_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.rhs as u32,
            check_row_bounds: blueprint.check_k_bounds,
            check_col_bounds: blueprint.check_n_bounds,
            matrix_layout: blueprint.rhs_layout,
            view_direction: ViewDirection::Row,
        };

        let out_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.out as u32,
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: blueprint.check_m_bounds,
            check_col_bounds: blueprint.check_n_bounds,
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
            specialization_tensor_config,
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
            specialization_tensor_config,
        };

        let writer_config = GlobalWriterConfig {
            gmem_config: out_gmem_config,
            smem_config: stage_config.out_smem_config(),
            role_rule_config: plane_role_config.rule,
            plane_dim,
        };

        Ok(SharedGlobalMatmulConfig {
            stage_config,
            num_planes: Self::cubedim_resource(blueprint)?.num_planes(blueprint.plane_dim)?,
            lhs_reader_config,
            rhs_reader_config,
            writer_config,
            must_sync_plane_after_execution: false,
        })
    }

    fn cubedim_resource(
        blueprint: &TilingBlueprint,
    ) -> Result<CubeDimResource, InvalidConfigError> {
        let resources = if !blueprint.load_specialization_config.has_specialization() {
            SMM::cubedim_resource(blueprint)
        } else {
            return Err(Box::new(
                "Error: Specialization is unavailable for simple matmul.",
            ));
        }?;

        Ok(resources)
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
        SMM::validate_blueprint(client, blueprint, (1, 1).into(), dtypes, line_sizes)
    }
}
