use crate::components::CubeDimResource;
use crate::components::global::memory::{GlobalMemoryConfig, ViewDirection};
use crate::components::global::multi_stage::EventLoadingMode;
use crate::components::global::read::LoadingValidation;
use crate::components::global::{
    GlobalReaderConfig, GlobalWriterConfig, MatmulPlaneCounts, SharedGlobalMatmulConfig,
};
use crate::components::global::{
    GlobalWriterFamily, multi_stage::double_buffering::DoubleBufferingMatmul,
};
use crate::components::global::{WriteTiling, read::PartialLoadingStrategy};
use crate::components::stage::StageConfig;
use crate::components::stage::StridedStageFamily;
use crate::components::{global::GlobalMatmulFamily, stage, stage::FilledStageFamily};
use crate::components::{global::MaxGlobalReaderPlanes, stage::NoTilingLayout};
use crate::definition::{InvalidConfigError, TilingBlueprint};
use crate::definition::{
    MatmulElems, MatmulLineSizes, MatmulPrecision, MatmulProblem, MatmulSetupError, MatrixLayout,
    StageIdent,
};
use cubecl::prelude::*;
use std::marker::PhantomData;

/// Double buffering matmul family for any precision
pub struct DoubleBufferingMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: PartialLoadingStrategy,
    RL: PartialLoadingStrategy,
    GW: GlobalWriterFamily,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
    _writer: PhantomData<GW>,
}

impl<SMM, LL, RL, GW> GlobalMatmulFamily for DoubleBufferingMatmulFamily<SMM, LL, RL, GW>
where
    SMM: stage::StageMatmulFamily<
            LhsStage = StridedStageFamily,
            RhsStage = StridedStageFamily,
            AccStage = FilledStageFamily,
            OutStage = GW::Stage,
        >,
    LL: PartialLoadingStrategy<Stage = StridedStageFamily>,
    RL: PartialLoadingStrategy<Stage = StridedStageFamily, SyncStrategy = LL::SyncStrategy>,
    GW: GlobalWriterFamily,
{
    type Matmul<MP: MatmulPrecision> = DoubleBufferingMatmul<
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
        let max_global_readers = blueprint
            .load_specialization_config
            .has_specialization()
            .then(|| {
                MaxGlobalReaderPlanes::new::<LL, RL>(
                    &blueprint.tiling_scheme,
                    &line_sizes,
                    blueprint.plane_dim,
                    dtypes,
                )
            });

        let stage_config = SMM::expand_config(
            blueprint,
            max_global_readers,
            (2, 2).into(),
            dtypes,
            line_sizes,
        )?;

        let plane_role_config = stage_config.plane_role_config();

        let plane_counts = MatmulPlaneCounts::new(
            blueprint.load_specialization_config,
            plane_role_config.plane_roles,
        );

        let precompute_job = blueprint.loading_precompute_strategy.into();
        let plane_dim = blueprint.plane_dim;
        let event_loading_mode = EventLoadingMode::Relaxed;
        let reader_mode = blueprint.reader_mode;

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
            specialization_tensor_config: blueprint.load_specialization_config.lhs,
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
            specialization_tensor_config: blueprint.load_specialization_config.rhs,
        };

        let writer_config = GlobalWriterConfig {
            gmem_config: out_gmem_config,
            smem_config: stage_config.out_smem_config(),
            role_rule_config: plane_role_config.rule,
            plane_dim: blueprint.plane_dim,
        };

        Ok(SharedGlobalMatmulConfig {
            stage_config,
            num_planes: plane_counts.total,
            lhs_reader_config,
            rhs_reader_config,
            writer_config,
            must_sync_plane_after_execution: false,
        })
    }

    fn cubedim_resource(
        blueprint: &TilingBlueprint,
    ) -> Result<CubeDimResource, InvalidConfigError> {
        todo!()
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<(), MatmulSetupError> {
        todo!();
        // LL::check(client, problem, &config.lhs_reader_config, dtypes)?;
        // RL::check(client, problem, &config.rhs_reader_config, dtypes)?;

        SMM::validate_blueprint(client, blueprint, (2, 2).into(), dtypes, line_sizes)
    }
}
