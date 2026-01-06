use crate::components::CubeDimResource;
use crate::components::global::multi_stage::EventLoadingMode;
use crate::components::global::{
    GlobalReaderConfig, GlobalWriterConfig, MatmulPlaneCounts, SharedGlobalMatmulConfig,
};
use crate::components::global::{GlobalWriterFamily, multi_stage::specialized::SpecializedMatmul};
use crate::components::global::{
    LoadSpecializationConfig, SpecializationTensorConfig, WriteTiling,
};
use crate::components::global::{
    memory::{GlobalMemoryConfig, ViewDirection},
    read::AsyncPartialLoadingStrategy,
};
use crate::components::stage::StageConfig;
use crate::components::{global::GlobalMatmulFamily, stage, stage::FilledStageFamily};
use crate::components::{global::MaxGlobalReaderPlanes, stage::NoTilingLayout};
use crate::definition::TilingBlueprint;
use crate::definition::{InvalidConfigError, MatmulLineSizes};
use crate::definition::{MatmulElems, MatmulSetupError};
use crate::definition::{MatmulPrecision, MatmulProblem};
use crate::definition::{MatrixLayout, StageIdent};
use cubecl::prelude::*;
use std::marker::PhantomData;

/// Double buffering matmul family for any precision
pub struct SpecializedMatmulFamily<
    SMM: stage::StageMatmulFamily,
    L: AsyncPartialLoadingStrategy,
    GW: GlobalWriterFamily,
> {
    _stage_matmul: PhantomData<SMM>,
    _loading: PhantomData<L>,
    _writer: PhantomData<GW>,
}

impl<SMM, L, GW> GlobalMatmulFamily for SpecializedMatmulFamily<SMM, L, GW>
where
    SMM: stage::StageMatmulFamily<
            LhsStage = L::Stage,
            RhsStage = L::Stage,
            AccStage = FilledStageFamily,
            OutStage = GW::Stage,
        >,
    L: AsyncPartialLoadingStrategy,
    GW: GlobalWriterFamily,
{
    type Matmul<MP: MatmulPrecision> = SpecializedMatmul<
        MP,
        SMM::Matmul<MP, L::TilingLayout, L::TilingLayout, NoTilingLayout, WriteTiling>,
        L,
        GW::Writer<MP::Acc>,
    >;
    type Config = SharedGlobalMatmulConfig<SMM::Config>;

    fn expand_config(
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        // Should be set from selection, but tests won't work properly. This algorithm fails without
        // specialization so it needs to be enabled.
        let mut blueprint = blueprint.clone();
        blueprint.load_specialization_config = LoadSpecializationConfig {
            lhs: SpecializationTensorConfig::LoadFlowOnly,
            rhs: SpecializationTensorConfig::LoadFlowOnly,
        };

        let max_global_readers = MaxGlobalReaderPlanes::new::<L, L>(
            &blueprint.tiling_scheme,
            line_sizes,
            blueprint.plane_dim,
            dtypes,
        );

        let stage_config = SMM::expand_config(
            &blueprint,
            Some(max_global_readers),
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
        L::validate_with_problem(problem, dtypes, StageIdent::Lhs)?;
        L::validate_with_problem(problem, dtypes, StageIdent::Rhs)?;
        SMM::validate_blueprint(client, blueprint, (2, 2).into(), dtypes, line_sizes)
    }
}
