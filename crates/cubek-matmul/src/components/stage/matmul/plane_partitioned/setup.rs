use crate::components::stage::matmul::plane_partitioned::{PlaneMatmul, PlanePartitioner};
use crate::components::{
    CubeDimResource,
    global::{PartitionedStage, PartitionedStageFamily, PlaneFlowConfig},
    stage::{
        NumStages, StageFamily, StageMatmulFamily,
        stage_matmul::{StageMatmul, StageMatmulKind},
    },
};
use crate::definition::{
    MatmulElems, MatmulSetupError, MatmulTypes, MatmulVectorSizes, TilingBlueprint,
};
use crate::{
    components::stage::TilingLayout,
    definition::{Acc, Lhs, Rhs},
};
use core::marker::PhantomData;
use cubecl::{ir::DeviceProperties, prelude::*};
use cubek_std::{InvalidConfigError, tile::Plane};

type STy<T> = crate::definition::Stage<T>;
type SSz<T> = crate::definition::StageSize<T>;

/// Plane Matmul family for any precision
pub struct PlaneMatmulFamily<StageLhs: StageFamily, StageRhs: StageFamily, StageAcc: StageFamily> {
    _phantom: PhantomData<(StageLhs, StageRhs, StageAcc)>,
}

impl<StageLhs: StageFamily, StageRhs: StageFamily, StageAcc: StageFamily> StageMatmulFamily
    for PlaneMatmulFamily<StageLhs, StageRhs, StageAcc>
{
    type Scope = Plane;
    type Partitioner = PlanePartitioner;
    type LhsStage = StageLhs;
    type RhsStage = StageRhs;
    type AccStage = StageAcc;
    type OutStage = PartitionedStageFamily;

    type Matmul<
        MP: MatmulTypes,
        TL: TilingLayout,
        TR: TilingLayout,
        TA: TilingLayout,
        TO: TilingLayout,
    > = PlaneMatmul<
        MP,
        StageLhs::Stage<STy<Lhs<MP>>, SSz<Lhs<MP>>, TL>,
        StageRhs::Stage<STy<Rhs<MP>>, SSz<Rhs<MP>>, TR>,
        StageAcc::Stage<STy<Acc<MP>>, SSz<Acc<MP>>, TA>,
        PartitionedStage<STy<Acc<MP>>, SSz<Acc<MP>>>,
    >;

    type Config = StageMatmul;

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        plane_flow_config: PlaneFlowConfig,
        num_stages: NumStages,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        StageMatmulKind::PlanePartitioned.expand_stage_matmul(
            device_props,
            blueprint,
            plane_flow_config,
            num_stages,
            dtypes,
            vector_sizes,
        )
    }

    fn cubedim_resource(
        blueprint: &TilingBlueprint,
    ) -> Result<CubeDimResource, InvalidConfigError> {
        StageMatmulKind::PlanePartitioned.cubedim_resource(blueprint)
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        StageMatmulKind::PlanePartitioned.validate_blueprint(
            client,
            blueprint,
            dtypes,
            vector_sizes,
        )
    }
}
