use crate::components::{
    CubeDimResource,
    global::{PartitionedStage, PartitionedStageFamily, PlaneFlowConfig},
    stage::{
        NumStages, StageFamily, StageMatmulFamily, TilingLayout,
        matmul::unit_partitioned::{UnitMatmul, UnitPartitioner},
        stage_matmul::{StageMatmul, StageMatmulKind},
    },
};
use crate::definition::{
    Acc, Lhs, MatmulElems, MatmulSetupError, MatmulTypes, MatmulVectorSizes, Rhs, TilingBlueprint,
};
use core::marker::PhantomData;
use cubecl::{ir::DeviceProperties, prelude::*};
use cubek_std::{InvalidConfigError, tile::Unit};

/// Unit Matmul family for any precision
pub struct UnitMatmulFamily<StageIn: StageFamily, StageAcc: StageFamily> {
    _phantom: PhantomData<(StageIn, StageAcc)>,
}

type STy<T> = crate::definition::Stage<T>;
type SSz<T> = crate::definition::StageSize<T>;

impl<StageIn: StageFamily, StageAcc: StageFamily> StageMatmulFamily
    for UnitMatmulFamily<StageIn, StageAcc>
{
    type Scope = Unit;
    type Partitioner = UnitPartitioner;
    type LhsStage = StageIn;
    type RhsStage = StageIn;
    type AccStage = StageAcc;
    type OutStage = PartitionedStageFamily;

    type Matmul<
        MP: MatmulTypes,
        TL: TilingLayout,
        TR: TilingLayout,
        TA: TilingLayout,
        TO: TilingLayout,
    > = UnitMatmul<
        MP,
        StageIn::Stage<STy<Lhs<MP>>, SSz<Lhs<MP>>, TL>,
        StageIn::Stage<STy<Rhs<MP>>, SSz<Rhs<MP>>, TR>,
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
        StageMatmulKind::UnitPartitioned.expand_stage_matmul(
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
        StageMatmulKind::UnitPartitioned.cubedim_resource(blueprint)
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        StageMatmulKind::UnitPartitioned.validate_blueprint(
            client,
            blueprint,
            dtypes,
            vector_sizes,
        )
    }
}
