use cubecl::{
    std::tensor::layout::Coords2d,
    {ir::DeviceProperties, prelude::*},
};
use cubek_std::{
    InvalidConfigError,
    stage::StageMemoryConfig,
    tile::{Tile, TileScope},
};

use crate::{
    components::{
        CubeDimResource,
        global::PlaneFlowConfig,
        stage::NumStages,
    },
    definition::{MatmulElems, MatmulSetupError, MatmulVectorSizes, TilingBlueprint},
};

use super::TilingLayout;

/// A family of [StageMatmul] implementations that operate with any [precision](MatmulPrecision).
pub trait StageMatmulFamily: Send + Sync + 'static {
    /// Compute primitive used by the underlying tile matmul
    type Scope: TileScope;

    /// Stage partitioner used by this family — picks the per-primitive
    /// coordinates that drive the PartitionScheduler. Exposed so callers
    /// that have cut away from the `StageMatmul` trait (PR 6 onwards) can
    /// reach the partitioner directly from the family.
    type Partitioner: crate::components::stage::matmul::partitioned_matmul::StagePartitioner<
            Scope = Self::Scope,
        >;

    /// Stage family for Lhs
    type LhsStage: StageFamily;
    /// Stage family for Rhs
    type RhsStage: StageFamily;
    /// Stage family for Acc
    type AccStage: StageFamily;
    /// Stage family for Out
    type OutStage: StageFamily<ReadWrite>;

    /// The configuration type associated with this matmul family.
    type Config;

    /// Constructs the configuration based on the matmul problem, selection, vector sizes,
    /// number of stages, maximum of tasks per plane, and whether the algorithm is an ordered variant
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    #[allow(clippy::too_many_arguments)]
    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        plane_flow_config: PlaneFlowConfig,
        num_stages: NumStages,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError>;

    /// Returns the compute resources required to run this matmul.
    fn cubedim_resource(blueprint: &TilingBlueprint)
    -> Result<CubeDimResource, InvalidConfigError>;

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError>;
}

pub use cubek_std::tile::PartitionBuffering;

/// Stage that can be divided into tiles, with the same kind used by the
/// tile matmul readers.
#[cube]
pub trait Stage<ES: Numeric, IO: SliceVisibility = ReadOnly>:
    CubeType + Clone + Send + Sync + 'static
{
    /// Slices a tile with offset (`row`, `col`) from the stage and returns it.
    ///
    /// The [Scope] generic lets the caller select the compute primitive that will consume
    /// this tile
    fn tile<Sc: TileScope>(this: &Self, tile: Coords2d) -> Tile<ES, Sc, IO>;
}

/// Stage family for any precision
pub trait StageFamily<IO: SliceVisibility = ReadOnly>: Send + Sync + 'static {
    /// The concrete stage type of this family, instantiated with the type and layout.
    /// `NS` parameterizes the underlying allocation's vector size; the produced
    /// `Stage<ES, IO>` no longer exposes it on its trait surface.
    type Stage<ES: Numeric, NS: Size, T: TilingLayout>: Stage<ES, IO>;
}

/// Stage family that can be used as the target of a loader
#[cube]
pub trait LoadStageFamily<IO: SliceVisibility = ReadOnly>: StageFamily {
    /// Create a new stage from the config and alignment
    fn create<ES: Numeric, NS: Size, T: TilingLayout>(
        #[comptime] alignment: usize,
        #[comptime] config: StageMemoryConfig,
    ) -> Self::Stage<ES, NS, T>;
    /// Return the same stage with a different buffer index
    fn with_buffer_index<ES: Numeric, NS: Size, T: TilingLayout>(
        stage: &Self::Stage<ES, NS, T>,
        buffer_index: u32,
    ) -> Self::Stage<ES, NS, T>;
    /// Free the stage
    fn free<ES: Numeric, NS: Size, T: TilingLayout>(stage: &Self::Stage<ES, NS, T>);
}

#[cube]
impl<ES: Numeric, IO: SliceVisibility, Inner: Stage<ES, IO>> Stage<ES, IO>
    for ComptimeOption<Inner>
{
    fn tile<Sc: TileScope>(this: &Self, tile: Coords2d) -> Tile<ES, Sc, IO> {
        #[comptime]
        if let ComptimeOption::Some(inner) = this {
            Inner::tile::<Sc>(inner, tile)
        } else {
            Tile::new_None()
        }
    }
}

#[cube]
impl<IO: SliceVisibility, S: LoadStageFamily<IO>> LoadStageFamily<IO> for Option<S> {
    fn create<ES: Numeric, NS: Size, T: TilingLayout>(
        #[comptime] alignment: usize,
        #[comptime] config: StageMemoryConfig,
    ) -> Self::Stage<ES, NS, T> {
        ComptimeOption::new_Some(S::create(alignment, config))
    }

    fn with_buffer_index<ES: Numeric, NS: Size, T: TilingLayout>(
        stage: &Self::Stage<ES, NS, T>,
        index: u32,
    ) -> Self::Stage<ES, NS, T> {
        stage.as_ref().map(|s| S::with_buffer_index(s, index))
    }

    fn free<ES: Numeric, NS: Size, T: TilingLayout>(stage: &Self::Stage<ES, NS, T>) {
        #[comptime]
        if let ComptimeOption::Some(inner) = stage {
            S::free(inner)
        }
    }
}

impl<IO: SliceVisibility, Inner: StageFamily<IO>> StageFamily<IO> for Option<Inner> {
    type Stage<ES: Numeric, NS: Size, T: TilingLayout> = ComptimeOption<Inner::Stage<ES, NS, T>>;
}
