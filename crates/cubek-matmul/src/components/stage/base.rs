use cubecl::{
    std::tensor::layout::Coords2d,
    {ir::DeviceProperties, prelude::*},
};
use cubek_std::{InvalidConfigError, stage::StageMemoryConfig};

use crate::{
    components::{
        CubeDimResource,
        global::{PlaneFlowConfig, WriteEventListener},
        stage::{NumStages, PartitionScheduler},
        tile::{TileConfig, Tilex},
    },
    definition::{
        Acc, Lhs, MatmulElems, MatmulSetupError, MatmulTypes, MatmulVectorSizes, Rhs,
        TilingBlueprint,
    },
};
use std::{fmt::Debug, hash::Hash};

use super::{StageEventListener, TilingLayout};

type Ty<T> = crate::definition::Stage<T>;
type Sz<T> = crate::definition::StageSize<T>;

/// A family of [StageMatmul] implementations that operate with any [precision](MatmulPrecision).
pub trait StageMatmulFamily: Send + Sync + 'static {
    /// The specific TileMatmul implementation associated with this family.
    type Matmul<MP: MatmulTypes, TL: TilingLayout, TR: TilingLayout, TA: TilingLayout, TO: TilingLayout>: StageMatmul<
            MP,
            Config = Self::Config,
            LhsStage = <Self::LhsStage as StageFamily>::Stage<Ty<Lhs<MP>>, Sz<Lhs<MP>>, TL>,
            RhsStage = <Self::RhsStage as StageFamily>::Stage<Ty<Rhs<MP>>, Sz<Rhs<MP>>, TR>,
            AccStage = <Self::AccStage as StageFamily>::Stage<Ty<Acc<MP>>, Sz<Acc<MP>>, TA>,
            OutStage = <Self::OutStage as StageFamily<ReadWrite>>::Stage<Ty<Acc<MP>>, Sz<Acc<MP>>, TO>,
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
    type Config: StageConfig;

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

#[cube]
/// Provides matrix multiplication operations at the stage level.
///
/// At the stage level,
///  - Inputs are assumed to be already staged into a shared memory.
///  - All main flow planes within a Cube are used to solve the problem
///  - Dimensions M, N and K are fixed to an integer, and the
///    matrix multiplication works only for size (M, K) · (K, N) = (M, N).
///    These integers are multiples of the underlying Tile matmul,
///    corresponding to the number of tiles in each dimension.
///
/// Assumptions:
///  - Data given as inputs by stage readers must always be valid. If the actual matrix multiplication
///    should be done on smaller sizes than M, N and K, padding with zeros must be done beforehand.
///  - Enough planes/units are launched to perform the whole computation
pub trait StageMatmul<MP: MatmulTypes>: 'static + Send + Sync {
    /// The configuration type associated with this Matmul.
    type Config: StageConfig;

    /// Contains the matrix multiplication output, that can be shared across the different planes of the cube.
    /// The same Accumulator will be added to across multiple executions of the Stage Matmul.
    type Accumulators: CubeType;

    /// Stage for Lhs
    type LhsStage: CubeType;
    /// Stage for Rhs
    type RhsStage: CubeType;
    /// Stage for Accumulator
    type AccStage: CubeType;
    /// Stage for Out
    type OutStage: CubeType;

    /// Lhs input of the underlying Tile Matmul
    type LhsTile: CubeType;
    /// Rhs input of the underlying Tile Matmul
    type RhsTile: CubeType;

    /// Executes the matrix multiplication of Lhs and Rhs, adding the result to the accumulator
    ///
    /// Equivalent to execute_with_listener with SEL:=NoEvent
    fn execute(
        lhs: &Self::LhsStage,
        rhs: &Self::RhsStage,
        instruction_lhs: &mut Self::LhsTile,
        instruction_rhs: &mut Self::RhsTile,
        acc: &mut Self::Accumulators,
        #[comptime] config: Self::Config,
        partition_scheduler: &PartitionScheduler,
    );

    /// Executes the matrix multiplication of Lhs and Rhs, with the addition of injected
    /// [event listener](StageEventListener).
    fn execute_with_listener<SEL: StageEventListener>(
        lhs: &Self::LhsStage,
        rhs: &Self::RhsStage,
        instruction_lhs: &mut Self::LhsTile,
        instruction_rhs: &mut Self::RhsTile,
        acc: &mut Self::Accumulators,
        #[comptime] config: Self::Config,
        listener: SEL,
        partition_scheduler: &PartitionScheduler,
    );

    /// Inits inputs of the underlying Tile Matmul
    fn init_tile_inputs(#[comptime] config: Self::Config) -> (Self::LhsTile, Self::RhsTile);

    /// Create an instance of the accumulators, without data
    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators;

    /// Load all accumulators in the stage from data
    fn load_accumulators(
        reader: &Self::AccStage,
        acc: &mut Self::Accumulators,
        #[comptime] config: Self::Config,
    );

    /// Reads the result of the accumulator and hands it to the stage writer
    fn write_results<W: WriteEventListener>(
        acc: &mut Self::Accumulators,
        stage: &mut Self::OutStage,
        listener: &mut W,
        partition_scheduler: &PartitionScheduler,
        #[comptime] stage_config: Self::Config,
    );

    fn init_scheduler(#[comptime] config: Self::Config) -> PartitionScheduler;
}

/// Configuration for the Stage matmul (SMM) level
pub trait StageConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    /// Underlying Tile matmul config
    type TileConfig: TileConfig;

    fn elements_in_stage_m(&self) -> u32;
    fn elements_in_stage_n(&self) -> u32;
    fn elements_in_stage_k(&self) -> u32;
    fn elements_in_tile_k(&self) -> u32;
    fn tiles_in_partition_mn(&self) -> u32;
    fn num_main_flow_planes(&self) -> u32;
    fn plane_dim(&self) -> u32;
    fn plane_flow_config(&self) -> PlaneFlowConfig;

    fn lhs_smem_config(&self) -> StageMemoryConfig;
    fn rhs_smem_config(&self) -> StageMemoryConfig;
    fn acc_smem_config(&self) -> StageMemoryConfig;
    fn out_smem_config(&self) -> StageMemoryConfig;
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PartitionBuffering {
    Single,
    #[default]
    Double,
}

/// Stage that can be divided into tiles, with the same kind used by the
/// tile matmul readers.
#[cube]
pub trait Stage<ES: Numeric, NS: Size, IO: SliceVisibility = ReadOnly>:
    CubeType + Clone + Send + Sync + 'static
{
    /// Slices a tile with offset (`row`, `col`) from the stage and returns it
    fn tile(this: &Self, tile: Coords2d) -> Tilex<ES, NS, IO>;
}

/// Stage family for any precision
pub trait StageFamily<IO: SliceVisibility = ReadOnly>: Send + Sync + 'static {
    /// The concrete stage type of this family, instantiated with the type and layout
    type Stage<ES: Numeric, NS: Size, T: TilingLayout>: Stage<ES, NS, IO>;
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
impl<ES: Numeric, NS: Size, IO: SliceVisibility, Inner: Stage<ES, NS, IO>> Stage<ES, NS, IO>
    for ComptimeOption<Inner>
{
    fn tile(this: &Self, tile: Coords2d) -> Tilex<ES, NS, IO> {
        #[comptime]
        if let ComptimeOption::Some(inner) = this {
            Inner::tile(inner, tile)
        } else {
            Tilex::new_None()
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
