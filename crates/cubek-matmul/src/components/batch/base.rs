use crate::launch::{InputRuntimeArg, MatmulArgs, OutputRuntimeArg};
use crate::{components::CubeDimResource, launch::RuntimeConfig};
use crate::{components::global::memory::GlobalLayoutConfig, launch::ConfigRuntimeArg};
use crate::{
    components::stage::NumStages,
    definition::{
        AccG, Blueprint, CubeMapping, CubeMappingLaunch, LhsG, MatmulElems, MatmulLineSizes,
        MatmulPrecision, MatmulProblem, MatmulSetupError, RhsG,
    },
};
use cubecl::{ir::DeviceProperties, prelude::*};
use std::{fmt::Debug, hash::Hash};

/// A family of [matmuls](BatchMatmul) working with any [precision](MatmulPrecision).
pub trait BatchMatmulFamily<RC: RuntimeConfig>: 'static + Send + Sync {
    /// The specific [BatchMatmul] implementation associated with this family.
    type Matmul<MP: MatmulPrecision>: BatchMatmul<RC, MP, Config = Self::Config>;

    /// The configuration type associated with this matmul family.
    type Config: BatchConfig;

    type Blueprint: Blueprint;

    /// Constructs the configuration based on the matmul problem, selection, and line sizes.
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &Self::Blueprint,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError>;

    fn num_stages() -> NumStages;

    /// Entry point
    ///
    /// # Safety
    ///
    /// Out-of-bounds can happen
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_unchecked<'a, MA: MatmulArgs<Config = RC>, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MA, R>,
        output: OutputRuntimeArg<'a, MA, R>,
        config: ConfigRuntimeArg<'a, MA, R>,
        cube_mapping: CubeMappingLaunch<'a, R>,
        blueprint: Self::Blueprint,
        dtypes: &MatmulElems,
    ) -> Result<(), LaunchError>;

    /// Returns the compute resources required to run this matmul.
    fn cubedim_resource(
        blueprint: &Self::Blueprint,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<CubeDimResource, MatmulSetupError>;

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<(), MatmulSetupError>;
}

#[cube]
/// Provides matrix multiplication operations at the batch level.
///
/// At the batch level,
///  - Inputs are whole tensors in global memory.
///  - All Cubes are used to solve the problem
///  - Dimensions M, N and K can be arbitrary large,
///    as well as the number of batches.
///
/// # Assumptions
/// - Line sizes of the inputs evenly divide the dimension they are aligned with.
///
/// # Safety
///
/// - It is not assumed that the matmul's dimensions match its inputs dimensions perfectly.
///   It is therefore important to use an underlying global matmul that performs check bounds,
/// - It is accepted to launch more Cube than necessary, providing a CubeCountInput that states
///   the max cube position
pub trait BatchMatmul<RC: RuntimeConfig, MP: MatmulPrecision>: 'static + Send + Sync {
    type Config: BatchConfig;

    /// Performs batchwise matrix multiplication over tensors.
    fn execute<Args: MatmulArgs<Config = RC>>(
        state: &mut Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
        cube_mapping: CubeMapping,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the [batch matmul](BatchMatmul) level.
pub trait BatchConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig;
    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig;
    fn out_global_layout_config(&self) -> GlobalLayoutConfig;
}
