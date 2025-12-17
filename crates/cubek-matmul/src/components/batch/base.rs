use crate::components::global::memory::GlobalLayoutConfig;
use crate::definition::{
    AccG, CubeCountInput, CubeCountInputArgs, CubeCountPlan, LhsG, MatmulElems, MatmulLineSizes,
    MatmulPrecision, MatmulProblem, MatmulSetupError, RhsG,
};
use crate::launch::{InputRuntimeArg, MatmulArgs, OutputRuntimeArg};
use cubecl::prelude::*;
use std::{fmt::Debug, hash::Hash};

/// A family of [matmuls](BatchMatmul) working with any [precision](MatmulPrecision).
pub trait BatchMatmulFamily: 'static + Send + Sync {
    /// The specific [BatchMatmul] implementation associated with this family.
    type Matmul<MP: MatmulPrecision>: BatchMatmul<MP, Config = Self::Config>;

    /// The configuration type associated with this matmul family.
    type Config: BatchConfig;

    type Blueprint;

    /// Constructs the configuration based on the matmul problem, selection, and line sizes.
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn expand_config<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        blueprint: &Self::Blueprint,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError>;

    /// Entry point
    ///
    /// # Safety
    ///
    /// Out-of-bounds can happen
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_unchecked<'a, MA: MatmulArgs, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MA, R>,
        output: OutputRuntimeArg<'a, MA, R>,
        cube_count_input: CubeCountInputArgs<'a, R>,
        config: Self::Config,
        dtypes: &MatmulElems,
    ) -> Result<(), LaunchError>;
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
pub trait BatchMatmul<MP: MatmulPrecision>: 'static + Send + Sync {
    type Config: BatchConfig;

    /// Performs batchwise matrix multiplication over tensors.
    fn execute<Args: MatmulArgs>(
        state: &mut Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
        cube_count_args: CubeCountInput,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the [batch matmul](BatchMatmul) level.
pub trait BatchConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    /// Returns the [CubeDim]
    fn cube_dim(&self) -> CubeDim;

    fn cube_count_plan(&self, problem: &MatmulProblem, max_cube_count: &CubeCount)
    -> CubeCountPlan;

    /// Returns the line sizes for Lhs, Rhs and output
    fn line_sizes(&self) -> MatmulLineSizes;

    /// Whether it may launch more cubes than the minimum required
    fn can_yield_extra_cubes(&self) -> bool;

    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig;
    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig;
    fn out_global_layout_config(&self) -> GlobalLayoutConfig;
}
