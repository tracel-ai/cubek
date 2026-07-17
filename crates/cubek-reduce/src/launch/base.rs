use crate::{
    ReduceError, ReducePrecision, VectorizationMode,
    components::{
        args::{NumericVector, ReduceArgs, TensorArgs, init_tensors},
        global::{
            cube::GlobalFullCubeReduce, plane::GlobalFullPlaneReduce, unit::GlobalFullUnitReduce,
        },
        instructions::*,
    },
    launch::{ReduceStrategy, RoutineStrategy, generate_vector_size},
    output_vectorization_axis,
    routines::{
        GlobalReduceBlueprint, ReduceBlueprint, ReduceLaunchSettings, ReduceProblem,
        ReduceVectorSettings, Routine, cube::CubeRoutine, plane::PlaneRoutine, unit::UnitRoutine,
    },
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[derive(Clone, Copy, Debug)]
pub struct ReduceDtypes {
    pub input: StorageType,
    pub output: StorageType,
    pub accumulation: StorageType,
}

/// Dtypes for a reduce that writes its values and their indices at once.
#[derive(Clone, Copy, Debug)]
pub struct ReduceWithIndicesDtypes {
    pub input: StorageType,
    /// Dtype of the values output.
    pub values: StorageType,
    /// Dtype of the indices output.
    pub indices: StorageType,
    pub accumulation: StorageType,
}

impl ReduceWithIndicesDtypes {
    fn values_dtypes(&self) -> ReduceDtypes {
        ReduceDtypes {
            input: self.input,
            output: self.values,
            accumulation: self.accumulation,
        }
    }
}

/// Analyse the problem and prepare the routine for launch: everything both entrypoints
/// share between validation and the actual kernel launch. `output` is the tensor whose
/// layout drives vectorization (the values tensor on the fused path).
///
/// `second_output` is the fused path's index tensor dtype: it shares the values tensor's
/// layout but not its dtype, so the chosen output width must be legal for both. Passing it
/// caps the width to one the second dtype also supports, rather than dropping to scalar on
/// any width mismatch. `None` for the single-output path.
///
/// Returns the blueprint, the launch settings, and the output vectorization axis.
#[allow(clippy::too_many_arguments)]
fn prepare_reduce_launch<Run: Runtime>(
    client: &ComputeClient<Run>,
    input: &TensorBinding<Run>,
    output: &TensorBinding<Run>,
    reduce_axis: usize,
    strategy: ReduceStrategy,
    dtypes: ReduceDtypes,
    inst: ReduceOperationConfig,
    address_type: AddressType,
    second_output: Option<StorageType>,
) -> Result<(ReduceBlueprint, ReduceLaunchSettings, usize), ReduceError> {
    // Number of distinct reductions = product of non-reduce input dims.
    let reduce_len = input.shape[reduce_axis];
    let input_elems: usize = input.shape.iter().copied().product();
    let reduce_count = input_elems / reduce_len;

    let problem = ReduceProblem {
        reduce_len,
        reduce_count,
        axis: reduce_axis,
        dtypes,
        instruction: inst,
        address_type,
    };
    let vectorization_mode = match input.strides[reduce_axis] {
        1 => VectorizationMode::Parallel,
        _ => VectorizationMode::Perpendicular,
    };

    let out_vec_axis = output_vectorization_axis(&input.strides, reduce_axis, vectorization_mode);

    let (vector_size_input, vector_size_output) = generate_vector_size::<Run>(
        client,
        input,
        output,
        reduce_axis,
        problem.dtypes.input,
        vectorization_mode,
        &strategy.vectorization,
    );
    // Both fused outputs share this width, so it must be legal for the index dtype too.
    // Cap to the largest width the index dtype supports that does not exceed the values
    // width (widths are powers of two, so the cap still divides the layout constraints the
    // values width already satisfied). Only drops to scalar if the index dtype truly cannot
    // vectorize; the common case (equal-width dtypes) is unchanged.
    let vector_size_output = match second_output {
        None => vector_size_output,
        Some(index_dtype) => client
            .io_optimized_vector_sizes(index_dtype.size())
            .filter(|&width| width <= vector_size_output)
            .max()
            .unwrap_or(1),
    };
    let settings = ReduceVectorSettings {
        vectorization_mode,
        vector_size_input,
        vector_size_output,
        unchecked_fast_paths: matches!(
            strategy.autotune_level,
            cubecl::config::autotune::AutotuneLevel::Full
        ),
    };

    let (blueprint, settings) = match strategy.routine {
        RoutineStrategy::Unit(strategy) => {
            let routine = UnitRoutine;
            routine.prepare(client, problem, settings, strategy)?
        }
        RoutineStrategy::Plane(strategy) => {
            let routine = PlaneRoutine;
            routine.prepare(client, problem, settings, strategy)?
        }
        RoutineStrategy::Cube(strategy) => {
            let routine = CubeRoutine;
            routine.prepare(client, problem, settings, strategy)?
        }
    };

    Ok((blueprint, settings, out_vec_axis))
}

/// Launch a reduce kernel. This function assumes that all parameters are already validated.
/// See the main entrypoint `reduce` in `lib.rs` for an example how to call this function
/// with the appropriate assumptions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_reduce<Run: Runtime>(
    client: &ComputeClient<Run>,
    input: TensorBinding<Run>,
    output: TensorBinding<Run>,
    reduce_axis: usize,
    strategy: ReduceStrategy,
    dtypes: ReduceDtypes,
    inst: ReduceOperationConfig,
) -> Result<(), ReduceError> {
    let address_type = input
        .required_address_type(dtypes.input.size())
        .max(output.required_address_type(dtypes.output.size()));

    let (blueprint, settings, out_vec_axis) = prepare_reduce_launch::<Run>(
        client,
        &input,
        &output,
        reduce_axis,
        strategy,
        dtypes,
        inst,
        address_type,
        None,
    )?;

    unsafe {
        reduce_kernel::launch_unchecked::<TensorArgs, Run>(
            client,
            settings.cube_count,
            settings.cube_dim,
            settings.address_type,
            settings.vector.vector_size_input,
            settings.vector.vector_size_output,
            input.into_tensor_arg(),
            output.into_tensor_arg(),
            reduce_axis,
            out_vec_axis,
            blueprint,
            inst,
            dtypes.input,
            dtypes.output,
            dtypes.accumulation,
        )
    };

    Ok(())
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn reduce_kernel<
    In: Numeric,
    InSize: Size,
    Out: Numeric,
    OutSize: Size,
    Acc: Numeric,
    RA: ReduceArgs,
>(
    input: &RA::Input<In, InSize>,
    output: &mut RA::Output<Out, OutSize>,
    reduce_axis: usize,
    out_vec_axis: usize,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] config: ReduceOperationConfig,
    #[define(In)] _input_dtype: StorageType,
    #[define(Out)] _output_dtype: StorageType,
    #[define(Acc)] _acc_dtype: StorageType,
) {
    let (input, mut output) = init_tensors::<RA, In, InSize, Out, OutSize>(input, output);
    reduce_kernel_virtual::<In, InSize, Out, OutSize, Acc>(
        &input,
        &mut output,
        reduce_axis,
        out_vec_axis,
        blueprint,
        config,
    );
}

/// Launch a reduce kernel writing both the values and their indices. This function assumes
/// that all parameters are already validated; see `reduce_with_indices` in `lib.rs`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_reduce_with_indices<Run: Runtime>(
    client: &ComputeClient<Run>,
    input: TensorBinding<Run>,
    values: TensorBinding<Run>,
    indices: TensorBinding<Run>,
    reduce_axis: usize,
    strategy: ReduceStrategy,
    dtypes: ReduceWithIndicesDtypes,
    k: usize,
) -> Result<(), ReduceError> {
    // Both halves are written, so the instruction always tracks coordinates
    // regardless of which top-k config the caller reached this path with. The
    // fused `to_output_both_*` conversions ignore this mode; it only sizes the
    // accumulator, and `Indices` is what turns coordinate tracking on.
    let config = TopKConfig {
        k,
        output: ReduceOutputMode::Indices,
    };

    let address_type = input
        .required_address_type(dtypes.input.size())
        .max(values.required_address_type(dtypes.values.size()))
        .max(indices.required_address_type(dtypes.indices.size()));

    let (blueprint, settings, out_vec_axis) = prepare_reduce_launch::<Run>(
        client,
        &input,
        &values,
        reduce_axis,
        strategy,
        dtypes.values_dtypes(),
        // Always size the blueprint as ArgTopK, never TopK: this path tracks
        // coordinates whichever config the caller passed, so the shared
        // accumulator needs its index slices too. Sizing it as TopK would
        // under-allocate shared memory by `k` u32 slices per accumulator.
        ReduceOperationConfig::ArgTopK(k),
        address_type,
        // The index output shares the values layout but not its dtype, so the
        // shared output width must stay legal for the index dtype too.
        Some(dtypes.indices),
    )?;

    unsafe {
        reduce_with_indices_kernel::launch_unchecked::<TensorArgs, TopK, Run>(
            client,
            settings.cube_count,
            settings.cube_dim,
            settings.address_type,
            settings.vector.vector_size_input,
            settings.vector.vector_size_output,
            settings.vector.vector_size_output,
            input.into_tensor_arg(),
            values.into_tensor_arg(),
            indices.into_tensor_arg(),
            reduce_axis,
            out_vec_axis,
            blueprint,
            config,
            dtypes.input,
            dtypes.values,
            dtypes.indices,
            dtypes.accumulation,
        )
    };

    Ok(())
}

/// Reduce `input` along `reduce_axis`, writing both the values and their indices.
///
/// The indices output goes through [`ReduceArgs`] like the value output, so both
/// can be virtualized the same way; today only [`TensorArgs`] is used here.
#[cube(launch_unchecked, address_type = "dynamic")]
pub fn reduce_with_indices_kernel<
    In: Numeric,
    InSize: Size,
    Out: Numeric,
    OutSize: Size,
    Idx: Numeric,
    IdxSize: Size,
    Acc: Numeric,
    RA: ReduceArgs,
    R: ReduceWithIndicesFamily,
>(
    input: &RA::Input<In, InSize>,
    output: &mut RA::Output<Out, OutSize>,
    indices: &mut RA::Output<Idx, IdxSize>,
    reduce_axis: usize,
    out_vec_axis: usize,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] config: R::Config,
    #[define(In)] _input_dtype: StorageType,
    #[define(Out)] _output_dtype: StorageType,
    #[define(Idx)] _indices_dtype: StorageType,
    #[define(Acc)] _acc_dtype: StorageType,
) {
    let (input_values, mut output) = init_tensors::<RA, In, InSize, Out, OutSize>(input, output);
    // Pairs the same input with the index output to build its virtual tensor;
    // the duplicate input tensor is comptime plumbing with no runtime cost.
    let (_input_indices, mut indices) =
        init_tensors::<RA, In, InSize, Idx, IdxSize>(input, indices);

    reduce_with_indices_kernel_inner::<(In, InSize, Acc), (Out, OutSize), (Idx, IdxSize), R>(
        &input_values,
        &mut output,
        &mut indices,
        reduce_axis,
        out_vec_axis,
        blueprint,
        config,
    );
}

#[cube]
fn reduce_with_indices_kernel_inner<
    P: ReducePrecision,
    Out: NumericVector,
    Idx: NumericVector,
    R: ReduceWithIndicesFamily,
>(
    input: &VirtualTensor<P::EI, P::SI>,
    output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
    indices: &mut VirtualTensor<Idx::T, Idx::N, ReadWrite>,
    reduce_axis: usize,
    out_vec_axis: usize,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] config: R::Config,
) {
    let inst = R::Instruction::<P>::from_config(config);

    match blueprint.global {
        GlobalReduceBlueprint::Cube(cube) => {
            GlobalFullCubeReduce::execute_with_indices::<P, Out, Idx, R::Instruction<P>>(
                input,
                output,
                indices,
                reduce_axis,
                out_vec_axis,
                &inst,
                blueprint.vectorization_mode,
                cube,
            )
        }
        GlobalReduceBlueprint::Plane(plane) => {
            GlobalFullPlaneReduce::execute_with_indices::<P, Out, Idx, R::Instruction<P>>(
                input,
                output,
                indices,
                reduce_axis,
                out_vec_axis,
                &inst,
                blueprint.vectorization_mode,
                plane,
            )
        }
        GlobalReduceBlueprint::Unit(unit) => {
            GlobalFullUnitReduce::execute_with_indices::<P, Out, Idx, R::Instruction<P>>(
                input,
                output,
                indices,
                reduce_axis,
                out_vec_axis,
                &inst,
                blueprint.vectorization_mode,
                unit,
            )
        }
    };
}

#[cube]
pub fn reduce_kernel_virtual<
    In: Numeric,
    InSize: Size,
    Out: Numeric,
    OutSize: Size,
    Acc: Numeric,
>(
    input: &VirtualTensor<In, InSize>,
    output: &mut VirtualTensor<Out, OutSize, ReadWrite>,
    reduce_axis: usize,
    out_vec_axis: usize,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] config: ReduceOperationConfig,
) {
    reduce_kernel_inner::<(In, InSize, Acc), (Out, OutSize), ReduceOperation>(
        input,
        output,
        reduce_axis,
        out_vec_axis,
        blueprint,
        config,
    )
}

#[cube]
fn reduce_kernel_inner<P: ReducePrecision, Out: NumericVector, R: ReduceFamily>(
    input: &VirtualTensor<P::EI, P::SI>,
    output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
    reduce_axis: usize,
    out_vec_axis: usize,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] config: R::Config,
) {
    let inst = R::Instruction::<P>::from_config(config);

    match blueprint.global {
        GlobalReduceBlueprint::Cube(cube) => {
            GlobalFullCubeReduce::execute::<P, Out, R::Instruction<P>>(
                input,
                output,
                reduce_axis,
                out_vec_axis,
                &inst,
                blueprint.vectorization_mode,
                cube,
            )
        }
        GlobalReduceBlueprint::Plane(plane) => {
            GlobalFullPlaneReduce::execute::<P, Out, R::Instruction<P>>(
                input,
                output,
                reduce_axis,
                out_vec_axis,
                &inst,
                blueprint.vectorization_mode,
                plane,
            )
        }
        GlobalReduceBlueprint::Unit(unit) => {
            GlobalFullUnitReduce::execute::<P, Out, R::Instruction<P>>(
                input,
                output,
                reduce_axis,
                out_vec_axis,
                &inst,
                blueprint.vectorization_mode,
                unit,
            )
        }
    };
}
