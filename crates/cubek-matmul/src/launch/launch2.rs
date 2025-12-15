use crate::components::batch::{BatchMatmulFamily, CubeCountInputArgs};
use crate::definition::{AvailableLineSizes, MatmulElems};
use crate::definition::{MatmulAvailabilityError, MatmulSetupError};
use crate::definition::{MatmulProblem, MatrixLayout};
use crate::launch::{
    ConcreteInputsFactory, ConcreteOutputFactory, InputArg, InputRuntimeArg, MatmulArgs, OutputArg,
    OutputRuntimeArg, TensorArgs, TensorMapArgs,
};
use crate::launch::{MatmulInputHandle, MatmulInputHandleRef, launch_kernel_concrete};
use crate::routines::{Routine, Selection};
use cubecl::features::TypeUsage;
use cubecl::prelude::*;
use cubecl::std::tensor::{MatrixBatchLayout, TensorHandle, matrix_batch_layout};
use cubecl::{Runtime, client::ComputeClient, frontend::TensorHandleRef};

/// Launch a matrix multiplication kernel.
///
/// Cmma will be used if enabled
/// Will fail if unavailable
#[allow(clippy::result_large_err)]
pub fn launch<R: Runtime, A: Routine>(
    client: &ComputeClient<R>,
    lhs: MatmulInputHandle<R>,
    rhs: MatmulInputHandle<R>,
    out: TensorHandle<R>,
    selection: &Selection<A::SelectionArgs>,
    mut dtypes: MatmulElems,
) -> Result<TensorHandle<R>, MatmulSetupError> {
    let result = launch_ref::<R, A>(
        client,
        &lhs.as_ref(),
        &rhs.as_ref(),
        &out.as_ref(),
        selection,
        &mut dtypes,
    );

    match result {
        Ok(_) => Ok(out),
        Err(e) => Err(e),
    }
}

/// Launch a matrix multiplication kernel.
///
/// Cmma will be used if available and enabled,
/// otherwise it will fall back on a non-cmma implementation
#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime, A: Routine>(
    client: &ComputeClient<R>,
    lhs: &MatmulInputHandleRef<'_, R>,
    rhs: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    selection: &Selection<A::SelectionArgs>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    let check_layout = |tensor: &TensorHandleRef<'_, R>| match matrix_batch_layout(tensor.strides) {
        MatrixBatchLayout::Contiguous => (false, false),
        MatrixBatchLayout::MildlyPermuted {
            transposed,
            batch_swap: _,
        } => (false, transposed),
        MatrixBatchLayout::HighlyPermuted => (true, false),
    };

    let (lhs_make_contiguous, lhs_transposed) = check_layout(lhs.data());
    let (rhs_make_contiguous, rhs_transposed) = check_layout(rhs.data());

    let lhs_owned;
    let rhs_owned;
    let lhs = if lhs_make_contiguous {
        lhs_owned = lhs.into_contiguous(client)?;
        &lhs_owned.as_ref()
    } else {
        lhs
    };
    let rhs = if rhs_make_contiguous {
        rhs_owned = rhs.into_contiguous(client)?;
        &rhs_owned.as_ref()
    } else {
        rhs
    };

    let line_sizes = AvailableLineSizes::from_type_sizes(
        client,
        lhs.data().elem_size,
        rhs.data().elem_size,
        out.elem_size,
    );

    launch_inner_ref::<R, TensorArgs, A>(
        client,
        lhs,
        rhs,
        out,
        (lhs_transposed, rhs_transposed),
        selection,
        line_sizes,
        dtypes,
    )
}

/// Launch a matrix multiplication kernel, with TMA restrictions enabled.
/// TMA doesn't support permuted batches, so checks are slightly different.
///
/// Cmma will be used if available and enabled,
/// otherwise it will fall back on a non-cmma implementation
#[allow(clippy::result_large_err)]
pub fn launch_ref_tma<R: Runtime, A: Routine>(
    client: &ComputeClient<R>,
    lhs: &MatmulInputHandleRef<'_, R>,
    rhs: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    selection: &Selection<A::SelectionArgs>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    let check_layout = |tensor: &TensorHandleRef<'_, R>| match matrix_batch_layout(tensor.strides) {
        MatrixBatchLayout::Contiguous => (false, false),
        MatrixBatchLayout::MildlyPermuted {
            transposed,
            batch_swap: false,
        } => (false, transposed),
        _ => (true, false),
    };

    let (lhs_make_contiguous, lhs_transposed) = check_layout(lhs.data());
    let (rhs_make_contiguous, rhs_transposed) = check_layout(rhs.data());

    let lhs_owned;
    let rhs_owned;
    let lhs = if lhs_make_contiguous {
        lhs_owned = lhs.into_contiguous(client)?;
        &lhs_owned.as_ref()
    } else {
        lhs
    };
    let rhs = if rhs_make_contiguous {
        rhs_owned = rhs.into_contiguous(client)?;
        &rhs_owned.as_ref()
    } else {
        rhs
    };

    let line_sizes = AvailableLineSizes::from_type_size_tma(client, out.elem_size);

    launch_inner_ref::<R, TensorMapArgs, A>(
        client,
        lhs,
        rhs,
        out,
        (lhs_transposed, rhs_transposed),
        selection,
        line_sizes,
        dtypes,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
fn launch_inner_ref<R: Runtime, MA: MatmulArgs, A: Routine>(
    client: &ComputeClient<R>,
    lhs_handle: &MatmulInputHandleRef<'_, R>,
    rhs_handle: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    transposed: (bool, bool),
    selection: &Selection<A::SelectionArgs>,
    line_sizes: AvailableLineSizes,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError>
where
    InputArg<MA>: ConcreteInputsFactory,
    OutputArg<MA>: ConcreteOutputFactory,
{
    let lhs_shape = lhs_handle.shape();
    let rhs_shape = rhs_handle.shape();

    let rank = lhs_shape.len();
    let lhs_elem = *dtypes.lhs_global;
    let rhs_elem = *dtypes.rhs_global;
    let acc_elem = *dtypes.acc_global;

    if !client
        .properties()
        .features
        .type_usage(lhs_elem)
        .contains(TypeUsage::Conversion)
        || !client
            .properties()
            .features
            .type_usage(rhs_elem)
            .contains(TypeUsage::Conversion)
        || !client
            .properties()
            .features
            .type_usage(acc_elem)
            .contains(TypeUsage::Conversion)
    {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::TypesUnavailable {
                lhs: lhs_elem,
                rhs: rhs_elem,
                output: acc_elem,
            },
        ));
    }

    let m = lhs_shape[rank - 2] as u32;
    let k = lhs_shape[rank - 1] as u32;
    let n = rhs_shape[rank - 1] as u32;

    let lhs_layout = match transposed.0 {
        true => MatrixLayout::ColMajor,
        false => MatrixLayout::RowMajor,
    };

    let rhs_layout = match transposed.1 {
        true => MatrixLayout::ColMajor,
        false => MatrixLayout::RowMajor,
    };

    let problem = MatmulProblem {
        m: m as usize,
        n: n as usize,
        k: k as usize,
        lhs_batches: lhs_shape[..lhs_shape.len() - 2].to_vec(),
        rhs_batches: rhs_shape[..rhs_shape.len() - 2].to_vec(),
        out_batches: out.shape[..out.shape.len() - 2].to_vec(),
        lhs_strides: lhs_handle.data().strides.to_vec(),
        rhs_strides: rhs_handle.data().strides.to_vec(),
        lhs_layout,
        rhs_layout,
    };

    let lhs = lhs_handle.data();
    let rhs = rhs_handle.data();

    let line_sizes = A::filter_line_sizes(line_sizes);
    let mut line_sizes = line_sizes
        .filter_lhs_with_tensor(lhs.strides, lhs.shape, problem.lhs_layout)
        .filter_rhs_with_tensor(rhs.strides, rhs.shape, problem.rhs_layout)
        .filter_out_with_tensor(out.strides, out.shape)
        .pick_max()?;

    // The large line size resulting from dequantizing ends up slower due to restrictions on
    // algorithms. Use this as a quick and dirty fix.
    if lhs_handle.scale().is_some() {
        line_sizes.lhs = 1;
    }
    if rhs_handle.scale().is_some() {
        line_sizes.rhs = 1;
    }

    let fix_plane_dim = |plane_dim: u32| {
        // Sometimes the GPU doesn't support plane instructions and doesn't report the
        // plane size, but we can still execute algorithms that don't use plane instructions.
        //
        // In this case, we set a plane size for the selector to work, defaulting to 32 as it
        // is a common plane size.
        if plane_dim == 0 { 32 } else { plane_dim }
    };

    let plane_dim = fix_plane_dim(A::select_plane_dim(client));

    launch_kernel_concrete::<MA, R, A>(
        client, lhs_handle, rhs_handle, out, problem, line_sizes, plane_dim, selection, dtypes,
    )
}

#[allow(clippy::too_many_arguments, clippy::result_large_err)]
pub fn launch_with_config<'a, MA: MatmulArgs, R: Runtime, A: Routine>(
    client: &ComputeClient<R>,
    cube_dim: CubeDim,
    cube_count: CubeCount,
    input: InputRuntimeArg<'a, MA, R>,
    output: OutputRuntimeArg<'a, MA, R>,
    cube_count_input: CubeCountInputArgs<'a, R>,
    config: <A::BatchMatmul as BatchMatmulFamily>::Config,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    let result = unsafe {
        A::BatchMatmul::launch_unchecked::<MA, R>(
            client,
            cube_dim,
            cube_count,
            input,
            output,
            cube_count_input,
            config,
            dtypes,
        )
    };

    match result {
        Ok(_) => Ok(()),
        Err(err) => Err(MatmulSetupError::Launch(err)),
    }
}
