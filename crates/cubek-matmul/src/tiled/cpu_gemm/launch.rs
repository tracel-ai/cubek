//! Launch wiring for the CpuGemm routine.

use cubecl::{client::Client, prelude::*};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_tile::{Axis, Geometry, KernelForm, Launcher, Space};

use crate::{
    definition::{
        AvailableVectorSizes, MatmulElems, MatmulIdent, MatmulProblem, MatmulSetupError,
        broadcast_batches,
    },
    routine::{BlueprintStrategy, DeviceSettings},
    tiled::cpu_gemm::{base::CpuGemmRoutine, kernel::cpu_gemm_kernel},
    tiled::{K, M, N, batch_axis, logical_dims, validate_stored_tile},
};

/// A strided matmul operand must be contiguous along one of its two innermost dims. Under storage
/// tiling those are the innermost fragments, which is the pair a leaf tile is read through, so the
/// same rule reads the same way on a tiled buffer.
#[allow(clippy::result_large_err)]
fn validate_strided(strides: &[usize]) -> Result<(), MatmulSetupError> {
    let n = strides.len();
    if strides[n - 1] == 1 || strides[n - 2] == 1 {
        Ok(())
    } else {
        Err(MatmulSetupError::InvalidConfig(Box::new(
            "CpuGemm: strided operand is contiguous in neither matrix axis".to_string(),
        )))
    }
}

/// CpuGemm reads each input where it already lies, so it carries one type per input from
/// global memory to the leaf, where the cast into the accumulator happens. A stage or register
/// type of its own is a second conversion this routine does not emit. The accumulator is not
/// fenced here: its element is stated on the operand and threaded as `EA`.
#[allow(clippy::result_large_err)]
fn validate_single_type(dtypes: &MatmulElems, ident: MatmulIdent) -> Result<(), MatmulSetupError> {
    let (global, stage, register) = (
        dtypes.global(ident),
        dtypes.stage(ident),
        dtypes.register(ident),
    );
    if stage == global && register == global {
        Ok(())
    } else {
        Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "CpuGemm runs {ident:?} at {global:?} throughout; stage {stage:?} and register \
             {register:?} would need a conversion it does not emit"
        ))))
    }
}

#[allow(clippy::result_large_err)]
pub fn launch_ref(
    client: &Client,
    lhs: InputBinding,
    rhs: InputBinding,
    out: TensorBinding,
    strategy: &BlueprintStrategy<(), CpuGemmRoutine>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    let sz = dtypes.acc_global.size();

    validate_strided(&lhs.data().strides)?;
    validate_strided(&rhs.data().strides)?;
    validate_strided(&out.strides)?;

    if matches!(lhs, InputBinding::Quantized { .. })
        || matches!(rhs, InputBinding::Quantized { .. })
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "CpuGemm does not support quantized inputs".to_string(),
        )));
    }

    validate_single_type(dtypes, MatmulIdent::Lhs)?;
    validate_single_type(dtypes, MatmulIdent::Rhs)?;

    // Logical dims folded from each operand's physical shape (it may be a higher-rank tiled
    // buffer): `k` on lhs's trailing axis, `n` on rhs's, leading dims each operand's own (possibly
    // broadcast) batch shape.
    let (lhs_batches, m, k) = logical_dims(lhs.data());
    let (rhs_batches, _, n) = logical_dims(rhs.data());
    let out_batches = broadcast_batches(&lhs_batches, &rhs_batches).ok_or_else(|| {
        MatmulSetupError::InvalidConfig(Box::new(format!(
            "CpuGemm: batch shapes do not broadcast, lhs:{lhs_batches:?} rhs:{rhs_batches:?}"
        )))
    })?;

    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type())
        .max(out.required_address_type(sz));

    // CpuGemm reads only `(m, n, k, batches)` + global dtypes off the problem; each operand's
    // physical layout rides in its own strides, so the matrix-layout args are placeholders.
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        lhs_batches[..].into(),
        rhs_batches[..].into(),
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        lhs.scheme(),
        rhs.scheme(),
        dtypes.as_global_elems(),
        address_type,
    );

    // Device context the heuristic reads: SIMD width (N alignment) and core count (parallelism
    // floor). The tiled path has no launch pipeline to derive it from, so it builds it here.
    let device_settings = DeviceSettings {
        client: client.clone(),
        plane_dim: 1,
        vector_sizes: AvailableVectorSizes::from_type_sizes(client, sz, sz, sz).pick_max()?,
        max_cube_count: client.properties().hardware.max_cube_count,
    };

    let blueprint = CpuGemmRoutine::blueprint(strategy, &problem, &device_settings)?;

    // Output batch dims that survive (extent > 1).
    let batch: Vec<usize> = (0..out_batches.len())
        .filter(|&p| out_batches[p] > 1)
        .collect();

    let batch_axes: Vec<Axis> = batch.iter().map(|&p| batch_axis(p)).collect();
    let extents: Vec<_> = (batch_axes.iter().zip(&batch))
        .map(|(&a, &p)| (a, out_batches[p]))
        .chain([(M, m), (N, n), (K, k)])
        .collect();

    // The kernel's own statement of the space and its levels, with this launch's extents stamped
    // on: geometry off the concrete extents, overhang checks derived per operand, a storage block
    // matched to its level, all inside the launcher.
    let space = Space::new(&extents);
    // A storage-tiled operand's tile must be the tile of one of this routine's levels, said here
    // on the host rather than by the launch on a worker thread.
    let partitioning = blueprint.partitioning(&space, &batch_axes);
    validate_stored_tile(lhs.data(), "lhs", &space, partitioning.levels(), (M, K))?;
    validate_stored_tile(rhs.data(), "rhs", &space, partitioning.levels(), (K, N))?;
    let plane_size = client.properties().hardware.plane_size_max;
    let launch = Launcher::partitioned(
        client,
        blueprint.partitioning(&space, &batch_axes),
        blueprint.grid(&space, &batch_axes, plane_size),
        KernelForm::Dynamic,
    );

    // One `N` line width shared by `rhs` and the output (the leaf writes the lines it reads);
    // `lhs` is always scalar (broadcast per `K`), so its layout never matters. The launcher
    // owns the gate: both operands unchecked and `N`-contiguous, the width dividing their
    // inner extents and the `N` leaf edge.
    let rhs = rhs.into_data();
    let v = launch.vector_size(
        N,
        &[
            (&Geometry::from(&rhs), &[K, N]),
            (&Geometry::from(&out), &[M, N]),
        ],
        sz,
    );

    // Bind each operand to its binding: the subspace comes off the operand, the batch list and
    // storage tiling are per-binding launch facts. All operands get the full output batch-axis
    // list; the builder right-aligns it to each operand's leading dims (numpy broadcast, size-1
    // dims drop out).
    let out_batch_axes: Vec<Axis> = (0..out_batches.len()).map(batch_axis).collect();
    let a = launch
        .arg(lhs.into_data())
        .subspace(&[M, K])
        .batches(&out_batch_axes)
        .build();
    let b = launch
        .arg(rhs)
        .subspace(&[K, N])
        .batches(&out_batch_axes)
        .vectorize(v)
        .build();
    let c = launch
        .arg(out)
        .subspace(&[M, N])
        .batches(&out_batch_axes)
        .vectorize(v)
        .build();
    cpu_gemm_kernel::launch(
        client,
        launch.cube_count(),
        launch.cube_dim(),
        a.vector_size,
        b.vector_size,
        c.vector_size,
        a.arg(),
        b.arg(),
        c.arg(),
        launch.space_arg(),
        blueprint.clone(),
        batch_axes,
        dtypes.lhs_global,
        dtypes.rhs_global,
        dtypes.acc_global,
        dtypes.acc_register,
    );
    Ok(())
}
