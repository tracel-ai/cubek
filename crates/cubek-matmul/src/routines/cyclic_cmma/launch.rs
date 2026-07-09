//! Launch wiring for the CyclicCmma routine.

use cubecl::{Runtime, client::ComputeClient, prelude::*};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_tile::{Axis, CubeAxis, Cut, Leaf, Schedule, Tiling, WalkOrder};

use crate::{
    definition::{
        AvailableVectorSizes, MatmulElems, MatmulProblem, MatmulSetupError, broadcast_batches,
    },
    routines::{
        BlueprintStrategy, DeviceSettings, K, M, N, batch_axis,
        cyclic_cmma::{base::CyclicCmmaRoutine, kernel::cyclic_cmma_kernel},
    },
};

/// A cmma operand must be row-major contiguous: the transport addresses each window
/// by a row stride off a scalar offset.
#[allow(clippy::result_large_err)]
fn validate_row_major(strides: &[usize]) -> Result<(), MatmulSetupError> {
    if strides.last() == Some(&1) {
        Ok(())
    } else {
        Err(MatmulSetupError::InvalidConfig(Box::new(
            "CyclicCmma: operand is not row-major contiguous".to_string(),
        )))
    }
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    strategy: &BlueprintStrategy<(), CyclicCmmaRoutine>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    if matches!(lhs, InputBinding::Quantized { .. })
        || matches!(rhs, InputBinding::Quantized { .. })
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "CyclicCmma does not support quantized inputs".to_string(),
        )));
    }
    validate_row_major(&lhs.data().strides)?;
    validate_row_major(&rhs.data().strides)?;
    validate_row_major(&out.strides)?;

    // Logical dims off each strided operand: trailing two axes are the matrix, leading
    // dims its own (possibly broadcast) batch shape.
    let split = |shape: &[usize]| {
        let r = shape.len();
        (shape[..r - 2].to_vec(), shape[r - 2], shape[r - 1])
    };
    let (lhs_batches, m, k) = split(lhs.shape());
    let (rhs_batches, _, n) = split(rhs.shape());
    let out_batches = broadcast_batches(&lhs_batches, &rhs_batches).ok_or_else(|| {
        MatmulSetupError::InvalidConfig(Box::new(format!(
            "CyclicCmma: batch shapes do not broadcast, lhs:{lhs_batches:?} rhs:{rhs_batches:?}"
        )))
    })?;

    let sz = dtypes.acc_global.size();
    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type())
        .max(out.required_address_type(sz));

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

    let device_settings = DeviceSettings {
        client: client.clone(),
        plane_dim: client.properties().hardware.plane_size_max,
        vector_sizes: AvailableVectorSizes::from_type_sizes(client, sz, sz, sz).pick_max()?,
        max_cube_count: client.properties().hardware.max_cube_count,
    };

    let blueprint = CyclicCmmaRoutine::blueprint(strategy, &problem, &device_settings)?;
    let (i, c) = (blueprint.instruction, blueprint.partition);
    let (stage_m, stage_n) = blueprint.stage();
    let stage_k = blueprint.stage_k;

    // Output batch dims that survive (extent > 1) ride one-per-cube on Z.
    let batch: Vec<usize> = (0..out_batches.len())
        .filter(|&p| out_batches[p] > 1)
        .collect();
    let batch_axes: Vec<_> = batch.iter().map(|&p| batch_axis(p)).collect();
    let extents: Vec<_> = (batch_axes.iter().zip(&batch))
        .map(|(&a, &p)| (a, out_batches[p]))
        .chain([(M, m), (N, n), (K, k)])
        .collect();

    // Four levels: the cube grid (double-buffered smem stages along `K`); one partition
    // per plane; the contraction-step walk staging each step's operand fragments
    // (`Staged`); the fragment grid the step contracts (`Direct`, walked statically).
    let space = Tiling::new()
        .extents(&extents)
        .level(WalkOrder::RowMajor, Schedule::DoubleBuffered, |l| {
            l.axes(&batch_axes, Cut::cube(CubeAxis::Z, 1))
                .axis(M, Cut::cube(CubeAxis::X, stage_m))
                .axis(N, Cut::cube(CubeAxis::Y, stage_n))
                .axis(K, Cut::sequential(stage_k))
        })
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axes(&batch_axes, Cut::sequential(1))
                .axis(M, Cut::plane(c.m * i.m))
                .axis(N, Cut::plane(c.n * i.n))
                .axis(K, Cut::sequential(stage_k))
        })
        .level(WalkOrder::RowMajor, Schedule::Staged, |l| {
            l.axes(&batch_axes, Cut::sequential(1))
                .axis(M, Cut::sequential(c.m * i.m))
                .axis(N, Cut::sequential(c.n * i.n))
                .axis(K, Cut::sequential(i.k))
        })
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axes(&batch_axes, Cut::sequential(1))
                .axis(M, Cut::sequential(i.m))
                .axis(N, Cut::sequential(i.n))
                .axis(K, Cut::sequential(i.k))
        })
        .leaf(Leaf::Cmma { k: i.k });

    let launch = space.launcher(client);

    // Line each operand's contiguous innermost axis (`K` on lhs, `N` on rhs/out) at the
    // widest width the launcher's gate allows, per-operand since dtypes differ.
    let lhs = lhs.into_data();
    let rhs = rhs.into_data();
    let v_lhs = launch.vector_size(K, &[(&lhs, &[M, K])], dtypes.lhs_global.size());
    let v_rhs = launch.vector_size(N, &[(&rhs, &[K, N])], dtypes.rhs_global.size());
    let v_out = launch.vector_size(N, &[(&out, &[M, N])], dtypes.acc_global.size());

    // Every operand gets the full output batch-axis list; the builder right-aligns it to each
    // operand's leading dims (numpy broadcast, size-1 dims drop out).
    let out_batch_axes: Vec<Axis> = (0..out_batches.len()).map(batch_axis).collect();
    cyclic_cmma_kernel::launch::<R>(
        client,
        launch.cube_count(),
        launch.cube_dim(),
        launch
            .arg(lhs)
            .subspace(&[M, K])
            .batches(&out_batch_axes)
            .vectorize(v_lhs)
            .build(),
        launch
            .arg(rhs)
            .subspace(&[K, N])
            .batches(&out_batch_axes)
            .vectorize(v_rhs)
            .build(),
        launch
            .arg(out)
            .subspace(&[M, N])
            .batches(&out_batch_axes)
            .vectorize(v_out)
            .build(),
        dtypes.lhs_global,
        dtypes.rhs_global,
        dtypes.acc_global,
    );

    Ok(())
}
