//! Launch wiring for the CyclicCmma routine: the strided entry ([`launch_ref`]) and its
//! TMA twin ([`launch_tma_ref`]), sharing the blueprint/space derivation and the 3-line
//! kernel body.

use cubecl::features::Tma as TmaFeature;
use cubecl::{Runtime, client::ComputeClient, prelude::*};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_tile::{Axis, CubeAxis, Cut, Leaf, Schedule, Space, Strided, Tiling, Tma, WalkOrder};

use crate::{
    definition::{
        AvailableVectorSizes, MatmulAvailabilityError, MatmulElems, MatmulProblem,
        MatmulSetupError, broadcast_batches,
    },
    routines::{
        BlueprintStrategy, DeviceSettings, K, M, N, batch_axis,
        cyclic_cmma::{
            base::{CyclicCmmaBlueprint, CyclicCmmaRoutine},
            kernel::cyclic_cmma_kernel,
        },
        tma_tile::operand_tma,
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

/// The derivation both entries share: reject what the routine can't run, build the
/// [`MatmulProblem`], and resolve the [`CyclicCmmaBlueprint`]. Returns the problem, the
/// plan, and the output's broadcast batch shape.
#[allow(clippy::result_large_err, clippy::type_complexity)]
fn setup<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: &InputBinding<R>,
    rhs: &InputBinding<R>,
    out: &TensorBinding<R>,
    strategy: &BlueprintStrategy<(), CyclicCmmaRoutine>,
    dtypes: &MatmulElems,
) -> Result<(MatmulProblem, CyclicCmmaBlueprint, Vec<usize>), MatmulSetupError> {
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
    Ok((problem, blueprint, out_batches.to_vec()))
}

/// The routine's 4-level space: the cube grid (double-buffered smem stages along `K`);
/// one partition per plane; the contraction-step walk staging each step's operand
/// fragments (`Staged`); the fragment grid the step contracts (`Direct`, walked
/// statically). `batch` lists the surviving (extent > 1) output batch axes, one per
/// cube on `Z`.
fn tile_space(
    blueprint: &CyclicCmmaBlueprint,
    (m, n, k): (usize, usize, usize),
    batch: &[(Axis, usize)],
) -> Space {
    let (i, c) = (blueprint.instruction, blueprint.partition);
    let (stage_m, stage_n) = blueprint.stage();
    let stage_k = blueprint.stage_k;

    let batch_axes: Vec<_> = batch.iter().map(|&(a, _)| a).collect();
    let extents: Vec<_> = batch
        .iter()
        .copied()
        .chain([(M, m), (N, n), (K, k)])
        .collect();

    Tiling::new()
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
        .leaf(Leaf::Cmma { k: i.k })
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
    let (problem, blueprint, out_batches) = setup(client, &lhs, &rhs, &out, strategy, dtypes)?;
    let (m, n, k) = (problem.m, problem.n, problem.k);

    // Output batch dims that survive (extent > 1) ride one-per-cube on Z.
    let batch: Vec<(Axis, usize)> = (0..out_batches.len())
        .filter(|&p| out_batches[p] > 1)
        .map(|p| (batch_axis(p), out_batches[p]))
        .collect();
    let space = tile_space(&blueprint, (m, n, k), &batch);

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
    cyclic_cmma_kernel::launch::<Strided, R>(
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

/// The TMA twin of [`launch_ref`]: both operands delivered by tensor-map bulk copies
/// (`Sync::of` rejects a mix, so the matrix is both-strided or both-TMA — one twin).
/// Gated host-side on the client's TMA feature: `Unavailable` on backends without it
/// (e.g. Metal), so on CUDA it runs or fails to compile — never silently degrades.
#[allow(clippy::result_large_err)]
pub fn launch_tma_ref<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    strategy: &BlueprintStrategy<(), CyclicCmmaRoutine>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    if !client.properties().features.tma.contains(TmaFeature::Base) {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::TmaUnavailable,
        ));
    }

    let (problem, blueprint, out_batches) = setup(client, &lhs, &rhs, &out, strategy, dtypes)?;
    let (m, n, k) = (problem.m, problem.n, problem.k);

    // The descriptor is 3-D `(batch, row, col)` and the operands' bounds are read off it;
    // surviving batch dims need a batch-aware descriptor path not wired yet.
    if out_batches.iter().any(|&b| b > 1) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "CyclicCmma TMA: batched problems are not supported yet".to_string(),
        )));
    }

    let space = tile_space(&blueprint, (m, n, k), &[]);
    let (stage_m, stage_n) = blueprint.stage();
    let stage_k = blueprint.stage_k;

    let launch = space.launcher(client);
    let lhs = lhs.into_data();
    let rhs = rhs.into_data();
    // TMA moves whole boxes, so the operands stay scalar; only the strided output lines up.
    let v_out = launch.vector_size(N, &[(&out, &[M, N])], dtypes.acc_global.size());

    // One bulk copy fills one double-buffered smem stage: the box is the stage.
    let a = operand_tma(
        lhs,
        (1, m, k),
        MatrixLayout::RowMajor,
        (stage_m, stage_k),
        dtypes.lhs_global,
        launch.space().project(&[M, K]),
    );
    let b = operand_tma(
        rhs,
        (1, k, n),
        MatrixLayout::RowMajor,
        (stage_k, stage_n),
        dtypes.rhs_global,
        launch.space().project(&[K, N]),
    );

    // The out binding may carry unit batch dims; labeling them lets the builder drop
    // them as broadcast omissions (surviving batches were rejected above).
    let out_batch_axes: Vec<Axis> = (0..out_batches.len()).map(batch_axis).collect();
    cyclic_cmma_kernel::launch::<Tma, R>(
        client,
        launch.cube_count(),
        launch.cube_dim(),
        a,
        b,
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
