//! Launch wiring for the Cmma routine: one entry ([`launch_ref`]) serving both
//! deliveries; the blueprint decides, and only the operand construction differs.

use cubecl::{Runtime, client::ComputeClient, prelude::*};
use cubek_std::launch::tma::tma_operand;
use cubek_std::{InputBinding, MatrixLayout};
use cubek_tile::{
    Axis, CubeAxis, Cut, Delivery, Leaf, Schedule, Space, Strided, Tiling, Tma, TmaArgLaunch,
    WalkOrder,
};

use crate::{
    components::global::read::stride_align_bits,
    definition::{
        AvailableVectorSizes, MatmulElems, MatmulProblem, MatmulSetupError, broadcast_batches,
    },
    routines::{
        BlueprintStrategy, DeviceSettings, K, M, N, batch_axis,
        cmma::{
            base::{CmmaBlueprint, CmmaRoutine},
            kernel::cmma_kernel,
        },
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
            "Cmma: operand is not row-major contiguous".to_string(),
        )))
    }
}

/// The derivation both entries share: reject what the routine can't run, build the
/// [`MatmulProblem`], and resolve the [`CmmaBlueprint`]. Returns the problem, the
/// plan, and the output's broadcast batch shape.
#[allow(clippy::result_large_err, clippy::type_complexity)]
fn setup<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: &InputBinding<R>,
    rhs: &InputBinding<R>,
    out: &TensorBinding<R>,
    strategy: &BlueprintStrategy<(), CmmaRoutine>,
    dtypes: &MatmulElems,
) -> Result<(MatmulProblem, CmmaBlueprint, Vec<usize>), MatmulSetupError> {
    if matches!(lhs, InputBinding::Quantized { .. })
        || matches!(rhs, InputBinding::Quantized { .. })
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "Cmma does not support quantized inputs".to_string(),
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
            "Cmma: batch shapes do not broadcast, lhs:{lhs_batches:?} rhs:{rhs_batches:?}"
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

    let blueprint = CmmaRoutine::blueprint(strategy, &problem, &device_settings)?;

    // The descriptor requires every non-contiguous stride 16-byte aligned; the problem's
    // strides are synthesized, so check the real bindings here.
    if blueprint.delivery.is_tma() {
        let aligned = |strides: &[usize], dtype: &StorageType| {
            stride_align_bits(strides, &MatrixLayout::RowMajor, dtype) >= 4
        };
        if !aligned(&lhs.data().strides, &dtypes.lhs_global)
            || !aligned(&rhs.data().strides, &dtypes.rhs_global)
        {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Cmma TMA: strides must be aligned to 16 bytes".to_string(),
            )));
        }
    }
    Ok((problem, blueprint, out_batches.to_vec()))
}

/// The routine's 4-level space: the cube grid (double-buffered smem stages along `K`);
/// one partition per plane; the contraction-step walk staging each step's operand
/// fragments (`Staged`); the fragment grid the step contracts (`Direct`, walked
/// statically). `batch` lists the surviving (extent > 1) output batch axes, one per
/// cube on `Z`.
fn tile_space(
    blueprint: &CmmaBlueprint,
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

/// The one entry for both deliveries: the shared geometry (space, launcher, out arg) is
/// built once, and only the operand construction dispatches on the blueprint's
/// [`Delivery`]. A TMA plan is fully validated by then, so on CUDA it runs or fails to
/// compile, never silently degrades.
#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    strategy: &BlueprintStrategy<(), CmmaRoutine>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    let (problem, blueprint, out_batches) = setup(client, &lhs, &rhs, &out, strategy, dtypes)?;
    let (m, n, k) = (problem.m, problem.n, problem.k);

    // Output batch dims that survive (extent > 1) ride one-per-cube on Z (none under TMA;
    // the blueprint rejected them).
    let batch: Vec<(Axis, usize)> = (0..out_batches.len())
        .filter(|&p| out_batches[p] > 1)
        .map(|p| (batch_axis(p), out_batches[p]))
        .collect();
    let space = tile_space(&blueprint, (m, n, k), &batch);

    let launch = space.launcher(client);
    let lhs = lhs.into_data();
    let rhs = rhs.into_data();

    // The out is strided under either delivery: lined at the widest width the launcher's
    // gate allows, labeled with the full output batch-axis list (the builder right-aligns
    // it, numpy broadcast, size-1 dims drop out).
    let v_out = launch.vector_size(N, &[(&out, &[M, N])], dtypes.acc_global.size());
    let out_batch_axes: Vec<Axis> = (0..out_batches.len()).map(batch_axis).collect();
    let c = launch
        .arg(out)
        .subspace(&[M, N])
        .batches(&out_batch_axes)
        .vectorize(v_out)
        .build();
    let (cube_count, cube_dim) = (launch.cube_count(), launch.cube_dim());

    match blueprint.delivery {
        // Line each operand's contiguous innermost axis (`K` on lhs, `N` on rhs) at the
        // widest width the launcher's gate allows, per-operand since dtypes differ.
        Delivery::Strided => {
            let v_lhs = launch.vector_size(K, &[(&lhs, &[M, K])], dtypes.lhs_global.size());
            let v_rhs = launch.vector_size(N, &[(&rhs, &[K, N])], dtypes.rhs_global.size());
            cmma_kernel::launch::<Strided, R>(
                client,
                cube_count,
                cube_dim,
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
                c,
                dtypes.lhs_global,
                dtypes.rhs_global,
                dtypes.acc_global,
            );
        }
        // One bulk copy fills one double-buffered smem stage: the box is the stage. The
        // operands stay scalar (TMA moves whole boxes).
        Delivery::Tma => {
            let (stage_m, stage_n) = blueprint.stage();
            let stage_k = blueprint.stage_k;
            let (map_a, transposed_a) = tma_operand(
                lhs,
                1,
                MatrixLayout::RowMajor,
                (stage_m, stage_k),
                dtypes.lhs_global,
                TensorMapSwizzle::None,
            );
            let a = TmaArgLaunch::tensor_map(
                map_a,
                launch.space().project(&[M, K]),
                (1, m as u32, k as u32),
                transposed_a,
            );
            let (map_b, transposed_b) = tma_operand(
                rhs,
                1,
                MatrixLayout::RowMajor,
                (stage_k, stage_n),
                dtypes.rhs_global,
                TensorMapSwizzle::None,
            );
            let b = TmaArgLaunch::tensor_map(
                map_b,
                launch.space().project(&[K, N]),
                (1, k as u32, n as u32),
                transposed_b,
            );
            cmma_kernel::launch::<Tma, R>(
                client,
                cube_count,
                cube_dim,
                a,
                b,
                c,
                dtypes.lhs_global,
                dtypes.rhs_global,
                dtypes.acc_global,
            );
        }
    }

    Ok(())
}
