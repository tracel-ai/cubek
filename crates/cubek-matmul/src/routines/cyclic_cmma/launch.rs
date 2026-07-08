//! Launch wiring for the CyclicCmma routine.

use cubecl::{Runtime, client::ComputeClient, prelude::*};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_tile::{Axis, CubeAxis, Cut, Leaf, Schedule, TileArgLaunch, Tiling, WalkOrder};

use crate::{
    definition::{
        AvailableVectorSizes, MatmulElems, MatmulProblem, MatmulSetupError, broadcast_batches,
    },
    routines::{
        BlueprintStrategy, DeviceSettings,
        cyclic_cmma::{
            base::{CyclicCmmaRoutine, K, M, N, batch_axis},
            kernel::cyclic_cmma_kernel,
        },
    },
};

/// A cmma operand must be row-major contiguous: the transport addresses each window by a
/// row stride off a scalar offset, which a col-major or permuted buffer doesn't afford.
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
    // Stage depth: the deepest `d·i.k` dividing `k` (d ≤ 8) whose two double-buffered
    // slots still fit shared memory — fewer, fatter K stages amortize the fill
    // rendezvous; within a stage each plane walks `d` leaf sub-tiles per fragment.
    let smem_budget = 32 * 1024;
    let row_bytes = stage_m * dtypes.lhs_global.size() + stage_n * dtypes.rhs_global.size();
    let stage_k = (1..=8usize)
        .rev()
        .map(|d| d * i.k)
        .find(|&sk| k.is_multiple_of(sk) && 2 * sk * row_bytes <= smem_budget)
        .unwrap_or(i.k);

    // Output batch dims that survive (extent > 1) ride one-per-cube on Z.
    let batch: Vec<usize> = (0..out_batches.len())
        .filter(|&p| out_batches[p] > 1)
        .collect();
    let batch_axes: Vec<_> = batch.iter().map(|&p| batch_axis(p)).collect();
    let extents: Vec<_> = (batch_axes.iter().zip(&batch))
        .map(|(&a, &p)| (a, out_batches[p]))
        .chain([(M, m), (N, n), (K, k)])
        .collect();

    // Three levels: the cube grid whose double-buffered walk rotates `stage_k`-deep smem
    // stages along `K` (filled cooperatively); the stage split one partition per plane;
    // the partition level the microkernel contracts with resident fragments.
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
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axes(&batch_axes, Cut::sequential(1))
                .axis(M, Cut::sequential(i.m))
                .axis(N, Cut::sequential(i.n))
                .axis(K, Cut::sequential(i.k))
        })
        .build()
        .with_leaf(Leaf::Cmma);

    let cube_count = space.cube_count();
    let cube_dim = space.cube_dim(client);

    // The kernel keys on a fully-dynamic space so distinct shapes reuse one compiled
    // kernel; the blueprint validated divisibility, so nothing is bounds-checked.
    let global_space = space.all_dynamic();

    // Line each operand's contiguous innermost axis (`K` on lhs, `N` on rhs/out) at the
    // widest width dividing every tile edge on it — the shape is exactly divisible, so the
    // fill moves whole lines; the cmma transport addresses the scalar buffer underneath.
    let line = |elem: usize, inner_edge: usize| {
        client
            .io_optimized_vector_sizes(elem)
            .filter(|&v| inner_edge.is_multiple_of(v))
            .max()
            .unwrap_or(1)
    };
    let v_lhs = line(dtypes.lhs_global.size(), i.k);
    let v_rhs = line(dtypes.rhs_global.size(), i.n);
    let v_out = line(dtypes.acc_global.size(), i.n);

    let rank = out_batches.len();
    let out_batch_axes: Vec<Axis> = (0..rank).map(batch_axis).collect();
    cyclic_cmma_kernel::launch::<R>(
        client,
        cube_count,
        cube_dim,
        TileArgLaunch::source(lhs.into_data())
            .space(&global_space)
            .subspace(&[M, K])
            .batches(&out_batch_axes[rank - lhs_batches.len()..])
            .vectorize(v_lhs)
            .checked(false)
            .build(),
        TileArgLaunch::source(rhs.into_data())
            .space(&global_space)
            .subspace(&[K, N])
            .batches(&out_batch_axes[rank - rhs_batches.len()..])
            .vectorize(v_rhs)
            .checked(false)
            .build(),
        TileArgLaunch::source(out)
            .space(&global_space)
            .subspace(&[M, N])
            .batches(&out_batch_axes)
            .vectorize(v_out)
            .checked(false)
            .build(),
        dtypes.lhs_global,
        dtypes.rhs_global,
        dtypes.acc_global,
    );

    Ok(())
}
