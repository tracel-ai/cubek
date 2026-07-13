//! Launch wiring for the CpuGemm routine.

use cubecl::{Runtime, client::ComputeClient, prelude::*};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_tile::{Axis, CubeAxis, Cut, Leaf, Schedule, Tiling, WalkOrder};

use crate::{
    definition::{
        AvailableVectorSizes, MatmulElems, MatmulProblem, MatmulSetupError, broadcast_batches,
    },
    routines::{
        BlueprintStrategy, DeviceSettings, K, M, N, batch_axis,
        cpu_gemm::{base::CpuGemmRoutine, kernel::cpu_gemm_kernel},
    },
};

/// A binding together with its storage-tiling depth: `levels` nested `[grid…, leaf]` splits per
/// matrix axis (`0` = a plain strided buffer). It's the one piece of physical layout that the
/// binding's own shape/strides don't reveal — a tiled buffer just looks like a higher-rank strided
/// one — so it's all production carries; row-vs-col-major rides in the strides, and the per-operand
/// view layout is built at load time by [`cubek_tile::StridedTileArgLaunch`].
pub struct WithLayout<B> {
    pub binding: B,
    pub levels: usize,
}

impl<R: Runtime> WithLayout<InputBinding<R>> {
    /// A plain strided operand (`levels = 0`). Errors on a binding contiguous in neither matrix axis.
    #[allow(clippy::result_large_err)]
    pub fn strided_input(binding: InputBinding<R>) -> Result<Self, MatmulSetupError> {
        validate_strided(&binding.data().strides)?;
        Ok(Self { binding, levels: 0 })
    }
}

impl<R: Runtime> WithLayout<TensorBinding<R>> {
    /// A plain strided operand (`levels = 0`). Errors on a binding contiguous in neither matrix axis.
    #[allow(clippy::result_large_err)]
    pub fn strided_output(binding: TensorBinding<R>) -> Result<Self, MatmulSetupError> {
        validate_strided(&binding.strides)?;
        Ok(Self { binding, levels: 0 })
    }
}

/// A strided matmul operand must be contiguous along one of its two matrix axes.
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

/// Fold a physical shape back to logical `(batches, rows, cols)` given its storage-tiling `levels`:
/// the trailing `2·(levels + 1)` dims are the matrix's level-major `[row, col]` factors (their
/// products are `rows`/`cols`), everything before them is the batch shape.
fn fold_logical(shape: &[usize], levels: usize) -> (Vec<usize>, usize, usize) {
    let split = shape.len() - 2 * (levels + 1);
    let (mut rows, mut cols) = (1, 1);
    for (i, &d) in shape[split..].iter().enumerate() {
        if i % 2 == 0 {
            rows *= d;
        } else {
            cols *= d;
        }
    }
    (shape[..split].to_vec(), rows, cols)
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: WithLayout<InputBinding<R>>,
    rhs: WithLayout<InputBinding<R>>,
    out: WithLayout<TensorBinding<R>>,
    strategy: &BlueprintStrategy<(), CpuGemmRoutine>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    let (lhs, lhs_levels) = (lhs.binding, lhs.levels);
    let (rhs, rhs_levels) = (rhs.binding, rhs.levels);
    let (out, out_levels) = (out.binding, out.levels);
    let sz = dtypes.acc_global.size();

    if matches!(lhs, InputBinding::Quantized { .. })
        || matches!(rhs, InputBinding::Quantized { .. })
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "CpuGemm does not support quantized inputs".to_string(),
        )));
    }

    // Logical dims folded from each operand's physical shape (it may be a higher-rank tiled
    // buffer): `k` on lhs's trailing axis, `n` on rhs's, leading dims each operand's own (possibly
    // broadcast) batch shape.
    let (lhs_batches, m, k) = fold_logical(lhs.shape(), lhs_levels);
    let (rhs_batches, _, n) = fold_logical(rhs.shape(), rhs_levels);
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
    // floor). CpuGemm isn't a BatchMatmulRoutine, so we build this bundle ourselves.
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

    // A cube owns a tile of `planes.m × planes.n` leaves; each plane (a CPU worker thread)
    // owns one leaf.
    let leaf = blueprint.instruction;
    let planes = blueprint.planes;
    let cube_m = planes.m * leaf.m;
    let cube_n = planes.n * leaf.n;

    let batch_axes: Vec<_> = batch.iter().map(|&p| batch_axis(p)).collect();
    let extents: Vec<_> = (batch_axes.iter().zip(&batch))
        .map(|(&a, &p)| (a, out_batches[p]))
        .chain([(M, m), (N, n), (K, k)])
        .collect();

    // One level per decomposition, coarse→fine: the cube grid (a serial loop on CPU), then the
    // plane split (the parallel worker threads). Batch axes ride one-per-cube on Z then iterate
    // sequentially; K is contracted sequentially in both leaves.
    let space = Tiling::new()
        .extents(&extents)
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axes(&batch_axes, Cut::cube(CubeAxis::Z, 1))
                .axis(M, Cut::cube(CubeAxis::X, cube_m))
                .axis(N, Cut::cube(CubeAxis::Y, cube_n))
                .axis(K, Cut::sequential(k))
        })
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axes(&batch_axes, Cut::sequential(1))
                .axis(M, Cut::plane(leaf.m))
                .axis(N, Cut::plane(leaf.n))
                .axis(K, Cut::sequential(leaf.k))
        })
        .leaf(Leaf::Register);

    // Geometry off the concrete extents, kernel space fully dynamic (one compiled kernel per
    // shape family), overhang checks derived per operand — all inside the launcher.
    let launch = space.launcher(client);

    // One `N` line width shared by `rhs` and the output (the leaf writes the lines it reads);
    // `lhs` is always scalar (broadcast per `K`), so its layout never matters. The launcher
    // owns the gate: both operands unchecked and `N`-contiguous, the width dividing their
    // inner extents and the `N` leaf edge.
    let rhs = rhs.into_data();
    let v = launch.vector_size(N, &[(&rhs, &[K, N]), (&out, &[M, N])], sz);

    // Load each operand through the tile builder over its subspace (the matrix `[row, col]` plus
    // batches). All operands get the full output batch-axis list; the builder right-aligns it to
    // each operand's leading dims (numpy broadcast, size-1 dims drop out).
    let out_batch_axes: Vec<Axis> = (0..out_batches.len()).map(batch_axis).collect();
    cpu_gemm_kernel::launch::<R>(
        client,
        launch.cube_count(),
        launch.cube_dim(),
        launch
            .arg(lhs.into_data())
            .subspace(&[M, K])
            .batches(&out_batch_axes)
            .levels(lhs_levels)
            .build(),
        launch
            .arg(rhs)
            .subspace(&[K, N])
            .batches(&out_batch_axes)
            .levels(rhs_levels)
            .vectorize(v)
            .build(),
        launch
            .arg(out)
            .subspace(&[M, N])
            .batches(&out_batch_axes)
            .levels(out_levels)
            .vectorize(v)
            .build(),
        dtypes.lhs_global,
        dtypes.rhs_global,
        dtypes.acc_global,
    );

    Ok(())
}
