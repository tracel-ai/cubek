//! Launch wiring for the CpuGemm routine.

use cubecl::{Runtime, client::ComputeClient, prelude::*};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_tile::{Axis, CubeAxis, Cut, Schedule, TileArgLaunch, Tiling, WalkOrder};

use crate::{
    definition::{
        AvailableVectorSizes, MatmulElems, MatmulProblem, MatmulSetupError, broadcast_batches,
    },
    routines::{
        BlueprintStrategy, DeviceSettings,
        cpu_gemm::{
            base::{CpuGemmRoutine, K, M, N, batch_axis},
            kernel::cpu_gemm_kernel,
        },
    },
};

/// A binding together with its storage-tiling depth: `levels` nested `[grid…, leaf]` splits per
/// matrix axis (`0` = a plain strided buffer). It's the one piece of physical layout that the
/// binding's own shape/strides don't reveal — a tiled buffer just looks like a higher-rank strided
/// one — so it's all production carries; row-vs-col-major rides in the strides, and the per-operand
/// view layout is built at load time by [`TileArgLaunch`].
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

/// Whether an operand vectorizes along `N`: its innermost physical axis is contiguous, so a kernel
/// reading `Vector<E, V>` lands on whole lines. That axis is `cols` for a plain row-major buffer and
/// the `N` leaf tile for a packed one — a tiled operand qualifies exactly like a strided one, the
/// vector just lands within a leaf block. Col-major (rows contiguous) does not.
fn vectorizes_n(strides: &[usize]) -> bool {
    strides.last() == Some(&1)
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
    let out = out.binding;
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

    // Vectorize `N` only when both `rhs` and the output are `N`-contiguous (innermost stride 1):
    // a plain row-major buffer, or a packed one whose `N` leaf tile is contiguous. Then a kernel
    // reading `Vector<E, V>` lands on whole lines. Col-major falls back to scalar (`V = 1`), as does
    // a width that doesn't divide the innermost extent — the logical `N` when strided, the leaf tile
    // edge when packed. `lhs` is always scalar (broadcast per `K`), so its layout never matters.
    let rhs_inner = *rhs.shape().last().unwrap();
    let out_inner = *out.shape.last().unwrap();
    let v = (vectorizes_n(&rhs.data().strides) && vectorizes_n(&out.strides))
        .then(|| {
            client
                .io_optimized_vector_sizes(sz)
                .filter(|&v| {
                    rhs_inner.is_multiple_of(v)
                        && out_inner.is_multiple_of(v)
                        && blueprint.instruction.n.is_multiple_of(v)
                })
                .max()
        })
        .flatten()
        .unwrap_or(1);

    // Output batch dims that survive (extent > 1).
    let batch: Vec<usize> = (0..out_batches.len())
        .filter(|&p| out_batches[p] > 1)
        .collect();

    // A cube owns a tile of `planes.m × planes.n` leaves; each plane (a CPU worker thread)
    // owns one leaf.
    let leaf = blueprint.instruction;
    let planes = blueprint.planes;
    let tile_n_lines = leaf.n / v;
    let cube_m = planes.m * leaf.m;
    let cube_n_lines = planes.n * tile_n_lines;

    let batch_axes: Vec<_> = batch.iter().map(|&p| batch_axis(p)).collect();
    let extents: Vec<_> = (batch_axes.iter().zip(&batch))
        .map(|(&a, &p)| (a, out_batches[p]))
        .chain([(M, m), (N, n / v), (K, k)])
        .collect();

    // One level per decomposition, coarse→fine: the cube grid (a serial loop on CPU), then the
    // plane split (the parallel worker threads). Batch axes ride one-per-cube on Z then iterate
    // sequentially; K is contracted sequentially in both leaves.
    let space = Tiling::new()
        .extents(&extents)
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axes(&batch_axes, Cut::cube(CubeAxis::Z, 1))
                .axis(M, Cut::cube(CubeAxis::X, cube_m))
                .axis(N, Cut::cube(CubeAxis::Y, cube_n_lines))
                .axis(K, Cut::sequential(k))
        })
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axes(&batch_axes, Cut::sequential(1))
                .axis(M, Cut::plane(leaf.m))
                .axis(N, Cut::plane(tile_n_lines))
                .axis(K, Cut::sequential(leaf.k))
        })
        .build();

    let cube_count = space.cube_count();
    let cube_dim = space.cube_dim(client);

    // The kernel keys on a fully-dynamic space (extents → runtime scalars) so distinct input
    // shapes reuse one compiled kernel. Consuming `space` into `kernel_space` means the grid
    // above must be read first — the wrong order won't compile.
    let global_space = space.all_dynamic();

    // The stage tile (`cube_m`/`cube_n`) is the overhang granularity for M/N — within a
    // cube the plane split is exact — and the leaf `k` for K.
    let check_m = !m.is_multiple_of(cube_m);
    let check_n = !n.is_multiple_of(planes.n * leaf.n);
    let check_k = !k.is_multiple_of(leaf.k);

    // Load each operand through the tile builder over its subspace (the matrix `[row, col]` plus
    // batches). Each operand's batches right-align to the output's `rank` batch axes (numpy
    // broadcast — a size-1 dim then drops out in the builder), so they're the trailing slice of
    // `out_batch_axes`. `lhs` stages scalar (`v = 1`), `rhs`/`out` carry the line size; each
    // bounds-checks its overhang. All the layout/broadcast mechanics live in the builder.
    let rank = out_batches.len();
    let out_batch_axes: Vec<Axis> = (0..rank).map(batch_axis).collect();
    cpu_gemm_kernel::launch::<R>(
        client,
        cube_count,
        cube_dim,
        TileArgLaunch::source(lhs.into_data())
            .space(&global_space)
            .subspace(&[M, K])
            .batches(&out_batch_axes[rank - lhs_batches.len()..])
            .checked(check_m || check_k)
            .build(),
        TileArgLaunch::source(rhs.into_data())
            .space(&global_space)
            .subspace(&[K, N])
            .batches(&out_batch_axes[rank - rhs_batches.len()..])
            .vectorize(v)
            .checked(check_k || check_n)
            .build(),
        TileArgLaunch::source(out)
            .space(&global_space)
            .subspace(&[M, N])
            .batches(&out_batch_axes)
            .vectorize(v)
            .checked(check_m || check_n)
            .build(),
        dtypes.lhs_global,
        dtypes.rhs_global,
        dtypes.acc_global,
    );

    Ok(())
}
