//! Launch wiring for the CpuGemm routine.

use cubecl::{CubeDim, Runtime, client::ComputeClient, prelude::*};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_tile::{
    Axis, ByAxis, ComputeScope, Coverage, CubeAxis, Distribution, Partitioner, Space, Spread,
    TileArgLaunch,
};

use crate::{
    definition::{
        AvailableVectorSizes, InnerLayout, MatmulElems, MatmulProblem, MatmulSetupError,
        broadcast_batches,
    },
    routines::{
        BlueprintStrategy, DeviceSettings,
        cpu_gemm::{
            base::{CpuGemmBlueprint, CpuGemmRoutine, K, M, N, batch_axis},
            kernel::cpu_gemm_kernel,
        },
    },
};

/// A binding together with the [`InnerLayout`] that folds its (possibly higher-rank,
/// tiled) physical shape back to the logical `(batches, rows, cols)`.
pub struct WithLayout<B> {
    pub binding: B,
    pub layout: InnerLayout,
}

impl<R: Runtime> WithLayout<InputBinding<R>> {
    /// Deduce a plain strided layout from the binding's strides. Valid only for
    /// non-tiled bindings; errors on a binding contiguous in neither matrix axis.
    #[allow(clippy::result_large_err)]
    pub fn strided_input(binding: InputBinding<R>) -> Result<Self, MatmulSetupError> {
        let layout = InnerLayout::from_shape_and_strides(binding.shape(), &binding.data().strides)?;
        Ok(Self { binding, layout })
    }
}

impl<R: Runtime> WithLayout<TensorBinding<R>> {
    /// Deduce a plain strided layout from the binding's strides. Valid only for
    /// non-tiled bindings; errors on a binding contiguous in neither matrix axis.
    #[allow(clippy::result_large_err)]
    pub fn strided_output(binding: TensorBinding<R>) -> Result<Self, MatmulSetupError> {
        let layout = InnerLayout::from_shape_and_strides(&binding.shape, &binding.strides)?;
        Ok(Self { binding, layout })
    }
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
    let (lhs, lhs_layout) = (lhs.binding, lhs.layout);
    let (rhs, rhs_layout) = (rhs.binding, rhs.layout);
    let (out, out_layout) = (out.binding, out.layout);

    if matches!(lhs, InputBinding::Quantized { .. })
        || matches!(rhs, InputBinding::Quantized { .. })
    {
        return Err({
            let msg = "CpuGemm does not support quantized inputs".to_string();
            MatmulSetupError::InvalidConfig(Box::new(msg))
        });
    }

    // Logical dims from each operand's imposed layout (its physical shape may be a
    // higher-rank tiled buffer). `k` rides lhs's trailing axis, `n` rhs's; the leading
    // dims are each operand's own (possibly broadcast) batch shape.
    let (lhs_batches, m, k) = lhs_layout.logical_dims(lhs.shape());
    let (rhs_batches, _, n) = rhs_layout.logical_dims(rhs.shape());
    let out_batches = broadcast_batches(&lhs_batches, &rhs_batches).ok_or_else(|| {
        MatmulSetupError::InvalidConfig(Box::new(format!(
            "CpuGemm: batch shapes do not broadcast, lhs:{lhs_batches:?} rhs:{rhs_batches:?}"
        )))
    })?;

    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type())
        .max(out.required_address_type(dtypes.acc_global.size()));

    // CpuGemm reads only `(m, n, k, batches)` and the global dtypes off the problem; the
    // physical layout lives in each operand's `InnerLayout`, so the problem's matrix
    // layout is a don't-care placeholder. Real batch shapes feed the heuristic and the
    // broadcast bookkeeping.
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

    // Device context the heuristic reads: SIMD width (for N alignment) and core count
    // (for the parallelism floor). CpuGemm isn't a BatchMatmulRoutine, so we build the
    // bundle ourselves rather than going through the pipeline.
    let sz = dtypes.acc_global.size();
    let device_settings = DeviceSettings {
        client: client.clone(),
        plane_dim: 1,
        vector_sizes: AvailableVectorSizes::from_type_sizes(client, sz, sz, sz).pick_max()?,
        max_cube_count: client.properties().hardware.max_cube_count,
    };

    let blueprint = CpuGemmRoutine::blueprint(strategy, &problem, &device_settings)?;

    // Vectorize `N` only when both `rhs` and the output keep it contiguous (both
    // row-major): then a kernel reading `Vector<E, V>` lands on whole lines. Any
    // other layout — col-major or tiled — falls back to scalar (`V = 1`). `lhs` is
    // always scalar (broadcast per `K`), so its layout never matters here.
    let vector_size = matches!(rhs_layout, InnerLayout::RowMajor)
        .then_some(matches!(out_layout, InnerLayout::RowMajor))
        .filter(|&x| x)
        .and_then(|_| {
            client
                .io_optimized_vector_sizes(sz)
                .filter(|&v| n.is_multiple_of(v) && blueprint.tile_n.is_multiple_of(v))
                .max()
        })
        .unwrap_or(1);

    let lhs_data = lhs.into_data();
    let rhs_data = rhs.into_data();

    launch_vectorized::<R>(
        client,
        lhs_data,
        rhs_data,
        out,
        lhs_layout,
        rhs_layout,
        out_layout,
        &lhs_batches,
        &rhs_batches,
        &out_batches,
        m,
        n,
        k,
        blueprint,
        vector_size,
        dtypes.lhs_global,
        dtypes.rhs_global,
        dtypes.acc_global,
    );

    Ok(())
}

/// Launch the kernel with line size `v`. `N` rides in line units — the space extent and
/// the `N` tile edge are divided by `v`, so each step covers one `Vector<E, v>`. `lhs` is
/// staged scalar; `rhs` and the output carry the line size.
#[allow(clippy::too_many_arguments)]
fn launch_vectorized<R: Runtime>(
    client: &ComputeClient<R>,
    lhs_data: TensorBinding<R>,
    rhs_data: TensorBinding<R>,
    out: TensorBinding<R>,
    lhs_layout: InnerLayout,
    rhs_layout: InnerLayout,
    out_layout: InnerLayout,
    lhs_batches: &[usize],
    rhs_batches: &[usize],
    out_batches: &[usize],
    m: usize,
    n: usize,
    k: usize,
    blueprint: CpuGemmBlueprint,
    v: usize,
    lhs_dtype: StorageType,
    rhs_dtype: StorageType,
    acc_dtype: StorageType,
) {
    // Output batch dims that survive (extent > 1)
    let batch: Vec<usize> = (0..out_batches.len())
        .filter(|&p| out_batches[p] > 1)
        .collect();

    let extents: Vec<_> = (batch.iter().map(|&p| (batch_axis(p), out_batches[p])))
        .chain([(M, m), (N, n / v), (K, k)])
        .collect();
    let edges: Vec<_> = (batch.iter().map(|&p| (batch_axis(p), 1)))
        .chain([
            (M, blueprint.tile_m),
            (N, blueprint.tile_n / v),
            (K, blueprint.tile_k),
        ])
        .collect();
    let dists: Vec<_> = (batch.iter().map(|&p| {
        (
            batch_axis(p),
            Distribution::Spatial {
                scope: ComputeScope::Cube(CubeAxis::Z),
                spread: Spread::Contiguous,
                coverage: Coverage::TilesEach(1),
            },
        )
    }))
    .chain([
        (
            M,
            Distribution::Spatial {
                scope: ComputeScope::Cube(CubeAxis::X),
                spread: Spread::Contiguous,
                coverage: Coverage::TilesEach(1),
            },
        ),
        (
            N,
            Distribution::Spatial {
                scope: ComputeScope::Cube(CubeAxis::Y),
                spread: Spread::Contiguous,
                coverage: Coverage::TilesEach(1),
            },
        ),
        (K, Distribution::Sequential),
    ])
    .collect();

    let partitioner = Partitioner::row_major(ByAxis::new(&edges), ByAxis::new(&dists)).staged();
    let space = Space::new(&extents).with_partitioner(partitioner.clone());
    let cube_count = partitioner.cube_count(&space);
    let cube_dim = CubeDim::new_single();

    let check_m = !m.is_multiple_of(blueprint.tile_m);
    let check_n = !n.is_multiple_of(blueprint.tile_n);
    let check_k = !k.is_multiple_of(blueprint.tile_k);

    // `lhs` always staged scalar (`v = 1`); `rhs`/`out` carry the line size. The output
    // rank left-aligns each operand's (possibly shorter) batch shape, numpy-style. Each
    // operand bounds-checks the edges its tile may overhang.
    let rank = out_batches.len();
    cpu_gemm_kernel::launch::<R>(
        client,
        cube_count,
        cube_dim,
        tile_arg(
            lhs_data,
            &lhs_layout,
            lhs_batches,
            rank,
            &space,
            [M, K],
            1,
            check_m || check_k,
        ),
        tile_arg(
            rhs_data,
            &rhs_layout,
            rhs_batches,
            rank,
            &space,
            [K, N],
            v,
            check_k || check_n,
        ),
        tile_arg(
            out,
            &out_layout,
            out_batches,
            rank,
            &space,
            [M, N],
            v,
            check_m || check_n,
        ),
        lhs_dtype,
        rhs_dtype,
        acc_dtype,
        v,
    );
}

/// Build one operand's launchable [`TileArgLaunch`]: the tensor arg, the [`Space`] it
/// projects, and its [`Storage`] (bounds-checked per `check`). Broadcasting is omission —
/// the operand drops each batch dim it keeps at size 1 (both the buffer dim and the axis),
/// so a dim of `batches` survives only when `> 1`. Its axis is `batch_axis(p)` at the
/// operand's *padded* batch position `p` (left-aligned to the output rank), matching the
/// binding's own leading dims; `matrix` closes out the projection (`[M,K]`/`[K,N]`/`[M,N]`).
#[allow(clippy::too_many_arguments)]
fn tile_arg<R: Runtime, E: Numeric, V: Size>(
    binding: TensorBinding<R>,
    layout: &InnerLayout,
    batches: &[usize],
    rank: usize,
    space: &Space,
    matrix: [Axis; 2],
    v: usize,
    check: bool,
) -> TileArgLaunch<'static, E, V, R> {
    let pad = rank - batches.len();

    let mut axes = Vec::new();
    let mut shape = Vec::new();
    let mut strides = Vec::new();
    for (j, &b) in batches.iter().enumerate() {
        if b > 1 {
            axes.push(batch_axis(pad + j));
            shape.push(binding.shape[j]);
            strides.push(binding.strides[j]);
        }
    }
    // The matrix (and, for a tiled buffer, its grid/tile) dims follow the batch prefix.
    shape.extend_from_slice(&binding.shape[batches.len()..]);
    strides.extend_from_slice(&binding.strides[batches.len()..]);
    axes.extend(matrix);

    let mut binding = binding;
    binding.shape = shape[..].into();
    binding.strides = strides[..].into();

    let (arg, storage) = layout.tensor_arg(binding, v);
    TileArgLaunch::new(arg, space.project(&axes), storage.checked(check))
}
