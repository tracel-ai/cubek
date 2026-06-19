//! Launch wiring for the CpuGemm routine.

use cubecl::{Runtime, client::ComputeClient, prelude::*};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_tile::{Axis, CubeAxis, Cut, Schedule, Space, TileArgLaunch, Tiling, WalkOrder};

use crate::{
    definition::{
        AvailableVectorSizes, InnerLayout, MatmulElems, MatmulProblem, MatmulSetupError,
        broadcast_batches,
    },
    routines::{
        BlueprintStrategy, DeviceSettings,
        cpu_gemm::{
            base::{CpuGemmRoutine, K, M, N, batch_axis},
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
    let sz = dtypes.acc_global.size();

    if matches!(lhs, InputBinding::Quantized { .. })
        || matches!(rhs, InputBinding::Quantized { .. })
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "CpuGemm does not support quantized inputs".to_string(),
        )));
    }

    // Logical dims from each operand's imposed layout (its physical shape may be a higher-rank
    // tiled buffer): `k` on lhs's trailing axis, `n` on rhs's, leading dims each operand's own
    // (possibly broadcast) batch shape.
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
        .max(out.required_address_type(sz));

    // CpuGemm reads only `(m, n, k, batches)` + global dtypes off the problem; the physical
    // layout lives in each operand's `InnerLayout`, so the matrix-layout args are placeholders.
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

    // Vectorize `N` only when both `rhs` and the output keep it contiguous (both
    // row-major): then a kernel reading `Vector<E, V>` lands on whole lines. Any
    // other layout — col-major or tiled — falls back to scalar (`V = 1`). `lhs` is
    // always scalar (broadcast per `K`), so its layout never matters here.
    let v = matches!(rhs_layout, InnerLayout::RowMajor)
        .then_some(matches!(out_layout, InnerLayout::RowMajor))
        .filter(|&x| x)
        .and_then(|_| {
            client
                .io_optimized_vector_sizes(sz)
                .filter(|&v| n.is_multiple_of(v) && blueprint.instruction.n.is_multiple_of(v))
                .max()
        })
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

    // `lhs` always staged scalar (`v = 1`); `rhs`/`out` carry the line size. Each operand
    // bounds-checks the edges its tile may overhang, and projects into the `rank`-deep output
    // batch space (numpy right-alignment of its own, possibly shorter, batch shape).
    let lhs = Operand::new(lhs.into_data(), &lhs_layout, &lhs_batches, [M, K]);
    let rhs = Operand::new(rhs.into_data(), &rhs_layout, &rhs_batches, [K, N]);
    let out = Operand::new(out, &out_layout, &out_batches, [M, N]);
    let rank = out_batches.len();
    cpu_gemm_kernel::launch::<R>(
        client,
        cube_count,
        cube_dim,
        lhs.tile_arg(rank, &global_space, 1, check_m || check_k),
        rhs.tile_arg(rank, &global_space, v, check_k || check_n),
        out.tile_arg(rank, &global_space, v, check_m || check_n),
        dtypes.lhs_global,
        dtypes.rhs_global,
        dtypes.acc_global,
        v,
    );

    Ok(())
}

/// One operand's identity, independent of any launch: its data, the [`InnerLayout`] that reads
/// it, its batch shape, and the two `matrix` axes that close the projection
/// (`[M,K]`/`[K,N]`/`[M,N]`). [`tile_arg`](Operand::tile_arg) situates it in a specific launch.
struct Operand<'a, R: Runtime> {
    binding: TensorBinding<R>,
    layout: &'a InnerLayout,
    batches: &'a [usize],
    matrix: [Axis; 2],
}

impl<'a, R: Runtime> Operand<'a, R> {
    fn new(
        binding: TensorBinding<R>,
        layout: &'a InnerLayout,
        batches: &'a [usize],
        matrix: [Axis; 2],
    ) -> Self {
        Operand {
            binding,
            layout,
            batches,
            matrix,
        }
    }

    /// Project the operand into a launch and build its [`TileArgLaunch`]: the tensor arg, the
    /// [`Space`] it projects, and its [`Storage`] (line size `v`, bounds-checked per `check`).
    /// Broadcasting is omission — the operand drops each batch dim it keeps at size 1 (both the
    /// buffer dim and the axis), so a dim of `batches` survives only when `> 1`. Its axis is
    /// `batch_axis(p)` at the operand's *padded* batch position `p` (left-aligned to the output
    /// `rank`), matching the binding's own leading dims; `matrix` closes out the projection.
    fn tile_arg<E: Numeric, V: Size>(
        self,
        rank: usize,
        space: &Space,
        v: usize,
        check: bool,
    ) -> TileArgLaunch<'static, E, V, R> {
        let Operand {
            mut binding,
            layout,
            batches,
            matrix,
        } = self;
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

        binding.shape = shape[..].into();
        binding.strides = strides[..].into();

        let (arg, storage) = layout.tensor_arg(binding, v);
        TileArgLaunch::new(arg, space.project(&axes), storage.checked(check))
    }
}
