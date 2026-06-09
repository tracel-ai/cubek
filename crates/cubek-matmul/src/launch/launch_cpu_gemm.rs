//! Launch wiring for the CpuGemm matmul.
//!
//! CpuGemm does not go through `MatmulArgs`/`Routine`: it builds three tile-DSL
//! [`Tile`]s straight from the input bindings and launches a kernel whose whole
//! body is `c.mma(a, b)`. Each operand arrives as a [`LaidOut`] binding carrying
//! its own [`InnerLayout`] (the layout can't be deduced — tiled layouts aren't
//! expressible as plain strides), and the layout builds the matching view — so the
//! same launch handles row-, col-, tiled-, or batch-major inputs. The partitioner
//! cuts the matrix axes into
//! `tile_size`-square sub-tiles (one output tile per cube, M→X / N→Y) and rides
//! the batch on cube Z.

use cubecl::{CubeDim, Runtime, client::ComputeClient, prelude::*};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_tile::{
    Axis, ByAxis, ComputeScope, Coverage, CubeAxis, Distribution, Partitioner, Space, Spread,
    TileArg, TileArgLaunch,
};

use crate::{
    definition::{AvailableVectorSizes, InnerLayout, MatmulElems, MatmulProblem, MatmulSetupError},
    routines::{
        BlueprintStrategy, DeviceSettings,
        cpu_gemm::{CpuGemmBlueprint, CpuGemmRoutine},
    },
};

// Matmul's axes — the labels this client gives the engine's opaque `Axis`. `B`
// is the (leading) batch axis; `M`/`N`/`K` are the matrix axes, with `K`
// contracted.
const B: Axis = Axis(0);
const M: Axis = Axis(1);
const N: Axis = Axis(2);
const K: Axis = Axis(3);

/// A binding together with the [`InnerLayout`] that folds its (possibly higher-rank,
/// tiled) physical shape back to the logical `(batch, rows, cols)`. The layout is a
/// property of the input, not of the strategy: it can't be recovered from strides —
/// tiled layouts aren't expressible as plain strides — so it rides with the binding.
/// Build a strided operand with [`strided_input`](Self::strided_input) /
/// [`strided_output`](Self::strided_output); impose a tiled one by setting the fields.
pub struct LaidOut<B> {
    pub binding: B,
    pub layout: InnerLayout,
}

impl<R: Runtime> LaidOut<InputBinding<R>> {
    /// Deduce a plain strided layout from the binding's strides. Valid only for
    /// non-tiled bindings; a tiled operand must impose its layout directly.
    pub fn strided_input(binding: InputBinding<R>) -> Self {
        let layout = InnerLayout::from_shape_and_strides(binding.shape(), &binding.data().strides);
        Self { binding, layout }
    }
}

impl<R: Runtime> LaidOut<TensorBinding<R>> {
    /// Deduce a plain strided layout from the binding's strides. Valid only for
    /// non-tiled bindings.
    pub fn strided_output(binding: TensorBinding<R>) -> Self {
        let layout = InnerLayout::from_shape_and_strides(&binding.shape, &binding.strides);
        Self { binding, layout }
    }
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: LaidOut<InputBinding<R>>,
    rhs: LaidOut<InputBinding<R>>,
    out: LaidOut<TensorBinding<R>>,
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

    // One element type drives the whole kernel, so the operands must agree.
    if dtypes.lhs_global != dtypes.rhs_global || dtypes.lhs_global != dtypes.acc_global {
        return Err({
            let msg = format!(
                "CpuGemm requires a single dtype, got lhs:{:?} rhs:{:?} acc:{:?}",
                dtypes.lhs_global, dtypes.rhs_global, dtypes.acc_global
            );
            MatmulSetupError::InvalidConfig(Box::new(msg))
        });
    }

    // Logical dims from each operand's imposed layout (its physical shape may be a
    // higher-rank tiled buffer). `k` rides lhs's trailing axis, `n` rhs's.
    let (b, m, k) = lhs_layout.logical_dims(lhs.shape());
    let (_, _, n) = rhs_layout.logical_dims(rhs.shape());

    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type())
        .max(out.required_address_type(dtypes.acc_global.size()));

    // CpuGemm reads only `(m, n, k, batch)` and the global dtypes off the problem; the
    // physical layout lives in each operand's `InnerLayout`, so the problem's matrix
    // layout is a don't-care placeholder.
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        [b][..].into(),
        [b][..].into(),
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

    launch_lined::<R>(
        client,
        lhs_data,
        rhs_data,
        out,
        lhs_layout,
        rhs_layout,
        out_layout,
        b,
        m,
        n,
        k,
        blueprint,
        vector_size,
        dtypes.acc_global,
    );

    Ok(())
}

/// Launch the kernel with line size `v`. `N` rides in line units — the space extent and
/// the `N` tile edge are divided by `v`, so each step covers one `Vector<E, v>`. `lhs` is
/// staged scalar; `rhs` and the output carry the line size.
#[allow(clippy::too_many_arguments)]
fn launch_lined<R: Runtime>(
    client: &ComputeClient<R>,
    lhs_data: TensorBinding<R>,
    rhs_data: TensorBinding<R>,
    out: TensorBinding<R>,
    lhs_layout: InnerLayout,
    rhs_layout: InnerLayout,
    out_layout: InnerLayout,
    b: usize,
    m: usize,
    n: usize,
    k: usize,
    blueprint: CpuGemmBlueprint,
    v: usize,
    dtype: StorageType,
) {
    // The full {B, M, N, K} space; each operand carries the batch plus its two matrix
    // axes (batch first, so `partition`'s trailing two axes are the matrix tile and the
    // batch is pinned). `N` is in line units.
    let space = Space::new(&[(B, b), (M, m), (N, n / v), (K, k)]);

    let partitioner = Partitioner::row_major(
        ByAxis::new(&[
            (B, 1),
            (M, blueprint.tile_m),
            (N, blueprint.tile_n / v),
            (K, blueprint.tile_k),
        ]),
        ByAxis::new(&[
            (B, {
                Distribution::Spatial {
                    scope: ComputeScope::Cube(CubeAxis::Z),
                    spread: Spread::Contiguous,
                    coverage: Coverage::TilesEach(1),
                }
            }),
            (M, {
                Distribution::Spatial {
                    scope: ComputeScope::Cube(CubeAxis::X),
                    spread: Spread::Contiguous,
                    coverage: Coverage::TilesEach(1),
                }
            }),
            (N, {
                Distribution::Spatial {
                    scope: ComputeScope::Cube(CubeAxis::Y),
                    spread: Spread::Contiguous,
                    coverage: Coverage::TilesEach(1),
                }
            }),
            (K, Distribution::Sequential),
        ]),
    )
    .staged();

    let space = space.with_partitioner(partitioner.clone());
    let cube_count = partitioner.cube_count(&space);
    let cube_dim = CubeDim::new_single();

    let (a_arg, a_storage) = lhs_layout.tensor_arg(lhs_data, b, m, k, 1);
    let (b_arg, b_storage) = rhs_layout.tensor_arg(rhs_data, b, k, n, v);
    let (c_arg, c_storage) = out_layout.tensor_arg(out, b, m, n, v);

    cpu_gemm_kernel::launch::<R>(
        client,
        cube_count,
        cube_dim,
        TileArgLaunch::new(a_arg, space.project(&[B, M, K]), a_storage),
        TileArgLaunch::new(b_arg, space.project(&[B, K, N]), b_storage),
        TileArgLaunch::new(c_arg, space.project(&[B, M, N]), c_storage),
        dtype,
        v,
    );
}

/// The whole body is `c.mma(a, b)`. `a` stays scalar (broadcast per `K`); `b` and `c`
/// carry the launch-chosen line size `V` along their contiguous `N` axis, so the leaf
/// contraction reads genuine `Vector<E, V>` lines.
#[cube(launch)]
pub fn cpu_gemm_kernel<E: Numeric, V: Size>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    #[define(E)] _dtype: StorageType,
    #[define(V)] _vector_size: usize,
) {
    let a = a.tile();
    let b = b.tile();
    let mut c = c.tile();
    c.mma(&a, &b);
}
