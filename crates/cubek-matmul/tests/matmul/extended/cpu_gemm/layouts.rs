#![allow(non_snake_case)]

use cubecl::std::tensor::{TensorHandle, layout::CoordsDyn};
use cubecl::{
    CubeCount, CubeDim, Runtime, TestRuntime, client::ComputeClient, frontend::CubePrimitive,
    ir::AddressType, prelude::*, zspace::Shape, zspace::shape,
};
use cubek_matmul::definition::{InnerLayout, MatmulElems, MatmulProblem};
use cubek_matmul::routines::BlueprintStrategy;
use cubek_matmul::routines::cpu_gemm::{
    CpuGemmBlueprint, Instruction, PlaneGrid, WithLayout, launch_ref,
};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::TestInput;
use cubek_tile::{Axis, Space, TileArg, TileArgLaunch};

use super::Dims;
use crate::matmul::assert_result;

// `B` leads (batch), then the matrix axes — so `partition`'s trailing two are
// the matrix and the batch is pinned.
const B: Axis = Axis(0);
const M: Axis = Axis(1);
const N: Axis = Axis(2);
const K: Axis = Axis(3);

/// Copy every logical `(d0, d1, d2)` element from `src` to `dst` through their
/// views — the layout-agnostic scatter/gather. Whatever physical layout each
/// view wraps, this moves data in logical order.
#[cube(launch)]
fn copy_logical<E: Numeric>(
    src: &TileArg<'_, E, Const<1>>,
    dst: &TileArg<'_, E, Const<1>>,
    #[define(E)] _dtype: StorageType,
) {
    let src = src.tile();
    let mut dst = dst.tile();
    let r = src.view();
    let mut w = dst.view_mut();
    let shape = r.shape();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for l in 0..shape[2] {
                let mut pos = CoordsDyn::new();
                pos.push(i);
                pos.push(j);
                pos.push(l);
                w.write(pos.clone(), r.read(pos));
            }
        }
    }
}

/// An operand: a physical buffer in some [`InnerLayout`], viewed in its logical
/// `(batch, rows, cols)` space.
struct Operand {
    handle: TensorHandle<TestRuntime>,
    layout: InnerLayout,
    space: Space,
    batch: usize,
    rows: usize,
    cols: usize,
}

impl Operand {
    /// A fresh (zeroed) operand of logical `(batch, rows, cols)` in `layout`.
    fn zeros(
        client: &ComputeClient<TestRuntime>,
        layout: InnerLayout,
        axes: [Axis; 3],
        batch: usize,
        rows: usize,
        cols: usize,
    ) -> Self {
        let handle = TestInput::builder(
            client.clone(),
            Shape::from(layout.physical_dims(&[batch], rows, cols)),
        )
        .zeros()
        .generate_without_host_data();
        Self::wrap(handle, layout, axes, batch, rows, cols)
    }

    /// Wrap an existing `handle` as an operand of the given layout/axes.
    fn wrap(
        handle: TensorHandle<TestRuntime>,
        layout: InnerLayout,
        axes: [Axis; 3],
        batch: usize,
        rows: usize,
        cols: usize,
    ) -> Self {
        Operand {
            handle,
            layout,
            space: Space::new(&[(axes[0], batch), (axes[1], rows), (axes[2], cols)]),
            batch,
            rows,
            cols,
        }
    }
}

/// Copy every logical element from `src` into `dst` through their views — moving
/// data between two physical layouts in logical order.
fn copy(client: &ComputeClient<TestRuntime>, src: &Operand, dst: &Operand) {
    copy_logical::launch::<TestRuntime>(
        client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        tile_arg(src, src.space.clone()),
        tile_arg(dst, dst.space.clone()),
        f32::as_type_native_unchecked().storage_type(),
    );
}

/// The operand's binding with the layout's physical strides realized on its buffer.
fn physical_binding(op: &Operand) -> TensorBinding<TestRuntime> {
    let mut binding = op.handle.clone().binding();
    binding.strides = op.layout.physical_strides(&[op.batch], op.rows, op.cols)[..].into();
    binding
}

/// The operand's launchable `TileArg`, viewed in `space`: its tensor arg (with the
/// layout's physical strides) and the matching [`Storage`]. Generic over the element
/// type so it fits a `#[define(E)]` kernel's launch arg by inference.
fn tile_arg<E: Numeric, V: Size>(
    op: &Operand,
    space: Space,
) -> TileArgLaunch<'static, E, V, TestRuntime> {
    let (tensor, storage) = op.layout.tensor_arg(physical_binding(op), 1);
    TileArgLaunch::new(tensor, space, storage)
}

/// Gather `src` (any layout) into a fresh logical row-major tensor.
fn gather(client: &ComputeClient<TestRuntime>, src: &Operand) -> TensorHandle<TestRuntime> {
    let logical = Operand::zeros(
        client,
        InnerLayout::RowMajor,
        [B, M, N],
        src.batch,
        src.rows,
        src.cols,
    );
    copy(client, src, &logical);
    logical.handle
}

/// Run `lhs_batch × (m, k) @ rhs_batch × (k, n)` with each operand in its chosen layout
/// and check it against the plain logical reference. When the batches differ (one side
/// `1`) this exercises a tiled (or strided) operand whose batch axis is *omitted* — the
/// layout's physical buffer crossed with the broadcast path.
fn run(lhs_layout: InnerLayout, rhs_layout: InnerLayout, out_layout: InnerLayout, dims: Dims) {
    let Dims {
        lhs_batch,
        rhs_batch,
        m,
        n,
        k,
        tile_size: tile,
    } = dims;
    let out_batch = lhs_batch.max(rhs_batch);
    let client = TestRuntime::client(&Default::default());
    let dtypes = MatmulElems::from_single_dtype(f32::as_type_native_unchecked());

    // Logical inputs (row-major) via cubek-test-utils, with host data for the
    // reference. The `problem` describes the *logical* matmul (the physical inner
    // layouts below don't change it); the reference broadcasts the batch per-dim.
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![lhs_batch],
        shape![rhs_batch],
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        dtypes.as_global_elems(),
        AddressType::U32,
    );
    let (a_handle, a_host) = TestInput::builder(client.clone(), shape![lhs_batch, m, k])
        .uniform(1234, -1., 1.)
        .generate_with_f32_host_data();
    let (b_handle, b_host) = TestInput::builder(client.clone(), shape![rhs_batch, k, n])
        .uniform(5678, -1., 1.)
        .generate_with_f32_host_data();

    // Operands in their chosen inner layouts; scatter the logical inputs in
    // through the views.
    let lhs = Operand::zeros(&client, lhs_layout, [B, M, K], lhs_batch, m, k);
    let rhs = Operand::zeros(&client, rhs_layout, [B, K, N], rhs_batch, k, n);
    let out = Operand::zeros(&client, out_layout, [B, M, N], out_batch, m, n);

    copy(
        &client,
        &Operand::wrap(a_handle, InnerLayout::RowMajor, [B, M, K], lhs_batch, m, k),
        &lhs,
    );
    copy(
        &client,
        &Operand::wrap(b_handle, InnerLayout::RowMajor, [B, K, N], rhs_batch, k, n),
        &rhs,
    );

    // Drive the production launch path, imposing each operand's inner layout via
    // `WithLayout` — this is where tiled (higher-rank) operands flow through `launch_ref`.
    launch_ref::<TestRuntime>(
        &client,
        WithLayout {
            binding: InputBinding::Normal(physical_binding(&lhs), dtypes.lhs_global),
            layout: lhs.layout.clone(),
        },
        WithLayout {
            binding: InputBinding::Normal(physical_binding(&rhs), dtypes.rhs_global),
            layout: rhs.layout.clone(),
        },
        WithLayout {
            binding: physical_binding(&out),
            layout: out.layout.clone(),
        },
        &BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: Instruction::new(tile, tile, tile),
            planes: PlaneGrid::new(2, 2),
        }),
        &dtypes,
    )
    .unwrap();

    // Gather the result into a logical row-major tensor and check it against the
    // CPU matmul reference.
    let result = gather(&client, &out);
    assert_result(&a_host, &b_host, &problem, &client, result, dtypes)
        .as_test_outcome()
        .enforce()
}

use InnerLayout::{ColMajor, RowMajor};

#[test]
fn all_row_major() {
    run(
        RowMajor,
        RowMajor,
        RowMajor,
        Dims {
            lhs_batch: 2,
            rhs_batch: 2,
            m: 8,
            n: 8,
            k: 8,
            tile_size: 4,
        },
    );
}

#[test]
fn row_col_natural() {
    run(
        RowMajor,
        ColMajor,
        RowMajor,
        Dims {
            lhs_batch: 2,
            rhs_batch: 2,
            m: 8,
            n: 8,
            k: 8,
            tile_size: 4,
        },
    );
}

#[test]
fn all_tiled() {
    // A single level of square `4 × 4` blocks.
    run(
        InnerLayout::square_tiled(4),
        InnerLayout::square_tiled(4),
        InnerLayout::square_tiled(4),
        Dims {
            lhs_batch: 2,
            rhs_batch: 2,
            m: 8,
            n: 8,
            k: 8,
            tile_size: 4,
        },
    );
}

#[test]
fn all_recursively_tiled() {
    // Two nested levels: `4 × 4` blocks each split into `2 × 2`.
    run(
        InnerLayout::Tiled {
            tiles: vec![(4, 4), (2, 2)],
        },
        InnerLayout::Tiled {
            tiles: vec![(4, 4), (2, 2)],
        },
        InnerLayout::Tiled {
            tiles: vec![(4, 4), (2, 2)],
        },
        Dims {
            lhs_batch: 2,
            rhs_batch: 2,
            m: 8,
            n: 8,
            k: 8,
            tile_size: 4,
        },
    );
}

#[test]
fn rectangular_tiled() {
    run(
        InnerLayout::Tiled {
            tiles: vec![(8, 4)],
        },
        InnerLayout::Tiled {
            tiles: vec![(4, 8)],
        },
        InnerLayout::Tiled {
            tiles: vec![(8, 8)],
        },
        Dims {
            lhs_batch: 2,
            rhs_batch: 2,
            m: 8,
            n: 8,
            k: 8,
            tile_size: 4,
        },
    );
}

#[test]
fn mixed_layouts() {
    run(
        InnerLayout::square_tiled(4),
        ColMajor,
        RowMajor,
        Dims {
            lhs_batch: 2,
            rhs_batch: 2,
            m: 8,
            n: 8,
            k: 8,
            tile_size: 4,
        },
    );
}

#[test]
fn tiled_inputs_recursive_output() {
    // Single-level `4 × 4` inputs, output recursively split into `2 × 2`.
    run(
        InnerLayout::square_tiled(4),
        InnerLayout::square_tiled(4),
        InnerLayout::Tiled {
            tiles: vec![(4, 4), (2, 2)],
        },
        Dims {
            lhs_batch: 2,
            rhs_batch: 2,
            m: 8,
            n: 8,
            k: 8,
            tile_size: 4,
        },
    );
}

/// The tiled storage (4×4 blocks, valid for 8×8) and the partitioner disagree: the GEMM
/// walks 3×3 tiles, which don't divide 8. The partition overhang is clipped against each
/// operand's logical bound (folded from the physical tiled shape), so masking is correct
/// even though the physical buffer is fully populated.
#[test]
fn tiled_storage_partitioner_overhang() {
    run(
        InnerLayout::square_tiled(4),
        InnerLayout::square_tiled(4),
        RowMajor,
        Dims {
            lhs_batch: 1,
            rhs_batch: 1,
            m: 8,
            n: 8,
            k: 8,
            tile_size: 3,
        },
    );
}

/// Broadcast crossed with a tiled buffer: `rhs` is batch-1 (broadcast) and tiled, so its
/// batch axis is omitted while its physical `[1, grid…, tile…]` buffer is reused across
/// all of `lhs`'s batch. `lhs` and the output are tiled and batched.
#[test]
fn broadcast_tiled_rhs() {
    run(
        InnerLayout::square_tiled(4),
        InnerLayout::square_tiled(4),
        InnerLayout::square_tiled(4),
        Dims {
            lhs_batch: 2,
            rhs_batch: 1,
            m: 8,
            n: 8,
            k: 8,
            tile_size: 4,
        },
    );
}

/// The mirror: `lhs` broadcasts (batch 1) and is tiled; `rhs`/out are batched. Also mixes
/// a strided output with tiled inputs.
#[test]
fn broadcast_tiled_lhs() {
    run(
        InnerLayout::square_tiled(4),
        InnerLayout::square_tiled(4),
        RowMajor,
        Dims {
            lhs_batch: 1,
            rhs_batch: 3,
            m: 8,
            n: 8,
            k: 8,
            tile_size: 4,
        },
    );
}
