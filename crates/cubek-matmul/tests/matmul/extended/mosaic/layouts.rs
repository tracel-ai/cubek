#![allow(non_snake_case)]

use cubecl::std::tensor::{TensorHandle, layout::CoordsDyn};
use cubecl::{
    CubeCount, CubeDim, Runtime, TestRuntime, client::ComputeClient, frontend::CubePrimitive,
    ir::AddressType, prelude::*, zspace::Shape, zspace::shape,
};
use cubek_matmul::definition::{InnerLayout, MatmulElems, MatmulProblem};
use cubek_matmul::launch::launch_mosaic::mosaic_kernel;
use cubek_std::MatrixLayout;
use cubek_test_utils::TestInput;
use cubek_tile::{
    Axis, ByAxis, ComputeScope, Coverage, CubeAxis, Distribution, Partitioner, Space, Spread,
    Storage, TileArg, TileArgLaunch,
};

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
    src: &TileArg<'_, E>,
    dst: &TileArg<'_, E>,
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
            Shape::from(layout.physical_dims(batch, rows, cols)),
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

fn spatial(dim: CubeAxis) -> Distribution {
    Distribution::Spatial {
        scope: ComputeScope::Cube(dim),
        spread: Spread::Contiguous,
        coverage: Coverage::TilesEach(1),
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

/// The operand's launchable `TileArg`, viewed in `space`: its tensor arg (with the
/// layout's physical strides) and the matching [`Storage`]. Generic over the element
/// type so it fits a `#[define(E)]` kernel's launch arg by inference.
fn tile_arg<E: Numeric>(op: &Operand, space: Space) -> TileArgLaunch<'static, E, TestRuntime> {
    let mut binding = op.handle.clone().binding();
    binding.strides = op.layout.physical_strides(op.batch, op.rows, op.cols)[..].into();
    let (tensor, storage) = op.layout.tensor_arg(binding, op.batch, op.rows, op.cols);
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

/// Run `batch × (m, k) @ (k, n)` with each operand in its chosen layout and check
/// it against the plain logical reference.
fn run(
    lhs_layout: InnerLayout,
    rhs_layout: InnerLayout,
    out_layout: InnerLayout,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    tile: usize,
) {
    let client = TestRuntime::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let dtypes = MatmulElems::from_single_dtype(f32::as_type_native_unchecked());

    // Logical inputs (row-major) via cubek-test-utils, with host data for the
    // reference. The `problem` describes the *logical* matmul (the physical inner
    // layouts below don't change it).
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![batch],
        shape![batch],
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        dtypes.as_global_elems(),
        AddressType::U32,
    );
    let (a_handle, a_host) = TestInput::builder(client.clone(), shape![batch, m, k])
        .uniform(1234, -1., 1.)
        .generate_with_f32_host_data();
    let (b_handle, b_host) = TestInput::builder(client.clone(), shape![batch, k, n])
        .uniform(5678, -1., 1.)
        .generate_with_f32_host_data();

    // Operands in their chosen inner layouts; scatter the logical inputs in
    // through the views.
    let lhs = Operand::zeros(&client, lhs_layout, [B, M, K], batch, m, k);
    let rhs = Operand::zeros(&client, rhs_layout, [B, K, N], batch, k, n);
    let out = Operand::zeros(&client, out_layout, [B, M, N], batch, m, n);

    copy(
        &client,
        &Operand::wrap(a_handle, InnerLayout::RowMajor, [B, M, K], batch, m, k),
        &lhs,
    );
    copy(
        &client,
        &Operand::wrap(b_handle, InnerLayout::RowMajor, [B, K, N], batch, k, n),
        &rhs,
    );

    // One output tile per cube (M→X, N→Y), batch on Z, K contracted in-cube.
    let partitioner = Partitioner::row_major(
        ByAxis::new(&[(B, 1), (M, tile), (N, tile), (K, tile)]),
        ByAxis::new(&[
            (B, spatial(CubeAxis::Z)),
            (M, spatial(CubeAxis::X)),
            (N, spatial(CubeAxis::Y)),
            (K, Distribution::Sequential),
        ]),
    )
    .staged();
    let space =
        Space::new(&[(B, batch), (M, m), (N, n), (K, k)]).with_partitioner(partitioner.clone());

    mosaic_kernel::launch::<TestRuntime>(
        &client,
        partitioner.cube_count(&space),
        CubeDim::new_single(),
        tile_arg(&lhs, space.project(&[B, M, K])),
        tile_arg(&rhs, space.project(&[B, K, N])),
        tile_arg(&out, space.project(&[B, M, N])),
        dtype,
    );

    // Gather the result into a logical row-major tensor and check it against the
    // CPU matmul reference.
    let result = gather(&client, &out);
    assert_result(&a_host, &b_host, &problem, &client, result, dtypes)
        .as_test_outcome()
        .enforce()
}

use InnerLayout::{ColMajor, RowMajor};

/// A single level of square `4 × 4` blocks.
fn tiled() -> InnerLayout {
    InnerLayout::square_tiled(4)
}

/// Two nested levels: `4 × 4` blocks each split into `2 × 2`.
fn recursive() -> InnerLayout {
    InnerLayout::Tiled {
        tiles: vec![(4, 4), (2, 2)],
    }
}

#[test]
fn all_row_major() {
    run(RowMajor, RowMajor, RowMajor, 2, 8, 8, 8, 4);
}

#[test]
fn row_col_natural() {
    run(RowMajor, ColMajor, RowMajor, 2, 8, 8, 8, 4);
}

#[test]
fn all_tiled() {
    run(tiled(), tiled(), tiled(), 2, 8, 8, 8, 4);
}

#[test]
fn all_recursively_tiled() {
    run(recursive(), recursive(), recursive(), 2, 8, 8, 8, 4);
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
        2,
        8,
        8,
        8,
        4,
    );
}

#[test]
fn mixed_layouts() {
    run(tiled(), ColMajor, RowMajor, 2, 8, 8, 8, 4);
}

#[test]
fn tiled_inputs_recursive_output() {
    run(tiled(), tiled(), recursive(), 2, 8, 8, 8, 4);
}
