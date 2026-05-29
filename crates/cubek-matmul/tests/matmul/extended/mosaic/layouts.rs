//! Mosaic layout laboratory: the same `c.mma(a, b)` kernel run with each operand
//! stored in a different [`InnerLayout`], verified against a plain logical
//! reference. Inputs are written into each operand's physical buffer *through its
//! view* and the output read back the same way (`copy_logical`), so correctness
//! needs no per-layout index math — adding a layout to the sweep is one line.
#![allow(non_snake_case)]

use cubecl::std::tensor::{TensorHandle, layout::CoordsDyn};
use cubecl::{
    CubeCount, CubeDim, Runtime, TestRuntime, client::ComputeClient, frontend::CubePrimitive,
    ir::AddressType, prelude::*, zspace::Shape, zspace::shape,
};
use cubek_matmul::components::batch::mosaic::mosaic_kernel;
use cubek_matmul::definition::{InnerLayout, MatmulElems, MatmulProblem};
use cubek_std::MatrixLayout;
use cubek_test_utils::TestInput;
use cubek_tile::{
    Axis, ByAxis, ComputePrimitive, Coverage, CubeDimension, Distribution, Partitioner, Space,
    Spread, Tile, TileKind, TileLaunch, cube_count_for,
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
fn copy_logical<E: Numeric, S: Size>(
    src: &Tile<'_, E, S>,
    dst: &mut Tile<'_, E, S>,
    #[define(E)] _dtype: StorageType,
    #[define(S)] _vector_size: usize,
) {
    let shape = src.view.shape();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for l in 0..shape[2] {
                let mut pos = CoordsDyn::new();
                pos.push(i);
                pos.push(j);
                pos.push(l);
                dst.view.write(pos.clone(), src.view.read(pos));
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

/// A trivial partitioner (edge 1, all sequential) for the logical copies, which
/// only walk views and never partition.
fn copy_partitioner() -> Partitioner {
    Partitioner::row_major(
        ByAxis::new(&[(B, 1), (M, 1), (N, 1), (K, 1)]),
        ByAxis::new(&[
            (B, Distribution::Sequential),
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
}

fn spatial(dim: CubeDimension) -> Distribution {
    Distribution::Spatial {
        unit: ComputePrimitive::Cube(dim),
        spread: Spread::Contiguous,
        coverage: Coverage::TilesEach(1),
    }
}

/// Copy every logical element from `src` into `dst` through their views — moving
/// data between two physical layouts in logical order.
fn copy(client: &ComputeClient<TestRuntime>, src: &Operand, dst: &Operand) {
    let part = copy_partitioner();
    copy_logical::launch::<TestRuntime>(
        client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        {
            let mut binding = src.handle.clone().binding();
            binding.strides = src.layout.physical_strides(src.batch, src.rows, src.cols)[..].into();
            TileLaunch::new(
                src.layout.view(binding, src.batch, src.rows, src.cols),
                part.launch(),
                src.space.clone(),
                TileKind::GmemWhole,
            )
        },
        {
            let mut binding = dst.handle.clone().binding();
            binding.strides = dst.layout.physical_strides(dst.batch, dst.rows, dst.cols)[..].into();
            TileLaunch::new(
                dst.layout.view(binding, dst.batch, dst.rows, dst.cols),
                part.launch(),
                dst.space.clone(),
                TileKind::GmemWhole,
            )
        },
        f32::as_type_native_unchecked().storage_type(),
        1,
    );
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
            (B, spatial(CubeDimension::Z)),
            (M, spatial(CubeDimension::X)),
            (N, spatial(CubeDimension::Y)),
            (K, Distribution::Sequential),
        ]),
    );
    let space = Space::new(&[(B, batch), (M, m), (N, n), (K, k)]);

    mosaic_kernel::launch::<TestRuntime>(
        &client,
        cube_count_for(&partitioner, &space),
        CubeDim::new_single(),
        {
            let mut binding = lhs.handle.clone().binding();
            binding.strides = lhs.layout.physical_strides(lhs.batch, lhs.rows, lhs.cols)[..].into();
            TileLaunch::new(
                lhs.layout.view(binding, lhs.batch, lhs.rows, lhs.cols),
                partitioner.launch(),
                lhs.space.clone(),
                TileKind::GmemWhole,
            )
        },
        {
            let mut binding = rhs.handle.clone().binding();
            binding.strides = rhs.layout.physical_strides(rhs.batch, rhs.rows, rhs.cols)[..].into();
            TileLaunch::new(
                rhs.layout.view(binding, rhs.batch, rhs.rows, rhs.cols),
                partitioner.launch(),
                rhs.space.clone(),
                TileKind::GmemWhole,
            )
        },
        {
            let mut binding = out.handle.clone().binding();
            binding.strides = out.layout.physical_strides(out.batch, out.rows, out.cols)[..].into();
            TileLaunch::new(
                out.layout.view(binding, out.batch, out.rows, out.cols),
                partitioner.launch(),
                out.space.clone(),
                TileKind::GmemWhole,
            )
        },
        dtype,
        1,
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
