//! Recursive (multi-level) tiling: a 2-level tiled `arange` read back through the
//! `tiled_view` must equal the element's flat physical index at each logical
//! coordinate. Exercises `TileInput`'s chained `.split` and cubecl's multi-level
//! `TiledViewLayout`.
#![allow(non_snake_case)]

use cubecl::std::tensor::layout::CoordsDyn;
use cubecl::{TestRuntime, prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput, TileInput, assert_equals_approx};
use cubek_tile::{Axis, ByAxis, Distribution, Partitioner, Space, Tile, TileKind, TileLaunch};

const M: Axis = Axis(0);
const N: Axis = Axis(1);

/// An 8×8 tile, two nested levels of 2×2 sub-tiles (so per axis `grid=2, level1=2,
/// level2=2`), filled with a physical-order arange. Reading it logically must
/// yield the mixed-radix physical index of each `(i, j)`.
#[test]
fn recursive_two_level_tiled_view() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (m, n) = (8usize, 8usize);

    let input = TileInput::builder(&client, Space::new(&[(M, m), (N, n)]))
        .split(&[2, 2])
        .split(&[2, 2])
        .arange();
    // Untiled output: its buffer is the logical shape itself, so `output[i * n + j]`
    // is the value read at logical `(i, j)`.
    let output = TileInput::builder(&client, Space::new(&[(M, m), (N, n)]))
        .untiled()
        .zeros();

    // The copy kernel only reads/writes through the views; the partitioner is
    // required to launch a `Tile` but unused here.
    let partitioner = Partitioner::row_major(
        ByAxis::new(&[(M, m), (N, n)]),
        ByAxis::new(&[(M, Distribution::Sequential), (N, Distribution::Sequential)]),
    );

    copy_logical::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TileLaunch::new(
            input.view(),
            partitioner.launch(),
            input.space(),
            TileKind::GmemWhole,
        ),
        TileLaunch::new(
            output.view(),
            partitioner.launch(),
            output.space(),
            TileKind::GmemWhole,
        ),
        f32::as_type_native_unchecked().storage_type(),
        1,
    );

    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);

    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            expected[i * n + j] = physical_index(i, j) as f32;
        }
    }
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();

    assert_equals_approx(&got, &expected, 1e-6)
        .as_test_outcome()
        .enforce()
}

/// Flat physical index of logical `(i, j)` in the two-level `[2,2,2,2,2,2]` buffer
/// (row-major strides `[32,16,8,4,2,1]`): each axis decomposes mixed-radix into
/// `(grid, level1, level2)` with edge 2.
fn physical_index(i: usize, j: usize) -> usize {
    let digits = |c: usize| (c / 4, (c / 2) % 2, c % 2);
    let (gi, a1i, a2i) = digits(i);
    let (gj, a1j, a2j) = digits(j);
    gi * 32 + gj * 16 + a1i * 8 + a1j * 4 + a2i * 2 + a2j
}

/// Copy every logical element of `input` into `output` through their views.
#[cube(launch)]
fn copy_logical<E: Numeric, S: Size>(
    input: Tile<'_, E, S, CoordsDyn>,
    mut output: Tile<'_, E, S, CoordsDyn>,
    #[define(E)] _dtype: StorageType,
    #[define(S)] _vector_size: usize,
) {
    let shape = input.view.shape();
    let rows = shape[0];
    let cols = shape[1];
    for i in 0..rows {
        for j in 0..cols {
            let mut pos = CoordsDyn::new();
            pos.push(i);
            pos.push(j);
            output.view.write(pos.clone(), input.view.read(pos));
        }
    }
}
