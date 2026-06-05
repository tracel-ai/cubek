//! Recursive (multi-level) tiling
#![allow(non_snake_case)]

use cubecl::std::tensor::layout::CoordsDyn;
use cubecl::{TestRuntime, prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput, TileInput, assert_equals_approx};
use cubek_tile::{Axis, Space, TileArg, TileArgLaunch};

use super::references;

const M: Axis = Axis(0);
const N: Axis = Axis(1);

/// An 8×8 tile, two nested levels of 2×2 sub-tiles
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

    // The copy kernel only reads/writes through the views — no partitioning, so the
    // spaces carry no partitioner.
    copy_logical::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TileArgLaunch::new(input.tensor_arg(1), input.space(), input.storage()),
        TileArgLaunch::new(output.tensor_arg(1), output.space(), output.storage()),
        f32::as_type_native_unchecked().storage_type(),
    );

    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);

    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            expected[i * n + j] = references::nested_index(i, j, n, &[2, 2]) as f32;
        }
    }
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();

    assert_equals_approx(&got, &expected, 1e-6)
        .as_test_outcome()
        .enforce()
}

/// Copy every logical element of `input` into `output` through their views.
#[cube(launch)]
fn copy_logical<E: Numeric>(
    input: &TileArg<'_, E>,
    output: &TileArg<'_, E>,
    #[define(E)] _dtype: StorageType,
) {
    let input = input.tile();
    let mut output = output.tile();
    let r = input.view();
    let mut w = output.view_mut();
    let shape = r.shape();
    let rows = shape[0];
    let cols = shape[1];
    for i in 0..rows {
        for j in 0..cols {
            let mut pos = CoordsDyn::new();
            pos.push(i);
            pos.push(j);
            w.write(pos.clone(), r.read(pos));
        }
    }
}
