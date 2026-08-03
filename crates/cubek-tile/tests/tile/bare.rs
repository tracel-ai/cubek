//! Parity for the bare-tensor construction: `Tile::of` must read and write
//! exactly what `StridedTileArg::tile` does for the same buffer and space —
//! the outputs of the two surfaces are compared bit-for-bit, no host
//! reference involved. Width 1 (scalar binding) and width 4
//! (`Vector<f32, 4>`-typed binding) both covered.
#![allow(non_snake_case)]

use cubecl::{TestRuntime, prelude::*};
use cubek_quant::scheme::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue};
use cubek_test_utils::{HostData, HostDataType, TileInput, assert_equals_approx};

use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

#[cube(launch)]
fn matmul_via_arg(
    a: &StridedTileArg<'_, f32>,
    b: &StridedTileArg<'_, f32>,
    c: &StridedTileArg<'_, f32>,
) {
    let a = a.tile();
    let b = b.tile();
    let mut c = c.tile();
    c.mma(&a, &b);
}

#[cube(launch)]
fn matmul_via_of<E: CubePrimitive<Scalar = f32>>(
    a: &Tensor<E>,
    b: &Tensor<E>,
    c: &Tensor<E>,
    #[comptime] spec_a: TileSpec,
    #[comptime] spec_b: TileSpec,
    #[comptime] spec_c: TileSpec,
) {
    let a = Tile::of(a, spec_a);
    let b = Tile::of(b, spec_b);
    let mut c = Tile::of(c, spec_c);
    c.mma(&a, &b);
}

/// Run the identical `c.mma(a, b)` through both surfaces at the given served
/// width; the two outputs must match exactly.
fn check_parity<E: CubePrimitive<Scalar = f32>>(vector_size: usize) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (m, n, k) = (8, 16, 8);

    let partitioner = Partitioner::row_major(
        ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .staged();
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(partitioner);

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c_arg = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();
    let c_of = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();
    let storage = Storage::of(2, 2);

    let cube_count = space.cube_count();
    let cube_dim = CubeDim::new_single();

    matmul_via_arg::launch::<TestRuntime>(
        &client,
        cube_count.clone(),
        cube_dim,
        StridedTileArgLaunch::strided(a.tensor_arg(1), vector_size, a.space(), storage),
        StridedTileArgLaunch::strided(b.tensor_arg(1), vector_size, b.space(), storage),
        StridedTileArgLaunch::strided(c_arg.tensor_arg(1), vector_size, c_arg.space(), storage),
    );
    let spec = |space| TileSpec { space, storage };
    matmul_via_of::launch::<E, TestRuntime>(
        &client,
        cube_count,
        cube_dim,
        a.tensor_arg(1),
        b.tensor_arg(1),
        c_of.tensor_arg(1),
        spec(a.space()),
        spec(b.space()),
        spec(c_of.space()),
    );

    let got_arg = HostData::from_tensor_handle(&client, c_arg.handle(), HostDataType::F32);
    let got_of = HostData::from_tensor_handle(&client, c_of.handle(), HostDataType::F32);
    assert_equals_approx(&got_of, &got_arg, 0.0)
        .as_test_outcome()
        .enforce()
}

#[test]
fn bare_matches_arg_scalar() {
    check_parity::<f32>(1);
}

#[test]
fn bare_matches_arg_width_4() {
    check_parity::<Vector<f32, Const<4>>>(4);
}

// ===========================================================================
// Quant: `Tile::of_dequant` vs `StridedTileArg::tile_dequant`, packed-u32.
// ===========================================================================

#[cube(launch)]
fn dequant_copy_via_arg(input: &StridedTileArg<'_, u32>, output: &StridedTileArg<'_, f32>) {
    let input = input.tile_dequant::<f32>();
    let mut output = output.tile();
    output.copy_from(&input);
}

#[cube(launch)]
fn dequant_copy_via_of(
    values: &Tensor<u32>,
    scales: &Tensor<f32>,
    output: &Tensor<Vector<f32, Const<4>>>,
    #[comptime] scheme: QuantScheme,
    #[comptime] spec_in: TileSpec,
    #[comptime] spec_out: TileSpec,
) {
    let input = Tile::<f32>::of_dequant(values, scales, scheme, spec_in);
    let mut output = Tile::of(output, spec_out);
    output.copy_from(&input);
}

/// The production quant shape (block Q8S, packed-u32, pack = 4): the same
/// dequantizing copy through both surfaces must agree bit-for-bit.
#[test]
fn bare_dequant_matches_arg_packed_u32() {
    let (m, n, bm, bn) = (8usize, 8usize, 4u8, 4u8);
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block([bm, bn]))
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q8S)
        .with_param(QuantParam::F32);
    let pack = scheme.num_quants();
    assert_eq!(pack, 4, "the bare kernel hardcodes the served width");

    let space = Space::new(&[(M, m), (N, n)]);
    let input = TileInput::builder(&client, space.clone())
        .untiled()
        .packed(&scheme)
        .arange();
    let out_arg = TileInput::builder(&client, space.clone()).untiled().zeros();
    let out_of = TileInput::builder(&client, space).untiled().zeros();

    dequant_copy_via_arg::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        StridedTileArgLaunch::strided(
            input.tile.tensor_arg(1),
            pack,
            input.tile.space(),
            input.tile.storage(),
        )
        .quantized(input.scales_arg(), scheme),
        StridedTileArgLaunch::strided(
            out_arg.tensor_arg(1),
            pack,
            out_arg.space(),
            out_arg.storage(),
        ),
    );
    dequant_copy_via_of::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        input.tile.tensor_arg(1),
        input.scales_arg(),
        out_of.tensor_arg(1),
        scheme,
        TileSpec {
            space: input.tile.space(),
            storage: input.tile.storage(),
        },
        TileSpec {
            space: out_of.space(),
            storage: out_of.storage(),
        },
    );

    let got_arg = HostData::from_tensor_handle(&client, out_arg.handle(), HostDataType::F32);
    let got_of = HostData::from_tensor_handle(&client, out_of.handle(), HostDataType::F32);
    assert_equals_approx(&got_of, &got_arg, 0.0)
        .as_test_outcome()
        .enforce()
}
