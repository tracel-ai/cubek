//! NVFP4 values stored in tiles and loaded a rectangle at a time: each load is four words, two
//! along `K` and two columns, 16 values of one scale block by 2 columns, and the decoding copy
//! lands every value where it belongs under its own column's scale.

use cubecl::zspace::shape;
use cubecl::{bytes::Bytes, prelude::*, quant::scheme::QuantValue, std::tensor::TensorHandle};
use cubecl_common::e2m1;
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::{
    Axis, Partitioning, Projection, Space, TileArg, TileArgLaunch, TileSpec,
    layout::{StorageLevels, VectorTile},
};

use crate::tile::uncut;

const N: Axis = Axis(0);
/// `K` as blocks of one scale make it: which block, and where inside it.
const KB: Axis = Axis(1);
const KI: Axis = Axis(2);

/// `output = unpack(values) ⊗ scales`, the values read `W` words a load.
#[cube(launch)]
fn tiled_scaled_copy<O: Numeric, V: Size, W: Size>(
    values: &TileArg<'_, u32, W>,
    scales: &TileArg<'_, f32, Const<1>>,
    output: &TileArg<'_, O, V>,
    space: Partitioning,
    #[define(O)] _dtype: ElemType,
) {
    let values = values.tile_as::<O>(comptime!(space.clone()));
    let scales = scales.tile(comptime!(space.clone()));
    let mut output = output.tile(comptime!(space.clone()));
    output.copy_from(&values.mul(&scales));
}

/// A `[n, kb, ki]` NVFP4 weight stored a word of 8 along `KI`, then 2 words along `KI` by 2
/// columns, the next such tile along `KB`: every load of four words is 16 values of one block by
/// 2 columns, each column under its own scale. Copied out four values a line along `KI`, every
/// value is its code times its column's block scale.
#[test]
fn a_rectangle_load_decodes_each_value_under_its_own_scale() {
    let (cols, blocks, block) = (4, 2, 16);
    let client = cubecl::test_device().client();

    let storage = StorageLevels::new(&[(KI, 8)])
        .tile(&[(KI, 2), (N, 2)])
        .grid(&[KB, N, KI]);
    let axes = [N, KB, KI];
    let geometry = storage
        .physical(&[(N, cols), (KB, blocks), (KI, block)])
        .unwrap();
    let labels = geometry.labels(&axes);
    assert_eq!(
        VectorTile::new(storage.tiles(), KI, 32).unwrap().extents(),
        &[(KI, 16), (N, 2)]
    );

    // Every code, cycled, laid down where the storage puts its value.
    let code = |n: usize, kb: usize, ki: usize| ((n * blocks + kb) * block + ki) % 16;
    let offset = |coords: [usize; 3]| {
        let mut rest = coords;
        let mut offset = 0;
        for d in (0..labels.len()).rev() {
            let a = axes.iter().position(|&a| a == labels[d]).unwrap();
            let extent = geometry.shape()[d];
            offset += (rest[a] % extent) * geometry.strides()[d];
            rest[a] /= extent;
        }
        offset
    };
    let values = cols * blocks * block;
    let mut words = vec![0u32; values / 8];
    for n in 0..cols {
        for kb in 0..blocks {
            for ki in 0..block {
                let at = offset([n, kb, ki]);
                words[at / 8] |= (code(n, kb, ki) as u32) << ((at % 8) * 4);
            }
        }
    }
    let w_tensor = TensorHandle::new(
        client.create(Bytes::from_elems(words)),
        geometry.shape().to_vec(),
        geometry.strides().to_vec(),
        u32::elem_type_native(),
    );
    // Halves, so the reference is exact.
    let s: Vec<f32> = (0..cols * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();
    let dtype = f32::elem_type_native();
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![cols, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![cols, blocks, block])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let mut w_spec = TileSpec::new(Projection::tiled(&axes, &labels)).packed(QuantValue::E2M1);
    w_spec.stored_tiles = storage.tiles().to_vec();
    let space = Space::new(&[(N, cols), (KB, blocks), (KI, block)]);
    tiled_scaled_copy::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        4,
        4,
        TileArgLaunch::new(w_tensor.binding().into_tensor_arg(), w_spec),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            TileSpec::direct(&[N, KB]),
        ),
        TileArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            TileSpec::direct(&axes),
        ),
        uncut(&client, &space, &space).partitioning_arg(),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    for n in 0..cols {
        for kb in 0..blocks {
            for ki in 0..block {
                let want = e2m1::from_bits(code(n, kb, ki) as u8).to_f32() * s[n * blocks + kb];
                let have = got.get_f32(&[n, kb, ki]);
                assert!(
                    (have - want).abs() < 1e-6,
                    "at (n {n}, kb {kb}, ki {ki}): got {have}, want {want}"
                );
            }
        }
    }
}
