//! Dequantization on its own: `out = weights ⊗ scales`, no contraction anywhere.
//!
//! A quantized tensor is values stored small beside a coarser tensor of scales. Decoding it is not
//! a matmul verb and not a quantization feature: it is the elementwise product of two tiles, where
//! one spans fewer axes than the other and so spreads each of its values across every position of
//! the axes it omits.
//!
//! Proven here with the axes doing all the work. `COL` is spelled `(CB, CI)` — which block of
//! columns, and where inside it — the weights address both, and the scales address `CB` alone. One
//! scale per block of columns is then a fact about which axes the operand distinguishes, with
//! nothing dividing anything.

use cubecl::{
    Runtime, TestRuntime, bytes::Bytes, prelude::*, quant::scheme::QuantValue,
    std::tensor::TensorHandle, zspace::shape,
};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;

const ROW: Axis = Axis(0);
/// Which block of columns, and where inside it. Together they are the column axis.
const CB: Axis = Axis(1);
const CI: Axis = Axis(2);

/// `out = weights ⊗ scales`. The weights arrive packed and unpack at the read; the scales
/// broadcast over the axis they omit. Neither fact is stated here — both are the operands'.
#[cube(launch)]
fn dequantize<E: Numeric, S: Numeric, W: Size>(
    weights: &TileArg<'_, u32, Const<1>>,
    scales: &TileArg<'_, S, Const<1>>,
    out: &TileArg<'_, E, W>,
    #[comptime] space: Space,
    #[define(E, S)] _dtypes: [ElemType; 2],
) {
    let weights = weights.tile_packed::<E>(comptime!(space.clone()));
    let scales = scales.tile(comptime!(space.clone()));
    let mut out = out.tile(space);
    out.mul(&weights, &scales);
}

#[test]
fn a_packed_tensor_decodes_against_its_scales() {
    let (rows, blocks, inside) = (4, 4, 8);
    let cols = blocks * inside;
    let field = QuantValue::Q4S;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();

    // Every value a signed 4-bit field represents, cycling.
    let span = 1i32 << bits;
    let w: Vec<i32> = (0..rows * cols)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = w
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    // Distinct per `(row, block)`, halves so the reference is exact.
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    // Shape and strides count values; the packing says how many share a stored word.
    let w_tensor = TensorHandle::<TestRuntime>::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Space::new(&[(ROW, rows), (CB, blocks), (CI, inside)]);

    dequantize::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        // A packed operand serves a whole stored word per line, so that is the width the walk
        // moves at; the scale of a line is one value, since the block is wider than the line.
        factor,
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[ROW, CB, CI],
                &[
                    PhysicalAxisMap::of(ROW),
                    PhysicalAxisMap::disjoint(&[(CB, inside), (CI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            // `CI` is not an axis of this operand at all: that absence is the broadcast, and it is
            // the whole statement that one scale covers a block of columns.
            TileSpec::new(Projection::new(
                &[ROW, CB],
                &[PhysicalAxisMap::of(ROW), PhysicalAxisMap::of(CB)],
            )),
        ),
        TileArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[ROW, CB, CI],
                &[
                    PhysicalAxisMap::of(ROW),
                    PhysicalAxisMap::disjoint(&[(CB, inside), (CI, 1)]),
                ],
            )),
        ),
        space,
        [dtype, dtype],
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    for row in 0..rows {
        for col in 0..cols {
            let want = w[row * cols + col] as f32 * s[row * blocks + col / inside];
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-6,
                "at ({row}, {col}): got {have}, want {want}"
            );
        }
    }
}
