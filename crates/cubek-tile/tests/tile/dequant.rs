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
fn dequantize<E: Numeric, S: Numeric, W: Size, F: Size>(
    weights: &TileArg<'_, u32, Const<1>>,
    scales: &TileArg<'_, S, F>,
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
    let (rows, blocks, inside, scale_lanes) = (4, 4, 8, 4);
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

    // The scales are an operand like the others, and the axis they omit is the whole statement
    // that one of their values covers a block of columns.
    let mut operands = (
        Operand::new(&[ROW, CB, CI], dtype),
        Operand::new(&[ROW, CB], dtype),
        Operand::new(&[ROW, CB, CI], dtype),
    );
    let space = Tiling::over(&mut operands, &[(ROW, rows), (CB, blocks), (CI, inside)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |level, _| {
            level.walk(&[(ROW, rows), (CB, blocks), (CI, inside)]);
        })
        .build();

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

    // `COL` is one physical dim that `(CB, CI)` partition, so each operand that spans both says
    // so; the scales span only `CB` and address it as it stands.
    let split = || {
        Projection::new(
            &[ROW, CB, CI],
            &[
                PhysicalAxisMap::of(ROW),
                PhysicalAxisMap::disjoint(&[(CB, inside), (CI, 1)]),
            ],
        )
    };
    let launcher = space.clone().launcher(&client);
    let w_op = launcher
        .arg(w_tensor.clone().binding())
        .gathered(split())
        .operand(&operands.0)
        .packed(field)
        .vectorize(factor)
        .build();
    // Four scales per read: one read covers four blocks of columns, and each of its lanes is
    // taken by the run of values that block holds.
    let s_op = launcher
        .bind(&operands.1, s_tensor.binding())
        .vectorize(scale_lanes)
        .build();
    let out_op = launcher
        .arg(out.clone().binding())
        .gathered(split())
        .operand(&operands.2)
        .vectorize(factor)
        .build();

    dequantize::launch::<TestRuntime>(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        factor,
        scale_lanes,
        w_op.arg(),
        s_op.arg(),
        out_op.arg(),
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
