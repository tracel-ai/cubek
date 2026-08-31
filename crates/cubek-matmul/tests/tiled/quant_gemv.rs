//! The quantized decode gemv, end to end: packed words in, one scale per block folded at the
//! contraction, no scale ever widened on the way.

use cubecl::{
    Runtime, TestRuntime, bytes::Bytes, prelude::*, quant::scheme::QuantValue,
    std::tensor::TensorHandle,
};
use cubek_matmul::{
    routine::BlueprintStrategy,
    tiled::quant_gemv::{QuantGemvBindings, QuantGemvElems, QuantGemvProblem, launch_ref},
};
use half::f16;

/// A weight's values, packed low field first, `32 / bits` of them per `u32`.
fn pack(values: &[i32], bits: usize) -> Vec<u32> {
    let mask = (1u32 << bits) - 1;
    values
        .chunks(32 / bits)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect()
}

/// A contiguous `[rows, cols]` handle over `data`. For the weight the shape counts *values*
/// while the buffer holds packed words, which is what the launch's `field` reconciles.
fn handle<E: Numeric + bytemuck::Pod>(
    client: &ComputeClient<TestRuntime>,
    data: Vec<E>,
    shape: [usize; 2],
) -> TensorHandle<TestRuntime> {
    TensorHandle::<TestRuntime>::new_contiguous(
        shape.to_vec(),
        client.create(Bytes::from_elems(data)),
        E::elem_type_native(),
    )
}

/// `y = (W ⊗ s) · x` against a host reference, with the scales stored at **f16** — the case the
/// widening pass existed for. The scales bind at their own element and are never cast.
fn decode_gemv_matches_the_reference(field: QuantValue, block: usize, rows: usize) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let plane = client.properties().hardware.plane_size_max as usize;

    // A shape the plans tile: `d_out` a multiple of the widest strip, `d_in` of the block.
    let (d_out, d_in) = (256, block * 8);
    let blocks = d_in / block;

    let span = 1i32 << field.size_bits();
    let w: Vec<i32> = (0..d_out * d_in)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let x: Vec<f16> = (0..rows * d_in)
        .map(|i| f16::from_f32((i % 7) as f32 - 3.0))
        .collect();
    let s: Vec<f16> = (0..d_out * blocks)
        .map(|i| f16::from_f32((i % 9) as f32 / 4.0 + 0.25))
        .collect();

    let out = handle(&client, vec![0f32; d_out * rows], [d_out, rows]);
    let written = out.handle.clone();

    let problem = QuantGemvProblem {
        d_out,
        d_in,
        rows,
        field,
        block,
    };
    launch_ref::<TestRuntime>(
        &client,
        QuantGemvBindings {
            x: handle(&client, x.clone(), [rows, d_in]).binding(),
            // The shape counts values; the buffer holds `d_out · d_in / factor` words.
            weights: TensorHandle::<TestRuntime>::new_contiguous(
                vec![d_out, d_in],
                client.create(Bytes::from_elems(pack(&w, field.size_bits()))),
                u32::elem_type_native(),
            )
            .binding(),
            scales: handle(&client, s.clone(), [d_out, blocks]).binding(),
            out: out.binding(),
        },
        &problem,
        &BlueprintStrategy::default(),
        QuantGemvElems {
            served: f32::elem_type_native(),
            x: f16::elem_type_native(),
            scales: f16::elem_type_native(),
            out: f32::elem_type_native(),
        },
    )
    .unwrap_or_else(|e| panic!("the gemv declined {problem:?}: {e}"));

    let bytes = client.read_one(written).unwrap();
    let got: &[f32] = bytemuck::cast_slice(&bytes);
    // `y^T`, so the weight's row is the outer dim and the activation row the inner one.
    for m in 0..d_out {
        for r in 0..rows {
            let want: f32 = (0..d_in)
                .map(|k| {
                    w[m * d_in + k] as f32
                        * s[m * blocks + k / block].to_f32()
                        * x[r * d_in + k].to_f32()
                })
                .sum();
            let tolerance = want.abs() * 1e-2 + 1e-2;
            let have = got[m * rows + r];
            assert!(
                (have - want).abs() <= tolerance,
                "at ({m}, {r}): got {have}, want {want} (plane {plane})"
            );
        }
    }
}

#[test]
fn eight_bit_weights_decode_against_f16_scales() {
    decode_gemv_matches_the_reference(QuantValue::Q8S, 32, 1);
}

#[test]
fn four_bit_weights_decode_against_f16_scales() {
    decode_gemv_matches_the_reference(QuantValue::Q4S, 32, 1);
}

/// More than one activation row against the one weight stream: `N` is sequential at every
/// level, so a lane holds that many partials against the line it already read.
#[test]
fn several_activation_rows_share_one_weight_stream() {
    decode_gemv_matches_the_reference(QuantValue::Q4S, 32, 3);
}
