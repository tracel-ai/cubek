//! `TileSpec::packed`: an operand whose values are fields of a stored word, said on its own.
//!
//! A packed tensor is *values*, stored small. Saying so takes one fact — how wide a field is and
//! how it reads back — and that fact belongs to the values, not to a quantization scheme: there
//! are no scales here, no block grid, no scale binding, nothing for a scheme to carry. The tile
//! serves what the words hold ([`TileArg::tile_packed`]) and the read unpacks.
//!
//! What a *quantized* operand adds on top is its scales, which are their own tensor and their own
//! operand; folding them in is a verb the kernel writes ([`Tile::mm_scaled`], see
//! [`scaled`](super::scaled)). The last test here is a q4 matmul spelled that way, end to end.

use cubecl::{
    Runtime, TestRuntime, bytes::Bytes, prelude::*, quant::scheme::QuantValue,
    std::tensor::TensorHandle, zspace::shape,
};
use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult};
use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// Pack `values` into words, `32 / bits` per word, low field first: what a q4 tensor holds.
/// Eight lines, no scheme.
fn pack(values: &[i32], bits: usize) -> Vec<u32> {
    let factor = 32 / bits;
    let mask = (1u32 << bits) - 1;
    values
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect()
}

/// The values a `bits`-wide signed field walks through, cycling so every representable one
/// (and its sign extension) appears.
fn field_values(count: usize, bits: usize) -> Vec<i32> {
    let span = 1i32 << bits;
    let lo = -(span / 2);
    (0..count).map(|i| lo + (i as i32 % span)).collect()
}

/// A `[rows, cols]` tensor of packed words, declared in *values*: the binding's shape and strides
/// count what the kernel serves, and the packing says how many of those sit in one stored word.
fn packed_tensor(
    client: &ComputeClient<TestRuntime>,
    values: &[i32],
    rows: usize,
    cols: usize,
    bits: usize,
) -> TensorHandle<TestRuntime> {
    let handle = client.create(Bytes::from_elems(pack(values, bits)));
    TensorHandle::new_contiguous(vec![rows, cols], handle, u32::elem_type_native())
}

/// Skip where the device's vectors are narrower than one word's worth of values: a packed line
/// serves a whole word, so `factor` is the width the tile is read at.
fn fits(client: &ComputeClient<TestRuntime>, factor: usize) -> bool {
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        )))
        .enforce();
        return false;
    }
    true
}

/// A packed operand copied into a plain one: the words unpack at the read, and nothing in the
/// kernel, the spec or the launch mentions a scale.
#[cube(launch)]
fn packed_copy<O: Numeric, V: Size>(
    input: &TileArg<'_, u32, Const<1>>,
    output: &TileArg<'_, O, V>,
    #[comptime] space: Space,
    #[define(O)] _dtype: ElemType,
) {
    let input = input.tile_packed::<O>(comptime!(space.clone()));
    let mut output = output.tile(space);
    output.copy_from(&input);
}

/// `c = (w ⊗ s) · x` with `w` packed: the q4 kernel in this spelling. Three tensors, three
/// operands, one verb.
#[cube(launch)]
fn packed_matmul<E: Numeric>(
    w: &TileArg<'_, u32, Const<1>>,
    x: &TileArg<'_, E, Const<1>>,
    scales: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let w = w.tile_packed::<E>(comptime!(space.clone()));
    let x = x.tile(comptime!(space.clone()));
    let scales = scales.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.mm_scaled(&w, &x, &scales, Semiring::SUM_PROD);
}

/// Run [`packed_copy`] over a `[ROWS, COLS]` tensor of `field`-wide values.
fn run_copy(field: QuantValue, rows: usize, cols: usize) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let bits = field.size_bits();
    let factor = 32 / bits;
    if !fits(&client, factor) {
        return;
    }

    let values = field_values(rows * cols, bits);
    let input = packed_tensor(&client, &values, rows, cols, bits);
    let dtype = f32::elem_type_native();
    let output = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Space::new(&[(M, rows), (N, cols)]);
    packed_copy::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        factor,
        TileArgLaunch::new(
            input.binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]).packed(field),
        ),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        space,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want = values[m * cols + n] as f32;
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-6,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **The one item 1 was blocking.** A q4 weight tensor, its scales beside it as their own
/// tensor, and one contraction: `C[m,n] = Σ_k W[m,k] · S[m, k/BLOCK] · X[k,n]`.
fn run_matmul(field: QuantValue) {
    const ROWS: usize = 4;
    const COLS: usize = 4;
    const DEPTH: usize = 32;
    /// Contracted values per scale.
    const BLOCK: usize = 8;
    const BLOCKS: usize = DEPTH / BLOCK;

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let bits = field.size_bits();
    let factor = 32 / bits;
    if !fits(&client, factor) {
        return;
    }

    let w = field_values(ROWS * DEPTH, bits);
    let x: Vec<f32> = (0..DEPTH * COLS).map(|i| (i % 7) as f32 - 3.0).collect();
    // Halves, so an f16 scale would be exact too.
    let s: Vec<f32> = (0..ROWS * BLOCKS).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let dtype = f32::elem_type_native();
    let w_tensor = packed_tensor(&client, &w, ROWS, DEPTH, bits);
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![DEPTH, COLS])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![ROWS, BLOCKS])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // One scale per `(row, block of K)`: `⌊k / BLOCK⌋` along the contracted axis.
    let scales_spec = TileSpec::new(Projection::new(
        &[M, K],
        &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(K).over(BLOCK)],
    ));
    // The K cut is the packed line, so one line is one word and never straddles a scale block.
    let space = Tiling::new()
        .extents(&[(M, ROWS), (N, COLS), (K, DEPTH)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(ROWS))
                .axis(N, Cut::sequential(COLS))
                .axis(K, Cut::sequential(factor))
        })
        .build()
        .with_instruction(Instruction::registers(16));

    packed_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            TileSpec::direct(&[M, K]).packed(field),
        ),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            TileSpec::direct(&[K, N]),
        ),
        TileArgLaunch::new(s_tensor.binding().into_tensor_arg(), scales_spec),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        space,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..ROWS {
        for n in 0..COLS {
            let want: f32 = (0..DEPTH)
                .map(|k| w[m * DEPTH + k] as f32 * s[m * BLOCKS + k / BLOCK] * x[k * COLS + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// Four 8-bit values per word.
#[test]
fn eight_bit_fields_unpack_on_read() {
    run_copy(QuantValue::Q8S, 8, 8);
}

/// Eight 4-bit values per word: the q4 store, and the one the plan names.
#[test]
fn four_bit_fields_unpack_on_read() {
    run_copy(QuantValue::Q4S, 8, 32);
}

/// Sixteen 2-bit values per word.
#[test]
fn two_bit_fields_unpack_on_read() {
    run_copy(QuantValue::Q2S, 4, 32);
}

/// The q4 matmul: packed values, a real scales operand, one verb.
#[test]
fn a_packed_operand_contracts_against_its_scales() {
    run_matmul(QuantValue::Q4S);
}

/// The same contraction over 8-bit fields, so the packing factor is not what makes it work.
#[test]
fn eight_bit_fields_contract_against_their_scales() {
    run_matmul(QuantValue::Q8S);
}
