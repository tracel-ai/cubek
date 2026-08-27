//! `TileSpec::packed`: an operand whose values are fields of a stored word, said on its own.
//!
//! A packed tensor is *values*, stored small. Saying so takes one fact — how wide a field is and
//! how it reads back — and that fact belongs to the values, not to a quantization scheme: there
//! are no scales here, no block grid, no scale binding, nothing for a scheme to carry. The tile
//! serves what the words hold ([`TileArg::tile_packed`]) and the read unpacks.
//!
//! What a *quantized* operand adds on top is its scales, which are their own tensor and their own
//! operand; folding them in is a verb the kernel writes ([`Tile::mm_scaled`], see
//! [`scaled`](super::scaled)). The matmuls here are spelled that way, up to a whole decode gemv.
//!
//! Every test packs its own words: values low field first, `32 / bits` of them per word, and a
//! binding whose shape counts *values* while the packing says how many sit in one stored word.

use cubecl::{
    Runtime, TestRuntime, bytes::Bytes, features::TypeUsage, prelude::*, quant::scheme::QuantValue,
    std::tensor::TensorHandle, zspace::shape,
};
use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult};
use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
/// The contraction as the two axes a scale block makes of it: which block, and where inside it.
const KB: Axis = Axis(2);
const KI: Axis = Axis(3);

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

/// `c = x · (w ⊗ s)` with `w` packed along its columns: the q4 kernel with the *weights on the
/// right*, which is the shape the shipped quant matmul has. Same verb, same body — only the
/// scales' axes differ, and that is what says which factor they meet.
#[cube(launch)]
fn packed_matmul_rhs<E: Numeric, V: Size>(
    x: &TileArg<'_, E, Const<1>>,
    w: &TileArg<'_, u32, Const<1>>,
    scales: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, V>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(comptime!(space.clone()));
    let w = w.tile_packed::<E>(comptime!(space.clone()));
    let scales = scales.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.mm_scaled(&x, &w, &scales, Semiring::SUM_PROD);
}

/// `c = (w ⊗ s) · x` with `w` an `i8` tensor: the native store, which needs no packing statement
/// at all. The binding says `i8`, the tile serves `i8`, and the contraction casts each value into
/// the accumulator's element as it always does — a value is whatever its tensor holds, for the
/// same reason a scale is.
#[cube(launch)]
fn native_matmul<E: Numeric>(
    w: &TileArg<'_, i8, Const<1>>,
    x: &TileArg<'_, E, Const<1>>,
    scales: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let w = w.tile(comptime!(space.clone()));
    let x = x.tile(comptime!(space.clone()));
    let scales = scales.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.mm_scaled(&w, &x, &scales, Semiring::SUM_PROD);
}

/// The decode gemv, whole: one row of activations against packed weights read straight from
/// global memory, their scales beside them, accumulating in registers. `N` spreads across cubes,
/// which is all a gemv has to spread.
#[cube(launch)]
fn packed_gemv<E: Numeric, V: Size>(
    x: &TileArg<'_, E, Const<1>>,
    w: &TileArg<'_, u32, Const<1>>,
    scales: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, V>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(comptime!(space.clone()));
    let w = w.tile_packed::<E>(comptime!(space.clone()));
    let scales = scales.tile(comptime!(space.clone()));
    let c = c.tile(space);
    let mut acc = c.accumulate::<E, _>(&x, Monoid::Sum);
    acc.mm_scaled(&x, &w, &scales, Semiring::SUM_PROD);
}

/// Four 8-bit values per word.
#[test]
fn eight_bit_fields_unpack_on_read() {
    let (field, rows, cols) = (QuantValue::Q8S, 8, 8);
    let bits = field.size_bits();
    // A packed line serves a whole word, so this is the width the tile is read at.
    let factor = 32 / bits;

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        )))
        .enforce();
        return;
    }

    // Cycling through every value a signed `bits`-wide field represents, sign extension included.
    let span = 1i32 << bits;
    let values: Vec<i32> = (0..rows * cols)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = values
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    // Shape and strides count values; the packing says how many share a stored word.
    let input = TensorHandle::<TestRuntime>::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );

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

/// Eight 4-bit values per word: the q4 store, and the one the plan names.
#[test]
fn four_bit_fields_unpack_on_read() {
    let (field, rows, cols) = (QuantValue::Q4S, 8, 32);
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        )))
        .enforce();
        return;
    }

    let span = 1i32 << bits;
    let values: Vec<i32> = (0..rows * cols)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = values
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let input = TensorHandle::<TestRuntime>::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );

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

/// Sixteen 2-bit values per word.
#[test]
fn two_bit_fields_unpack_on_read() {
    let (field, rows, cols) = (QuantValue::Q2S, 4, 32);
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        )))
        .enforce();
        return;
    }

    let span = 1i32 << bits;
    let values: Vec<i32> = (0..rows * cols)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = values
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let input = TensorHandle::<TestRuntime>::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );

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

/// **The q4 matmul.** A packed weight tensor, its scales beside it as their own tensor, and one
/// contraction: `C[m,n] = Σ_k W[m,k] · S[m, k/block] · X[k,n]`.
#[test]
fn a_packed_operand_contracts_against_its_scales() {
    let (field, rows, cols, block, blocks) = (QuantValue::Q4S, 4, 4, 8, 4);
    let depth = block * blocks;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        )))
        .enforce();
        return;
    }

    let span = 1i32 << bits;
    let w: Vec<i32> = (0..rows * depth)
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
    let x: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    // Halves, so an f16 scale would be exact too.
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let dtype = f32::elem_type_native();
    let w_tensor = TensorHandle::<TestRuntime>::new_contiguous(
        vec![rows, depth],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // A region sits inside one block, and the packed line is one word of it.
    let space = Tiling::new()
        .extents(&[(M, rows), (N, cols), (KB, blocks), (KI, block)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(rows))
                .axis(N, Cut::sequential(cols))
                .axis(KB, Cut::sequential(1))
                .axis(KI, Cut::sequential(factor))
        })
        .build()
        .with_instruction(Instruction::registers(16));

    packed_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            // One scale per `(row, block)`: `KI` is carried and addresses nothing.
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        space,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| w[m * depth + k] as f32 * s[m * blocks + k / block] * x[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The same contraction over 8-bit fields, so the packing factor is not what makes it work.
#[test]
fn eight_bit_fields_contract_against_their_scales() {
    let (field, rows, cols, block, blocks) = (QuantValue::Q8S, 4, 4, 8, 4);
    let depth = block * blocks;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        )))
        .enforce();
        return;
    }

    let span = 1i32 << bits;
    let w: Vec<i32> = (0..rows * depth)
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
    let x: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let dtype = f32::elem_type_native();
    let w_tensor = TensorHandle::<TestRuntime>::new_contiguous(
        vec![rows, depth],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[(M, rows), (N, cols), (KB, blocks), (KI, block)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(rows))
                .axis(N, Cut::sequential(cols))
                .axis(KB, Cut::sequential(1))
                .axis(KI, Cut::sequential(factor))
        })
        .build()
        .with_instruction(Instruction::registers(16));

    packed_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        space,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| w[m * depth + k] as f32 * s[m * blocks + k / block] * x[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The q4 matmul with the weights on the right: what the shipped quant matmul does. One column
/// block per packed line, so a line never straddles a scale.
#[test]
fn a_packed_rhs_contracts_against_its_scales() {
    let (field, rows, block_k, blocks_k) = (QuantValue::Q4S, 4, 8, 4);
    let depth = block_k * blocks_k;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        )))
        .enforce();
        return;
    }
    // Two packed lines wide; `bn` says how many columns share a scale, and a line takes one.
    let (cols, bn) = (factor * 2, factor);
    let blocks_n = cols / bn;

    let x: Vec<f32> = (0..rows * depth).map(|i| (i % 7) as f32 - 3.0).collect();
    let span = 1i32 << bits;
    let w: Vec<i32> = (0..depth * cols)
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
    let s: Vec<f32> = (0..blocks_k * blocks_n)
        .map(|i| (i as f32 + 1.0) / 2.0)
        .collect();

    let dtype = f32::elem_type_native();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let w_tensor = TensorHandle::<TestRuntime>::new_contiguous(
        vec![depth, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![blocks_k, blocks_n])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[(M, rows), (N, cols), (KB, blocks_k), (KI, block_k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(rows))
                .axis(N, Cut::sequential(cols))
                .axis(KB, Cut::sequential(1))
                .axis(KI, Cut::sequential(block_k))
        })
        .build()
        .with_instruction(Instruction::registers(16));

    packed_matmul_rhs::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        factor,
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            // The contraction splits, because that is what the scales need an axis for. The
            // columns keep the rational spelling: splitting an axis the *accumulator* spans needs
            // the output's edges stated, which nothing does yet.
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(N).over(bn)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        space,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| {
                    x[m * depth + k] * w[k * cols + n] as f32 * s[(k / block_k) * blocks_n + n / bn]
                })
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The same over 8-bit fields.
#[test]
fn an_eight_bit_packed_rhs_contracts_against_its_scales() {
    let (field, rows, block_k, blocks_k) = (QuantValue::Q8S, 4, 8, 4);
    let depth = block_k * blocks_k;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        )))
        .enforce();
        return;
    }
    let (cols, bn) = (factor * 2, factor);
    let blocks_n = cols / bn;

    let x: Vec<f32> = (0..rows * depth).map(|i| (i % 7) as f32 - 3.0).collect();
    let span = 1i32 << bits;
    let w: Vec<i32> = (0..depth * cols)
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
    let s: Vec<f32> = (0..blocks_k * blocks_n)
        .map(|i| (i as f32 + 1.0) / 2.0)
        .collect();

    let dtype = f32::elem_type_native();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let w_tensor = TensorHandle::<TestRuntime>::new_contiguous(
        vec![depth, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![blocks_k, blocks_n])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[(M, rows), (N, cols), (KB, blocks_k), (KI, block_k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(rows))
                .axis(N, Cut::sequential(cols))
                .axis(KB, Cut::sequential(1))
                .axis(KI, Cut::sequential(block_k))
        })
        .build()
        .with_instruction(Instruction::registers(16));

    packed_matmul_rhs::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        factor,
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(N).over(bn)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        space,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| {
                    x[m * depth + k] * w[k * cols + n] as f32 * s[(k / block_k) * blocks_n + n / bn]
                })
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// A block covering both lines: several lines share a scale, which is the direction that is
/// always sound. The other one — a block narrower than the line reading it — is refused by the
/// contraction (`mm_scaled: ... scale blocks must cover whole lines`), and that refusal is a
/// comptime panic inside the kernel, so it lands on a worker thread rather than in a
/// `should_panic` test.
#[test]
fn several_lines_may_share_one_scale() {
    let (field, rows, block_k, blocks_k) = (QuantValue::Q8S, 4, 8, 4);
    let depth = block_k * blocks_k;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        )))
        .enforce();
        return;
    }
    // Two packed lines wide, and one scale spanning both of them.
    let (cols, bn) = (factor * 2, factor * 2);
    let blocks_n = cols / bn;

    let x: Vec<f32> = (0..rows * depth).map(|i| (i % 7) as f32 - 3.0).collect();
    let span = 1i32 << bits;
    let w: Vec<i32> = (0..depth * cols)
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
    let s: Vec<f32> = (0..blocks_k * blocks_n)
        .map(|i| (i as f32 + 1.0) / 2.0)
        .collect();

    let dtype = f32::elem_type_native();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let w_tensor = TensorHandle::<TestRuntime>::new_contiguous(
        vec![depth, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![blocks_k, blocks_n])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[(M, rows), (N, cols), (KB, blocks_k), (KI, block_k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(rows))
                .axis(N, Cut::sequential(cols))
                .axis(KB, Cut::sequential(1))
                .axis(KI, Cut::sequential(block_k))
        })
        .build()
        .with_instruction(Instruction::registers(16));

    packed_matmul_rhs::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        factor,
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(N).over(bn)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        space,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| {
                    x[m * depth + k] * w[k * cols + n] as f32 * s[(k / block_k) * blocks_n + n / bn]
                })
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **The native store needs no engine feature.** `i8` weights, their scales beside them, one
/// contraction — the tile serves the element its binding names and the block casts it, so
/// `Packing::Native` never has to be *stated* for a store that carries no scales in its element.
#[test]
fn an_i8_operand_contracts_against_its_scales() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let depth = block * blocks;

    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no native i8".to_string(),
        ))
        .enforce();
        return;
    }

    // Cycling through every 8-bit signed value, sign extension included.
    let w: Vec<i32> = (0..rows * depth).map(|i| -128 + (i as i32 % 256)).collect();
    let x: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let dtype = f32::elem_type_native();
    let (w_tensor, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(i8::elem_type_native())
        .custom(w.iter().map(|&v| v as f32).collect::<Vec<_>>())
        .generate_with_f32_host_data();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[(M, rows), (N, cols), (KB, blocks), (KI, block)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(rows))
                .axis(N, Cut::sequential(cols))
                .axis(KB, Cut::sequential(1))
                .axis(KI, Cut::sequential(block))
        })
        .build()
        .with_instruction(Instruction::registers(16));

    native_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        space,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| w[m * depth + k] as f32 * s[m * blocks + k / block] * x[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **The q4 decode gemv, end to end.** A packed weight tensor read in place, its scales as their
/// own operand, one row of activations, `N` across cubes, and the partials living in registers for
/// the whole `K` walk. Every piece of it is a thing the plan had to build: packed values with no
/// scheme, a scales operand, the rhs side, and a promoted accumulator.
#[test]
fn a_packed_decode_gemv_runs_in_this_spelling() {
    let (field, block_k, blocks_k) = (QuantValue::Q4S, 8, 4);
    let depth = block_k * blocks_k;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        )))
        .enforce();
        return;
    }
    // Two cubes, each owning one packed line of columns.
    let (cols, bn) = (factor * 2, factor);
    let blocks_n = cols / bn;

    let x: Vec<f32> = (0..depth).map(|i| (i % 7) as f32 - 3.0).collect();
    let span = 1i32 << bits;
    let w: Vec<i32> = (0..depth * cols)
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
    let s: Vec<f32> = (0..blocks_k * blocks_n)
        .map(|i| (i as f32 + 1.0) / 2.0)
        .collect();

    let dtype = f32::elem_type_native();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![1, depth])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let w_tensor = TensorHandle::<TestRuntime>::new_contiguous(
        vec![depth, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![blocks_k, blocks_n])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![1, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[(M, 1), (N, cols), (KB, blocks_k), (KI, block_k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(1))
                .axis(N, Cut::cube(CubeAxis::X, bn))
                .axis(KB, Cut::sequential(1))
                .axis(KI, Cut::sequential(block_k))
        })
        .build()
        .with_instruction(Instruction::registers(16));

    // One entry per level: the accumulator opens at the outermost and lives to the leaf.
    let mut residence = vec![Residence::InPlace; space.partitioner().depth()];
    residence[0] = Residence::Register;

    packed_gemv::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        factor,
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(N).over(bn)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]).residence(&residence),
        ),
        space,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for n in 0..cols {
        let want: f32 = (0..depth)
            .map(|k| x[k] * w[k * cols + n] as f32 * s[(k / block_k) * blocks_n + n / bn])
            .sum();
        let have = got.get_f32(&[0, n]);
        assert!(
            (have - want).abs() < 1e-3,
            "at {n}: got {have}, want {want}"
        );
    }
}

/// The same over 8-bit fields, so it runs on a device whose vectors cap at four.
#[test]
fn an_eight_bit_decode_gemv_runs_in_this_spelling() {
    let (field, block_k, blocks_k) = (QuantValue::Q8S, 8, 4);
    let depth = block_k * blocks_k;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        )))
        .enforce();
        return;
    }
    let (cols, bn) = (factor * 2, factor);
    let blocks_n = cols / bn;

    let x: Vec<f32> = (0..depth).map(|i| (i % 7) as f32 - 3.0).collect();
    let span = 1i32 << bits;
    let w: Vec<i32> = (0..depth * cols)
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
    let s: Vec<f32> = (0..blocks_k * blocks_n)
        .map(|i| (i as f32 + 1.0) / 2.0)
        .collect();

    let dtype = f32::elem_type_native();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![1, depth])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let w_tensor = TensorHandle::<TestRuntime>::new_contiguous(
        vec![depth, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![blocks_k, blocks_n])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![1, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[(M, 1), (N, cols), (KB, blocks_k), (KI, block_k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(1))
                .axis(N, Cut::cube(CubeAxis::X, bn))
                .axis(KB, Cut::sequential(1))
                .axis(KI, Cut::sequential(block_k))
        })
        .build()
        .with_instruction(Instruction::registers(16));

    let mut residence = vec![Residence::InPlace; space.partitioner().depth()];
    residence[0] = Residence::Register;

    packed_gemv::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        factor,
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(N).over(bn)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]).residence(&residence),
        ),
        space,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for n in 0..cols {
        let want: f32 = (0..depth)
            .map(|k| x[k] * w[k * cols + n] as f32 * s[(k / block_k) * blocks_n + n / bn])
            .sum();
        let have = got.get_f32(&[0, n]);
        assert!(
            (have - want).abs() < 1e-3,
            "at {n}: got {have}, want {want}"
        );
    }
}
