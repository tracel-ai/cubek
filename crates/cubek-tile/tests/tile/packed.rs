//! `TileSpec::packed`: an operand whose values are fields of a stored word, said on its own.
//!
//! A packed tensor is *values*, stored small. Saying so takes one fact (how wide a field is and
//! how it reads back), and that fact belongs to the values, not to a quantization scheme: there
//! are no scales here, no block grid, no scale binding, nothing for a scheme to carry. The tile
//! serves what the words hold ([`TileArg::tile_as`]) and the read unpacks.
//!
//! What a *quantized* operand adds on top is its scales, which are their own tensor and their own
//! operand; folding them in is a verb the kernel writes ([`Tile::mm_scaled`], see
//! [`scaled`](super::scaled)). The matmuls here are spelled that way, up to a whole decode gemv.
//!
//! Every test packs its own words: values low field first, `32 / bits` of them per word, and a
//! binding whose shape counts *values* while the packing says how many sit in one stored word.

use cubecl::{
    bytes::Bytes, features::TypeUsage, ir::FloatKind, ir::types::Fp8Format, prelude::*,
    quant::scheme::QuantValue, std::tensor::TensorHandle, zspace::shape,
};
use cubecl_common::{e2m1, e4m3};
use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult};
use cubek_tile::*;
use half::f16;

use super::matmul::require_cmma_8x8x8_f32;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
/// The contraction as the two axes a scale block makes of it: which block, and where inside it.
const KB: Axis = Axis(2);
const KI: Axis = Axis(3);
/// The columns, likewise, where a scale covers a block of them.
const NB: Axis = Axis(4);
const NI: Axis = Axis(5);

/// The register block every contraction here runs under.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// A packed operand copied into a plain one: the words unpack at the read, and nothing in the
/// kernel, the spec or the launch mentions a scale.
#[cube(launch)]
fn packed_copy<O: Numeric, V: Size>(
    input: &TileArg<'_, u32, Const<1>>,
    output: &TileArg<'_, O, V>,
    space: Partitioning,
    #[define(O)] _dtype: ElemType,
) {
    let input = input.tile_as::<O>(comptime!(space.clone()));
    let mut output = output.tile(comptime!(space.clone()));
    output.copy_from(&input);
}

/// `c = (w ⊗ s) · x` with `w` packed: the q4 kernel in this spelling. Three tensors, three
/// operands, one verb.
#[cube(launch)]
fn packed_matmul<E: Numeric>(
    w: &TileArg<'_, u32, Const<1>>,
    x: &TileArg<'_, E, Const<1>>,
    scale: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let w = w
        .tile_as::<E>(comptime!(space.clone()))
        .with_scale(&scale.tile(comptime!(space.clone())));
    let x = x.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));
    c.zero();
    for region in space.over(&level) {
        let mut c_r = c.at(&region);
        c_r.mma_scaled_with(
            &w.at(&region),
            &x.at(&region).plain(),
            REGISTER_BLOCK,
            Semiring::SUM_PROD,
        );
    }
}

/// [`packed_matmul`] with two scale levels: `nvfp4`'s shape.
#[cube(launch)]
fn nvfp4_shaped_matmul<E: Numeric>(
    w: &TileArg<'_, u32, Const<1>>,
    x: &TileArg<'_, E, Const<1>>,
    blocks: &TileArg<'_, E, Const<1>>,
    global: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    // Two levels, said twice: the blocks, then the factor over the whole tensor.
    let w = w
        .tile_as::<E>(comptime!(space.clone()))
        .with_scale(&blocks.tile(comptime!(space.clone())))
        .with_scale(&global.tile(comptime!(space.clone())));
    let x = x.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));
    c.zero();
    for region in space.over(&level) {
        let mut c_r = c.at(&region);
        c_r.mma_scaled_with(
            &w.at(&region),
            &x.at(&region).plain(),
            REGISTER_BLOCK,
            Semiring::SUM_PROD,
        );
    }
}

/// **`nvfp4`'s shape, end to end.** `e2m1` values eight to a word, a scale per sixteen of them, and
/// one factor over the whole tensor.
///
/// The only thing standing between this and the format itself is where the block scales are
/// *stored*: `nvfp4` puts them in `ue4m3`, which needs a device that loads it. Everything the design
/// has to get right is here: the value decode, two levels in order, the coarser one spanning no
/// axis, and a block of sixteen against words of eight.
///
/// Nothing in the kernel names the format, the block, or the number of levels.
///
/// Every operand here is bound one element wide, so unlike the read tests below this asks nothing
/// of the device's vector cap and runs everywhere. That matters: the decode it covers is the one
/// that lowers differently per backend, so a device that skipped this would be the device most
/// worth running it on.
#[test]
fn nvfp4_shaped_decode() {
    let (field, rows, cols, block, blocks) = (QuantValue::E2M1, 4, 4, 16, 2);
    let depth = block * blocks;
    let factor = 32 / field.size_bits();

    let client = cubecl::test_device().client();

    // Every code, cycled, so the whole value ladder and both signs are read back.
    let codes: Vec<u32> = (0..rows * depth).map(|i| (i % 16) as u32).collect();
    let words: Vec<u32> = codes
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &c)| acc | (c << (j * field.size_bits())))
        })
        .collect();
    let x: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    // Halves and a quarter, so the reference is exact.
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();
    let g = vec![0.25f32];

    let dtype = f32::elem_type_native();
    let w_tensor = TensorHandle::new_contiguous(
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
    let (g_tensor, _) = TestInput::builder(client.clone(), shape![1])
        .dtype(dtype)
        .custom(g.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            vec![Level::walk(&[(M, rows), (N, cols), (KB, 1), (KI, factor)])],
        ),
        KernelForm::Static,
    );

    nvfp4_shaped_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
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
            // One scale per `(row, block of sixteen)`.
            TileSpec::new(Projection::new(
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            g_tensor.binding().into_tensor_arg(),
            // The same axes, addressing neither: one value over all their tiles.
            TileSpec::new(Projection::new(&[M, KB], &[PhysicalAxisMap::broadcast()])),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| {
                    let value = e2m1::from_bits(codes[m * depth + k] as u8).to_f32();
                    value * s[m * blocks + k / block] * g[0] * x[k * cols + n]
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

/// `c = x · (w ⊗ s)` with `w` packed along its columns: the q4 kernel with the *weights on the
/// right*, which is the shape the shipped quant matmul has. Same verb, same body: the scales are
/// written on the other factor, and their axes are checked against it.
#[cube(launch)]
fn packed_matmul_rhs<E: Numeric, V: Size>(
    x: &TileArg<'_, E, Const<1>>,
    w: &TileArg<'_, u32, Const<1>>,
    scale: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, V>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(comptime!(space.clone()));
    let w = w
        .tile_as::<E>(comptime!(space.clone()))
        .with_scale(&scale.tile(comptime!(space.clone())));
    let mut c = c.tile(comptime!(space.clone()));
    c.zero();
    for region in space.over(&level) {
        let mut c_r = c.at(&region);
        c_r.mma_scaled_with(
            &x.at(&region).plain(),
            &w.at(&region),
            REGISTER_BLOCK,
            Semiring::SUM_PROD,
        );
    }
}

/// `c = (w ⊗ s) · x` with `w` an `i8` tensor: the native store, which needs no packing statement
/// at all. The binding says `i8`, the tile serves `i8`, and the contraction casts each value into
/// the accumulator's element as it always does: a value is whatever its tensor holds, for the
/// same reason a scale is.
#[cube(launch)]
fn native_matmul<E: Numeric>(
    w: &TileArg<'_, i8, Const<1>>,
    x: &TileArg<'_, E, Const<1>>,
    scale: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let w = w.tile(comptime!(space.clone()));
    let x = x.tile(comptime!(space.clone()));
    let w = w.with_scale(&scale.tile(comptime!(space.clone())));
    let mut c = c.tile(comptime!(space.clone()));
    c.zero();
    for region in space.over(&level) {
        let mut c_r = c.at(&region);
        c_r.mma_scaled_with(
            &w.at(&region),
            &x.at(&region).plain(),
            REGISTER_BLOCK,
            Semiring::SUM_PROD,
        );
    }
}

/// The decode gemv, whole: one row of activations against packed weights read straight from
/// global memory, their scales beside them, accumulating in a register block opened above the
/// walk. `N` spreads across cubes, which is all a gemv has to spread.
#[cube(launch)]
fn packed_gemv<E: Numeric, V: Size>(
    x: &TileArg<'_, E, Const<1>>,
    w: &TileArg<'_, u32, Const<1>>,
    scale: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, V>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(comptime!(space.clone()));
    let values = w.tile_as::<E>(comptime!(space.clone()));
    let w = values.with_scale(&scale.tile(comptime!(space.clone())));
    let c = c.tile(comptime!(space.clone()));
    for cube in space {
        let x = x.at(&cube);
        let values = values.at(&cube);
        let w = w.at(&cube);
        let c = c.at(&cube);
        // The accumulator lives in registers across the whole walk and drains once.
        let mut acc = c.block_accumulator::<E, E, E>(
            &x,
            &values,
            comptime!(Fragments::below(&c, &x)),
            REGISTER_BLOCK,
            Monoid::Sum,
        );
        acc.zero();
        for step in cube {
            let mut acc_s = acc.at(&step);
            acc_s.mma_scaled(&x.at(&step).plain(), &w.at(&step), Semiring::SUM_PROD);
        }
        for r0 in c.walk().unrolled() {
            let mut c_w = c.at(&r0);
            c_w.copy_cast_from(&acc.at(&r0));
        }
    }
}

/// [`packed_matmul`] with the scales stored as bytes, four to a word, served one at a time: a
/// `ue8m0` or `ue4m3` scale is read in its own width, and nothing widens it on the host.
#[cube(launch)]
fn packed_matmul_byte_scales<E: Numeric>(
    w: &TileArg<'_, u32, Const<1>>,
    x: &TileArg<'_, E, Const<1>>,
    scale: &TileArg<'_, u32, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let w = w
        .tile_as::<E>(comptime!(space.clone()))
        .with_scale(&scale.tile_as::<E>(comptime!(space.clone())));
    let x = x.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));
    c.zero();
    for region in space.over(&level) {
        let mut c_r = c.at(&region);
        c_r.mma_scaled_with(
            &w.at(&region),
            &x.at(&region).plain(),
            REGISTER_BLOCK,
            Semiring::SUM_PROD,
        );
    }
}

/// [`packed_gemv`] with byte scales on the rhs, into the promoted block.
#[cube(launch)]
fn packed_gemv_byte_scales<E: Numeric, V: Size>(
    x: &TileArg<'_, E, Const<1>>,
    w: &TileArg<'_, u32, Const<1>>,
    scale: &TileArg<'_, u32, Const<1>>,
    c: &TileArg<'_, E, V>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(comptime!(space.clone()));
    let values = w.tile_as::<E>(comptime!(space.clone()));
    let w = values.with_scale(&scale.tile_as::<E>(comptime!(space.clone())));
    let c = c.tile(comptime!(space.clone()));
    for cube in space {
        let x = x.at(&cube);
        let values = values.at(&cube);
        let w = w.at(&cube);
        let c = c.at(&cube);
        let mut acc = c.block_accumulator::<E, E, E>(
            &x,
            &values,
            comptime!(Fragments::below(&c, &x)),
            REGISTER_BLOCK,
            Monoid::Sum,
        );
        acc.zero();
        for step in cube {
            let mut acc_s = acc.at(&step);
            acc_s.mma_scaled(&x.at(&step).plain(), &w.at(&step), Semiring::SUM_PROD);
        }
        for r0 in c.walk().unrolled() {
            let mut c_w = c.at(&r0);
            c_w.copy_cast_from(&acc.at(&r0));
        }
    }
}

/// The prefill shape on the tensor cores: a packed weight on the rhs with byte scales, landed
/// unpacked and scaled by the plane's lanes, contracted through the plain cmma instruction.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn packed_cmma_rhs<E: Numeric>(
    x: &TileArg<'_, E, Const<1>>,
    w: &TileArg<'_, u32, Const<1>>,
    scale: &TileArg<'_, u32, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] planes: usize,
    #[comptime] lanes: usize,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(comptime!(space.clone()));
    let w = w
        .tile_as::<E>(comptime!(space.clone()))
        .with_landing(planes, lanes)
        .with_scale(&scale.tile_as::<E>(comptime!(space.clone())));
    let c = c.tile(comptime!(space.clone()));
    let mut acc = c.cmma_accumulator::<E, E>(
        &x,
        comptime!(Fragments::new(
            &c.space,
            &x.space,
            std::slice::from_ref(&level)
        )),
        Monoid::Sum,
    );
    acc.zero();
    // The level cuts the columns into two fragments and walks `K`: unrolled, so each region
    // selects its fragment at comptime.
    for region in space.over(&level).unrolled() {
        let mut acc_r = acc.at(&region);
        acc_r.mma_scaled(&x.at(&region).plain(), &w.at(&region), Semiring::SUM_PROD);
    }
    for r0 in c.over(&level).unrolled() {
        let mut c_w = c.at(&r0);
        c_w.copy_cast_from(&acc.at(&r0));
    }
}

/// Four 8-bit values per word.
#[test]
fn eight_bit_fields_unpack_on_read() {
    let (field, rows, cols) = (QuantValue::Q8S, 8, 8);
    let bits = field.size_bits();
    // A packed line serves a whole word, so this is the width the tile is read at.
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
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
    let input = TensorHandle::new_contiguous(
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
    packed_copy::launch(
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
        space.launch_arg(&space),
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

    let client = cubecl::test_device().client();
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
    let input = TensorHandle::new_contiguous(
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
    packed_copy::launch(
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
        space.launch_arg(&space),
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

/// Eight `e2m1` codes per word, reinterpreted rather than sign-extended.
///
/// The field of `nvfp4` and `mxfp4`. Its bits are an index into the format's sixteen values, not a
/// small integer, so reading it as one would answer plausible nonsense — `0b0111` is `6.0`, and as
/// a signed nibble it is `7`. Nothing here states a scheme: the packing names the field, and the
/// values a word holds are whatever that field decodes to.
#[test]
fn fp4_codes_unpack_on_read() {
    let (field, rows, cols) = (QuantValue::E2M1, 4, 32);
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        )))
        .enforce();
        return;
    }

    // Every code, cycled, so both signs and the whole value ladder are read back.
    let codes: Vec<u32> = (0..rows * cols).map(|i| (i % 16) as u32).collect();
    let words: Vec<u32> = codes
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &c)| acc | (c << (j * bits)))
        })
        .collect();
    let input = TensorHandle::new_contiguous(
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
    packed_copy::launch(
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
        space.launch_arg(&space),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want = e2m1::from_bits(codes[m * cols + n] as u8).to_f32();
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

    let client = cubecl::test_device().client();
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
    let input = TensorHandle::new_contiguous(
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
    packed_copy::launch(
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
        space.launch_arg(&space),
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

    let client = cubecl::test_device().client();
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
    let w_tensor = TensorHandle::new_contiguous(
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
    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            vec![Level::walk(&[(M, rows), (N, cols), (KB, 1), (KI, factor)])],
        ),
        KernelForm::Static,
    );

    packed_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
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
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
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

    let client = cubecl::test_device().client();
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
    let w_tensor = TensorHandle::new_contiguous(
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

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            vec![Level::walk(&[(M, rows), (N, cols), (KB, 1), (KI, factor)])],
        ),
        KernelForm::Static,
    );

    packed_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
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
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
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

    let client = cubecl::test_device().client();
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
    let w_tensor = TensorHandle::new_contiguous(
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

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (M, rows),
                (NB, blocks_n),
                (NI, bn),
                (KB, blocks_k),
                (KI, block_k),
            ]),
            vec![Level::walk(&[
                (M, rows),
                (NB, blocks_n),
                (NI, bn),
                (KB, 1),
                (KI, block_k),
            ])],
        ),
        KernelForm::Static,
    );

    packed_matmul_rhs::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
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
                &[KB, KI, NB, NI],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            // One scale per `(block of K, block of N)`: `KI` is carried and addressed by
            // nothing, and the position inside a column block is not an axis of this operand at
            // all, which leaves its innermost axis one it actually varies over.
            TileSpec::new(Projection::new(
                &[KB, KI, NB],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(NB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
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

    let client = cubecl::test_device().client();
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
    let w_tensor = TensorHandle::new_contiguous(
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

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (M, rows),
                (NB, blocks_n),
                (NI, bn),
                (KB, blocks_k),
                (KI, block_k),
            ]),
            vec![Level::walk(&[
                (M, rows),
                (NB, blocks_n),
                (NI, bn),
                (KB, 1),
                (KI, block_k),
            ])],
        ),
        KernelForm::Static,
    );

    packed_matmul_rhs::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
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
                &[KB, KI, NB, NI],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            // One scale per `(block of K, block of N)`: `KI` is carried and addressed by
            // nothing, and the position inside a column block is not an axis of this operand at
            // all, which leaves its innermost axis one it actually varies over.
            TileSpec::new(Projection::new(
                &[KB, KI, NB],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(NB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
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
/// always sound. The other one (a block narrower than the line reading it) is refused by the
/// contraction (`mm_scaled: ... scale blocks must cover whole lines`), and that refusal is a
/// comptime panic inside the kernel, so it lands on a worker thread rather than in a
/// `should_panic` test.
#[test]
fn several_lines_may_share_one_scale() {
    let (field, rows, block_k, blocks_k) = (QuantValue::Q8S, 4, 8, 4);
    let depth = block_k * blocks_k;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
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
    let w_tensor = TensorHandle::new_contiguous(
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

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (M, rows),
                (NB, blocks_n),
                (NI, bn),
                (KB, blocks_k),
                (KI, block_k),
            ]),
            vec![Level::walk(&[
                (M, rows),
                (NB, blocks_n),
                (NI, bn),
                (KB, 1),
                (KI, block_k),
            ])],
        ),
        KernelForm::Static,
    );

    packed_matmul_rhs::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
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
                &[KB, KI, NB, NI],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            // One scale per `(block of K, block of N)`: `KI` is carried and addressed by
            // nothing, and the position inside a column block is not an axis of this operand at
            // all, which leaves its innermost axis one it actually varies over.
            TileSpec::new(Projection::new(
                &[KB, KI, NB],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(NB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
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
/// contraction: the tile serves the element its binding names and the block casts it, so
/// `Packing::Native` never has to be *stated* for a store that carries no scales in its element.
#[test]
fn an_i8_operand_contracts_against_its_scales() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
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

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            vec![Level::walk(&[(M, rows), (N, cols), (KB, 1), (KI, block)])],
        ),
        KernelForm::Static,
    );

    native_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
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
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
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

    let client = cubecl::test_device().client();
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
    let w_tensor = TensorHandle::new_contiguous(
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

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (M, 1),
                (NB, blocks_n),
                (NI, bn),
                (KB, blocks_k),
                (KI, block_k),
            ]),
            vec![Level::cubes(&[(NB, 1)]), Level::walk(&[(KB, 1)])],
        ),
        KernelForm::Static,
    );

    packed_gemv::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
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
                &[KB, KI, NB, NI],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            // One scale per `(block of K, block of N)`: `KI` is carried and addressed by
            // nothing, and the position inside a column block is not an axis of this operand at
            // all, which leaves its innermost axis one it actually varies over.
            TileSpec::new(Projection::new(
                &[KB, KI, NB],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(NB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
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

    let client = cubecl::test_device().client();
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
    let w_tensor = TensorHandle::new_contiguous(
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

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (M, 1),
                (NB, blocks_n),
                (NI, bn),
                (KB, blocks_k),
                (KI, block_k),
            ]),
            vec![Level::cubes(&[(NB, 1)]), Level::walk(&[(KB, 1)])],
        ),
        KernelForm::Static,
    );

    packed_gemv::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
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
                &[KB, KI, NB, NI],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            // One scale per `(block of K, block of N)`: `KI` is carried and addressed by
            // nothing, and the position inside a column block is not an axis of this operand at
            // all, which leaves its innermost axis one it actually varies over.
            TileSpec::new(Projection::new(
                &[KB, KI, NB],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(NB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
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

/// [`packed_gemv`] with the scales taken away: a packed rhs contracting into a promoted
/// accumulator through the plain `mm`.
#[cube(launch)]
fn packed_gemv_unscaled<E: Numeric, V: Size>(
    x: &TileArg<'_, E, Const<1>>,
    w: &TileArg<'_, u32, Const<1>>,
    c: &TileArg<'_, E, V>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(comptime!(space.clone()));
    let w = w.tile_as::<E>(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    for cube in space {
        let x = x.at(&cube);
        let w = w.at(&cube);
        let c = c.at(&cube);
        let mut acc = c.block_accumulator::<E, E, E>(
            &x,
            &w,
            comptime!(Fragments::below(&c, &x)),
            REGISTER_BLOCK,
            Monoid::Sum,
        );
        acc.zero();
        for step in cube {
            let mut acc_s = acc.at(&step);
            acc_s.mma(&x.at(&step), &w.at(&step), Semiring::SUM_PROD);
        }
        for r0 in c.walk().unrolled() {
            let mut c_w = c.at(&r0);
            c_w.copy_cast_from(&acc.at(&r0));
        }
    }
}

/// A packed rhs drains from a promoted accumulator, exactly as its scaled twin does.
///
/// [`a_packed_decode_gemv_runs_in_this_spelling`] runs this block with the scales folded in, and
/// the two reach the same `block::contract` through the same `RegisterData`, so if a packed rhs
/// were what a promoted accumulator could not drain, the gemv could not be spelled either. What
/// makes both work is the accumulator being declared at the *served* width (`V = factor`); a
/// narrower one is refused by the width assert, and that is the only thing packing owes here.
#[test]
fn a_packed_rhs_drains_from_a_promoted_accumulator() {
    let (field, block_k, blocks_k) = (QuantValue::Q8S, 8, 4);
    let depth = block_k * blocks_k;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
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

    let dtype = f32::elem_type_native();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![1, depth])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let w_tensor = TensorHandle::new_contiguous(
        vec![depth, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let c = TestInput::builder(client.clone(), shape![1, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, 1), (N, cols), (KB, blocks_k), (KI, block_k)]),
            vec![Level::cubes(&[(N, bn)]), Level::walk(&[(KB, 1)])],
        ),
        KernelForm::Static,
    );

    packed_gemv_unscaled::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
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
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for n in 0..cols {
        let want: f32 = (0..depth).map(|k| x[k] * w[k * cols + n] as f32).sum();
        let have = got.get_f32(&[0, n]);
        assert!(
            (have - want).abs() < 1e-3,
            "at col {n}: got {have}, want {want}"
        );
    }
}

/// Pack bytes four to a word, low byte first.
fn words_of_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &b)| acc | ((b as u32) << (j * 8)))
        })
        .collect()
}

/// **An 8-bit float field unpacks on read.** `e4m3` codes four to a word, decoded through the
/// format rather than sign-extended; the NaN codes are left out, since one of those has nothing
/// to compare against.
#[test]
fn e4m3_fields_unpack_on_read() {
    let (field, rows, cols) = (QuantValue::E4M3, 8, 8);
    let factor = 32 / field.size_bits();

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        )))
        .enforce();
        return;
    }

    let bytes: Vec<u8> = (0..rows * cols).map(|i| (i * 4) as u8).collect();
    let input = TensorHandle::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words_of_bytes(&bytes))),
        u32::elem_type_native(),
    );
    let dtype = f32::elem_type_native();
    let output = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Space::new(&[(M, rows), (N, cols)]);
    packed_copy::launch(
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
        space.launch_arg(&space),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want = e4m3::from_bits(bytes[m * cols + n]).to_f32();
            let have = got.get_f32(&[m, n]);
            assert!(
                have.to_bits() == want.to_bits(),
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **Scales stored as `ue8m0` bytes are read in their own width.** Four codes to a word, the
/// scales operand bound one word wide and served one scale at a time
/// ([`TileSpec::subword`]), on the lhs of the memory-backed leaf, which steps the block index at
/// runtime. A `ue8m0` code is a bare exponent, so the scales are powers of two and the answer is
/// exact.
#[test]
fn ue8m0_scales_are_read_as_bytes() {
    check_ue8m0_scales(8, 4);
}

/// The same with two blocks a row: a row of scales is half a word, so a word straddles two
/// rows and the slot is the line's index through the whole layout, not its column.
#[test]
fn byte_scale_rows_need_no_word_alignment() {
    check_ue8m0_scales(16, 2);
}

fn check_ue8m0_scales(block: usize, blocks: usize) {
    let (field, rows, cols) = (QuantValue::Q8S, 4, 4);
    let depth = block * blocks;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
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
    // Exponents -2..=2: the code is the exponent biased by 127.
    let exponents: Vec<i32> = (0..rows * blocks).map(|i| (i % 5) as i32 - 2).collect();
    let s: Vec<f32> = exponents.iter().map(|&e| 2f32.powi(e)).collect();
    let codes: Vec<u8> = exponents.iter().map(|&e| (e + 127) as u8).collect();

    let dtype = f32::elem_type_native();
    let w_tensor = TensorHandle::new_contiguous(
        vec![rows, depth],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    // Shape and strides count scales; a row of `blocks` codes is one word.
    let s_tensor = TensorHandle::new_contiguous(
        vec![rows, blocks],
        client.create(Bytes::from_elems(words_of_bytes(&codes))),
        u32::elem_type_native(),
    );
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            vec![Level::walk(&[(M, rows), (N, cols), (KB, 1), (KI, factor)])],
        ),
        KernelForm::Static,
    );

    let w_projection = Projection::new(
        &[M, KB, KI],
        &[
            PhysicalAxisMap::of(M),
            PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
        ],
    );
    packed_matmul_byte_scales::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            TileSpec::new(w_projection.clone()).packed(field),
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
            TileSpec::new(w_projection.scales_per(KB)).subword(Fp8Format::UE8M0, 1),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
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

/// **Scales stored at their own float width are read as fields of a word too.** Two `f16`
/// scales share a word, so a row of them need not start on one and the slot is picked at the
/// read, exactly as a byte field's is.
#[test]
fn half_scales_are_read_as_fields_of_a_word() {
    check_float_scales(FloatKind::F16, 16, 2);
}

/// An `f32` scale fills the word it is stored in, which is the case with no slot to find: the
/// read is the word, reinterpreted.
#[test]
fn full_width_scales_fill_the_word_they_are_read_from() {
    check_float_scales(FloatKind::F32, 8, 4);
}

/// [`check_ue8m0_scales`] with the scales stored as whole floats of `kind`. Halves and quarters,
/// which every width here holds exactly.
fn check_float_scales(kind: FloatKind, block: usize, blocks: usize) {
    let (field, rows, cols) = (QuantValue::Q8S, 4, 4);
    let depth = block * blocks;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
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
    let s: Vec<f32> = (0..rows * blocks)
        .map(|i| 0.25 * (1 + i % 4) as f32)
        .collect();
    let scale_words = match kind {
        FloatKind::F32 => s.iter().map(|v| v.to_bits()).collect(),
        _ => s
            .chunks(2)
            .map(|pair| {
                pair.iter().enumerate().fold(0u32, |acc, (j, &v)| {
                    acc | ((f16::from_f32(v).to_bits() as u32) << (j * 16))
                })
            })
            .collect::<Vec<u32>>(),
    };

    let dtype = f32::elem_type_native();
    let w_tensor = TensorHandle::new_contiguous(
        vec![rows, depth],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    // Shape and strides count scales; how many share a word is the field's to say.
    let s_tensor = TensorHandle::new_contiguous(
        vec![rows, blocks],
        client.create(Bytes::from_elems(scale_words)),
        u32::elem_type_native(),
    );
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            vec![Level::walk(&[(M, rows), (N, cols), (KB, 1), (KI, factor)])],
        ),
        KernelForm::Static,
    );

    let w_projection = Projection::new(
        &[M, KB, KI],
        &[
            PhysicalAxisMap::of(M),
            PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
        ],
    );
    packed_matmul_byte_scales::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            TileSpec::new(w_projection.clone()).packed(field),
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
            TileSpec::new(w_projection.scales_per(KB)).subword(kind, 1),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
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

/// **Byte scales reach the promoted block on the rhs.** `e4m3` codes for the decode gemv's
/// scales, one per `(block of K, block of N)`, read one at a time under the block's constant
/// column ordinal. Halves, which `e4m3` holds exactly.
#[test]
fn e4m3_scales_reach_the_promoted_block() {
    let (field, block_k, blocks_k) = (QuantValue::Q8S, 8, 4);
    let depth = block_k * blocks_k;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        )))
        .enforce();
        return;
    }
    // Four cubes of one packed line each, so a row of the scales is one word of four codes.
    let (cols, bn) = (factor * 4, factor);
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
    let codes: Vec<u8> = s.iter().map(|&v| e4m3::from_f32(v).to_bits()).collect();

    let dtype = f32::elem_type_native();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![1, depth])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let w_tensor = TensorHandle::new_contiguous(
        vec![depth, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let s_tensor = TensorHandle::new_contiguous(
        vec![blocks_k, blocks_n],
        client.create(Bytes::from_elems(words_of_bytes(&codes))),
        u32::elem_type_native(),
    );
    let c = TestInput::builder(client.clone(), shape![1, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (M, 1),
                (NB, blocks_n),
                (NI, bn),
                (KB, blocks_k),
                (KI, block_k),
            ]),
            vec![Level::cubes(&[(NB, 1)]), Level::walk(&[(KB, 1)])],
        ),
        KernelForm::Static,
    );

    packed_gemv_byte_scales::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
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
                &[KB, KI, NB, NI],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, NB],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(NB)],
            ))
            .subword(QuantValue::E4M3, 1),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
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

/// **A packed rhs with byte scales reaches the tensor cores.** Eight activation rows against a
/// `q8` weight packed along its columns, `e4m3` scales per `(block of K, block of N)` read one
/// at a time, the weight landed unpacked and scaled and loaded as the `B` fragment of the M2's
/// `8x8x8` instruction. The prefill shape, one fragment a region.
#[test]
fn a_packed_rhs_reaches_the_tensor_cores() {
    let (field, rows, block_k, blocks_k) = (QuantValue::Q8S, 8, 8, 4);
    let depth = block_k * blocks_k;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        )))
        .enforce();
        return;
    }
    let lanes = client.properties().hardware.plane_size_min as usize;
    // Four column blocks of one packed line each; a fragment covers two of them.
    let (cols, bn) = (factor * 4, factor);
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
    let codes: Vec<u8> = s.iter().map(|&v| e4m3::from_f32(v).to_bits()).collect();

    let dtype = f32::elem_type_native();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let w_tensor = TensorHandle::new_contiguous(
        vec![depth, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let s_tensor = TensorHandle::new_contiguous(
        vec![blocks_k, blocks_n],
        client.create(Bytes::from_elems(words_of_bytes(&codes))),
        u32::elem_type_native(),
    );
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (M, rows),
                (NB, blocks_n),
                (NI, bn),
                (KB, blocks_k),
                (KI, block_k),
            ]),
            vec![Level::walk(&[
                (M, rows),
                (NB, 8 / bn),
                (NI, bn),
                (KB, 1),
                (KI, block_k),
            ])],
        ),
        KernelForm::Static,
    );

    packed_cmma_rhs::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
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
                &[KB, KI, NB, NI],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, NB],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(NB)],
            ))
            .subword(QuantValue::E4M3, 1),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
        1,
        lanes,
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
