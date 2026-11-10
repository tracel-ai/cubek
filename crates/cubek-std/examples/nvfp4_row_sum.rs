//! NVFP4 row sum, twice: once in the tile DSL, once as a plain cubecl kernel.
//!
//! `out[n] = g · Σ_b s[n, b] · Σ_{k in b} e2m1(w[n, k])`: a one-row product against a vector of
//! ones, so it moves exactly the bytes a decode-step gemv moves of its weight (the words and their
//! block scales) and nothing else. Both kernels read the same buffers, laid as a checkpoint
//! stores them: `e2m1` codes eight to a `u32`, low nibble first, rows along `K`; `ue4m3` block
//! scales of sixteen values, four to a `u32`, rows along `K`; one `f32` per-tensor scale.
//!
//! Every kernel is timed with its operands cold (a rotation through 512 MB of copies), in device
//! time, the best of interleaved rounds. The columns, in GB/s of words and block scales:
//!
//! - `ceiling`: cubecl's read probe at the shape's footprint, a persistent loop over one launch.
//! - `plain`: the same bytes read flat in one launch sized to the device, nothing decoded.
//! - `loads`: the raw kernel's loads, one plane a row, nothing decoded.
//! - `raw`: the raw kernel, its planes walking rows grid-stride.
//! - `reduce`, `red×s`: the DSL's `reduce_axis`, the scales attached to the words or multiplied
//!   in, its planes walking the rows grid-stride as the raw kernel's do.
//! - `dot`: the DSL's one-row product against a row of ones, the spelling its gemv takes.
//!
//! Under each row, the same one-row product written by hand, one plane a row as the DSL launches
//! it, then again in the access pattern of a DSL leaf four blocks deep: what `dot` would reach if
//! the DSL's codegen were the hand kernel's, its reread of the lhs for every row included.
//!
//! The DSL kernels need the cubecl of branch `perf/tile-nvfp4-reduce-gap` (on top of this
//! workspace's pin): a reduction that applies its scales, leaves that keep their loads in flight,
//! and `PlaneOrder::capped`. Their spelling, read off the AMDGPU ISA: every walk unrolled
//! (`Walk::unrolled`), a leaf one scale block deep so every load the plane issues is one run, the
//! block scales bound a byte a block (a `ue4m3` byte is an `e4m3` one with its sign clear) so a
//! leaf need not span a word of them, a one-wide accumulator cell so each run is scaled once, and,
//! for the reductions, two cubes a processor.
//!
//! ```sh
//! cargo run --release -p cubek-std --example nvfp4_row_sum --features cubecl/hip
//! ```
//!
//! `NVFP4_DEBUG=1` runs one small shape and prints the reduce's sums against their parts;
//! `NVFP4_ROUNDS`, `NVFP4_PLANES`, `NVFP4_ROWS` and `NVFP4_CUBES_PER_PROCESSOR` (0 for a cube a
//! group of rows) set the rounds and the raw kernel's launch. `NVFP4_SHAPE` keeps one shape.
//! The DSL's spelling: `NVFP4_DSL_LEAF_BLOCKS` (1), `NVFP4_DSL_SCALE_BYTES` (1; 0 binds words of
//! four), `NVFP4_DSL_SCALE_WIDTH` (4, for words) and `NVFP4_DSL_CUBES_PER_PROCESSOR` (2; 0 for a
//! plane a row). `CUBECL_DEBUG_PLIRON=<dir>` dumps every kernel's LLVM IR (`amdgpu.ll`) and ISA
//! (`amdgpu.s`).
//!
//! `NVFP4_GEMV_ONLY=1` times only the persistent row sum against the split-K gemv
//! ([`raw_gemv_split_k`]), a real `f16` lhs checked against an `f64` reference: the gemv's
//! ceiling. `NVFP4_GEMV_LINES` (1), `NVFP4_GEMV_COLUMNS` (8) and `NVFP4_GEMV_CUBES_PER_PROCESSOR`
//! (4) set its launch. On the Radeon 8060S it reads 98–101% of the row sum on Qwen3-8B's
//! projections; three lines a lane (four planes for `K` = 12288) falls to 68%.

#![allow(clippy::needless_range_loop)]

use std::time::Duration;

use cubecl::client::Client;
use cubecl::future::block_on;
use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::prelude::*;
use cubecl::quant::scheme::{QuantValue, ScaleDtype};
use cubecl::std::tensor::TensorHandle;
use cubecl::std::throughput::{measure_launch_overhead, memory_bound};
use cubecl::throughput::MemoryAccess;
use cubecl::tile::kind::Field;
use cubecl::tile::launch::Grid;
use cubecl::tile::layout::PhysicalAxisMap;
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubecl::tile::{
    Axis, Instruction, Launcher, Levels, Monoid, Partitioning, Projection, RegisterBlock, Semiring,
    Space, TileTensor, TileTensorLaunch,
};
use cubecl::tune::Work;
use cubecl::zspace::{Shape, metadata::Metadata};
use half::f16;

/// The rows.
const N: Axis = Axis(0);
/// The output's one column: the DSL's register reducer drains into a matrix.
const C: Axis = Axis(3);
/// Which block of sixteen values along `K`, and where inside it.
const KB: Axis = Axis(1);
const KI: Axis = Axis(2);

const BLOCK: usize = 16;
const CODES_PER_WORD: usize = 8;
const SCALES_PER_WORD: usize = 4;
/// Values one 16-byte load brings: four words.
const VALUES_PER_LINE: usize = 32;
/// Values a unit takes a step in both kernels: one word of scales, two lines of words.
const VALUES_PER_STEP: usize = BLOCK * SCALES_PER_WORD;
/// Rows a cube holds, one a plane.
const PLANES_PER_CUBE: usize = 8;
/// Cubes a processor runs of the raw kernel, its planes walking the rows grid-stride: on the
/// Radeon 8060S two beat one cube a row by 3 to 20%, and four, eight and sixteen fall back.
const RAW_CUBES_PER_PROCESSOR: usize = 2;

/// What the raw kernel's placed values are short of: `2^14` for an `e2m1` code moved onto an
/// `f16`'s exponent and mantissa, `2^8` for a `ue4m3` scale moved the same way.
const PLACED_LIFT: f64 = (1u32 << 22) as f64;

/// The register block the DSL reduction accumulates in.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);
/// The instruction the DSL contraction runs: registers, a lane's partials kept to the drain.
const INSTRUCTION: Instruction = Instruction::Registers {
    config: RegisterBlock::new(16),
};

const WORD: ElemType = ElemType::UInt(UIntKind::U32);
const F32: ElemType = ElemType::Float(FloatKind::F32);

// ---------------------------------------------------------------------------------------------
// The tile DSL.
// ---------------------------------------------------------------------------------------------

/// Every plane sums its row: each unit reduces the values it reads, under the scales the operand
/// carries, and the drain sums the units' partials across the plane.
#[cube(launch_unchecked)]
fn dsl_row_sum(w: &TileTensor<'_>, out: &TileTensor<'_>, partitioning: Partitioning) {
    let w = w.tile(&partitioning);
    let out = out.tile(&partitioning);
    for plane in &partitioning {
        let (o, w) = (out.at(&plane), w.at(&plane));
        let mut sum = o.block_reducer::<f32>(&w, REGISTER_BLOCK, Monoid::Sum);
        sum.init::<f32>(0.0f32);
        for step in plane.walk_within(&w).unrolled() {
            for unit in step {
                let mut unit_sum = sum.at(&unit);
                unit_sum.reduce_axis_accumulate(&w.at(&unit), Monoid::Sum);
            }
        }
        sum.drained_into(&o);
    }
}

/// [`dsl_row_sum`] with the scales separate operands, multiplied in by the kernel as the DSL's
/// own `nvfp4` decode states them: the words, under the blocks, under the per-tensor scale.
#[cube(launch_unchecked)]
fn dsl_row_sum_product(
    w: &TileTensor<'_>,
    blocks: &TileTensor<'_>,
    global: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
) {
    let w = w
        .tile(&partitioning)
        .mul(&blocks.tile(&partitioning))
        .mul(&global.tile(&partitioning));
    let out = out.tile(&partitioning);
    for plane in &partitioning {
        let (o, w) = (out.at(&plane), w.at(&plane));
        let mut sum = o.block_reducer::<f32>(&w, REGISTER_BLOCK, Monoid::Sum);
        sum.init::<f32>(0.0f32);
        for step in plane.walk_within(&w).unrolled() {
            for unit in step {
                let mut unit_sum = sum.at(&unit);
                unit_sum.reduce_axis_accumulate(&w.at(&unit), Monoid::Sum);
            }
        }
        sum.drained_into(&o);
    }
}

/// The same sum as the one-row product against a row of ones: the spelling a decode-step gemv
/// takes, the words' scales pulled out as factors the drain applies.
#[cube(launch_unchecked)]
fn dsl_row_dot(
    x: &TileTensor<'_>,
    w: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
) {
    let a = x.tile(&partitioning);
    let b = w.tile(&partitioning);
    let c = out.tile(&partitioning);
    for plane in &partitioning {
        let (mut c, a, b) = (c.at(&plane), a.at(&plane), b.at(&plane));
        let (a, a_factor) = a.factor(c.space(), INSTRUCTION);
        let (b, b_factor) = b.factor(c.space(), INSTRUCTION);
        let mut sum = c.accumulator::<f32>(&a, &b, INSTRUCTION, Semiring::SUM_PROD);
        sum.zero();
        for step in plane.walk_within(&b).unrolled() {
            for unit in step {
                let mut unit_sum = sum.at(&unit);
                unit_sum.mma(&a.at(&unit), &b.at(&unit));
            }
        }
        c.dequantize_from(&sum.scaled(&a_factor).scaled(&b_factor));
    }
}

/// How the DSL kernel states the sum.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Spelling {
    /// `reduce_axis` over the words, their scales attached to them.
    Reduce,
    /// `reduce_axis` over the words times their scales, each scale its own operand.
    ReduceProduct,
    /// A one-row product against ones.
    Contract,
}

/// The DSL launch: a plane a row, its units taking `K`'s steps of [`VALUES_PER_STEP`] in turns,
/// the words served a line at a time under their scale words and the per-tensor scale.
fn dsl_launch(
    client: &Client,
    data: &Device,
    rows: usize,
    k: usize,
    spelling: Spelling,
) -> (Launcher, Vec<TileTensorLaunch>) {
    let plane = client.properties().hardware.plane_size_max as usize;
    let dsl_line = std::env::var("NVFP4_DSL_LINE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(BLOCK);
    let space = Space::new(&[(N, rows), (C, 1), (KB, k / BLOCK), (KI, BLOCK)]);
    // A unit's leaf, in blocks of sixteen along `K`: one, so every load the plane issues is one
    // run of the row, its units side by side. Deeper leaves span a word of scales, which the
    // scales bound as words need, and read four lines a unit a quarter-used.
    let leaf_blocks = env_usize("NVFP4_DSL_LEAF_BLOCKS", 1);
    let levels = Levels::leaf(&[(N, 1), (C, 1), (KB, leaf_blocks), (KI, BLOCK)])
        .units(&[(KB, plane)])
        .walk_every(&[KB]);
    // `NVFP4_DSL_CUBES_PER_PROCESSOR` caps the launch at that many cubes a processor, its planes
    // walking the rows grid-stride; 0 launches a plane a row.
    let order = PlaneOrder::new(&[(N, PLANES_PER_CUBE)], CubeOrder::RowMajor);
    // The reductions only: a capped one-row product hoists its row-invariant lhs out of the row
    // walk, more registers than a long `K` has, and spills.
    let cap = match spelling {
        Spelling::Contract => 0,
        Spelling::Reduce | Spelling::ReduceProduct => env_usize("NVFP4_DSL_CUBES_PER_PROCESSOR", 2),
    };
    let order = match cap {
        0 => order,
        per => {
            let processors = client
                .properties()
                .hardware
                .num_streaming_multiprocessors
                .unwrap_or(32) as usize;
            order.capped(per * processors)
        }
    };
    let levels = levels.launch_planes(&[N], order).build();
    let launcher = Launcher::new(
        client,
        Partitioning::new(space.clone(), levels),
        &space,
        Grid::FromLevels,
    )
    .expect("the launch");

    // Words: a binding over values, `[rows, k]`, its strides in values; the packing says eight
    // share a word.
    let mut words = data.words.clone().binding();
    words.shape = [rows, k].into();
    words.strides = [k, 1].into();
    let words = launcher
        .arg(words, WORD)
        .gathered(Projection::new(
            &[N, KB, KI],
            &[
                PhysicalAxisMap::of(N),
                PhysicalAxisMap::disjoint(&[(KB, BLOCK), (KI, 1)]),
            ],
        ))
        .vectorize(dsl_line)
        .packed(QuantValue::E2M1);

    // Block scales: a `ue4m3` field a block, four to a word, a word read at once.
    let blocks = k / BLOCK;
    let mut scales = data.scales.clone().binding();
    scales.shape = [rows, blocks].into();
    scales.strides = [blocks, 1].into();
    let scale_projection =
        Projection::new(&[N, KB], &[PhysicalAxisMap::of(N), PhysicalAxisMap::of(KB)]);
    let scales = match env_usize("NVFP4_DSL_SCALE_BYTES", 1) {
        // Each scale its own element: a `ue4m3` byte is an `e4m3` one with its sign clear, so it
        // reads as one, a byte a block, and a unit's leaf need not span a word of them.
        1 => launcher
            .arg(scales, ElemType::Float(FloatKind::E4M3))
            .gathered(scale_projection)
            .vectorize(1),
        _ => launcher
            .arg(scales, WORD)
            .gathered(scale_projection)
            .vectorize(env_usize("NVFP4_DSL_SCALE_WIDTH", SCALES_PER_WORD))
            .packed(Field::of_scale(ScaleDtype::UE4M3)),
    }
    .build()
    .expect("the block scales")
    .arg();

    // The per-tensor scale: one word over every axis the words carry but the position in a block.
    let mut global = data.global.clone().binding();
    global.shape = [1, 1].into();
    global.strides = [1, 1].into();
    let global = launcher
        .arg(global, WORD)
        .gathered(Projection::new(
            &[N, KB],
            &[PhysicalAxisMap::broadcast(), PhysicalAxisMap::broadcast()],
        ))
        .packed(Field::of_scale(ScaleDtype::F32))
        .build()
        .expect("the per-tensor scale")
        .arg();

    let words = words.build().expect("the words").arg();
    let (w, separate) = match spelling {
        Spelling::ReduceProduct => (words, Some([scales, global])),
        _ => (words.scales(scales).scales(global), None),
    };

    let mut out = data.out.clone().binding();
    let out = match spelling {
        Spelling::Reduce | Spelling::ReduceProduct => {
            out.shape = [rows, 1].into();
            out.strides = [1, 1].into();
            launcher.arg(out, F32).axes(&[N, C])
        }
        Spelling::Contract => {
            out.shape = [1, rows].into();
            out.strides = [rows, 1].into();
            launcher.arg(out, F32).axes(&[C, N])
        }
    }
    .build()
    .expect("the output")
    .arg();
    match spelling {
        Spelling::Reduce => (launcher, vec![w, out]),
        Spelling::ReduceProduct => {
            let [blocks, global] = separate.expect("separate scales");
            (launcher, vec![w, blocks, global, out])
        }
        Spelling::Contract => {
            let mut ones = data.ones.clone().binding();
            ones.shape = [1, k].into();
            ones.strides = [k, 1].into();
            let x = launcher
                .arg(ones, F32)
                .gathered(Projection::new(
                    &[C, KB, KI],
                    &[
                        PhysicalAxisMap::of(C),
                        PhysicalAxisMap::disjoint(&[(KB, BLOCK), (KI, 1)]),
                    ],
                ))
                .vectorize(dsl_line)
                .build()
                .expect("the ones")
                .arg();
            (launcher, vec![x, w, out])
        }
    }
}

// ---------------------------------------------------------------------------------------------
// No abstraction.
// ---------------------------------------------------------------------------------------------

/// The four `e2m1` pairs of `word` placed as `f16` pairs, summed: codes `j` and `j + 4` sit
/// sixteen bits apart, so one mask and shift moves both onto an `f16`'s exponent and mantissa and
/// their sign onto its sign, `2^14` short of their values.
#[cube]
fn word_pair_sum(word: u32) -> Vector<f16, Const<2>> {
    let mut sum = Vector::<f16, Const<2>>::new(f16::new(0.0));
    #[unroll]
    for j in 0..4u32 {
        let codes = word >> (4 * j);
        let bits = ((codes & 0x0007_0007) << 9) | ((codes & 0x0008_0008) << 12);
        sum += Vector::<f16, Const<2>>::reinterpret(bits);
    }
    sum
}

/// Every plane sums `rows` consecutive rows. Those rows are one run of memory, and unit `u` reads
/// lines `u`, `u + P`, `u + 2P`, … of it, `P` the plane's units, so every load the plane issues is
/// one run of `16 · P` bytes, beside the scale word of the line, which neighbours share. All of a
/// unit's loads are issued before any decode: `lines`, a row's lines per unit, and `rows` are
/// known at compile time, so which row line `i` falls in is too.
#[cube(launch_unchecked)]
fn raw_row_sum<V: Size>(
    words: &Tensor<Vector<u32, V>>,
    scales: &Tensor<u32>,
    global: &Tensor<f32>,
    out: &mut Tensor<f32>,
    #[comptime] lines: usize,
    #[comptime] rows: usize,
) {
    let plane_dim = PLANE_DIM as usize;
    let lane = UNIT_POS_PLANE as usize;
    let planes_per_cube = CUBE_DIM as usize / plane_dim;
    let plane = CUBE_POS * planes_per_cube + UNIT_POS as usize / plane_dim;
    // A launch of fewer planes than row groups walks them grid-stride; one of as many takes one.
    let planes = CUBE_COUNT * planes_per_cube;
    let mut first_row = plane * rows;
    while first_row < out.len() {
        raw_rows::<V>(words, scales, global, out, first_row, lane, lines, rows);
        first_row += planes * rows;
    }
}

/// The `rows` rows from `first_row` summed by one plane, as [`raw_row_sum`] says.
#[cube]
#[allow(clippy::too_many_arguments)]
fn raw_rows<V: Size>(
    words: &Tensor<Vector<u32, V>>,
    scales: &Tensor<u32>,
    global: &Tensor<f32>,
    out: &mut Tensor<f32>,
    first_row: usize,
    lane: usize,
    #[comptime] lines: usize,
    #[comptime] rows: usize,
) {
    let plane_dim = PLANE_DIM as usize;
    let row_lines = lines * plane_dim;
    let line_base = first_row * row_lines;
    let scale_base = first_row * (row_lines / 2);

    let mut loaded = Array::<Vector<u32, V>>::new(lines * rows);
    let mut scale_words = Array::<u32>::new(lines * rows);
    #[unroll]
    for i in 0..lines * rows {
        let line = i * plane_dim + lane;
        loaded[i] = words[line_base + line];
        scale_words[i] = scales[scale_base + line / 2];
    }

    let mut sums = Array::<f32>::new(rows);
    #[unroll]
    for r in 0..rows {
        sums[r] = 0.0f32;
    }
    #[unroll]
    for i in 0..lines * rows {
        let line = i * plane_dim + lane;
        let words = loaded[i];
        // The line's two blocks are bytes `2 (line % 2)` and the next of its scale word.
        let pair = scale_words[i] >> (16 * (line % 2) as u32);
        let scale_bits = ((pair & 0x7F) | ((pair & 0x7F00) << 8)) << 7;
        let scale =
            Vector::<f32, Const<2>>::cast_from(Vector::<f16, Const<2>>::reinterpret(scale_bits));
        let first = Vector::<f32, Const<2>>::cast_from(
            word_pair_sum(words.extract(0usize)) + word_pair_sum(words.extract(1usize)),
        );
        let second = Vector::<f32, Const<2>>::cast_from(
            word_pair_sum(words.extract(2usize)) + word_pair_sum(words.extract(3usize)),
        );
        let r = comptime!(i / lines);
        sums[r] += (first.extract(0usize) + first.extract(1usize)) * scale.extract(0usize)
            + (second.extract(0usize) + second.extract(1usize)) * scale.extract(1usize);
    }

    let lift = global[0] * f32::new(PLACED_LIFT);
    #[unroll]
    for r in 0..rows {
        let total = plane_sum(sums[r]);
        if lane == 0 {
            out[first_row + r] = total * lift;
        }
    }
}

/// The eight values of `word` against the eight of `x` they meet, `x` the values `8w` to `8w + 7`
/// as two lines: codes `j` and `j + 4` are one placed `f16` pair, so pair `j` meets `x[j]` and
/// `x[j + 4]`.
#[cube]
fn word_dot(word: u32, lo: Vector<f32, Const<4>>, hi: Vector<f32, Const<4>>) -> f32 {
    let mut sum = 0.0f32;
    #[unroll]
    for j in 0..4u32 {
        let codes = word >> (4 * j);
        let bits = ((codes & 0x0007_0007) << 9) | ((codes & 0x0008_0008) << 12);
        let pair = Vector::<f32, Const<2>>::cast_from(Vector::<f16, Const<2>>::reinterpret(bits));
        sum += pair.extract(0usize) * lo.extract(j as usize)
            + pair.extract(1usize) * hi.extract(j as usize);
    }
    sum
}

/// The raw kernel as a one-row product against `x`, one plane a row as the DSL's launch has it:
/// every line of words read beside the eight lines of `x` its values meet. What the DSL's `dot`
/// spelling moves, written by hand: it says how much of that spelling's cost is the lhs it reads
/// again for every row.
#[cube(launch_unchecked)]
fn raw_row_dot<V: Size>(
    words: &Tensor<Vector<u32, V>>,
    scales: &Tensor<u32>,
    global: &Tensor<f32>,
    x: &Tensor<Vector<f32, Const<4>>>,
    out: &mut Tensor<f32>,
    #[comptime] lines: usize,
) {
    let plane_dim = PLANE_DIM as usize;
    let lane = UNIT_POS_PLANE as usize;
    let row = CUBE_POS * (CUBE_DIM as usize / plane_dim) + UNIT_POS as usize / plane_dim;
    let row_lines = lines * plane_dim;
    let line_base = row * row_lines;
    let scale_base = row * (row_lines / 2);

    let mut sum = 0.0f32;
    #[unroll]
    for i in 0..lines {
        let line = i * plane_dim + lane;
        let words = words[line_base + line];
        let pair = scales[scale_base + line / 2] >> (16 * (line % 2) as u32);
        let scale_bits = ((pair & 0x7F) | ((pair & 0x7F00) << 8)) << 7;
        let scale =
            Vector::<f32, Const<2>>::cast_from(Vector::<f16, Const<2>>::reinterpret(scale_bits));
        let mut blocks = Array::<f32>::new(2usize);
        #[unroll]
        for b in 0..2usize {
            let mut block = 0.0f32;
            #[unroll]
            for w in 0..2usize {
                let word = 2 * b + w;
                let at = line * 8 + word * 2;
                block += word_dot(words.extract(word), x[at], x[at + 1]);
            }
            blocks[b] = block;
        }
        sum += blocks[0] * scale.extract(0usize) + blocks[1] * scale.extract(1usize);
    }

    let total = plane_sum(sum);
    if lane == 0 {
        out[row] = total * global[0] * f32::new(PLACED_LIFT);
    }
}

/// [`raw_row_dot`] with the DSL's access pattern: each unit takes a run of 64 values a step (one
/// word of scales), read as four 8-byte loads of 16 values, the units side by side. Every load
/// the plane issues spans `32 · P` bytes and uses a quarter of them.
#[cube(launch_unchecked)]
fn raw_row_dot_blocked(
    words: &Tensor<Vector<u32, Const<2>>>,
    scales: &Tensor<u32>,
    global: &Tensor<f32>,
    x: &Tensor<Vector<f32, Const<4>>>,
    out: &mut Tensor<f32>,
    #[comptime] steps: usize,
) {
    let plane_dim = PLANE_DIM as usize;
    let lane = UNIT_POS_PLANE as usize;
    let row = CUBE_POS * (CUBE_DIM as usize / plane_dim) + UNIT_POS as usize / plane_dim;
    let row_runs = steps * plane_dim;
    let mut sum = 0.0f32;
    #[unroll]
    for s in 0..steps {
        let run = row * row_runs + s * plane_dim + lane;
        let scale_word = scales[run];
        let in_row = s * plane_dim + lane;
        #[unroll]
        for b in 0..4usize {
            let pair_words = words[run * 4 + b];
            let byte = (scale_word >> (8 * b as u32)) & 0x7F;
            let scale =
                f32::cast_from(Vector::<f16, Const<2>>::reinterpret(byte << 7).extract(0usize));
            let at = (in_row * 4 + b) * 4;
            let block = word_dot(pair_words.extract(0usize), x[at], x[at + 1])
                + word_dot(pair_words.extract(1usize), x[at + 2], x[at + 3]);
            sum += block * scale;
        }
    }
    let total = plane_sum(sum);
    if lane == 0 {
        out[row] = total * global[0] * f32::new(PLACED_LIFT);
    }
}

/// The raw kernel's loads, decoding nothing: each unit folds the bits it read into one word, so
/// no load can be dropped. The ceiling of this access pattern.
#[cube(launch_unchecked)]
fn raw_loads_only<V: Size>(
    words: &Tensor<Vector<u32, V>>,
    scales: &Tensor<u32>,
    out: &mut Tensor<u32>,
    #[comptime] lines: usize,
    #[comptime] rows: usize,
) {
    let plane_dim = PLANE_DIM as usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane = CUBE_POS * (CUBE_DIM as usize / plane_dim) + UNIT_POS as usize / plane_dim;
    let first_row = plane * rows;
    let row_lines = lines * plane_dim;
    let line_base = first_row * row_lines;
    let scale_base = first_row * (row_lines / 2);

    let mut folded = Vector::<u32, V>::new(0u32);
    let mut scale_fold = 0u32;
    #[unroll]
    for i in 0..lines * rows {
        let line = i * plane_dim + lane;
        folded ^= words[line_base + line];
        scale_fold ^= scales[scale_base + line / 2];
    }
    let mut word = scale_fold;
    #[unroll]
    for v in 0..V::value() {
        word ^= folded.extract(v);
    }
    out[plane * plane_dim + lane] = word;
}

/// The decode-step gemv from first principles: `out[n] = Σ_k x[k] · w[n, k]`, `x` an `f16` row.
///
/// `K` is split across the cube's `planes` planes, and each lane owns `lines` 16-byte lines of
/// every row, `PLANE_DIM` lines apart, so every load the plane issues is one run of 512 bytes. The
/// lane's share of `x`, the same for every row, is read once and kept in registers for the
/// cube's whole life. The cubes then walk the rows grid-stride, `columns` at a time: every load of
/// the group is issued before any decode, each plane sums its share of a row, and the planes'
/// partials meet in shared memory, one barrier a group. The partials are double-buffered, so the
/// next group's writes never race this group's reads.
#[cube(launch_unchecked)]
fn raw_gemv_split_k(
    words: &Tensor<Vector<u32, Const<4>>>,
    scales: &Tensor<u32>,
    global: &Tensor<f32>,
    x: &Tensor<Vector<f16, Const<4>>>,
    out: &mut Tensor<f32>,
    #[comptime] lines: usize,
    #[comptime] planes: usize,
    #[comptime] columns: usize,
) {
    let plane_dim = PLANE_DIM as usize;
    let lane = UNIT_POS_PLANE as usize;
    let p = UNIT_POS as usize / plane_dim;
    let row_lines = lines * plane_dim * planes;
    let first_line = p * lines * plane_dim + lane;

    // The lane's `x`: line `i` holds values `32 line` to `32 line + 31`, eight lines of four.
    let mut xs = Array::<Vector<f32, Const<4>>>::new(lines * 8);
    #[unroll]
    for i in 0..lines {
        let line = first_line + i * plane_dim;
        #[unroll]
        for q in 0..8usize {
            xs[i * 8 + q] = Vector::<f32, Const<4>>::cast_from(x[line * 8 + q]);
        }
    }

    let mut partials = Shared::<[f32]>::new_slice(2 * planes * columns);
    let lift = global[0] * f32::new(PLACED_LIFT);
    // A line's two scale bytes sit in the half of a word its parity picks: `row_lines` and the
    // lane strides are even, so that parity is the lane's.
    let half = (16 * (lane % 2)) as u32;
    let groups = out.len() / columns;
    let mut group = CUBE_POS;
    let mut parity = 0usize;
    while group < groups {
        let first_row = group * columns;
        let mut loaded = Array::<Vector<u32, Const<4>>>::new(columns * lines);
        let mut scale_words = Array::<u32>::new(columns * lines);
        #[unroll]
        for c in 0..columns {
            #[unroll]
            for i in 0..lines {
                let at = (first_row + c) * row_lines + first_line + i * plane_dim;
                loaded[c * lines + i] = words[at];
                scale_words[c * lines + i] = scales[at / 2];
            }
        }
        #[unroll]
        for c in 0..columns {
            let mut sum = 0.0f32;
            #[unroll]
            for i in 0..lines {
                let line_words = loaded[c * lines + i];
                let pair = scale_words[c * lines + i] >> half;
                let scale_bits = ((pair & 0x7F) | ((pair & 0x7F00) << 8)) << 7;
                let scale = Vector::<f32, Const<2>>::cast_from(
                    Vector::<f16, Const<2>>::reinterpret(scale_bits),
                );
                let first = word_dot(line_words.extract(0usize), xs[i * 8], xs[i * 8 + 1])
                    + word_dot(line_words.extract(1usize), xs[i * 8 + 2], xs[i * 8 + 3]);
                let second = word_dot(line_words.extract(2usize), xs[i * 8 + 4], xs[i * 8 + 5])
                    + word_dot(line_words.extract(3usize), xs[i * 8 + 6], xs[i * 8 + 7]);
                sum += first * scale.extract(0usize) + second * scale.extract(1usize);
            }
            let total = plane_sum(sum);
            if lane == 0 {
                partials[(parity * planes + p) * columns + c] = total;
            }
        }
        sync_cube();
        let unit = UNIT_POS as usize;
        if unit < columns {
            let mut total = 0.0f32;
            #[unroll]
            for q in 0..planes {
                total += partials[(parity * planes + q) * columns + unit];
            }
            out[first_row + unit] = total * lift;
        }
        parity = 1 - parity;
        group += CUBE_COUNT;
    }
}

/// The units [`plain_read`] launches on `client`'s device.
fn plain_units(client: &Client) -> usize {
    let processors = client
        .properties()
        .hardware
        .num_streaming_multiprocessors
        .unwrap_or(32) as usize;
    processors * PLAIN_UNITS_PER_PROCESSOR
}

/// Reads each unit of [`plain_read`] issues together on a turn of its walk.
const PLAIN_READS_IN_FLIGHT: usize = 4;
/// Units [`plain_read`] launches per processor.
const PLAIN_UNITS_PER_PROCESSOR: usize = 1024;

/// The same bytes read as two flat runs of lines, no rows: every unit walks the words and then the
/// scales grid-stride, [`PLAIN_READS_IN_FLIGHT`] loads a turn, a launch sized to the device's
/// processors. What one launch reading these bytes can reach when nothing about the problem
/// shapes its reads.
#[cube(launch_unchecked)]
fn plain_read<V: Size>(
    words: &Tensor<Vector<u32, V>>,
    scales: &Tensor<Vector<u32, V>>,
    out: &mut Tensor<Vector<u32, V>>,
) {
    let unit = CUBE_POS * CUBE_DIM as usize + UNIT_POS as usize;
    let units = out.len();
    let mut folded = Vector::<u32, V>::new(0u32);
    let mut turn = unit;
    while turn < words.len() {
        #[unroll]
        for i in 0..PLAIN_READS_IN_FLIGHT {
            let at = turn + i * units;
            if at < words.len() {
                folded ^= words[at];
            }
        }
        turn += PLAIN_READS_IN_FLIGHT * units;
    }
    let mut turn = unit;
    while turn < scales.len() {
        folded ^= scales[turn];
        turn += units;
    }
    out[unit] = folded;
}

// ---------------------------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------------------------

/// One copy of the operands on the device.
struct Device {
    words: TensorHandle,
    scales: TensorHandle,
    global: TensorHandle,
    out: TensorHandle,
    scratch: TensorHandle,
    /// Where each unit of [`plain_read`] writes its fold, a line a unit.
    plain_out: TensorHandle,
    /// A row of `k` ones, the contraction's lhs.
    ones: TensorHandle,
    /// [`Host::x`] as `f16`, the gemv's lhs.
    x16: TensorHandle,
}

/// The operands on the host and what each row sums to.
struct Host {
    words: Vec<u32>,
    scale_words: Vec<u32>,
    global: f32,
    expected: Vec<f64>,
    /// The gemv's lhs, every value exact in `f16`.
    x: Vec<f32>,
    /// What each row of the gemv against [`Host::x`] comes to.
    expected_gemv: Vec<f64>,
}

const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn ue4m3(byte: u8) -> f64 {
    let (exp, mant) = ((byte >> 3) & 0xF, byte & 0x7);
    match exp {
        0 => (mant as f64 / 8.0) * 2f64.powi(-6),
        _ => (1.0 + mant as f64 / 8.0) * 2f64.powi(exp as i32 - 7),
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

impl Host {
    fn new(rows: usize, k: usize) -> Self {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let codes: Vec<u8> = (0..rows * k).map(|_| (rng.next() % 16) as u8).collect();
        // Normal scales from 2^-3 to 2^4, subnormals now and then; never 0x7F, `ue4m3`'s NaN.
        let scales: Vec<u8> = (0..rows * k / BLOCK)
            .map(|i| match i % 37 {
                0 => (rng.next() % 8) as u8,
                _ => (0x20 + rng.next() % 0x40) as u8,
            })
            .collect();
        let global = 0.371;

        let words = codes
            .chunks(CODES_PER_WORD)
            .map(|word| {
                word.iter()
                    .enumerate()
                    .fold(0u32, |acc, (j, &c)| acc | ((c as u32) << (4 * j)))
            })
            .collect();
        let scale_words = scales
            .chunks(SCALES_PER_WORD)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
            .collect();
        let blocks = k / BLOCK;
        let expected = (0..rows)
            .map(|n| {
                let sum: f64 = (0..blocks)
                    .map(|b| {
                        let block: f64 = (0..BLOCK)
                            .map(|i| {
                                let code = codes[n * k + b * BLOCK + i];
                                let magnitude = E2M1[(code & 7) as usize];
                                if code & 8 != 0 { -magnitude } else { magnitude }
                            })
                            .sum();
                        block * ue4m3(scales[n * blocks + b])
                    })
                    .sum();
                sum * global as f64
            })
            .collect();
        // Eighths from -2 to 2, never all one sign or one magnitude, so a misplaced `x` shows.
        let x: Vec<f32> = (0..k)
            .map(|i| {
                let v = 0.25 + ((i * 37) % 16) as f32 / 8.0;
                if i % 5 == 0 { -v } else { v }
            })
            .collect();
        let expected_gemv = (0..rows)
            .map(|n| {
                let sum: f64 = (0..blocks)
                    .map(|b| {
                        let block: f64 = (0..BLOCK)
                            .map(|i| {
                                let at = b * BLOCK + i;
                                let code = codes[n * k + at];
                                let magnitude = E2M1[(code & 7) as usize];
                                let value = if code & 8 != 0 { -magnitude } else { magnitude };
                                value * x[at] as f64
                            })
                            .sum();
                        block * ue4m3(scales[n * blocks + b])
                    })
                    .sum();
                sum * global as f64
            })
            .collect();
        Host {
            words,
            scale_words,
            global,
            expected,
            x,
            expected_gemv,
        }
    }

    fn upload(&self, client: &Client, rows: usize, k: usize, plane: usize) -> Device {
        let words: Vec<u8> = self.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let scales: Vec<u8> = self
            .scale_words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let tensor = |bytes: &[u8], elem| {
            let n = bytes.len() / 4;
            TensorHandle::from_metadata(
                client.create_from_slice(bytes),
                Metadata::new(Shape::new([n]), vec![1]),
                elem,
            )
        };
        Device {
            words: tensor(&words, WORD),
            scales: tensor(&scales, WORD),
            global: tensor(&self.global.to_le_bytes(), F32),
            out: tensor(&vec![0u8; rows * 4], F32),
            scratch: tensor(&vec![0u8; rows * plane * 4], WORD),
            plain_out: tensor(&vec![0u8; plain_units(client) * 16], WORD),
            ones: tensor(
                &(0..k).flat_map(|_| 1f32.to_le_bytes()).collect::<Vec<u8>>(),
                F32,
            ),
            x16: TensorHandle::from_metadata(
                client.create_from_slice(
                    &self
                        .x
                        .iter()
                        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                        .collect::<Vec<u8>>(),
                ),
                Metadata::new(Shape::new([k]), vec![1]),
                ElemType::Float(FloatKind::F16),
            ),
        }
    }
}

impl Host {
    /// The codes of block `b` of `row`, summed, unscaled.
    fn block_codes(&self, row: usize, b: usize, k: usize) -> f64 {
        (0..BLOCK)
            .map(|i| {
                let at = row * k + b * BLOCK + i;
                let code = (self.words[at / CODES_PER_WORD] >> (4 * (at % CODES_PER_WORD))) & 0xF;
                let magnitude = E2M1[(code & 7) as usize];
                if code & 8 != 0 { -magnitude } else { magnitude }
            })
            .sum()
    }

    /// The row's sum over `range` of its blocks.
    fn scaled_blocks(&self, row: usize, k: usize, range: std::ops::Range<usize>) -> f64 {
        let per_row = k / BLOCK;
        range
            .map(|b| {
                let at = row * per_row + b;
                let byte =
                    (self.scale_words[at / SCALES_PER_WORD] >> (8 * (at % SCALES_PER_WORD))) as u8;
                self.block_codes(row, b, k) * ue4m3(byte)
            })
            .sum::<f64>()
            * self.global as f64
    }

    /// The row's sum over its first `blocks` blocks only.
    fn partial(&self, row: usize, k: usize, blocks: usize) -> f64 {
        let per_row = k / BLOCK;
        (0..blocks)
            .map(|b| {
                let at = row * per_row + b;
                let byte =
                    (self.scale_words[at / SCALES_PER_WORD] >> (8 * (at % SCALES_PER_WORD))) as u8;
                self.block_codes(row, b, k) * ue4m3(byte)
            })
            .sum::<f64>()
            * self.global as f64
    }
}

fn read_f32(client: &Client, handle: &TensorHandle) -> Vec<f32> {
    client
        .read_one_unchecked(handle.handle.clone())
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// The largest error against the reference, relative to the row's magnitude.
fn worst_error(got: &[f32], host: &Host) -> f64 {
    worst_error_against(got, &host.expected)
}

/// The largest error against `expected`, relative to the row's magnitude.
fn worst_error_against(got: &[f32], expected: &[f64]) -> f64 {
    got.iter()
        .zip(expected)
        .map(|(&got, &want)| (got as f64 - want).abs() / want.abs().max(1.0))
        .fold(0.0, f64::max)
}

/// A launch with every argument built, waiting to be issued.
type Prepared = Box<dyn FnOnce() + Send>;

/// The device time of one launch, cycling through `copies` copies of the operands so none is
/// served from a cache the previous launch warmed. The launches' arguments are built before the
/// profiled window opens, so a kernel whose arguments take long to build on the host is not
/// timed on its host.
fn time(client: &Client, copies: usize, prepare: impl Fn(usize) -> Prepared) -> Duration {
    for copy in 0..copies {
        prepare(copy)();
    }
    block_on(client.sync()).expect("sync");
    let launches = (400 / copies).max(4) * copies;
    let prepared: Vec<Prepared> = (0..launches).map(|i| prepare(i % copies)).collect();
    let ((), duration) = client
        .profile(
            move || {
                for launch in prepared {
                    launch();
                }
            },
            "nvfp4 row sum",
        )
        .expect("profile");
    let ticks = block_on(duration.resolve()).expect("device ticks");
    ticks.duration() / launches as u32
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let client = cubecl::test_device().client();
    let plane = client.properties().hardware.plane_size_max as usize;
    let props = client.properties();
    // The device's read-only stream ceiling, as cubecl's roofline probe measures it.
    let ceiling = memory_bound(
        &client,
        MemoryAccess::Read,
        Work {
            compute_ops: 0,
            bytes: 64 << 20,
        },
        1.0,
    )
    .resource
    .peak_per_s;
    println!(
        "plane {plane}, memory alignment {} B, read ceiling {:.1} GB/s, launch overhead {:?}",
        props.memory.alignment,
        ceiling / 1e9,
        measure_launch_overhead(&client),
    );

    // Qwen3-8B's projections as a decode step reads them, `[out, in]`.
    let shapes = match std::env::var("NVFP4_DEBUG") {
        Ok(_) => vec![("tiny", 16usize, VALUES_PER_STEP * plane * 2)],
        Err(_) => vec![
            ("qkv", 6144, 4096),
            ("o", 4096, 4096),
            ("gate/up", 24576, 4096),
            ("down", 4096, 12288),
        ],
    };
    // Twice the largest last-level cache in this class: a rotation through this many bytes leaves
    // nothing of a copy cached when it comes round again.
    const ROTATION: usize = 512 << 20;

    println!(
        "{:<8} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}   {:>8} {:>8} {:>8} {:>8}",
        "shape",
        "MB",
        "ceiling",
        "plain",
        "loads",
        "raw",
        "reduce",
        "red×s",
        "dot",
        "raw err",
        "red err",
        "red×s er",
        "dot err"
    );
    // `NVFP4_SHAPE` keeps one shape by name: what a compiler dump of one shape's kernels runs.
    let only = std::env::var("NVFP4_SHAPE").ok();
    for (name, rows, k) in shapes {
        if only.as_deref().is_some_and(|only| only != name) {
            continue;
        }
        assert!(rows.is_multiple_of(PLANES_PER_CUBE));
        assert!(k.is_multiple_of(VALUES_PER_STEP * plane));
        let host = Host::new(rows, k);
        let bytes = rows * k / 2 + rows * k / BLOCK;
        let copies = ROTATION.div_ceil(bytes).max(2);
        let data: Vec<Device> = (0..copies)
            .map(|_| host.upload(&client, rows, k, plane))
            .collect();

        let lines = k / VALUES_PER_LINE / plane;
        let rows_per_plane = env_usize("NVFP4_ROWS", 1);
        let planes = env_usize("NVFP4_PLANES", PLANES_PER_CUBE);
        assert!(rows.is_multiple_of(rows_per_plane * planes));
        let cube_count = CubeCount::new_1d((rows / (rows_per_plane * planes)) as u32);
        let cube_dim = CubeDim::new_2d(plane as u32, planes as u32);
        let line_arg = |h: &TensorHandle| h.clone().binding().into_tensor_arg();
        // The raw kernel's cubes: one a group of rows, or, where `NVFP4_CUBES_PER_PROCESSOR` says,
        // that many a processor, walking the groups grid-stride.
        let raw_cubes = match env_usize("NVFP4_CUBES_PER_PROCESSOR", RAW_CUBES_PER_PROCESSOR) {
            0 if env_usize("NVFP4_RAW_CUBES", 0) > 0 => {
                CubeCount::new_1d(env_usize("NVFP4_RAW_CUBES", 0) as u32)
            }
            0 => cube_count.clone(),
            per => {
                let processors = props.hardware.num_streaming_multiprocessors.unwrap_or(32);
                let groups = (rows / (rows_per_plane * planes)) as u32;
                CubeCount::new_1d((per as u32 * processors).min(groups))
            }
        };

        let raw = |d: &Device| -> Prepared {
            let (client, cube_count) = (client.clone(), raw_cubes.clone());
            let args = [&d.words, &d.scales, &d.global, &d.out].map(line_arg);
            Box::new(move || unsafe {
                let [words, scales, global, out] = args;
                raw_row_sum::launch_unchecked(
                    &client,
                    cube_count,
                    cube_dim,
                    4,
                    words,
                    scales,
                    global,
                    out,
                    lines,
                    rows_per_plane,
                )
            })
        };
        // The split-K gemv: `NVFP4_GEMV_LINES` lines a lane, `NVFP4_GEMV_COLUMNS` rows a group,
        // `NVFP4_GEMV_CUBES_PER_PROCESSOR` cubes a processor.
        let gemv_lines = env_usize("NVFP4_GEMV_LINES", 1);
        let gemv_columns = env_usize("NVFP4_GEMV_COLUMNS", 8);
        let gemv_planes = k / (gemv_lines * VALUES_PER_LINE * plane);
        assert_eq!(gemv_planes * gemv_lines * VALUES_PER_LINE * plane, k);
        assert!(rows.is_multiple_of(gemv_columns));
        let gemv_cubes = {
            let processors = props.hardware.num_streaming_multiprocessors.unwrap_or(32);
            let per = env_usize("NVFP4_GEMV_CUBES_PER_PROCESSOR", 4) as u32;
            CubeCount::new_1d((per * processors).min((rows / gemv_columns) as u32))
        };
        let gemv = |d: &Device| -> Prepared {
            let (client, cube_count) = (client.clone(), gemv_cubes.clone());
            let args = [&d.words, &d.scales, &d.global, &d.x16, &d.out].map(line_arg);
            Box::new(move || unsafe {
                let [words, scales, global, x, out] = args;
                raw_gemv_split_k::launch_unchecked(
                    &client,
                    cube_count,
                    CubeDim::new_2d(plane as u32, gemv_planes as u32),
                    words,
                    scales,
                    global,
                    x,
                    out,
                    gemv_lines,
                    gemv_planes,
                    gemv_columns,
                )
            })
        };
        if std::env::var("NVFP4_GEMV_ONLY").is_ok() {
            gemv(&data[0])();
            let gemv_err =
                worst_error_against(&read_f32(&client, &data[0].out), &host.expected_gemv);
            raw(&data[0])();
            let raw_err = worst_error(&read_f32(&client, &data[0].out), &host);
            let mut best = [Duration::MAX; 2];
            for _ in 0..env_usize("NVFP4_ROUNDS", 5) {
                let times = [
                    time(&client, copies, |c| raw(&data[c])),
                    time(&client, copies, |c| gemv(&data[c])),
                ];
                for (best, time) in best.iter_mut().zip(times) {
                    *best = (*best).min(time);
                }
            }
            let gbs = |t: Duration| bytes as f64 / t.as_secs_f64() / 1e9;
            println!(
                "{:<8} row sum {:>6.1} GB/s (err {:.1e})   split-K gemv {:>6.1} GB/s = {:>5.1}% \
                 (err {:.1e}; {} planes, {} lines, {} columns, {:?} cubes)",
                name,
                gbs(best[0]),
                raw_err,
                gbs(best[1]),
                100.0 * best[0].as_secs_f64() / best[1].as_secs_f64(),
                gemv_err,
                gemv_planes,
                gemv_lines,
                gemv_columns,
                gemv_cubes,
            );
            continue;
        }
        let loads = |d: &Device| -> Prepared {
            let (client, cube_count) = (client.clone(), cube_count.clone());
            let args = [&d.words, &d.scales, &d.scratch].map(line_arg);
            Box::new(move || unsafe {
                let [words, scales, out] = args;
                raw_loads_only::launch_unchecked(
                    &client,
                    cube_count,
                    cube_dim,
                    4,
                    words,
                    scales,
                    out,
                    lines,
                    rows_per_plane,
                )
            })
        };
        let plain = |d: &Device| -> Prepared {
            let client = client.clone();
            let units = plain_units(&client);
            let args = [&d.words, &d.scales, &d.plain_out].map(line_arg);
            Box::new(move || unsafe {
                // `plain_out` holds `units` lines: one a unit.
                let [words, scales, out] = args;
                plain_read::launch_unchecked(
                    &client,
                    CubeCount::new_1d((units / 256) as u32),
                    CubeDim::new_1d(256),
                    4,
                    words,
                    scales,
                    out,
                )
            })
        };
        let dsl = |d: &Device| -> Prepared {
            let client = client.clone();
            let (launcher, args) = dsl_launch(&client, d, rows, k, Spelling::Reduce);
            let [w, out]: [TileTensorLaunch; 2] = args.try_into().ok().expect("two operands");
            Box::new(move || unsafe {
                dsl_row_sum::launch_unchecked(
                    &client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    w,
                    out,
                    launcher.partitioning_arg(),
                )
            })
        };
        let dot = |d: &Device| -> Prepared {
            let client = client.clone();
            let (launcher, args) = dsl_launch(&client, d, rows, k, Spelling::Contract);
            let [x, w, out]: [TileTensorLaunch; 3] = args.try_into().ok().expect("three operands");
            Box::new(move || unsafe {
                dsl_row_dot::launch_unchecked(
                    &client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    x,
                    w,
                    out,
                    launcher.partitioning_arg(),
                )
            })
        };

        let product = |d: &Device| -> Prepared {
            let client = client.clone();
            let (launcher, args) = dsl_launch(&client, d, rows, k, Spelling::ReduceProduct);
            let [w, blocks, global, out]: [TileTensorLaunch; 4] =
                args.try_into().ok().expect("four operands");
            Box::new(move || unsafe {
                dsl_row_sum_product::launch_unchecked(
                    &client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    w,
                    blocks,
                    global,
                    out,
                    launcher.partitioning_arg(),
                )
            })
        };

        product(&data[0])();
        let product_err = worst_error(&read_f32(&client, &data[0].out), &host);
        raw(&data[0])();
        let raw_err = worst_error(&read_f32(&client, &data[0].out), &host);
        dsl(&data[0])();
        let dsl_err = worst_error(&read_f32(&client, &data[0].out), &host);
        if std::env::var("NVFP4_DEBUG").is_ok() {
            let got = read_f32(&client, &data[0].out);
            let bad: Vec<(usize, f32, f64)> = got
                .iter()
                .zip(&host.expected)
                .enumerate()
                .filter(|(_, (g, w))| (**g as f64 - **w).abs() > 1e-3 * w.abs().max(1.0))
                .map(|(i, (g, w))| (i, *g, *w))
                .collect();
            let first = read_f32(&client, &data[0].out);
            println!(
                "reduce err now {:e} (was {:e})",
                worst_error(&first, &host),
                dsl_err
            );
            println!(
                "reduce: {} bad rows of {}: {:?}",
                bad.len(),
                got.len(),
                &bad[..bad.len().min(6)]
            );
            for row in 0..4 {
                let codes = |b: usize| -> f64 { host.block_codes(row, b, k) };
                let blocks = k / BLOCK;
                let unscaled: f64 = (0..blocks).map(codes).sum();
                println!(
                    "row {row}: dsl {} want {} unscaled codes {} first-half {}",
                    got[row],
                    host.expected[row],
                    unscaled,
                    host.partial(row, k, blocks / 2),
                );
                let turn =
                    |t: usize| -> f64 { (t * blocks / 2..(t + 1) * blocks / 2).map(codes).sum() };
                let lane_first: f64 = (0..blocks).filter(|b| (b / 4) % 2 == 0).map(codes).sum();
                let first_line: f64 = (0..blocks).filter(|b| b % 4 < 2).map(codes).sum();
                let first_block: f64 = (0..blocks).filter(|b| b % 4 == 0).map(codes).sum();
                println!(
                    "   turn0 {} turn1 {} even-lanes {} first-line-of-step {} first-block-of-step {}",
                    turn(0),
                    turn(1),
                    lane_first,
                    first_line,
                    first_block
                );
            }
        }

        dot(&data[0])();
        let dot_err = worst_error(&read_f32(&client, &data[0].out), &host);
        if std::env::var("NVFP4_DEBUG").is_ok() {
            let got = read_f32(&client, &data[0].out);
            let blocks = k / BLOCK;
            for row in 0..4 {
                let scaled = |range: std::ops::Range<usize>| host.scaled_blocks(row, k, range);
                println!(
                    "dot row {row}: got {} want {} turn0 {} turn1 {} unscaled-global {}",
                    got[row],
                    host.expected[row],
                    scaled(0..blocks / 2),
                    scaled(blocks / 2..blocks),
                    host.expected[row] / host.global as f64,
                );
            }
        }
        let gbs = |t: Duration| bytes as f64 / t.as_secs_f64() / 1e9;
        // Rounds in turn, the best of each kernel kept: on an APU whatever else runs shares the
        // bus, so a kernel's best round is the one least disturbed.
        let raw_dot = |d: &Device| -> Prepared {
            let client = client.clone();
            let args = [&d.words, &d.scales, &d.global, &d.ones, &d.out].map(line_arg);
            Box::new(move || unsafe {
                let [words, scales, global, x, out] = args;
                raw_row_dot::launch_unchecked(
                    &client,
                    CubeCount::new_1d((rows / PLANES_PER_CUBE) as u32),
                    CubeDim::new_2d(plane as u32, PLANES_PER_CUBE as u32),
                    4,
                    words,
                    scales,
                    global,
                    x,
                    out,
                    lines,
                )
            })
        };
        raw_dot(&data[0])();
        let raw_dot_err = worst_error(&read_f32(&client, &data[0].out), &host);
        let raw_dot_blocked = |d: &Device| -> Prepared {
            let client = client.clone();
            let args = [&d.words, &d.scales, &d.global, &d.ones, &d.out].map(line_arg);
            let steps = k / VALUES_PER_STEP / plane;
            Box::new(move || unsafe {
                let [words, scales, global, x, out] = args;
                raw_row_dot_blocked::launch_unchecked(
                    &client,
                    CubeCount::new_1d((rows / PLANES_PER_CUBE) as u32),
                    CubeDim::new_2d(plane as u32, PLANES_PER_CUBE as u32),
                    words,
                    scales,
                    global,
                    x,
                    out,
                    steps,
                )
            })
        };
        raw_dot_blocked(&data[0])();
        let raw_dot_blocked_err = worst_error(&read_f32(&client, &data[0].out), &host);

        let mut best = [Duration::MAX; 8];
        for _ in 0..env_usize("NVFP4_ROUNDS", 5) {
            let times = [
                time(&client, copies, |c| plain(&data[c])),
                time(&client, copies, |c| loads(&data[c])),
                time(&client, copies, |c| raw(&data[c])),
                time(&client, copies, |c| dsl(&data[c])),
                time(&client, copies, |c| product(&data[c])),
                time(&client, copies, |c| dot(&data[c])),
                time(&client, copies, |c| raw_dot(&data[c])),
                time(&client, copies, |c| raw_dot_blocked(&data[c])),
            ];
            for (best, time) in best.iter_mut().zip(times) {
                *best = (*best).min(time);
            }
        }
        let [
            t_plain,
            t_loads,
            t_raw,
            t_dsl,
            t_product,
            t_dot,
            t_raw_dot,
            t_raw_dot_blocked,
        ] = best;
        // The read ceiling at this shape's own footprint.
        let at_size = memory_bound(
            &client,
            MemoryAccess::Read,
            Work {
                compute_ops: 0,
                bytes,
            },
            1.0,
        )
        .resource
        .peak_per_s
            / 1e9;
        println!(
            "{:<8} {:>6.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1}   \
             {:>8.1e} {:>8.1e} {:>8.1e} {:>8.1e}",
            name,
            bytes as f64 / 1e6,
            at_size,
            gbs(t_plain),
            gbs(t_loads),
            gbs(t_raw),
            gbs(t_dsl),
            gbs(t_product),
            gbs(t_dot),
            raw_err,
            dsl_err,
            product_err,
            dot_err,
        );
        println!(
            "{:<8} raw one-row product against x, one plane a row: {:.1} GB/s, err {:.1e}",
            "",
            gbs(t_raw_dot),
            raw_dot_err
        );
        println!(
            "{:<8} the same, the DSL's access pattern (64 values a unit, 8-byte loads): {:.1} GB/s, \
             err {:.1e}",
            "",
            gbs(t_raw_dot_blocked),
            raw_dot_blocked_err
        );
    }
}
