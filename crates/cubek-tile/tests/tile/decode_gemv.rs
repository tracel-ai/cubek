//! The decode gemv's serving geometry, in the split-axis spelling.
//!
//! `y = W · x` with `W` the weight's physical `[d_out, d_in]` buffer as the *lhs* — the
//! orientation a decode step streams, where the contraction runs along the buffer's contiguous
//! direction. `K` is spelled `(KB, KI)`, so one scale per block is the operand omitting `KI`.
//!
//! Two shapes, and the engine picks which by what it will accept. Where the accumulator sits in
//! memory the activation is read one `K`-contiguous line a step, which is what lets a step fold
//! whole lines; where it is promoted to registers the block lines its cells along the
//! accumulator instead, so the activation has to be read cell by cell. Both compute the same
//! answer and neither is a ladder rung of the other. [`cubek-matmul`'s QuantGemv routine] is
//! built on the first.

use cubecl::{
    Runtime, TestRuntime, bytes::Bytes, prelude::*, quant::scheme::QuantValue,
    std::tensor::TensorHandle, zspace::shape,
};
use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult};
use cubek_tile::*;

/// The weight's output dimension: rows of the lhs.
const M: Axis = Axis(0);
/// Activation rows projected in one launch (1 at decode).
const N: Axis = Axis(1);
/// The contraction as the two axes a scale block makes of it.
const KB: Axis = Axis(2);
const KI: Axis = Axis(3);

/// `edge`-wide tiles dealt across `lanes` lanes of the plane.
fn unit(edge: usize, spread: Spread, lanes: usize) -> Cut {
    Cut::new(
        edge,
        Distribution::Spatial {
            scope: ComputeScope::Unit,
            spread,
            coverage: Coverage::Instances(lanes),
        },
    )
}

/// The partials fold through the sink's element between `K` steps.
#[cube(launch)]
fn decode_gemv<E: Numeric, S: Numeric, VX: Size, VO: Size>(
    w: &TileArg<'_, u32, Const<1>>,
    x: &TileArg<'_, E, VX>,
    scale: &TileArg<'_, S, Const<1>>,
    out: &TileArg<'_, E, VO>,
    #[comptime] space: Space,
    #[define(E, S)] _dtypes: [ElemType; 2],
) {
    let w = w.tile_packed::<E>(comptime!(space.clone()));
    let x = x.tile(comptime!(space.clone()));
    let mut scales = Sequence::new();
    scales.push(scale.tile(comptime!(space.clone())));
    let mut out = out.tile(space);
    out.mm_scaled(&w, &x, &scales, Semiring::SUM_PROD);
}

/// The same, with the partials living in registers for the whole `K` walk: `out` states
/// `Register` at the outermost level, so the accumulator opens above the walk and drains once.
///
/// The activation is read *scalar* here, and has to be: a promoted block lines its cells along
/// the accumulator, so a rhs lined along the contraction folds a whole step into one cell and
/// `RegisterData::mma_scaled` refuses it. Which is why the shipping shape below keeps its
/// accumulator in memory — the two are a trade, not a ladder.
#[cube(launch)]
fn decode_gemv_promoted<E: Numeric, S: Numeric, VX: Size, VO: Size>(
    w: &TileArg<'_, u32, Const<1>>,
    x: &TileArg<'_, E, VX>,
    scale: &TileArg<'_, S, Const<1>>,
    out: &TileArg<'_, E, VO>,
    #[comptime] space: Space,
    #[define(E, S)] _dtypes: [ElemType; 2],
) {
    let w = w.tile_packed::<E>(comptime!(space.clone()));
    let x = x.tile(comptime!(space.clone()));
    let mut scales = Sequence::new();
    scales.push(scale.tile(comptime!(space.clone())));
    let out = out.tile(space);
    let mut acc = out.accumulate::<E, _>(&w, Monoid::Sum);
    acc.mm_scaled(&w, &x, &scales, Semiring::SUM_PROD);
}

#[test]
fn the_serving_geometry_computes_the_decode_gemv() {
    serving_geometry(false);
}

#[test]
fn a_promoted_accumulator_spans_the_whole_decode_walk() {
    serving_geometry(true);
}

fn serving_geometry(promoted: bool) {
    let field = QuantValue::Q8S;
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
    let plane = client.properties().hardware.plane_size_max as usize;

    // The block, and the geometry cut from it: a lane takes one stored word per step, so a
    // group of `block / factor` lanes covers one block of `K`, and the rest of the plane
    // carries rows.
    let (block, blocks) = (32, 4);
    let (d_in, n) = (block * blocks, 1);
    let group_lanes = block / factor;
    if !plane.is_multiple_of(group_lanes) {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "a {plane}-lane plane does not split into {group_lanes}-lane groups"
        )))
        .enforce();
        return;
    }
    let groups = plane / group_lanes;
    let rows_per_lane = 2;
    let rows_per_plane = groups * rows_per_lane;
    let planes = 2;
    let rows_per_cube = planes * rows_per_plane;
    let d_out = rows_per_cube * 2;

    let span = 1i32 << bits;
    let w: Vec<i32> = (0..d_out * d_in)
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
    let x: Vec<f32> = (0..d_in).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..d_out * blocks)
        .map(|i| (i % 9) as f32 / 4.0 + 0.25)
        .collect();

    // The activation is read one `K`-contiguous line a step where the accumulator sits in
    // memory, and cell by cell where it is promoted (see the kernel above).
    let dtype = f32::elem_type_native();
    let x_axes: &[Axis] = if promoted { &[KB, KI, N] } else { &[N, KB, KI] };
    let mut ops = (
        Operand::new(&[M, KB, KI], dtype),
        Operand::new(x_axes, dtype),
        Operand::new(&[M, KB], dtype),
        Operand::new(&[M, N], dtype),
    );
    let space = Tiling::over(&mut ops, &[(M, d_out), (N, n), (KB, blocks), (KI, block)])
        // A strip of output rows per cube, walking all of `K`.
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, o| {
            l.axis(M, Cut::cube(CubeAxis::X, rows_per_cube))
                .axis(N, Cut::sequential(n))
                .axis(KB, Cut::sequential(blocks))
                .axis(KI, Cut::sequential(block));
            if promoted {
                o.3.stage(Residence::Register);
            }
        })
        // One plane per group of rows.
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::plane(rows_per_plane))
                .axis(N, Cut::sequential(n))
                .axis(KB, Cut::sequential(blocks))
                .axis(KI, Cut::sequential(block));
        })
        // The fold: `rows_per_lane` rows per lane group, the group's lanes interleaving one
        // stored word each along `KI`, so a step reads one contiguous span of the block.
        .instruction(Instruction::registers(rows_per_lane * factor), |l, _| {
            l.axis(M, unit(rows_per_lane, Spread::Contiguous, groups))
                .axis(N, Cut::sequential(n))
                .axis(KB, Cut::sequential(1))
                .axis(KI, unit(factor, Spread::Interleaved, group_lanes));
        })
        .build();

    let w_tensor = TensorHandle::<TestRuntime>::new_contiguous(
        vec![d_out, d_in],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let x_shape = if promoted {
        shape![d_in, n]
    } else {
        shape![n, d_in]
    };
    let (x_tensor, _) = TestInput::builder(client.clone(), x_shape)
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![d_out, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![d_out, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = space.clone().launcher_over(&client, &[]);
    let w_op = launcher
        .arg(w_tensor.binding())
        .gathered(Projection::new(
            &[M, KB, KI],
            &[
                PhysicalAxisMap::of(M),
                PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
            ],
        ))
        .operand(&ops.0)
        .packed(field)
        .vectorize(factor)
        .build();
    let x_projection = if promoted {
        Projection::new(
            &[KB, KI, N],
            &[
                PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                PhysicalAxisMap::of(N),
            ],
        )
    } else {
        Projection::new(
            &[N, KB, KI],
            &[
                PhysicalAxisMap::of(N),
                PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
            ],
        )
    };
    let x_op = launcher
        .arg(x_tensor.binding())
        .gathered(x_projection)
        .operand(&ops.1)
        .vectorize(if promoted { 1 } else { factor })
        .build();
    // One scale per `(row, block of K)`: `KI` is carried and addressed by nothing.
    let s_op = launcher.bind(&ops.2, s_tensor.binding()).build();
    let out_op = launcher.bind(&ops.3, out.clone().binding()).build();

    let (count, dim) = (launcher.cube_count(), launcher.cube_dim());
    let kernel_space = launcher.space().clone();
    if promoted {
        decode_gemv_promoted::launch::<TestRuntime>(
            &client,
            count,
            dim,
            x_op.vector_size,
            out_op.vector_size,
            w_op.arg(),
            x_op.arg(),
            s_op.arg(),
            out_op.arg(),
            kernel_space,
            [dtype, dtype],
        );
    } else {
        decode_gemv::launch::<TestRuntime>(
            &client,
            count,
            dim,
            x_op.vector_size,
            out_op.vector_size,
            w_op.arg(),
            x_op.arg(),
            s_op.arg(),
            out_op.arg(),
            kernel_space,
            [dtype, dtype],
        );
    }

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    for m in 0..d_out {
        let want: f32 = (0..d_in)
            .map(|k| w[m * d_in + k] as f32 * s[m * blocks + k / block] * x[k])
            .sum();
        let have = got.get_f32(&[m, 0]);
        assert!(
            (have - want).abs() < 1e-2,
            "at row {m}: got {have}, want {want}"
        );
    }
}
