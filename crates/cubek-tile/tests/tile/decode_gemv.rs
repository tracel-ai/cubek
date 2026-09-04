//! The decode gemv's serving geometry, in the split-axis spelling.
//!
//! `y = W · x` with `W` the weight's physical `[d_out, d_in]` buffer as the *lhs*, the
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
    bytes::Bytes, prelude::*, quant::scheme::QuantValue, std::tensor::TensorHandle, zspace::shape,
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

/// Every scale level windowed to `region`.
#[cube]
fn at_all<S: Numeric>(scales: &Sequence<Tile<S>>, region: &Region) -> Sequence<Tile<S>> {
    let mut at = Sequence::new();
    #[unroll]
    for k in 0..scales.len() {
        at.push(scales.index(k).at(region));
    }
    at
}

/// The partials fold through the sink's element between `K` steps. Three levels: this cube's
/// strip of rows, this plane's group of rows, this lane's rows against its share of the
/// contraction, the leaf running under a block of `budget` scalars.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn decode_gemv<E: Numeric, S: Numeric, VX: Size, VO: Size>(
    w: &TileArg<'_, u32, Const<1>>,
    x: &TileArg<'_, E, VX>,
    scale: &TileArg<'_, S, Const<1>>,
    out: &TileArg<'_, E, VO>,
    #[comptime] space: Space,
    #[comptime] cube: Level,
    #[comptime] plane: Level,
    #[comptime] lane: Level,
    #[comptime] budget: usize,
    #[define(E, S)] _dtypes: [ElemType; 2],
) {
    let w = w.tile_packed::<E>(comptime!(space.clone()));
    let x = x.tile(comptime!(space.clone()));
    let mut scales = Sequence::new();
    scales.push(scale.tile(comptime!(space.clone())));
    let out = out.tile(comptime!(space.clone()));
    // This instance's windows of `out`, each initialized once: the level projected
    // onto `out`'s own axes walks nothing it does not span.
    for region in out.runtime_space().level(comptime!(cube.clone())) {
        let mut out_w = out.at(&region);
        out_w.zero();
    }
    for region in out.op_space(&w, &x).level(comptime!(cube.clone())) {
        let out_cube = out.at(&region);
        let w_cube = w.at(&region);
        let x_cube = x.at(&region);
        let scales_cube = at_all(&scales, &region);
        for region in out_cube
            .op_space(&w_cube, &x_cube)
            .level(comptime!(plane.clone()))
        {
            let out_plane = out_cube.at(&region);
            let w_plane = w_cube.at(&region);
            let x_plane = x_cube.at(&region);
            let scales_plane = at_all(&scales_cube, &region);
            for region in out_plane
                .op_space(&w_plane, &x_plane)
                .level(comptime!(lane.clone()))
            {
                let mut out_lane = out_plane.at(&region);
                out_lane.mma_scaled_with(
                    &w_plane.at(&region),
                    &x_plane.at(&region),
                    &at_all(&scales_plane, &region),
                    comptime!(RegisterBlock::new(budget)),
                    Semiring::SUM_PROD,
                );
            }
        }
    }
}

/// The same, with the partials living in registers for the whole `K` walk: the kernel opens a
/// register block above the walk and drains it once.
///
/// The activation is read *scalar* here, and has to be: a promoted block lines its cells along
/// the accumulator, so a rhs lined along the contraction folds a whole step into one cell and
/// `RegisterData::mma_scaled` refuses it. Which is why the shipping shape below keeps its
/// accumulator in memory: the two are a trade, not a ladder.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn decode_gemv_promoted<E: Numeric, S: Numeric, VX: Size, VO: Size>(
    w: &TileArg<'_, u32, Const<1>>,
    x: &TileArg<'_, E, VX>,
    scale: &TileArg<'_, S, Const<1>>,
    out: &TileArg<'_, E, VO>,
    #[comptime] space: Space,
    #[comptime] cube: Level,
    #[comptime] plane: Level,
    #[comptime] lane: Level,
    #[comptime] budget: usize,
    #[define(E, S)] _dtypes: [ElemType; 2],
) {
    let w = w.tile_packed::<E>(comptime!(space.clone()));
    let x = x.tile(comptime!(space.clone()));
    let mut scales = Sequence::new();
    scales.push(scale.tile(comptime!(space.clone())));
    let mut out = out.tile(comptime!(space.clone()));
    let mut acc = out.block_accumulator::<E, E>(
        &w,
        comptime!(Fragments::new(
            &out.space,
            &w.space,
            &[cube.clone(), plane.clone(), lane.clone()]
        )),
        comptime!(RegisterBlock::new(budget)),
        Monoid::Sum,
    );
    acc.zero();
    for region in acc.op_space(&w, &x).level(comptime!(cube.clone())) {
        let acc_cube = acc.at(&region);
        let w_cube = w.at(&region);
        let x_cube = x.at(&region);
        let scales_cube = at_all(&scales, &region);
        for region in acc_cube
            .op_space(&w_cube, &x_cube)
            .level(comptime!(plane.clone()))
        {
            let acc_plane = acc_cube.at(&region);
            let w_plane = w_cube.at(&region);
            let x_plane = x_cube.at(&region);
            let scales_plane = at_all(&scales_cube, &region);
            for region in acc_plane
                .op_space(&w_plane, &x_plane)
                .level(comptime!(lane.clone()))
            {
                let mut acc_lane = acc_plane.at(&region);
                acc_lane.mma_scaled(
                    &w_plane.at(&region),
                    &x_plane.at(&region),
                    &at_all(&scales_plane, &region),
                    Semiring::SUM_PROD,
                );
            }
        }
    }
    acc.drain_cast_into(&mut out);
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

    let client = cubecl::test_device().client();
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
    let num_planes = 2;
    let rows_per_cube = num_planes * rows_per_plane;
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
    let nest = Nest::new(
        Space::new(&[(M, d_out), (N, n), (KB, blocks), (KI, block)]),
        vec![],
    )
    // A strip of output rows per cube, walking all of `K`.
    .level(|l| {
        l.distribute(cubes(CubeAxis::X), &[(M, rows_per_cube)])
            .walk(&[(N, n), (KB, blocks), (KI, block)]);
    })
    // One plane per group of rows.
    .level(|l| {
        l.distribute(planes(), &[(M, rows_per_plane)])
            .walk(&[(N, n), (KB, blocks), (KI, block)]);
    })
    // The fold: `rows_per_lane` rows per lane group, the group's lanes interleaving one
    // stored word each along `KI`, so a step reads one contiguous span of the block.
    .level(|l| {
        l.distribute(lanes(groups), &[(M, rows_per_lane)])
            .distribute(lanes(group_lanes).interleaved(), &[(KI, factor)])
            .walk(&[(N, n), (KB, 1)]);
    });
    // The leaf's budget: one scalar per row a lane owns, per value of the word it takes a step.
    let budget = rows_per_lane * factor;

    let w_tensor = TensorHandle::new_contiguous(
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

    let launcher = Launcher::new(&client, &nest, KernelForm::Static);
    let w_op = launcher
        .arg(w_tensor.binding())
        .gathered(Projection::new(
            &[M, KB, KI],
            &[
                PhysicalAxisMap::of(M),
                PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
            ],
        ))
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
        .vectorize(if promoted { 1 } else { factor })
        .build();
    // One scale per `(row, block of K)`: `KI` is carried and addressed by nothing.
    let s_op = launcher.arg(s_tensor.binding()).subspace(&[M, KB]).build();
    let out_op = launcher
        .arg(out.clone().binding())
        .subspace(&[M, N])
        .build();

    let (count, dim) = (launcher.cube_count(), launcher.cube_dim());
    if promoted {
        decode_gemv_promoted::launch(
            &client,
            count,
            dim,
            x_op.vector_size,
            out_op.vector_size,
            w_op.arg(),
            x_op.arg(),
            s_op.arg(),
            out_op.arg(),
            launcher.space().clone(),
            launcher.concrete().at(0),
            launcher.concrete().at(1),
            launcher.concrete().at(2),
            budget,
            [dtype, dtype],
        );
    } else {
        decode_gemv::launch(
            &client,
            count,
            dim,
            x_op.vector_size,
            out_op.vector_size,
            w_op.arg(),
            x_op.arg(),
            s_op.arg(),
            out_op.arg(),
            launcher.space().clone(),
            launcher.concrete().at(0),
            launcher.concrete().at(1),
            launcher.concrete().at(2),
            budget,
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
