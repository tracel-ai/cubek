//! The attention fold end-to-end on tiles: the score and mix leaves
//! ([`ops::attention`]) interleaved with the softmax step, the walk owned by
//! the kernel: the miniature of the routine a client (metabolic) launches.
//! GQA rides the axes: `q` stacks its group over the same query block
//! (group-major rows), `k`/`v` simply omit the group axis, and the probe's
//! `q_rows` maps rows back to query positions for the causal predicate.

use cubecl::{Runtime, TestRuntime, client::ComputeClient, prelude::*, zspace::Shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::{
    Axis, Buffering, Cut, MaskProbe, MemData, RowState, Space, StagePlan, StreamFold, TileArg,
    TileArgLaunch, TileSpec, Tiling, Walk, WalkOrder,
};

const G: Axis = Axis(0); // GQA group member
const QP: Axis = Axis(1); // query position
const S: Axis = Axis(2); // key/value position (the reduced axis)
const D: Axis = Axis(3); // head dim (contracted by the score matmul)
const V: Axis = Axis(4); // value dim
// Local labels for the kernel-allocated smem tiles.
const R: Axis = Axis(5); // score rows = G × QP, group-major
const C: Axis = Axis(6); // score cols = one S block

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn attention_fold_kernel<W: Size>(
    q: &TileArg<'_, f32, W>,           // {G, QP, D}
    k: &TileArg<'_, f32, W>,           // {S, D}: omits the group axis
    v: &TileArg<'_, f32, W>,           // {S, V}
    mask: &TileArg<'_, u32, Const<1>>, // 1-cell dummy (materialized = false)
    out: &mut Tensor<f32>,             // [G·QP·V] flat
    scale: f32,
    bound: u32,
    #[comptime] space: Space,
    #[comptime] units: usize,
    #[comptime] causal: bool,
    #[comptime] block: usize,
    #[comptime] row_chunk: usize,
) {
    let q = q.tile(comptime!(space.clone()));
    let k = k.tile(comptime!(space.clone()));
    let v = v.tile(comptime!(space.clone()));
    let mask_tile = mask.tile(space);

    let rows = comptime!(q.space.extent(G) * q.space.extent(QP));
    let q_rows = comptime!(q.space.extent(QP));
    let val_dim = comptime!(v.space.extent(V));

    // The stage: q resident in smem for the whole walk (read cols-fold by the
    // score leaf), score/p/factors/acc the fold's working set.
    let mut q_s = MemData::<f32>::smem(
        comptime!(q.space.clone()),
        q.vector_size(),
        comptime!(StagePlan::in_place()),
    );
    q_s.copy_from(&q);
    let score_space = comptime!(Space::new(&[(R, rows), (C, block)]));
    let mut score = MemData::<f32>::smem(
        score_space.clone(),
        1usize,
        comptime!(StagePlan::in_place()),
    );
    let mut p = MemData::<f32>::smem(score_space, 1usize, comptime!(StagePlan::in_place()));
    let row_space = comptime!(Space::new(&[(R, rows)]));
    let mut factors =
        MemData::<f32>::smem(row_space.clone(), 1usize, comptime!(StagePlan::in_place()));
    let acc_space = comptime!(Space::new(&[(R, rows), (V, val_dim)]));
    let mut acc = MemData::<f32>::smem(acc_space, 1usize, comptime!(StagePlan::in_place()));
    acc.zero();
    let mut state = RowState::<f32>::new(row_space, units);
    let rpu = comptime!(state.rows_per_unit);
    let bound_s = bound as usize;
    sync_cube();

    // The fold: one S block per region.
    for region in Walk::over(k.runtime_space()) {
        let kb = k.at(&region);
        let vb = v.at(&region);
        let s0 = region.coord(S) * block;

        // Clip the ragged tail: no reads past the attended prefix.
        let cols_bound = max(bound_s, s0) - s0;
        score.score_columns(&q_s, &kb, cols_bound, row_chunk);
        sync_cube();

        let probe = MaskProbe {
            origin_q: 0,
            origin_s: s0,
            bound_q: q_rows.runtime(),
            bound_s,
            q_rows,
            causal,
            materialized: false,
        };
        let corr = score.softmax::<f32>(&mut p, &mut state, &probe, &mask_tile, scale);
        factors.store_rows(&corr, rpu);
        sync_cube();

        // The mix folds the rescale in; stale cache beyond the attended
        // prefix must not ride a zero probability into the accumulator.
        acc.mix_columns(&p, &vb, &factors, cols_bound, row_chunk);
        sync_cube();
    }

    // Epilogue: owners publish 1/l, the cube normalizes and drains.
    let mut recip = Array::<f32>::new(rpu);
    for ri in 0..rpu {
        recip[ri] = state.recip_l(ri);
    }
    factors.store_rows(&recip, rpu);
    sync_cube();
    acc.scale_rows(&factors);
    sync_cube();

    let size!(W1) = 1usize;
    let acc_flat = acc.flat::<W1>();
    let total = comptime!(rows * val_dim);
    let workers = CUBE_DIM as usize;
    let mut i = UNIT_POS as usize;
    while i < total {
        out[i] = acc_flat.read(i).extract(0usize);
        i += workers;
    }
}

/// Launch the fold at a GQA geometry and check against direct host math.
fn run(
    (units, g, qp, s_total, block, d, val_dim): (usize, usize, usize, usize, usize, usize, usize),
    bound_s: usize,
    causal: bool,
    vec: usize,
) {
    let client: ComputeClient<TestRuntime> = <TestRuntime as Runtime>::client(&Default::default());
    let units = units.min(client.properties().hardware.max_units_per_cube as usize);
    let rows = g * qp;
    let scale = 1. / (d as f32).sqrt();
    // The register-blocking knob a routine would size from the register budget.
    let row_chunk = 4;

    let f32_ty = f32::elem_type_native();
    let u32_ty = u32::elem_type_native();
    let wobble =
        |i: usize, salt: usize| ((i * 2654435761 + salt * 40503) % 2048) as f32 / 512. - 2.;

    let (q_handle, q_data) = TestInput::builder(client.clone(), Shape::new([g, qp, d]))
        .dtype(f32_ty)
        .custom((0..g * qp * d).map(|i| wobble(i, 1)).collect())
        .generate_with_f32_host_data();
    let (k_handle, k_data) = TestInput::builder(client.clone(), Shape::new([s_total, d]))
        .dtype(f32_ty)
        .custom((0..s_total * d).map(|i| wobble(i, 2)).collect())
        .generate_with_f32_host_data();
    let (v_handle, v_data) = TestInput::builder(client.clone(), Shape::new([s_total, val_dim]))
        .dtype(f32_ty)
        .custom((0..s_total * val_dim).map(|i| wobble(i, 3) / 2.).collect())
        .generate_with_f32_host_data();
    let mask_handle = TestInput::builder(client.clone(), Shape::new([1]))
        .dtype(u32_ty)
        .zeros()
        .generate_without_host_data();
    let out_handle = TestInput::builder(client.clone(), Shape::new([rows, val_dim]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    // The one attention space: every operand projects its axes out of it. The
    // walk cuts S into blocks; every other axis rides whole.
    let space = Tiling::new()
        .extents(&[
            (G, g),
            (QP, qp),
            (S, s_total),
            (D, d),
            (V, val_dim),
            (R, 1),
            (C, 1),
        ])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(G, Cut::sequential(g))
                .axis(QP, Cut::sequential(qp))
                .axis(S, Cut::sequential(block))
                .axis(D, Cut::sequential(d))
                .axis(V, Cut::sequential(val_dim))
                .axis(R, Cut::sequential(1))
                .axis(C, Cut::sequential(1))
        })
        .build();

    attention_fold_kernel::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_2d(units as u32, 1),
        vec,
        TileArgLaunch::new(
            q_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[G, QP, D]),
        ),
        TileArgLaunch::new(
            k_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[S, D]),
        ),
        TileArgLaunch::new(
            v_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[S, V]),
        ),
        TileArgLaunch::new(
            mask_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[R, C]),
        ),
        out_handle.clone().binding().into_tensor_arg(),
        scale,
        bound_s as u32,
        space,
        units,
        causal,
        block,
        row_chunk,
    );

    let out = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);

    for gi in 0..g {
        for qi in 0..qp {
            let r = gi * qp + qi;
            let mut scores = Vec::new();
            for j in 0..s_total {
                let masked = j >= bound_s || (causal && j > qi);
                if !masked {
                    let dot: f32 = (0..d)
                        .map(|p| q_data.get_f32(&[gi, qi, p]) * k_data.get_f32(&[j, p]))
                        .sum();
                    scores.push((dot * scale, j));
                }
            }
            if scores.is_empty() {
                for vi in 0..val_dim {
                    assert_eq!(
                        out.get_f32(&[r, vi]),
                        0.,
                        "fully-masked row {r}: out must be exactly 0"
                    );
                }
                continue;
            }
            let m = scores.iter().fold(f32::NEG_INFINITY, |m, (s, _)| m.max(*s));
            let l: f32 = scores.iter().map(|(s, _)| (s - m).exp()).sum();
            for vi in 0..val_dim {
                let expected: f32 = scores
                    .iter()
                    .map(|(s, j)| (s - m).exp() / l * v_data.get_f32(&[*j, vi]))
                    .sum();
                let got = out.get_f32(&[r, vi]);
                assert!(
                    (got - expected).abs() <= 1e-4 * expected.abs().max(1.),
                    "row ({gi},{qi}) v {vi}: out {got} vs direct {expected}"
                );
            }
        }
    }
}

/// The decode shape: one query per group member, no causal, ragged prefix.
#[test]
fn fold_decode_gqa() {
    run((32, 4, 1, 64, 16, 16, 16), 50, false, 2);
}

/// Prefill with GQA and causal: the probe's `q_rows` row→query mapping.
#[test]
fn fold_prefill_gqa_causal() {
    run((8, 2, 8, 32, 8, 8, 8), 29, true, 2);
}

/// Scalar reads (vector width 1), block not dividing the prefix.
#[test]
fn fold_scalar_odd_bound() {
    run((16, 4, 4, 24, 8, 8, 4), 13, true, 1);
}

/// The split fold: teams on the cube's y dim each fold a disjoint slice of
/// the S walk with their own running state into their own window of
/// split-wide smem tiles, then the states merge cross-team
/// ([`merge_splits`](cubek_tile::Tile)) and the drain folds the split
/// weights and the normalizer in. `splits == 1` degenerates to the plain
/// fold: one code path for both.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn attention_fold_split_kernel<W: Size>(
    q: &TileArg<'_, f32, W>,           // {G, QP, D}
    k: &TileArg<'_, f32, W>,           // {S, D}: omits the group axis
    v: &TileArg<'_, f32, W>,           // {S, V}
    mask: &TileArg<'_, u32, Const<1>>, // 1-cell dummy (materialized = false)
    out: &mut Tensor<f32>,             // [G·QP·V] flat
    scale: f32,
    bound: u32,
    #[comptime] space: Space,
    #[comptime] team: usize,
    #[comptime] splits: usize,
    #[comptime] causal: bool,
    #[comptime] block: usize,
    #[comptime] row_chunk: usize,
) {
    let q = q.tile(comptime!(space.clone()));
    let k = k.tile(comptime!(space.clone()));
    let v = v.tile(comptime!(space.clone()));
    let mask_tile = mask.tile(space);

    let rows = comptime!(q.space.extent(G) * q.space.extent(QP));
    let q_rows = comptime!(q.space.extent(QP));
    let val_dim = comptime!(v.space.extent(V));

    let mut q_s = MemData::<f32>::smem(
        comptime!(q.space.clone()),
        q.vector_size(),
        comptime!(StagePlan::in_place()),
    );
    q_s.copy_from(&q);

    // Split-wide working set: a leading `splits` slice on every tile, one
    // window per team.
    let split_rows = comptime!(splits * rows);
    let score_space = comptime!(
        Tiling::new()
            .extents(&[(R, split_rows), (C, block)])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                l.axis(R, Cut::sequential(rows))
                    .axis(C, Cut::sequential(block))
            })
            .build()
    );
    let row_space = comptime!(
        Tiling::new()
            .extents(&[(R, split_rows)])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                l.axis(R, Cut::sequential(rows))
            })
            .build()
    );
    let acc_space = comptime!(
        Tiling::new()
            .extents(&[(R, split_rows), (V, val_dim)])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                l.axis(R, Cut::sequential(rows))
                    .axis(V, Cut::sequential(val_dim))
            })
            .build()
    );
    let score_all = MemData::<f32>::smem(
        score_space.clone(),
        1usize,
        comptime!(StagePlan::in_place()),
    );
    let p_all = MemData::<f32>::smem(score_space, 1usize, comptime!(StagePlan::in_place()));
    let mut factors_all =
        MemData::<f32>::smem(row_space.clone(), 1usize, comptime!(StagePlan::in_place()));
    let m_all = MemData::<f32>::smem(row_space.clone(), 1usize, comptime!(StagePlan::in_place()));
    let l_all = MemData::<f32>::smem(row_space.clone(), 1usize, comptime!(StagePlan::in_place()));
    let mut acc_all = MemData::<f32>::smem(acc_space, 1usize, comptime!(StagePlan::in_place()));
    let mut recip = MemData::<f32>::smem(
        comptime!(Space::new(&[(R, rows)])),
        1usize,
        comptime!(StagePlan::in_place()),
    );
    acc_all.zero();

    // This team's windows.
    let t = UNIT_POS_Y as usize;
    let tw = Walk::over(score_all.runtime_space());
    let mut score = score_all.at(&tw.region(t));
    let mut p = p_all.at(&tw.region(t));
    let rw = Walk::over(factors_all.runtime_space());
    let mut factors = factors_all.at(&rw.region(t));
    let mut m_win = m_all.at(&rw.region(t));
    let mut l_win = l_all.at(&rw.region(t));
    let aw = Walk::over(acc_all.runtime_space());
    let mut acc = acc_all.at(&aw.region(t));

    let kept = comptime!(Space::new(&[(R, rows)]));
    let mut state = RowState::<f32>::new(kept, team);
    let rpu = comptime!(state.rows_per_unit);
    let bound_s = bound as usize;
    sync_cube();

    // Interleaved split walk: team t folds blocks t, t + splits, …; every
    // team runs every round (the barriers must stay uniform), an out-of-range
    // block just skips its compute.
    let k_walk = Walk::over(k.runtime_space());
    let blocks = bound_s.div_ceil(block);
    let rounds = blocks.div_ceil(splits);
    for round in 0..rounds {
        let blk = round * splits + t;
        let live = blk < blocks;
        let region = k_walk.region(blk);
        let s0 = region.coord(S) * block;
        let cols_bound = max(bound_s, s0) - s0;

        if live {
            let kb = k.at(&region);
            score.score_columns(&q_s, &kb, cols_bound, row_chunk);
        }
        sync_cube();

        if live {
            let probe = MaskProbe {
                origin_q: 0,
                origin_s: s0,
                bound_q: q_rows.runtime(),
                bound_s,
                q_rows,
                causal,
                materialized: false,
            };
            let corr = score.softmax::<f32>(&mut p, &mut state, &probe, &mask_tile, scale);
            factors.store_rows(&corr, rpu);
        }
        sync_cube();

        if live {
            let vb = v.at(&region);
            acc.mix_columns(&p, &vb, &factors, cols_bound, row_chunk);
        }
        sync_cube();
    }

    // Publish each team's running state, merge across splits, drain with the
    // split weights and the normalizer folded in.
    m_win.store_rows(&state.m, rpu);
    l_win.store_rows(&state.l, rpu);
    sync_cube();
    factors_all.merge_splits(&mut recip, &m_all, &l_all, splits);
    sync_cube();

    let size!(W1) = 1usize;
    let acc_flat = acc_all.flat::<W1>();
    let w_flat = factors_all.flat::<W1>();
    let r_flat = recip.flat::<W1>();
    let total = comptime!(rows * val_dim);
    let workers = CUBE_DIM as usize;
    let mut i = UNIT_POS as usize;
    while i < total {
        let r = i / val_dim;
        let vi = i % val_dim;
        let mut sum = 0.0f32;
        for ti in 0..splits {
            let sr = ti * rows + r;
            sum +=
                acc_flat.read(sr * val_dim + vi).extract(0usize) * w_flat.read(sr).extract(0usize);
        }
        out[i] = sum * r_flat.read(r).extract(0usize);
        i += workers;
    }
}

/// Launch the split fold and check against direct host math.
#[allow(clippy::too_many_arguments)]
fn run_split(
    (team, splits, g, qp, s_total, block, d, val_dim): (
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ),
    bound_s: usize,
    causal: bool,
    vec: usize,
) {
    let client: ComputeClient<TestRuntime> = <TestRuntime as Runtime>::client(&Default::default());
    let cap = client.properties().hardware.max_units_per_cube as usize;
    let team = team.min((cap / splits).max(1));
    let rows = g * qp;
    let scale = 1. / (d as f32).sqrt();
    // The register-blocking knob a routine would size from the register budget.
    let row_chunk = 4;

    let f32_ty = f32::elem_type_native();
    let u32_ty = u32::elem_type_native();
    let wobble =
        |i: usize, salt: usize| ((i * 2654435761 + salt * 40503) % 2048) as f32 / 512. - 2.;

    let (q_handle, q_data) = TestInput::builder(client.clone(), Shape::new([g, qp, d]))
        .dtype(f32_ty)
        .custom((0..g * qp * d).map(|i| wobble(i, 1)).collect())
        .generate_with_f32_host_data();
    let (k_handle, k_data) = TestInput::builder(client.clone(), Shape::new([s_total, d]))
        .dtype(f32_ty)
        .custom((0..s_total * d).map(|i| wobble(i, 2)).collect())
        .generate_with_f32_host_data();
    let (v_handle, v_data) = TestInput::builder(client.clone(), Shape::new([s_total, val_dim]))
        .dtype(f32_ty)
        .custom((0..s_total * val_dim).map(|i| wobble(i, 3) / 2.).collect())
        .generate_with_f32_host_data();
    let mask_handle = TestInput::builder(client.clone(), Shape::new([1]))
        .dtype(u32_ty)
        .zeros()
        .generate_without_host_data();
    let out_handle = TestInput::builder(client.clone(), Shape::new([rows, val_dim]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    // The one attention space, as in [`run`].
    let space = Tiling::new()
        .extents(&[
            (G, g),
            (QP, qp),
            (S, s_total),
            (D, d),
            (V, val_dim),
            (R, 1),
            (C, 1),
        ])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(G, Cut::sequential(g))
                .axis(QP, Cut::sequential(qp))
                .axis(S, Cut::sequential(block))
                .axis(D, Cut::sequential(d))
                .axis(V, Cut::sequential(val_dim))
                .axis(R, Cut::sequential(1))
                .axis(C, Cut::sequential(1))
        })
        .build();

    attention_fold_split_kernel::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_2d(team as u32, splits as u32),
        vec,
        TileArgLaunch::new(
            q_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[G, QP, D]),
        ),
        TileArgLaunch::new(
            k_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[S, D]),
        ),
        TileArgLaunch::new(
            v_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[S, V]),
        ),
        TileArgLaunch::new(
            mask_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[R, C]),
        ),
        out_handle.clone().binding().into_tensor_arg(),
        scale,
        bound_s as u32,
        space,
        team,
        splits,
        causal,
        block,
        row_chunk,
    );

    let out = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);

    for gi in 0..g {
        for qi in 0..qp {
            let r = gi * qp + qi;
            let mut scores = Vec::new();
            for j in 0..s_total {
                let masked = j >= bound_s || (causal && j > qi);
                if !masked {
                    let dot: f32 = (0..d)
                        .map(|p| q_data.get_f32(&[gi, qi, p]) * k_data.get_f32(&[j, p]))
                        .sum();
                    scores.push((dot * scale, j));
                }
            }
            if scores.is_empty() {
                for vi in 0..val_dim {
                    assert_eq!(
                        out.get_f32(&[r, vi]),
                        0.,
                        "fully-masked row {r}: out must be exactly 0"
                    );
                }
                continue;
            }
            let m = scores.iter().fold(f32::NEG_INFINITY, |m, (s, _)| m.max(*s));
            let l: f32 = scores.iter().map(|(s, _)| (s - m).exp()).sum();
            for vi in 0..val_dim {
                let expected: f32 = scores
                    .iter()
                    .map(|(s, j)| (s - m).exp() / l * v_data.get_f32(&[*j, vi]))
                    .sum();
                let got = out.get_f32(&[r, vi]);
                assert!(
                    (got - expected).abs() <= 1e-4 * expected.abs().max(1.),
                    "row ({gi},{qi}) v {vi}: out {got} vs direct {expected}"
                );
            }
        }
    }
}

/// The decode shape split four ways, with an idle team in the last round
/// (7 blocks over 4 teams) and a ragged prefix.
#[test]
fn split_fold_decode_gqa() {
    run_split((8, 4, 4, 1, 64, 8, 16, 16), 50, false, 2);
}

/// Causal prefill split two ways.
#[test]
fn split_fold_prefill_gqa_causal() {
    run_split((8, 2, 2, 8, 32, 8, 8, 8), 29, true, 2);
}

/// One split: the degenerate path must match the plain fold.
#[test]
fn split_fold_degenerates_to_one() {
    run_split((16, 1, 4, 1, 24, 8, 8, 4), 13, false, 1);
}

/// More teams than blocks: whole teams idle, weight zero on their own.
#[test]
fn split_fold_idle_teams() {
    run_split((4, 4, 2, 1, 16, 8, 8, 8), 10, false, 1);
}

/// The streaming fold: the decode shape with no score tile, where each plane (one
/// per split team on the cube's y dim) streams its S slice through
/// [`StreamFold`], and the same split ending as the shared-memory fold
/// (publish, [`merge_splits`](cubek_tile::Tile), weighted drain) closes it.
/// No barriers until the ending.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn attention_stream_test_kernel<W: Size>(
    q: &TileArg<'_, f32, W>,   // {G, QP(=1), D}
    k: &TileArg<'_, f32, W>,   // {S, D}
    v: &TileArg<'_, f32, W>,   // {S, V}
    out: &TileArg<'_, f32, W>, // {G, QP(=1), V}
    scale: f32,
    bound: u32,
    #[comptime] space: Space,
    #[comptime] lanes: usize,
    #[comptime] splits: usize,
    #[comptime] block: usize,
) {
    let q = q.tile(comptime!(space.clone()));
    let k = k.tile(comptime!(space.clone()));
    let v = v.tile(comptime!(space.clone()));
    let mut out = out.tile(space);

    let rank = comptime!(q.space.rank());
    let d = comptime!(q.space.extent_at(rank - 1));
    let rows = comptime!(q.space.tile_size() / d);

    let kept = comptime!(Space::new(&[(R, rows)]));
    let size!(N) = q.vector_size();
    let mut fold = StreamFold::<f32, N>::new(&q, lanes, kept);

    // This team's contiguous slice of the walk: no barriers anywhere.
    let t = UNIT_POS_Y as usize;
    let bound_s = bound as usize;
    let k_walk = Walk::over(k.runtime_space());
    let blocks = bound_s.div_ceil(block);
    let per_team = blocks.div_ceil(splits);
    let start_b = t * per_team;
    let end_b = min(start_b + per_team, blocks);
    let mut blk = start_b;
    while blk < end_b {
        let region = k_walk.region(blk);
        let s0 = region.coord(S) * block;
        let cols_bound = max(bound_s, s0) - s0;
        fold.absorb(&k.at(&region), &v.at(&region), scale, cols_bound);
        blk += 1;
    }

    // The one-barrier merged store.
    fold.store(&mut out, splits);
}

/// Launch the streaming fold and check against direct host math.
fn run_stream(
    (splits, g, s_total, block, d): (usize, usize, usize, usize, usize),
    bound_s: usize,
    vec: usize,
) {
    let client: ComputeClient<TestRuntime> = <TestRuntime as Runtime>::client(&Default::default());
    let lanes = client.properties().hardware.plane_size_max as usize;
    let cap = client.properties().hardware.max_units_per_cube as usize;
    let splits = splits.min((cap / lanes).max(1));
    let rows = g;
    let val_dim = d;
    let scale = 1. / (d as f32).sqrt();

    let f32_ty = f32::elem_type_native();
    let wobble =
        |i: usize, salt: usize| ((i * 2654435761 + salt * 40503) % 2048) as f32 / 512. - 2.;

    let (q_handle, q_data) = TestInput::builder(client.clone(), Shape::new([g, 1, d]))
        .dtype(f32_ty)
        .custom((0..g * d).map(|i| wobble(i, 1)).collect())
        .generate_with_f32_host_data();
    let (k_handle, k_data) = TestInput::builder(client.clone(), Shape::new([s_total, d]))
        .dtype(f32_ty)
        .custom((0..s_total * d).map(|i| wobble(i, 2)).collect())
        .generate_with_f32_host_data();
    let (v_handle, v_data) = TestInput::builder(client.clone(), Shape::new([s_total, val_dim]))
        .dtype(f32_ty)
        .custom((0..s_total * val_dim).map(|i| wobble(i, 3) / 2.).collect())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), Shape::new([rows, val_dim]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    // The one attention space: q/k/v/out project their axes out of it.
    let space = Tiling::new()
        .extents(&[(G, g), (QP, 1), (S, s_total), (D, d), (V, val_dim)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(G, Cut::sequential(g))
                .axis(QP, Cut::sequential(1))
                .axis(S, Cut::sequential(block))
                .axis(D, Cut::sequential(d))
                .axis(V, Cut::sequential(val_dim))
        })
        .build();

    attention_stream_test_kernel::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_2d(lanes as u32, splits as u32),
        vec,
        TileArgLaunch::new(
            q_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[G, QP, D]),
        ),
        TileArgLaunch::new(
            k_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[S, D]),
        ),
        TileArgLaunch::new(
            v_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[S, V]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[G, QP, V]),
        ),
        scale,
        bound_s as u32,
        space,
        lanes,
        splits,
        block,
    );

    let out = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);

    for gi in 0..g {
        let mut scores = Vec::new();
        for j in 0..s_total.min(bound_s) {
            let dot: f32 = (0..d)
                .map(|p| q_data.get_f32(&[gi, 0, p]) * k_data.get_f32(&[j, p]))
                .sum();
            scores.push((dot * scale, j));
        }
        let m = scores.iter().fold(f32::NEG_INFINITY, |m, (s, _)| m.max(*s));
        let l: f32 = scores.iter().map(|(s, _)| (s - m).exp()).sum();
        for vi in 0..val_dim {
            let expected: f32 = scores
                .iter()
                .map(|(s, j)| (s - m).exp() / l * v_data.get_f32(&[*j, vi]))
                .sum();
            let got = out.get_f32(&[gi, vi]);
            assert!(
                (got - expected).abs() <= 1e-4 * expected.abs().max(1.),
                "row {gi} v {vi}: out {got} vs direct {expected}"
            );
        }
    }
}

/// The decode shape split four ways: ragged prefix, idle team in the tail.
#[test]
fn stream_fold_decode_gqa() {
    run_stream((4, 4, 64, 8, 16), 50, 2);
}

/// One split, scalar reads, bound not on a block edge.
#[test]
fn stream_fold_single_split_scalar() {
    run_stream((1, 4, 24, 8, 8), 13, 1);
}

/// More teams than blocks: whole teams idle, weight zero through the merge.
#[test]
fn stream_fold_idle_teams() {
    run_stream((4, 2, 16, 8, 8), 10, 1);
}
