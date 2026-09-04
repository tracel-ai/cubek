//! The attention fold end-to-end on tiles: the score and mix leaves
//! ([`ops::attention`]) interleaved with the softmax step, the walk owned by
//! the kernel: the miniature of the routine a client (metabolic) launches.
//! GQA rides the axes: `q` stacks its group over the same query block
//! (group-major rows), `k`/`v` simply omit the group axis, and the probe's
//! `q_rows` maps rows back to query positions for the causal predicate.

use cubecl::{client::Client, prelude::*, zspace::Shape};
use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult};
use cubek_tile::{
    Axis, MaskProbe, MemData, RegisterBlock, RowState, Space, StageStorage, StreamFold, TileArg,
    TileArgLaunch, TileSpec, Tiling, Walk,
};

const G: Axis = Axis(0); // GQA group member
const QP: Axis = Axis(1); // query position
const S: Axis = Axis(2); // key/value position (the reduced axis)
const D: Axis = Axis(3); // head dim (contracted by the score matmul)
const V: Axis = Axis(4); // value dim
// Local labels for the kernel-allocated smem tiles.
const R: Axis = Axis(5); // score rows = G × QP, group-major
const C: Axis = Axis(6); // score cols = one S block
const T: Axis = Axis(7); // split team, one window per team

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
    #[comptime] budget: usize,
    #[comptime] in_place: bool,
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
        StageStorage::Strided,
        0usize,
    );
    q_s.copy_from(&q);
    // The register block the two matmuls contract through: the software instruction, whose
    // budget caps the rows a visit keeps live.
    let config = comptime!(RegisterBlock::new(budget));
    let score_space = comptime!(Space::new(&[(R, rows), (C, block)]));
    let mut score =
        MemData::<f32>::smem(score_space.clone(), 1usize, StageStorage::Strided, 0usize);
    let mut p = MemData::<f32>::smem(score_space, 1usize, StageStorage::Strided, 0usize);
    let row_space = comptime!(Space::new(&[(R, rows)]));
    let mut factors =
        MemData::<f32>::smem(row_space.clone(), 1usize, StageStorage::Strided, 0usize);
    let acc_space = comptime!(Space::new(&[(R, rows), (V, val_dim)]));
    let mut acc = MemData::<f32>::smem(acc_space, 1usize, StageStorage::Strided, 0usize);
    acc.zero();
    let mut state = RowState::<f32>::new(row_space, units);
    let share = comptime!(state.share);
    let rpu = comptime!(share.rows());
    let bound_s = bound as usize;
    sync_cube();

    // The fold: one S block per region.
    for region in Walk::over(k.runtime_space()) {
        let kb = k.at(&region);
        let vb = v.at(&region);
        let s0 = region.coord(S) * block;

        // Clip the ragged tail: no reads past the attended prefix.
        let cols_bound = max(bound_s, s0) - s0;
        score.score_columns(&q_s, &kb, cols_bound, config);
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
        if comptime!(in_place) {
            let corr = score.softmax_in_place(&mut state, &probe, &mask_tile, scale);
            acc.rescale_rows(&corr, share);
        } else {
            let corr = score.softmax::<f32>(&mut p, &mut state, &probe, &mask_tile, scale);
            acc.rescale_rows(&corr, share);
        }
        sync_cube();

        // Stale cache beyond the attended prefix must not ride a zero
        // probability into the accumulator.
        if comptime!(in_place) {
            acc.mix_columns(&score, &vb, cols_bound, config);
        } else {
            acc.mix_columns(&p, &vb, cols_bound, config);
        }
        sync_cube();
    }

    // Epilogue: owners publish 1/l, the cube normalizes and drains.
    let mut recip = Array::<f32>::new(rpu);
    for ri in 0..rpu {
        recip[ri] = state.recip_l(ri);
    }
    factors.store_rows(&recip, share);
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
    in_place: bool,
) {
    let client: Client = cubecl::test_device().client();
    let units = units.min(client.properties().hardware.max_units_per_cube as usize);
    let rows = g * qp;
    let scale = 1. / (d as f32).sqrt();
    // The register budget a routine would size from the hardware: 4 rows of `vec`-wide
    // accumulators live at once.
    let budget = 4 * vec;

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
    let space = Tiling::over(&[
        (G, g),
        (QP, qp),
        (S, s_total),
        (D, d),
        (V, val_dim),
        (R, 1),
        (C, 1),
    ])
    .level(|l| {
        l.walk(&[
            (G, g),
            (QP, qp),
            (S, block),
            (D, d),
            (V, val_dim),
            (R, 1),
            (C, 1),
        ]);
    })
    .build();

    attention_fold_kernel::launch(
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
        budget,
        in_place,
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
    run((32, 4, 1, 64, 16, 16, 16), 50, false, 2, false);
}

/// Prefill with GQA and causal: the probe's `q_rows` row→query mapping.
#[test]
fn fold_prefill_gqa_causal() {
    run((8, 2, 8, 32, 8, 8, 8), 29, true, 2, false);
}

/// Scalar reads (vector width 1), block not dividing the prefix.
#[test]
fn fold_scalar_odd_bound() {
    run((16, 4, 4, 24, 8, 8, 4), 13, true, 1, false);
}

/// The probabilities left in place over the scores on the column arm: the register mix reads the
/// score tile as P and no P tile is written.
#[test]
fn fold_in_place() {
    run((8, 2, 8, 32, 8, 8, 8), 29, true, 2, true);
}

/// The same fold with both matmuls on tensor cores: the score and mix leaves are the tensor-core
/// ones, so a plane owns whole fragments where a unit owned columns, and the softmax between them
/// is unchanged scalar shared-memory work.
///
/// The flow llama.cpp's `fa.metal` runs: Q@K into fragments stored straight to shared memory, the
/// online softmax on plain floats, the running total rescaled where it lies, then P@V folding onto
/// it through an accumulator fragment loaded back from shared memory. Nothing persists in a
/// fragment across a barrier.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn attention_fold_cmma_kernel<E: Float>(
    q: &TileArg<'_, E, Const<1>>,      // {QP, D}
    k: &TileArg<'_, E, Const<1>>,      // {S, D}
    v: &TileArg<'_, E, Const<1>>,      // {S, V}
    mask: &TileArg<'_, u32, Const<1>>, // 1-cell dummy (materialized = false)
    out: &mut Tensor<f32>,             // [QP·V] flat
    scale: f32,
    bound: u32,
    #[comptime] space: Space,
    #[comptime] units: usize,
    #[comptime] causal: bool,
    #[comptime] block: usize,
    #[comptime] frag: usize,
    #[comptime] planes: usize,
    #[comptime] in_place: bool,
    #[comptime] score_vec: usize,
    #[comptime] lanes: usize,
    #[define(E)] _dtype: ElemType,
) {
    let q = q.tile(comptime!(space.clone()));
    let k = k.tile(comptime!(space.clone()));
    let v = v.tile(comptime!(space.clone()));
    let mask_tile = mask.tile(space);

    let rows = comptime!(q.space.extent(QP));
    let d = comptime!(q.space.extent(D));
    let val_dim = comptime!(v.space.extent(V));
    // The queries stage with the grid their lhs role reads: `frag` rows against `frag` of the
    // contracted head dim. Both matmuls contract through the tensor-core leaves.
    let q_space = comptime!(
        Tiling::over(&[(QP, rows), (D, d)])
            .level(|l| {
                l.walk(&[(QP, frag), (D, frag)]);
            })
            .build()
    );
    let mut q_s = MemData::<E>::smem(q_space, 1usize, StageStorage::Strided, 0usize);
    q_s.copy_from(&q);

    // With `planes > 1` the space states each plane's slice of the grid above the instruction:
    // the leaf then holds its slice's accumulators across the contraction instead of walking the
    // whole grid one fragment at a time.
    // The score (and P) tile may be lined: the softmax passes read a row a line at a time and the
    // fragments store and load it through its element slice either way.
    let score_space = comptime!(sliced(&[(R, rows), (C, block)], planes, frag));
    let mut score = MemData::<f32>::smem(
        score_space.clone(),
        score_vec,
        StageStorage::Strided,
        0usize,
    );
    let mut p = MemData::<f32>::smem(score_space, score_vec, StageStorage::Strided, 0usize);
    let row_space = comptime!(Space::new(&[(R, rows)]));
    let mut factors =
        MemData::<f32>::smem(row_space.clone(), 1usize, StageStorage::Strided, 0usize);
    let acc_space = comptime!(sliced(&[(R, rows), (V, val_dim)], planes, frag));
    let mut acc = MemData::<f32>::smem(acc_space, 1usize, StageStorage::Strided, 0usize);
    acc.zero();
    // `lanes > 1` puts the softmax at plane ownership, the row-slice per plane and its lanes
    // splitting the lines; one lane is the unit arm.
    let mut state = match comptime!(lanes > 1) {
        true => RowState::<f32>::over_planes(row_space, units, lanes),
        false => RowState::<f32>::new(row_space, units),
    };
    let share = comptime!(state.share);
    let rpu = comptime!(share.rows());
    let bound_s = bound as usize;
    sync_cube();

    for region in Walk::over(k.runtime_space()) {
        let kb = k.at(&region);
        let vb = v.at(&region);
        let s0 = region.coord(S) * block;

        let cols_bound = max(bound_s, s0) - s0;
        score.score_fragments(&q_s, &kb, cols_bound);
        sync_cube();

        let probe = MaskProbe {
            origin_q: 0,
            origin_s: s0,
            bound_q: rows.runtime(),
            bound_s,
            q_rows: rows,
            causal,
            materialized: false,
        };
        if comptime!(in_place) {
            let corr = score.softmax_in_place(&mut state, &probe, &mask_tile, scale);
            acc.rescale_rows(&corr, share);
        } else {
            let corr = score.softmax::<f32>(&mut p, &mut state, &probe, &mask_tile, scale);
            acc.rescale_rows(&corr, share);
        }
        sync_cube();

        if comptime!(in_place) {
            acc.mix_fragments(&score, &vb, cols_bound);
        } else {
            acc.mix_fragments(&p, &vb, cols_bound);
        }
        sync_cube();
    }

    let mut recip = Array::<f32>::new(rpu);
    for ri in 0..rpu {
        recip[ri] = state.recip_l(ri);
    }
    factors.store_rows(&recip, share);
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

/// `{rows, cols}` cut into `frag × frag` fragments, under one slice of `cols / planes` per plane
/// when more than one is stated.
fn sliced(extents: &[(Axis, usize); 2], planes: usize, frag: usize) -> Space {
    let (rows, cols) = (extents[0], extents[1]);
    let tiling = Tiling::over(extents);
    let tiling = match planes {
        1 => tiling,
        planes => tiling.level(|l| {
            l.walk(&[rows, (cols.0, cols.1 / planes)]);
        }),
    };
    tiling
        .level(|l| {
            l.walk(&[(rows.0, frag), (cols.0, frag)]);
        })
        .build()
}

/// Launch the tensor-core fold and check against direct host math.
///
/// `planes` above one states each plane's slice of the fragment grid, and requires the cube to be
/// exactly that many planes of the width the device commits to; skipped where it does not.
///
/// `spanned` gives `k` and `v` a leading axis the launch spans rather than iterates — the shape a
/// client's cache has, where a KV head is an axis of the operand and a cube owns one position of
/// it.
#[allow(clippy::too_many_arguments)]
fn run_cmma<E: Float + CubeElement>(
    (units, rows, s_total, block, d, val_dim, frag): (
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
    spanned: bool,
    planes: usize,
    in_place: bool,
    score_vec: usize,
    planar: bool,
) {
    let client: Client = cubecl::test_device().client();
    let hw = &client.properties().hardware;
    let exact = hw.plane_size_min == hw.plane_size_max;
    if planes > 1 && (!exact || units != planes * hw.plane_size_min as usize) {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "a stated slice needs a cube of exactly {planes} planes; this device commits to none"
        )))
        .enforce();
        return;
    }
    if planar && !exact {
        TestOutcome::Validated(ValidationResult::Skipped(
            "a planar softmax needs a plane width the device commits to".into(),
        ))
        .enforce();
        return;
    }
    let lanes = if planar {
        hw.plane_size_min as usize
    } else {
        1
    };
    let f32_ty = f32::elem_type_native();
    let e_ty = E::elem_type_native();
    let supported = client.properties().features.matmul.cmma.iter().any(|cfg| {
        cfg.a_type == e_ty
            && cfg.b_type == e_ty
            && cfg.cd_type == f32_ty
            && (cfg.m as usize, cfg.n as usize, cfg.k as usize) == (frag, frag, frag)
    });
    if !supported {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device has no {frag}x{frag}x{frag} {e_ty:?} cmma fragment accumulating at f32"
        )))
        .enforce();
        return;
    }
    let scale = 1. / (d as f32).sqrt();

    let u32_ty = u32::elem_type_native();
    let wobble =
        |i: usize, salt: usize| ((i * 2654435761 + salt * 40503) % 2048) as f32 / 512. - 2.;

    let (q_handle, q_data) = TestInput::builder(client.clone(), Shape::new([rows, d]))
        .dtype(e_ty)
        .custom((0..rows * d).map(|i| wobble(i, 1)).collect())
        .generate_with_f32_host_data();
    // The spanned arm binds the same values under one more dim: same cells, one axis above the
    // two the leaf contracts.
    let kv_shape = |rows: usize, cols: usize| {
        if spanned {
            Shape::new([1, rows, cols])
        } else {
            Shape::new([rows, cols])
        }
    };
    let (k_handle, k_data) = TestInput::builder(client.clone(), kv_shape(s_total, d))
        .dtype(e_ty)
        .custom((0..s_total * d).map(|i| wobble(i, 2)).collect())
        .generate_with_f32_host_data();
    let (v_handle, v_data) = TestInput::builder(client.clone(), kv_shape(s_total, val_dim))
        .dtype(e_ty)
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

    // `R` and `C` are the score tile's own axes, declared degenerate here: the launch walks `S`
    // in blocks and nothing else.
    let space = Tiling::over(&[
        (G, 1),
        (QP, rows),
        (S, s_total),
        (D, d),
        (V, val_dim),
        (R, 1),
        (C, 1),
    ])
    .level(|l| {
        l.walk(&[
            (G, 1),
            (QP, rows),
            (S, block),
            (D, d),
            (V, val_dim),
            (R, 1),
            (C, 1),
        ]);
    })
    .build();
    // The axis is on the operand only where its binding has a dim for it: a spec naming one it
    // does not is a different operand, not the same one spanned.
    let (k_axes, v_axes): (&[Axis], &[Axis]) = if spanned {
        (&[G, S, D], &[G, S, V])
    } else {
        (&[S, D], &[S, V])
    };

    attention_fold_cmma_kernel::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(units as u32),
        TileArgLaunch::new(
            q_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[QP, D]),
        ),
        TileArgLaunch::new(
            k_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(k_axes),
        ),
        TileArgLaunch::new(
            v_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(v_axes),
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
        frag,
        planes,
        in_place,
        score_vec,
        lanes,
        e_ty,
    );

    let out = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);

    for r in 0..rows {
        let mut scores = Vec::new();
        for j in 0..s_total {
            let masked = j >= bound_s || (causal && j > r);
            if !masked {
                let dot: f32 = (0..d)
                    .map(|p| {
                        let key = if spanned {
                            k_data.get_f32(&[0, j, p])
                        } else {
                            k_data.get_f32(&[j, p])
                        };
                        q_data.get_f32(&[r, p]) * key
                    })
                    .sum();
                scores.push((dot * scale, j));
            }
        }
        let m = scores.iter().fold(f32::NEG_INFINITY, |m, (s, _)| m.max(*s));
        let l: f32 = scores.iter().map(|(s, _)| (s - m).exp()).sum();
        for vi in 0..val_dim {
            let expected: f32 = scores
                .iter()
                .map(|(s, j)| {
                    let value = if spanned {
                        v_data.get_f32(&[0, *j, vi])
                    } else {
                        v_data.get_f32(&[*j, vi])
                    };
                    (s - m).exp() / l * value
                })
                .sum();
            let got = out.get_f32(&[r, vi]);
            assert!(
                (got - expected).abs() <= 1e-4 * expected.abs().max(1.),
                "row {r} v {vi}: out {got} vs direct {expected}"
            );
        }
    }
}

/// The hardware arm folds the same attention the scalar one does: one plane, one fragment per
/// matmul, two KV blocks. A fragment that never ran, or a running total that lost what the block
/// before it folded in, reads back here as zeros or as one block's answer.
#[test]
fn fold_cmma_single_fragment() {
    run_cmma::<f32>(
        (32, 8, 16, 8, 8, 8, 8),
        16,
        false,
        false,
        1,
        false,
        1,
        false,
    );
}

/// Fragments are owned, not shared, and a contraction deeper than one fragment closes: two planes
/// over a 2x2 grid, the head dim and the block each two steps deep. A plane taking the wrong share
/// leaves whole fragments stale; a step dropped leaves half a dot product.
#[test]
fn fold_cmma_fragment_grid() {
    run_cmma::<f32>(
        (64, 16, 32, 16, 16, 16, 8),
        32,
        false,
        false,
        1,
        false,
        1,
        false,
    );
}

/// The attended prefix may end inside a block: the mask probe owns the tail, so score fragments
/// that straddle the bound are still contracted whole and only whole value steps past it are
/// skipped. Splitting that the other way lets stale cache ride a zero probability into the
/// accumulator.
#[test]
fn fold_cmma_causal_ragged_bound() {
    run_cmma::<f32>(
        (64, 16, 32, 16, 16, 16, 8),
        24,
        true,
        false,
        1,
        false,
        1,
        false,
    );
}

/// The keys and values a client hands over are not bare matrices: a KV head is an axis of the
/// cache and a cube owns one position of it, so the block reaching the leaf carries that axis
/// above the two it contracts. The fragment arm reads the trailing two, as the column arm does,
/// and spans what is above them — a leaf reading `axis_at(0)` as the rows contracts a 1×block
/// matrix here and returns zeros.
#[test]
fn fold_cmma_spanned_leading_axis() {
    run_cmma::<f32>(
        (64, 16, 32, 16, 16, 16, 8),
        24,
        true,
        true,
        1,
        false,
        1,
        false,
    );
}

/// The space states each plane's slice of the grid, so a plane holds its slice's accumulators
/// across the whole contraction and loads each P fragment once for all of them. Same rows, same
/// ragged causal bound and same spanned operands as the cases above — the slice changes who
/// holds what, never the answer. A plane reading its neighbour's slice, or a P fragment reused
/// against the wrong column, comes out here as a wrong row rather than zeros.
#[test]
fn fold_cmma_plane_slices() {
    run_cmma::<f32>(
        (64, 16, 32, 16, 16, 16, 8),
        24,
        true,
        true,
        2,
        false,
        1,
        false,
    );
}

/// A slice one fragment wide: `cn = 1`, so the reuse loop degenerates and the accumulator
/// bookkeeping is all that is left to get wrong.
#[test]
fn fold_cmma_plane_slices_one_fragment_wide() {
    run_cmma::<f32>(
        (64, 16, 32, 16, 16, 16, 8),
        32,
        false,
        false,
        2,
        false,
        1,
        false,
    );
}

/// The probabilities left in place over the scores: the mix contracts the score tile itself,
/// and no P tile is written. Same fold as `fold_cmma_plane_slices`; a mix reading stale scores
/// (the pre-exponentiation cells, or the previous block's) comes out here as wrong rows.
#[test]
fn fold_cmma_in_place() {
    run_cmma::<f32>(
        (64, 16, 32, 16, 16, 16, 8),
        24,
        true,
        true,
        2,
        true,
        1,
        false,
    );
}

/// Half operands with the probabilities at the accumulate element: the mix's instruction takes
/// P at f32 against values at f16, the mixed-type form the hand-written Metal kernels contract
/// through. Skipped where the device states no half fragment accumulating at f32.
#[test]
fn fold_cmma_in_place_half_operands() {
    run_cmma::<half::f16>(
        (64, 16, 32, 16, 16, 16, 8),
        24,
        true,
        true,
        2,
        true,
        1,
        false,
    );
}

/// The same half operands through a written P tile at f32 — the mix is mixed-type either way,
/// so a device that refuses it fails here as well, not only in place.
#[test]
fn fold_cmma_half_operands() {
    run_cmma::<half::f16>(
        (64, 16, 32, 16, 16, 16, 8),
        24,
        true,
        true,
        2,
        false,
        1,
        false,
    );
}

/// The softmax at plane ownership under the fragment arm: each plane owns a slice of the rows
/// and its lanes split the columns, with the plane reductions closing each row.
#[test]
fn fold_cmma_planar_softmax() {
    run_cmma::<f32>(
        (64, 16, 32, 16, 16, 16, 8),
        24,
        true,
        true,
        2,
        true,
        1,
        true,
    );
}

/// A lined score tile: the passes read a row two columns at a time, the fragments store into
/// and load from the same lines, and the mix contracts them in place. A pass indexing lines as
/// columns reads the wrong half of every row.
#[test]
fn fold_cmma_lined_scores() {
    run_cmma::<f32>(
        (64, 16, 32, 16, 16, 16, 8),
        24,
        true,
        true,
        2,
        true,
        2,
        false,
    );
}

/// Lined and planar at once, the shape a fold on a plane-width device takes: 16 columns in lines
/// of 4 leave 4 lines for 32 lanes, so the lane guard is live and most lanes own no line.
#[test]
fn fold_cmma_planar_lined_scores() {
    run_cmma::<f32>(
        (64, 16, 32, 16, 16, 16, 8),
        24,
        true,
        true,
        2,
        true,
        4,
        true,
    );
}

/// The lined P tile written rather than left in place: the cast-write moves whole lines.
#[test]
fn fold_cmma_planar_lined_written_p() {
    run_cmma::<f32>(
        (64, 16, 32, 16, 16, 16, 8),
        24,
        true,
        true,
        2,
        false,
        2,
        true,
    );
}

/// Half operands, lined scores, planar softmax, P in place: every density at once.
#[test]
fn fold_cmma_planar_lined_half_operands() {
    run_cmma::<half::f16>(
        (64, 16, 32, 16, 16, 16, 8),
        24,
        true,
        true,
        2,
        true,
        2,
        true,
    );
}

/// The split fold: teams on the cube's y dim each fold a disjoint slice of
/// the S walk with their own running state into their own window of
/// split-wide smem tiles, then the states merge cross-team
/// ([`merge_splits`](cubek_tile::Tile)) and the drain folds the split
/// weights and the normalizer in. `splits == 1` degenerates to the plain
/// fold: one code path for both.
///
/// `split_inner` flips where the split axis sits on the row lanes, which is
/// the one thing `merge_splits` reads off the space. Both orders run the same
/// cuts and the same op and must give the same answer; a cross-cube merge
/// lays the split innermost so its drain can contract it, and nothing but a
/// test here says that layout works.
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
    #[comptime] budget: usize,
    #[comptime] split_inner: bool,
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
        StageStorage::Strided,
        0usize,
    );
    q_s.copy_from(&q);

    // Split-wide working set: a leading `splits` slice on every tile, one
    // window per team.
    //
    // Only the row lanes name the split as an axis: they are what
    // `merge_splits` reads. The score and the accumulator stack it into their
    // row axis, which is what the rank-2 rowwise leaves read.
    let split_rows = comptime!(splits * rows);
    let config = comptime!(RegisterBlock::new(budget));
    let score_space = comptime!(
        Tiling::over(&[(R, split_rows), (C, block)])
            .level(|l| {
                l.walk(&[(R, rows), (C, block)]);
            })
            .build()
    );
    // The split outermost gives a team one contiguous run of rows; innermost
    // gives it a strided column. Only the declared order differs: the cuts
    // below are the same either way, and so is every op that reads them.
    let row_extents = comptime!(if split_inner {
        [(R, rows), (T, splits)]
    } else {
        [(T, splits), (R, rows)]
    });
    let row_space = comptime!(
        Tiling::over(&row_extents)
            .level(|l| {
                l.walk(&[(T, 1), (R, rows)]);
            })
            .build()
    );
    let acc_space = comptime!(
        Tiling::over(&[(R, split_rows), (V, val_dim)])
            .level(|l| {
                l.walk(&[(R, rows), (V, val_dim)]);
            })
            .build()
    );
    let score_all =
        MemData::<f32>::smem(score_space.clone(), 1usize, StageStorage::Strided, 0usize);
    let p_all = MemData::<f32>::smem(score_space, 1usize, StageStorage::Strided, 0usize);
    let mut factors_all =
        MemData::<f32>::smem(row_space.clone(), 1usize, StageStorage::Strided, 0usize);
    let m_all = MemData::<f32>::smem(row_space.clone(), 1usize, StageStorage::Strided, 0usize);
    let l_all = MemData::<f32>::smem(row_space.clone(), 1usize, StageStorage::Strided, 0usize);
    let mut acc_all = MemData::<f32>::smem(acc_space, 1usize, StageStorage::Strided, 0usize);
    acc_all.zero();

    // This team's windows.
    let t = UNIT_POS_Y as usize;
    let tw = Walk::over(score_all.runtime_space());
    let mut score = score_all.at(&tw.region(t));
    let mut p = p_all.at(&tw.region(t));
    let rw = Walk::over(factors_all.runtime_space());
    let mut m_win = m_all.at(&rw.region(t));
    let mut l_win = l_all.at(&rw.region(t));
    let aw = Walk::over(acc_all.runtime_space());
    let mut acc = acc_all.at(&aw.region(t));

    let kept = comptime!(Space::new(&[(R, rows)]));
    let mut state = RowState::<f32>::new(kept, team);
    let share = comptime!(state.share);
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
            score.score_columns(&q_s, &kb, cols_bound, config);
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
            acc.rescale_rows(&corr, share);
        }
        sync_cube();

        if live {
            let vb = v.at(&region);
            acc.mix_columns(&p, &vb, cols_bound, config);
        }
        sync_cube();
    }

    // Publish each team's running state, merge across splits, drain with the
    // split weights and the normalizer folded in.
    m_win.store_rows(&state.m, share);
    l_win.store_rows(&state.l, share);
    sync_cube();
    factors_all.merge_splits(&m_all, &l_all, T);
    sync_cube();

    let size!(W1) = 1usize;
    let acc_flat = acc_all.flat::<W1>();
    let w_flat = factors_all.flat::<W1>();
    let total = comptime!(rows * val_dim);
    let workers = CUBE_DIM as usize;
    let mut i = UNIT_POS as usize;
    while i < total {
        let r = i / val_dim;
        let vi = i % val_dim;
        let mut sum = 0.0f32;
        for ti in 0..splits {
            // The accumulator always stacks the split into its row axis; only
            // the weights follow `split_inner`.
            let sr = ti * rows + r;
            let w = if comptime!(split_inner) {
                r * splits + ti
            } else {
                sr
            };
            sum +=
                acc_flat.read(sr * val_dim + vi).extract(0usize) * w_flat.read(w).extract(0usize);
        }
        out[i] = sum;
        i += workers;
    }
}

/// Launch the split fold and check against direct host math, once per row-lane
/// layout: the answer cannot depend on where the space puts the split axis.
#[allow(clippy::too_many_arguments)]
fn run_split(
    shape: (usize, usize, usize, usize, usize, usize, usize, usize),
    bound_s: usize,
    causal: bool,
    vec: usize,
) {
    for split_inner in [false, true] {
        run_split_at(shape, bound_s, causal, vec, split_inner);
    }
}

/// One launch, at the stated layout.
#[allow(clippy::too_many_arguments)]
fn run_split_at(
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
    split_inner: bool,
) {
    let client: Client = cubecl::test_device().client();
    let cap = client.properties().hardware.max_units_per_cube as usize;
    let team = team.min((cap / splits).max(1));
    let rows = g * qp;
    let scale = 1. / (d as f32).sqrt();
    // The register budget a routine would size from the hardware: 4 rows of `vec`-wide
    // accumulators live at once.
    let budget = 4 * vec;

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
    let space = Tiling::over(&[
        (G, g),
        (QP, qp),
        (S, s_total),
        (D, d),
        (V, val_dim),
        (R, 1),
        (C, 1),
    ])
    .level(|l| {
        l.walk(&[
            (G, g),
            (QP, qp),
            (S, block),
            (D, d),
            (V, val_dim),
            (R, 1),
            (C, 1),
        ]);
    })
    .build();

    attention_fold_split_kernel::launch(
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
        budget,
        split_inner,
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
    let client: Client = cubecl::test_device().client();
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
    let space = Tiling::over(&[(G, g), (QP, 1), (S, s_total), (D, d), (V, val_dim)])
        .level(|l| {
            l.walk(&[(G, g), (QP, 1), (S, block), (D, d), (V, val_dim)]);
        })
        .build();

    attention_stream_test_kernel::launch(
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
