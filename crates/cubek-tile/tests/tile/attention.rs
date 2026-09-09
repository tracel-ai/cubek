//! The attention fold end-to-end on tiles: the score and mix leaves
//! ([`ops::attention`]) interleaved with the softmax step, the walk owned by
//! the kernel: the miniature of the routine a client (metabolic) launches.
//! GQA rides the axes: `q` stacks its group over the same query block
//! (group-major rows), `k`/`v` simply omit the group axis, and the probe's
//! `q_rows` maps rows back to query positions for the causal predicate.

use cubecl::{client::Client, prelude::*, zspace::Shape};
use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult};
use cubek_tile::{
    Axis, Fragments, KernelForm, Launcher, Level, MaskProbe, MemData, Monoid, Partitioning,
    RegisterBlock, RowState, Semiring, Space, StageStorage, StreamFold, TileArg, TileArgLaunch,
    TileSpec,
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
    space: Partitioning,
    #[comptime] blocks: Level,
    #[comptime] units: usize,
    #[comptime] causal: bool,
    #[comptime] block: usize,
    #[comptime] budget: usize,
    #[comptime] in_place: bool,
) {
    let q = q.tile(comptime!(space.clone()));
    let k = k.tile(comptime!(space.clone()));
    let v = v.tile(comptime!(space.clone()));
    let mask_tile = mask.tile(comptime!(space.clone()));

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

    // The probe states the masking once, and the walk takes it as its bound: a block every row
    // masks throughout is one the walk never steps to. Its rows still mask per element, for the
    // last block's tail and its diagonal.
    let probe = MaskProbe {
        origin_q: 0,
        row_origin: 0,
        origin_s: 0,
        bound_q: q_rows.runtime(),
        bound_s,
        q_rows,
        causal,
        materialized: false,
    };

    // The fold: one S block per region.
    for region in k.over(&blocks).window(0, probe.blocks(block)) {
        let kb = k.at(&region);
        let vb = v.at(&region);
        let s0 = region.coord(S) * block;

        // Clip the ragged tail: no reads past the attended prefix.
        let cols_bound = max(bound_s, s0) - s0;
        score.score_columns(&q_s, &kb, cols_bound, config);
        sync_cube();

        let probe = probe.step_s(s0);
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
    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (G, g),
                (QP, qp),
                (S, s_total),
                (D, d),
                (V, val_dim),
                (R, 1),
                (C, 1),
            ]),
            vec![Level::walk(&[
                (G, g),
                (QP, qp),
                (S, block),
                (D, d),
                (V, val_dim),
                (R, 1),
                (C, 1),
            ])],
        ),
        KernelForm::Static,
    );

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
        launcher.partitioning_arg(),
        launcher.level(0),
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
/// The fold on the hardware instruction, written as the matmul kernel is. Each plane owns
/// `rows / planes` query rows through all three phases: the score is a plane-resident
/// accumulator stored into the plane's window of the smem score tile, the softmax runs on that
/// window, and the mix contracts into an accumulator opened once before the walk, kept in
/// fragments across it and rescaled where it sits. K and V are staged per block for the whole
/// cube, and those two fills are the only cube barriers.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn attention_fold_cmma_kernel<E: Float>(
    q: &TileArg<'_, E, Const<1>>,      // {QP, D}
    k: &TileArg<'_, E, Const<1>>,      // {S, D}
    v: &TileArg<'_, E, Const<1>>,      // {S, V}
    mask: &TileArg<'_, u32, Const<1>>, // 1-cell dummy (materialized = false)
    out: &TileArg<'_, f32, Const<1>>,  // {QP, V}
    scale: f32,
    bound: u32,
    space: Partitioning,
    #[comptime] blocks: Level,
    #[comptime] causal: bool,
    #[comptime] block: usize,
    #[comptime] frag: usize,
    #[comptime] planes: usize,
    #[comptime] score_vec: usize,
    #[comptime] lanes: usize,
    #[define(E)] _dtype: ElemType,
) {
    let q = q.tile(comptime!(space.clone()));
    let k = k.tile(comptime!(space.clone()));
    let v = v.tile(comptime!(space.clone()));
    let mask_tile = mask.tile(comptime!(space.clone()));
    let out = out.tile(comptime!(space.clone()));

    let rows = comptime!(q.space.extent(QP));
    let d = comptime!(q.space.extent(D));
    let val_dim = comptime!(v.space.extent(V));
    let rows_p = comptime!(rows / planes);
    let (rm, cn, vn, ks) = comptime!((rows_p / frag, block / frag, val_dim / frag, d / frag));

    let mut q_s = MemData::<E>::smem(
        comptime!(Space::new(&[(QP, rows), (D, d)])),
        1usize,
        StageStorage::Strided,
        0usize,
    );
    q_s.copy_from(&q);
    let score = MemData::<f32>::smem(
        comptime!(Space::new(&[(QP, rows), (S, block)])),
        score_vec,
        StageStorage::Strided,
        0usize,
    );
    // The block's window of K and of V, staged by the cube and read by every plane.
    let k_walk = k.over(&blocks);
    let k_probe = k.at(&k_walk.region(0usize));
    let v_probe = v.at(&k_walk.region(0usize));
    let mut k_stage = MemData::<E>::smem(
        comptime!(k_probe.space.clone()),
        1usize,
        StageStorage::Strided,
        0usize,
    );
    let mut v_stage = MemData::<E>::smem(
        comptime!(v_probe.space.clone()),
        1usize,
        StageStorage::Strided,
        0usize,
    );
    let bound_s = bound as usize;
    sync_cube();

    for plane in space.over(&comptime!(Level::planes(&[(QP, rows_p)]))) {
        let q_w = q_s.at(&plane);
        let mut score_w = score.at(&plane);
        let out_w = out.at(&plane);
        // The stages carry no query axis, so the plane's window is the whole block, one level
        // down like everything else it reads.
        let k_w = k_stage.at(&plane);
        let v_w = v_stage.at(&plane);
        let row_origin = plane.coord(QP) * rows_p;

        let mut state = RowState::<f32>::over_plane(comptime!(Space::new(&[(QP, rows_p)])), lanes);
        let share = comptime!(state.share);
        let mut acc = out_w
            .cmma_accumulator::<f32, f32>(
                &score_w,
                comptime!(Fragments {
                    m_tiles: rm,
                    n_tiles: vn,
                    m: frag,
                    n: frag,
                    k: frag,
                }),
                Monoid::Sum,
            )
            .with_scratch(planes, lanes);
        acc.zero();
        // The fragment grids of every operand, cells in row-major order.
        let acc_cells = out_w.over(&comptime!(Level::walk(&[(QP, frag), (V, frag)])));
        let p_cells = score_w.over(&comptime!(Level::walk(&[(QP, frag), (S, frag)])));
        let q_cells = q_w.over(&comptime!(Level::walk(&[(QP, frag), (D, frag)])));
        let k_cells = k_w.over(&comptime!(Level::walk(&[(S, frag), (D, frag)])));
        let v_cells = v_w.over(&comptime!(Level::walk(&[(S, frag), (V, frag)])));

        // The probe states the masking once, and the walk takes it as its bound: a block every
        // row masks throughout is one the walk never steps to, rather than two contractions
        // whose scores the softmax then discards. Its rows still mask per element, for the last
        // block's tail and its diagonal.
        let probe = MaskProbe {
            origin_q: 0,
            row_origin,
            origin_s: 0,
            bound_q: rows.runtime(),
            bound_s,
            q_rows: rows,
            causal,
            materialized: false,
        };

        for region in k.over(&blocks).window(0, probe.blocks(block)) {
            let s0 = region.coord(S) * block;
            let cols_bound = max(bound_s, s0) - s0;
            // Every plane is through the previous block before its stages are overwritten.
            sync_cube();
            k_stage.copy_from(&k.at(&region));
            v_stage.copy_from(&v.at(&region));
            sync_cube();

            // The score: `q · kᵀ`, the keys' window read col-major by the leaf.
            let mut s = score_w.cmma_accumulator::<f32, E>(
                &q_w,
                comptime!(Fragments {
                    m_tiles: rm,
                    n_tiles: cn,
                    m: frag,
                    n: frag,
                    k: frag,
                }),
                Monoid::Sum,
            );
            s.zero();
            #[unroll]
            for si in 0..ks {
                #[unroll]
                for mi in 0..rm {
                    #[unroll]
                    for ni in 0..cn {
                        let mut cell = s.at(&p_cells.region(comptime!(mi * cn + ni)));
                        cell.mma(
                            &q_w.at(&q_cells.region(comptime!(mi * ks + si))),
                            &k_w.at(&k_cells.region(comptime!(ni * ks + si))),
                            Semiring::SUM_PROD,
                        );
                    }
                }
            }
            #[unroll]
            for i in 0..comptime!(rm * cn) {
                let cell = p_cells.region(i);
                let mut window = score_w.at(&cell);
                window.copy_cast_from(&s.at(&cell));
            }
            sync_plane();

            let probe = probe.step_s(s0);
            let corr = score_w.softmax_in_place(&mut state, &probe, &mask_tile, scale);
            acc.rescale_rows(&corr, share);
            sync_plane();

            // The mix: `p · v`, steps at or past the prefix skipped so stale cache never rides
            // a zero probability.
            #[unroll]
            for si in 0..cn {
                if si * frag < cols_bound {
                    #[unroll]
                    for mi in 0..rm {
                        #[unroll]
                        for ni in 0..vn {
                            let mut cell = acc.at(&acc_cells.region(comptime!(mi * vn + ni)));
                            cell.mma(
                                &score_w.at(&p_cells.region(comptime!(mi * cn + si))),
                                &v_w.at(&v_cells.region(comptime!(si * vn + ni))),
                                Semiring::SUM_PROD,
                            );
                        }
                    }
                }
            }
            sync_plane();
        }

        let mut recip = Array::<f32>::new(rows_p);
        #[unroll]
        for ri in 0..rows_p {
            recip[ri] = state.recip_l(ri);
        }
        acc.rescale_rows(&recip, share);
        sync_plane();
        #[unroll]
        for i in 0..comptime!(rm * vn) {
            let cell = acc_cells.region(i);
            let mut window = out_w.at(&cell);
            window.copy_cast_from(&acc.at(&cell));
        }
    }
}

/// Launch the hardware fold: `planes` planes of `lanes`, each owning `rows / planes` rows, over
/// `block`-wide steps of `s_total` keys with `frag` fragments, and check against direct host
/// math.
#[allow(clippy::too_many_arguments)]
fn run_cmma<E: Float + CubeElement>(
    (rows, s_total, block, d, val_dim, frag): (usize, usize, usize, usize, usize, usize),
    bound_s: usize,
    causal: bool,
    spanned: bool,
    planes: usize,
    score_vec: usize,
) {
    let client: Client = cubecl::test_device().client();
    let hw = &client.properties().hardware;
    if hw.plane_size_min != hw.plane_size_max {
        TestOutcome::Validated(ValidationResult::Skipped(
            "a plane-owned fold needs a plane width the device commits to".into(),
        ))
        .enforce();
        return;
    }
    let lanes = hw.plane_size_min as usize;
    if lanes * planes > hw.max_units_per_cube as usize {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "{planes} planes of {lanes} do not fit one cube here"
        )))
        .enforce();
        return;
    }
    let f32_ty = f32::elem_type_native();
    let e_ty = E::elem_type_native();
    let supported = client.properties().features.matmul.cmma.iter().any(|cfg| {
        cfg.a_type == e_ty
            && cfg.b_type == e_ty
            && cfg.cd_type == f32_ty
            && (cfg.m as usize, cfg.n as usize, cfg.k as usize) == (frag, frag, frag)
    });
    let mix_supported = client.properties().features.matmul.cmma.iter().any(|cfg| {
        cfg.a_type == f32_ty
            && cfg.b_type == e_ty
            && cfg.cd_type == f32_ty
            && (cfg.m as usize, cfg.n as usize, cfg.k as usize) == (frag, frag, frag)
    });
    if !supported || !mix_supported {
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

    // The launch walks `S` in blocks and nothing else; the planes' cut on `QP` is the kernel's.
    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (G, 1),
                (QP, rows),
                (S, s_total),
                (D, d),
                (V, val_dim),
                (R, 1),
                (C, 1),
            ]),
            vec![Level::walk(&[
                (G, 1),
                (QP, rows),
                (S, block),
                (D, d),
                (V, val_dim),
                (R, 1),
                (C, 1),
            ])],
        ),
        KernelForm::Static,
    );
    let (k_axes, v_axes): (&[Axis], &[Axis]) = if spanned {
        (&[G, S, D], &[G, S, V])
    } else {
        (&[S, D], &[S, V])
    };

    attention_fold_cmma_kernel::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_2d(lanes as u32, planes as u32),
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
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[QP, V]),
        ),
        scale,
        bound_s as u32,
        launcher.partitioning_arg(),
        launcher.level(0),
        causal,
        block,
        frag,
        planes,
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

/// One fragment: 8 rows on one plane, one head-dim step, one block of 8 keys. A leaf reading
/// the wrong window comes out as zeros or as one wrong block.
#[test]
fn fold_cmma_single_fragment() {
    run_cmma::<f32>((8, 16, 8, 8, 8, 8), 16, false, false, 1, 1);
}

/// A grid on one plane: 16 rows, two fragments tall, the head dim and the block two steps deep.
/// A step dropped leaves half a dot product; a fragment stored to the wrong cell a wrong row.
#[test]
fn fold_cmma_fragment_grid() {
    run_cmma::<f32>((16, 32, 16, 16, 16, 8), 32, false, false, 1, 1);
}

/// The attended prefix ends inside a block: the mask probe owns the tail, score fragments that
/// straddle the bound are contracted whole, and only whole value steps past it are skipped.
#[test]
fn fold_cmma_causal_ragged_bound() {
    run_cmma::<f32>((16, 32, 16, 16, 16, 8), 24, true, false, 1, 1);
}

/// K and V carry an axis above the two the contraction reads, spanned at one position.
#[test]
fn fold_cmma_spanned_leading_axis() {
    run_cmma::<f32>((16, 32, 16, 16, 16, 8), 24, true, true, 1, 1);
}

/// Two planes, each owning eight of sixteen rows through the score, the softmax and the mix,
/// with the accumulator resident in fragments and rescaled by the plane's own corrections. A
/// plane reading another's rows, a correction applied to the wrong row, or a drain landing on
/// the wrong window comes out here as a wrong row.
#[test]
fn fold_cmma_two_planes() {
    run_cmma::<f32>((16, 32, 16, 16, 16, 8), 24, true, true, 2, 1);
}

/// Four planes, one fragment of rows each, the accumulator several fragments wide.
#[test]
fn fold_cmma_four_planes() {
    run_cmma::<f32>((32, 32, 16, 16, 32, 8), 32, true, false, 4, 1);
}

/// Half operands: `f16` queries, keys and values, scores and the running total at `f32`, the mix
/// contracting `f32` probabilities against `f16` values.
#[test]
fn fold_cmma_half_operands() {
    run_cmma::<half::f16>((16, 32, 16, 16, 16, 8), 24, true, true, 2, 1);
}

/// A lined score tile: the fragments store and load it through its element slice, and the
/// softmax reads a row a line at a time.
#[test]
fn fold_cmma_lined_scores() {
    run_cmma::<f32>((16, 32, 16, 16, 16, 8), 24, true, false, 2, 4);
}

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
    space: Partitioning,
    #[comptime] blocks: Level,
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
    let mask_tile = mask.tile(comptime!(space.clone()));

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
    let score_space = comptime!(Space::new(&[(R, split_rows), (C, block)]));
    // The split outermost gives a team one contiguous run of rows; innermost
    // gives it a strided column. Only the declared order differs: the cuts
    // below are the same either way, and so is every op that reads them.
    let row_extents = comptime!(if split_inner {
        [(R, rows), (T, splits)]
    } else {
        [(T, splits), (R, rows)]
    });
    let row_space = comptime!(Space::new(&row_extents));
    let acc_space = comptime!(Space::new(&[(R, split_rows), (V, val_dim)]));
    let score_all =
        MemData::<f32>::smem(score_space.clone(), 1usize, StageStorage::Strided, 0usize);
    let p_all = MemData::<f32>::smem(score_space, 1usize, StageStorage::Strided, 0usize);
    let mut factors_all =
        MemData::<f32>::smem(row_space.clone(), 1usize, StageStorage::Strided, 0usize);
    let m_all = MemData::<f32>::smem(row_space.clone(), 1usize, StageStorage::Strided, 0usize);
    let l_all = MemData::<f32>::smem(row_space.clone(), 1usize, StageStorage::Strided, 0usize);
    let mut acc_all = MemData::<f32>::smem(acc_space, 1usize, StageStorage::Strided, 0usize);
    acc_all.zero();

    // This team's windows: one slice of rows per team, the levels stated here on the
    // scratch spaces the kernel owns.
    let t = UNIT_POS_Y as usize;
    let team_scores = comptime!(Level::walk(&[(R, rows), (C, block)]));
    let tw = score_all.over(&team_scores);
    let mut score = score_all.at(&tw.region(t));
    let mut p = p_all.at(&tw.region(t));
    let team_rows = comptime!(Level::walk(&[(T, 1), (R, rows)]));
    let rw = factors_all.over(&team_rows);
    let mut m_win = m_all.at(&rw.region(t));
    let mut l_win = l_all.at(&rw.region(t));
    let team_acc = comptime!(Level::walk(&[(R, rows), (V, val_dim)]));
    let aw = acc_all.over(&team_acc);
    let mut acc = acc_all.at(&aw.region(t));

    let kept = comptime!(Space::new(&[(R, rows)]));
    let mut state = RowState::<f32>::new(kept, team);
    let share = comptime!(state.share);
    let bound_s = bound as usize;
    sync_cube();

    // Interleaved split walk: team t folds blocks t, t + splits, …; every
    // team runs every round (the barriers must stay uniform), an out-of-range
    // block just skips its compute.
    let probe = MaskProbe {
        origin_q: 0,
        row_origin: 0,
        origin_s: 0,
        bound_q: q_rows.runtime(),
        bound_s,
        q_rows,
        causal,
        materialized: false,
    };
    let k_walk = k.over(&blocks);
    // The same bound the plain fold walks to, so a causal split runs the rounds its rows can
    // read rather than every round the axis holds.
    let blocks = probe.blocks(block);
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
            let probe = probe.step_s(s0);
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

    // The one attention nest, as in [`run`].
    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (G, g),
                (QP, qp),
                (S, s_total),
                (D, d),
                (V, val_dim),
                (R, 1),
                (C, 1),
            ]),
            vec![Level::walk(&[
                (G, g),
                (QP, qp),
                (S, block),
                (D, d),
                (V, val_dim),
                (R, 1),
                (C, 1),
            ])],
        ),
        KernelForm::Static,
    );

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
        launcher.partitioning_arg(),
        launcher.level(0),
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
    space: Partitioning,
    #[comptime] blocks: Level,
    #[comptime] lanes: usize,
    #[comptime] splits: usize,
    #[comptime] block: usize,
) {
    let q = q.tile(comptime!(space.clone()));
    let k = k.tile(comptime!(space.clone()));
    let v = v.tile(comptime!(space.clone()));
    let mut out = out.tile(comptime!(space.clone()));

    let rank = comptime!(q.space.rank());
    let d = comptime!(q.space.extent_at(rank - 1));
    let rows = comptime!(q.space.tile_size() / d);

    let kept = comptime!(Space::new(&[(R, rows)]));
    let size!(N) = q.vector_size();
    let mut fold = StreamFold::<f32, N>::new(&q, lanes, kept);

    // This team's contiguous slice of the walk: no barriers anywhere.
    let t = UNIT_POS_Y as usize;
    let bound_s = bound as usize;
    let k_walk = k.over(&blocks);
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
    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(G, g), (QP, 1), (S, s_total), (D, d), (V, val_dim)]),
            vec![Level::walk(&[
                (G, g),
                (QP, 1),
                (S, block),
                (D, d),
                (V, val_dim),
            ])],
        ),
        KernelForm::Static,
    );

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
        launcher.partitioning_arg(),
        launcher.level(0),
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

/// What the probe's bound decides, measured rather than inferred from a result the per-element
/// masking would make right either way: the walk's own step count, written out.
#[cube(launch)]
fn visited_blocks_kernel(
    k: &TileArg<'_, f32, Const<1>>,
    out: &mut Tensor<f32>,
    bound: u32,
    space: Partitioning,
    #[comptime] blocks: Level,
    #[comptime] block: usize,
    #[comptime] q_rows: usize,
    #[comptime] causal: bool,
) {
    let k = k.tile(comptime!(space.clone()));
    let probe = MaskProbe {
        origin_q: 0,
        row_origin: 0,
        origin_s: 0,
        bound_q: comptime!(q_rows).runtime(),
        bound_s: bound as usize,
        q_rows,
        causal,
        materialized: false,
    };
    let mut visited = 0u32;
    for _region in k.over(&blocks).window(0, probe.blocks(block)) {
        visited += 1;
    }
    out[0] = f32::cast_from(visited);
}

/// `S` total, in blocks of `BLOCK`: four blocks to skip out of.
const VISIT_S: usize = 16;
const VISIT_BLOCK: usize = 4;
const VISIT_D: usize = 4;

fn visited_blocks(bound_s: usize, q_rows: usize, causal: bool) -> usize {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(S, VISIT_S), (D, VISIT_D)]),
            vec![Level::walk(&[(S, VISIT_BLOCK)])],
        ),
        KernelForm::Static,
    );

    let (k_handle, _) = TestInput::builder(client.clone(), Shape::new([VISIT_S, VISIT_D]))
        .dtype(f32_ty)
        .custom(vec![0.0; VISIT_S * VISIT_D])
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), Shape::new([1]))
        .dtype(f32_ty)
        .custom(vec![999.0])
        .generate_without_host_data();

    visited_blocks_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            k_handle.binding().into_tensor_arg(),
            TileSpec::direct(&[S, D]),
        ),
        out_handle.clone().binding().into_tensor_arg(),
        bound_s as u32,
        launcher.partitioning_arg(),
        launcher.level(0),
        VISIT_BLOCK,
        q_rows,
        causal,
    );

    HostData::from_tensor_handle(&client, out_handle, HostDataType::F32).get_f32(&[0]) as usize
}

/// A block every row masks throughout is never stepped to. Without the bound each case walks all
/// four blocks and the softmax throws the extra scores away, which is what this measures the
/// absence of.
#[test]
fn the_probe_bounds_the_walk_to_the_blocks_its_rows_can_read() {
    // Nothing masked: the whole axis, so the bound costs nothing.
    assert_eq!(visited_blocks(VISIT_S, VISIT_S, false), 4);
    // Causal with four query rows: only the first block holds a key any row may see.
    assert_eq!(visited_blocks(VISIT_S, 4, true), 1);
    // Causal with eight: two blocks.
    assert_eq!(visited_blocks(VISIT_S, 8, true), 2);
    // A ragged bound alone, no causality: six keys is two blocks, the second partial.
    assert_eq!(visited_blocks(6, VISIT_S, false), 2);
    // Both, the tighter one deciding.
    assert_eq!(visited_blocks(6, 4, true), 1);
    // Nothing to read: the walk takes no steps at all.
    assert_eq!(visited_blocks(0, VISIT_S, true), 0);
}
