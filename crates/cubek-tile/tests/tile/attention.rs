//! The attention fold end-to-end on tiles: the score and mix leaves
//! ([`ops::attention`]) interleaved with the softmax step, the walk owned by
//! the kernel — the miniature of the routine a client (metabolic) launches.
//! GQA rides the axes: `q` stacks its group over the same query block
//! (group-major rows), `k`/`v` simply omit the group axis, and the probe's
//! `q_rows` maps rows back to query positions for the causal predicate.

use cubecl::{Runtime, TestRuntime, client::ComputeClient, prelude::*, zspace::Shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::{
    Axis, Cut, Leaf, MaskProbe, MemData, RowState, Schedule, Space, StagePlan, Storage,
    StridedTileArg, StridedTileArgLaunch, Tiling, Walk, WalkOrder,
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
fn attention_fold_kernel(
    q: &StridedTileArg<'_, f32>,    // {G, QP, D}
    k: &StridedTileArg<'_, f32>,    // {S, D} — omits the group axis
    v: &StridedTileArg<'_, f32>,    // {S, V}
    mask: &StridedTileArg<'_, u32>, // 1-cell dummy (materialized = false)
    out: &mut Tensor<f32>,          // [G·QP·V] flat
    scale: f32,
    bound: u32,
    #[comptime] units: usize,
    #[comptime] causal: bool,
    #[comptime] block: usize,
) {
    let q = q.tile();
    let k = k.tile();
    let v = v.tile();
    let mask_tile = mask.tile();

    let rows = comptime!(q.space.extent(G) * q.space.extent(QP));
    let q_rows = comptime!(q.space.extent(QP));
    let val_dim = comptime!(v.space.extent(V));

    // The stage: q resident in smem for the whole walk (read cols-fold by the
    // score leaf), score/p/factors/acc the fold's working set.
    let mut q_s = MemData::<f32>::smem(
        comptime!(q.space.clone()),
        q.vector_size(),
        comptime!(StagePlan::strided()),
    );
    q_s.copy_from(&q);
    let score_space = comptime!(Space::new(&[(R, rows), (C, block)]));
    let mut score =
        MemData::<f32>::smem(score_space.clone(), 1usize, comptime!(StagePlan::strided()));
    let mut p = MemData::<f32>::smem(score_space, 1usize, comptime!(StagePlan::strided()));
    let row_space = comptime!(Space::new(&[(R, rows)]));
    let mut factors =
        MemData::<f32>::smem(row_space.clone(), 1usize, comptime!(StagePlan::strided()));
    let acc_space = comptime!(Space::new(&[(R, rows), (V, val_dim)]));
    let mut acc = MemData::<f32>::smem(acc_space, 1usize, comptime!(StagePlan::strided()));
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

        score.score_columns(&q_s, &kb);
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

        acc.scale_rows(&factors);
        sync_cube();

        // Clip the ragged tail: stale cache beyond the attended prefix must
        // not ride a zero probability into the accumulator.
        let cols_bound = max(bound_s, s0) - s0;
        acc.mix_columns(&p, &vb, cols_bound);
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
        out[i] = acc_flat.read(i).extract(0);
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

    let f32_ty = f32::as_type_native_unchecked().storage_type();
    let u32_ty = u32::as_type_native_unchecked().storage_type();
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

    // q is final (no walk of its own); k/v walk S in blocks.
    let q_space = Space::new(&[(G, g), (QP, qp), (D, d)]);
    let k_space = Tiling::new()
        .extents(&[(S, s_total), (D, d)])
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(S, Cut::sequential(block))
                .axis(D, Cut::sequential(d))
        })
        .leaf(Leaf::Register);
    let v_space = Tiling::new()
        .extents(&[(S, s_total), (V, val_dim)])
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(S, Cut::sequential(block))
                .axis(V, Cut::sequential(val_dim))
        })
        .leaf(Leaf::Register);
    let mask_space = Space::new(&[(R, 1), (C, 1)]);

    attention_fold_kernel::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_2d(units as u32, 1),
        StridedTileArgLaunch::strided(
            q_handle.clone().binding().into_tensor_arg(),
            vec,
            q_space,
            Storage::of(3, 3),
        ),
        StridedTileArgLaunch::strided(
            k_handle.clone().binding().into_tensor_arg(),
            vec,
            k_space,
            Storage::of(2, 2),
        ),
        StridedTileArgLaunch::strided(
            v_handle.clone().binding().into_tensor_arg(),
            vec,
            v_space,
            Storage::of(2, 2),
        ),
        StridedTileArgLaunch::strided(
            mask_handle.clone().binding().into_tensor_arg(),
            1,
            mask_space,
            Storage::of(2, 2),
        ),
        out_handle.clone().binding().into_tensor_arg(),
        scale,
        bound_s as u32,
        units,
        causal,
        block,
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
