//! Numerics tests for the attention softmax leaf.
//!
//! The kernel below is a miniature of the future Fold body: walk S blocks,
//! stage each score block into an smem tile, run `Tile::softmax`,
//! rescale a running accumulator by the returned correction (val_dim = 1
//! surrogate for the value matmul), then the epilogue normalizes by `l` and
//! stores `lse = m + ln(l)`. Checked against direct (non-online) host math,
//! including exact zeros and exact -inf lse on fully-masked rows.

use cubecl::std::tensor::layout::CoordsDyn;
use cubecl::{Runtime, TestRuntime, client::ComputeClient, prelude::*, zspace::Shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::{
    Axis, MaskProbe, MemData, RowState, Space, StagePlan, Storage, StridedTileArg,
    StridedTileArgLaunch,
};

const Q: Axis = Axis(0);
const S: Axis = Axis(1);

#[cube(launch)]
fn softmax_walk_kernel(
    score_in: &StridedTileArg<'_, f32>, // {Q: rows, S: total_cols} raw scores
    mask: &StridedTileArg<'_, u32>,     // {Q, S} boolean, nonzero = masked
    values: &Tensor<f32>,               // [total_cols]
    out: &mut Tensor<f32>,              // [rows]
    lse: &mut Tensor<f32>,              // [rows]
    scale: f32,
    bound_s: u32,
    #[comptime] block_space: Space, // {Q: rows, S: block cols}
    #[comptime] units: usize,
    #[comptime] causal: bool,
    #[comptime] materialized: bool,
    #[comptime] num_blocks: usize,
) {
    let score_gmem = score_in.tile();
    let mask_tile = mask.tile();
    let mut score =
        MemData::<f32>::smem(block_space.clone(), 1usize, comptime!(StagePlan::strided()));
    let mut p = MemData::<f32>::smem(block_space.clone(), 1usize, comptime!(StagePlan::strided()));

    let rows = comptime!(block_space.extent(Q));
    let cols = comptime!(block_space.extent(S));
    let kept_space = comptime!(Space::new(&[(Q, rows)]));
    let mut state = RowState::<f32>::new(kept_space, units);
    let rpu = comptime!(state.rows_per_unit);
    let mut acc = Array::<f32>::new(rpu);
    for ri in 0..rpu {
        acc[ri] = 0.0;
    }

    for blk in 0..num_blocks {
        // Stage the block: each unit fills its own rows, so no syncs.
        let gmem = score_gmem.view::<Const<1>>();
        let mut smem = score.view_mut::<Const<1>>();
        for ri in 0..rpu {
            let r = UNIT_POS_X as usize * rpu + ri;
            if r < rows {
                for c in 0..cols {
                    let mut src = CoordsDyn::new();
                    src.push(r as u32);
                    src.push((blk * cols + c) as u32);
                    let mut dst = CoordsDyn::new();
                    dst.push(r as u32);
                    dst.push(c as u32);
                    smem.write(dst, gmem.read(src));
                }
            }
        }

        let probe = MaskProbe {
            origin_q: 0,
            origin_s: blk * cols,
            bound_q: rows,
            bound_s: bound_s as usize,
            q_rows: rows,
            causal,
            materialized,
        };
        let corr = score.softmax::<f32>(&mut p, &mut state, &probe, &mask_tile, scale);

        // The block update `O = corr·O + P·V`, on a scalar accumulator.
        let p_view = p.view::<Const<1>>();
        for ri in 0..rpu {
            let r = UNIT_POS_X as usize * rpu + ri;
            if r < rows {
                acc[ri] *= corr[ri];
                for c in 0..cols {
                    let mut pos = CoordsDyn::new();
                    pos.push(r as u32);
                    pos.push(c as u32);
                    acc[ri] += p_view.read(pos).extract(0) * values[blk * cols + c];
                }
            }
        }
    }

    for ri in 0..rpu {
        let r = UNIT_POS_X as usize * rpu + ri;
        if r < rows {
            out[r] = acc[ri] * state.recip_l(ri);
            lse[r] = state.lse(ri);
        }
    }
}

/// Launch the walk kernel and check out/lse against direct host math.
fn run(
    (units, rows, cols, num_blocks): (usize, usize, usize, usize),
    bound_s: usize,
    causal: bool,
    mask_fn: Option<fn(usize, usize) -> bool>,
) {
    let client: ComputeClient<TestRuntime> = <TestRuntime as Runtime>::client(&Default::default());
    let total_cols = cols * num_blocks;
    let scale = 0.125f32;

    // The CPU runtime caps units per cube at core count; ownership only needs
    // a consistent unit count, so clamp (rows_per_unit grows to compensate).
    let units = units.min(client.properties().hardware.max_units_per_cube as usize);

    let f32_ty = f32::as_type_native_unchecked().storage_type();
    let u32_ty = u32::as_type_native_unchecked().storage_type();

    // Deterministic host-built data so device and host math see identical bits.
    let wobble =
        |i: usize, salt: usize| ((i * 2654435761 + salt * 40503) % 2048) as f32 / 512. - 2.;
    let (score_handle, score_data) =
        TestInput::builder(client.clone(), Shape::new([rows, total_cols]))
            .dtype(f32_ty)
            .custom((0..rows * total_cols).map(|i| wobble(i, 1)).collect())
            .generate_with_f32_host_data();

    let mask_values: Vec<f32> = (0..rows * total_cols)
        .map(|lin| {
            let (i, j) = (lin / total_cols, lin % total_cols);
            mask_fn.map(|f| f(i, j)).unwrap_or(false) as u32 as f32
        })
        .collect();
    let mask_handle = TestInput::builder(client.clone(), Shape::new([rows, total_cols]))
        .dtype(u32_ty)
        .custom(mask_values)
        .generate_without_host_data();

    let (values_handle, values_data) = TestInput::builder(client.clone(), Shape::new([total_cols]))
        .dtype(f32_ty)
        .custom((0..total_cols).map(|i| wobble(i, 2) / 2.).collect())
        .generate_with_f32_host_data();

    let out_handle = TestInput::builder(client.clone(), Shape::new([rows]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();
    let lse_handle = TestInput::builder(client.clone(), Shape::new([rows]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let gmem_space = Space::new(&[(Q, rows), (S, total_cols)]);
    let block_space = Space::new(&[(Q, rows), (S, cols)]);

    softmax_walk_kernel::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        // Explicit x = units so UNIT_POS_X is the owner index on every
        // backend (CubeDim::new packs by plane size: y-major on CPU).
        CubeDim::new_2d(units as u32, 1),
        StridedTileArgLaunch::strided(
            score_handle.clone().binding().into_tensor_arg(),
            1,
            gmem_space.clone(),
            Storage::of(2, 2),
        ),
        StridedTileArgLaunch::strided(
            mask_handle.clone().binding().into_tensor_arg(),
            1,
            gmem_space,
            Storage::of(2, 2),
        ),
        values_handle.clone().binding().into_tensor_arg(),
        out_handle.clone().binding().into_tensor_arg(),
        lse_handle.clone().binding().into_tensor_arg(),
        scale,
        bound_s as u32,
        block_space,
        units,
        causal,
        mask_fn.is_some(),
        num_blocks,
    );

    let out = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    let lse = HostData::from_tensor_handle(&client, lse_handle, HostDataType::F32);

    // Guard against the cubecl-cpu smem-aliases-input bug: the kernel never
    // writes its inputs, so any change means Shared aliased a live binding.
    let score_after = HostData::from_tensor_handle(&client, score_handle, HostDataType::F32);
    let values_after = HostData::from_tensor_handle(&client, values_handle, HostDataType::F32);
    let _keep_alive = mask_handle;
    for i in 0..rows {
        for j in 0..total_cols {
            assert_eq!(
                score_after.get_f32(&[i, j]),
                score_data.get_f32(&[i, j]),
                "input score corrupted at ({i},{j}): smem aliased an input binding"
            );
        }
    }
    for j in 0..total_cols {
        assert_eq!(
            values_after.get_f32(&[j]),
            values_data.get_f32(&[j]),
            "input values corrupted at {j}: smem aliased an input binding"
        );
    }

    for i in 0..rows {
        // Direct math: gather unmasked scaled scores, then one-pass softmax.
        let mut scores = Vec::new();
        for j in 0..total_cols {
            let oob = j >= bound_s;
            let causal_masked = causal && j > i;
            let mat_masked = mask_fn.map(|f| f(i, j)).unwrap_or(false);
            if !(oob || causal_masked || mat_masked) {
                scores.push((
                    score_data.get_f32(&[i, j]) * scale,
                    values_data.get_f32(&[j]),
                ));
            }
        }

        let got_out = out.get_f32(&[i]);
        let got_lse = lse.get_f32(&[i]);

        if scores.is_empty() {
            assert_eq!(got_out, 0., "fully-masked row {i}: out must be exactly 0");
            assert_eq!(
                got_lse,
                f32::NEG_INFINITY,
                "fully-masked row {i}: lse must be exactly -inf"
            );
            continue;
        }

        let m = scores.iter().fold(f32::NEG_INFINITY, |m, (s, _)| m.max(*s));
        let l: f32 = scores.iter().map(|(s, _)| (s - m).exp()).sum();
        let expected_lse = m + l.ln();
        let expected_out: f32 = scores.iter().map(|(s, v)| (s - m).exp() / l * v).sum();

        assert!(
            (got_out - expected_out).abs() <= 1e-5 * expected_out.abs().max(1.),
            "row {i}: out {got_out} vs direct {expected_out}"
        );
        assert!(
            (got_lse - expected_lse).abs() <= 1e-5 * expected_lse.abs().max(1.),
            "row {i}: lse {got_lse} vs direct {expected_lse}"
        );
    }
}

const V: Axis = Axis(2);

/// The fold with a shared-memory `{Q, V}` accumulator, the shape the attention
/// routine takes: softmax row owners publish their correction through a rank-1
/// factors tile ([`store_rows`](cubek_tile::Tile)), the whole cube rescales the
/// accumulator ([`scale_rows`](cubek_tile::Tile)), and the value accumulate
/// runs under a *different* ownership (cyclic) than the softmax rows — the
/// cross-unit handoff the smem path exists for.
#[cube(launch)]
fn softmax_smem_acc_kernel(
    score_in: &StridedTileArg<'_, f32>, // {Q: rows, S: total_cols} raw scores
    mask: &StridedTileArg<'_, u32>,     // unused (materialized = false)
    values: &Tensor<f32>,               // [total_cols, val_dim] row-major
    out: &mut Tensor<f32>,              // [rows, val_dim] row-major
    lse: &mut Tensor<f32>,              // [rows]
    scale: f32,
    bound_s: u32,
    #[comptime] block_space: Space, // {Q: rows, S: block cols}
    #[comptime] units: usize,
    #[comptime] causal: bool,
    #[comptime] num_blocks: usize,
    #[comptime] val_dim: usize,
) {
    let score_gmem = score_in.tile();
    let mask_tile = mask.tile();
    let mut score =
        MemData::<f32>::smem(block_space.clone(), 1usize, comptime!(StagePlan::strided()));
    let mut p = MemData::<f32>::smem(block_space.clone(), 1usize, comptime!(StagePlan::strided()));

    let rows = comptime!(block_space.extent(Q));
    let cols = comptime!(block_space.extent(S));
    let kept_space = comptime!(Space::new(&[(Q, rows)]));
    let mut state = RowState::<f32>::new(kept_space.clone(), units);
    let rpu = comptime!(state.rows_per_unit);

    let mut factors = MemData::<f32>::smem(kept_space, 1usize, comptime!(StagePlan::strided()));
    let acc_space = comptime!(Space::new(&[(Q, rows), (V, val_dim)]));
    let mut acc = MemData::<f32>::smem(acc_space, 1usize, comptime!(StagePlan::strided()));
    acc.zero();

    for blk in 0..num_blocks {
        // Stage the block: each unit fills its own rows.
        let gmem = score_gmem.view::<Const<1>>();
        let mut smem = score.view_mut::<Const<1>>();
        for ri in 0..rpu {
            let r = UNIT_POS_X as usize * rpu + ri;
            if r < rows {
                for c in 0..cols {
                    let mut src = CoordsDyn::new();
                    src.push(r as u32);
                    src.push((blk * cols + c) as u32);
                    let mut dst = CoordsDyn::new();
                    dst.push(r as u32);
                    dst.push(c as u32);
                    smem.write(dst, gmem.read(src));
                }
            }
        }

        let probe = MaskProbe {
            origin_q: 0,
            origin_s: blk * cols,
            bound_q: rows,
            bound_s: bound_s as usize,
            q_rows: rows,
            causal,
            materialized: false,
        };
        let corr = score.softmax::<f32>(&mut p, &mut state, &probe, &mask_tile, scale);
        factors.store_rows(&corr, rpu);
        sync_cube();

        // `O = corr·O`, every cell exactly once, whoever owns the rows.
        acc.scale_rows(&factors);
        sync_cube();

        // `O += P·V_blk`, cyclic ownership — deliberately not the softmax rows'.
        let p_view = p.view::<Const<1>>();
        let mut acc_view = acc.view_mut::<Const<1>>();
        let total = comptime!(rows * val_dim);
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            let r = i / val_dim;
            let v = i % val_dim;
            let mut pos = CoordsDyn::new();
            pos.push(r as u32);
            pos.push(v as u32);
            let mut cell = acc_view.read(pos.clone());
            for c in 0..cols {
                let mut ppos = CoordsDyn::new();
                ppos.push(r as u32);
                ppos.push(c as u32);
                let prob = p_view.read(ppos).extract(0);
                cell += Vector::cast_from(prob * values[(blk * cols + c) * val_dim + v]);
            }
            acc_view.write(pos, cell);
            i += workers;
        }
        sync_cube();
    }

    // Epilogue: owners publish `1/l`, the cube normalizes, everyone drains.
    let mut recip = Array::<f32>::new(rpu);
    for ri in 0..rpu {
        recip[ri] = state.recip_l(ri);
    }
    factors.store_rows(&recip, rpu);
    sync_cube();
    acc.scale_rows(&factors);
    sync_cube();

    let acc_view = acc.view::<Const<1>>();
    let total = comptime!(rows * val_dim);
    let workers = CUBE_DIM as usize;
    let mut i = UNIT_POS as usize;
    while i < total {
        let mut pos = CoordsDyn::new();
        pos.push((i / val_dim) as u32);
        pos.push((i % val_dim) as u32);
        out[i] = acc_view.read(pos).extract(0);
        i += workers;
    }
    for ri in 0..rpu {
        let r = UNIT_POS_X as usize * rpu + ri;
        if r < rows {
            lse[r] = state.lse(ri);
        }
    }
}

/// Launch the smem-accumulator kernel and check out/lse against direct host math.
fn run_smem_acc(
    (units, rows, cols, num_blocks, val_dim): (usize, usize, usize, usize, usize),
    bound_s: usize,
    causal: bool,
) {
    let client: ComputeClient<TestRuntime> = <TestRuntime as Runtime>::client(&Default::default());
    let total_cols = cols * num_blocks;
    let scale = 0.125f32;
    let units = units.min(client.properties().hardware.max_units_per_cube as usize);

    let f32_ty = f32::as_type_native_unchecked().storage_type();
    let u32_ty = u32::as_type_native_unchecked().storage_type();

    let wobble =
        |i: usize, salt: usize| ((i * 2654435761 + salt * 40503) % 2048) as f32 / 512. - 2.;
    let (score_handle, score_data) =
        TestInput::builder(client.clone(), Shape::new([rows, total_cols]))
            .dtype(f32_ty)
            .custom((0..rows * total_cols).map(|i| wobble(i, 1)).collect())
            .generate_with_f32_host_data();
    let mask_handle = TestInput::builder(client.clone(), Shape::new([rows, total_cols]))
        .dtype(u32_ty)
        .zeros()
        .generate_without_host_data();
    let (values_handle, values_data) =
        TestInput::builder(client.clone(), Shape::new([total_cols, val_dim]))
            .dtype(f32_ty)
            .custom(
                (0..total_cols * val_dim)
                    .map(|i| wobble(i, 2) / 2.)
                    .collect(),
            )
            .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), Shape::new([rows, val_dim]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();
    let lse_handle = TestInput::builder(client.clone(), Shape::new([rows]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let gmem_space = Space::new(&[(Q, rows), (S, total_cols)]);
    let block_space = Space::new(&[(Q, rows), (S, cols)]);

    softmax_smem_acc_kernel::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_2d(units as u32, 1),
        StridedTileArgLaunch::strided(
            score_handle.clone().binding().into_tensor_arg(),
            1,
            gmem_space.clone(),
            Storage::of(2, 2),
        ),
        StridedTileArgLaunch::strided(
            mask_handle.clone().binding().into_tensor_arg(),
            1,
            gmem_space,
            Storage::of(2, 2),
        ),
        values_handle.clone().binding().into_tensor_arg(),
        out_handle.clone().binding().into_tensor_arg(),
        lse_handle.clone().binding().into_tensor_arg(),
        scale,
        bound_s as u32,
        block_space,
        units,
        causal,
        num_blocks,
        val_dim,
    );

    let out = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    let lse = HostData::from_tensor_handle(&client, lse_handle, HostDataType::F32);

    for i in 0..rows {
        let mut scores = Vec::new();
        for j in 0..total_cols {
            let oob = j >= bound_s;
            let causal_masked = causal && j > i;
            if !(oob || causal_masked) {
                scores.push((score_data.get_f32(&[i, j]) * scale, j));
            }
        }

        let got_lse = lse.get_f32(&[i]);
        if scores.is_empty() {
            assert_eq!(
                got_lse,
                f32::NEG_INFINITY,
                "fully-masked row {i}: lse must be exactly -inf"
            );
            for v in 0..val_dim {
                assert_eq!(
                    out.get_f32(&[i, v]),
                    0.,
                    "fully-masked row {i}: out must be exactly 0"
                );
            }
            continue;
        }

        let m = scores.iter().fold(f32::NEG_INFINITY, |m, (s, _)| m.max(*s));
        let l: f32 = scores.iter().map(|(s, _)| (s - m).exp()).sum();
        let expected_lse = m + l.ln();
        assert!(
            (got_lse - expected_lse).abs() <= 1e-5 * expected_lse.abs().max(1.),
            "row {i}: lse {got_lse} vs direct {expected_lse}"
        );
        for v in 0..val_dim {
            let expected: f32 = scores
                .iter()
                .map(|(s, j)| (s - m).exp() / l * values_data.get_f32(&[*j, v]))
                .sum();
            let got = out.get_f32(&[i, v]);
            assert!(
                (got - expected).abs() <= 1e-5 * expected.abs().max(1.),
                "row ({i},{v}): out {got} vs direct {expected}"
            );
        }
    }
}

/// The decode shape: a handful of GQA rows, a full plane, deep-ish walk.
#[test]
fn smem_acc_decode_rows() {
    run_smem_acc((32, 4, 16, 4, 8), 60, false);
}

/// Causal with several rows per unit and a val_dim wider than the plane.
#[test]
fn smem_acc_causal_multi_rows() {
    run_smem_acc((8, 16, 8, 3, 40), 20, true);
}

/// Rows 0..2 fully masked, row 2 first-half, row 3 second-half (running max
/// starts at -inf / all-masked late blocks), scattered elsewhere.
fn crafted_mask(i: usize, j: usize) -> bool {
    i < 2 || (i == 2 && j < 16) || (i == 3 && j >= 16) || (i + j).is_multiple_of(5)
}

/// One unit owns every row: the unit/CPU shape (`SoftmaxKind::Direct` twin).
#[test]
fn direct_unit_masked() {
    run((1, 8, 8, 4), 27, false, Some(crafted_mask));
}

/// One row per unit across a full plane.
#[test]
fn plane_masked() {
    run((32, 32, 16, 2), 30, false, Some(crafted_mask));
}

/// Several rows per unit (rows/units = 2).
#[test]
fn multi_rows_per_unit_masked_causal() {
    run((4, 8, 8, 2), 14, true, Some(crafted_mask));
}

#[test]
fn causal_only() {
    run((16, 16, 8, 2), 16, true, None);
}

/// Causal with bound past the diagonal: early rows see one unmasked block
/// then fully-masked blocks; late rows the reverse.
#[test]
fn causal_odd_bound() {
    run((8, 8, 4, 3), 10, true, None);
}
