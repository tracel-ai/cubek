//! The chunked packed-lhs register microkernel: a packed-u32 quantized **lhs** whose binding is
//! wider than one word per line. The served width is the binding width times the packing factor
//! (a four-word q4 line serves 32 values), which the fused dequant view cannot materialize as a
//! register line; the chunked walk loads whole word lines and decodes them in registers instead.

use cubecl::{TestRuntime, ir::ElemType, prelude::*};
use cubek_quant::scheme::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue};
use cubek_test_utils::{
    HostData, HostDataType, TestInput, TestOutcome, TileInput, ValidationResult,
    assert_equals_approx,
};
use cubek_tile::{
    Axis, ComputeScope, Coverage, Cut, DequantAt, Distribution, QuantTileArg, Schedule, Space,
    Spread, TileArg, Tiling, WalkOrder,
};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// `edge`-wide tiles spread across a plane's lanes with a pinned instance count — the same
/// constructor the production plane-fold plans use (`Cut::unit` would defer the count to the
/// launcher, where these tests derive their edges *from* it).
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

fn scheme(value: QuantValue, bm: usize, bk: usize) -> QuantScheme {
    QuantScheme::default()
        .with_level(QuantLevel::block([bm as u8, bk as u8]))
        .with_store(QuantStore::PackedU32(0))
        .with_value(value)
        .with_param(QuantParam::F32)
}

/// Whole-tile direct matmul with a packed lhs bound `words` u32 per line: q4 at four words
/// serves 32 values per line, the case the fused view could never register. One scale block
/// per line and multi-line blocks both run.
#[test]
fn register_matmul_packed_lhs_multiword_line() {
    // (value, words per line): served line = words * pack.
    run_packed_lhs_direct(QuantValue::Q4S, 4, 32); // 32-value line = one block
    run_packed_lhs_direct(QuantValue::Q4S, 2, 32); // 16-value line, two lines per block
    run_packed_lhs_direct(QuantValue::Q8S, 4, 32); // 16-value line, q8
    run_packed_lhs_direct(QuantValue::Q8S, 2, 32); // 8-value line
}

/// The multi-word binding still matches the fused view's numbers at one word per line: the
/// chunked walk replaces the old packed-lhs path wholesale, so the narrow case must not move.
#[test]
fn register_matmul_packed_lhs_single_word_line() {
    run_packed_lhs_direct(QuantValue::Q4S, 1, 32);
    run_packed_lhs_direct(QuantValue::Q8S, 1, 32);
}

fn run_packed_lhs_direct(value: QuantValue, words: usize, bk: usize) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let scheme = scheme(value, 1, bk);
    let pack = scheme.num_quants();
    let lw = words * pack;
    let (m, n, k) = (2usize, 4usize, 64usize);
    // The stored line has to tile the reduction, and line and scale block have
    // to nest — the two facts the chunked walk asserts at the leaf.
    assert!(k.is_multiple_of(lw));
    assert!(bk.is_multiple_of(lw) || lw.is_multiple_of(bk));

    let seq = |edge| Cut::sequential(edge);
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(M, seq(m)).axis(N, seq(n)).axis(K, seq(k))
        })
        .build();

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .packed(&scheme, DequantAt::Read)
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    let (i_dtype, e_dtype) = (u32::elem_type_native(), f32::elem_type_native());
    launch_matmul_quant_lhs::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        words,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
        i_dtype,
        e_dtype,
    );

    let got = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // Row-major arange rhs: b(p, j) = p·n + j; lhs value q[i·k + p] scaled by its K-block.
    let blocks = k / bk;
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| {
                    let q = a.q[i * k + p] as f32;
                    let scale = a.scale_values[i * blocks + p / bk];
                    q * scale * ((p * n + j) as f32)
                })
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, cubecl::zspace::shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&got, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// The production plane-fold shape in miniature: one output row per aligned 8-lane group, the
/// group's lanes taking interleaved 32-value chunks of `K` as four-word q4 lines — a packed lhs
/// under a `Group` lane share, whose partials the engine drains with the segmented butterfly.
/// This is exactly the space the quantized gemv's col arm emits.
#[test]
fn register_matmul_packed_lhs_plane_fold() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let lanes = client.properties().hardware.plane_size_max as usize;
    let group = 8usize;
    if lanes == 0 || !lanes.is_multiple_of(group) {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "plane width {lanes} does not divide into {group}-lane groups"
        )))
        .enforce();
        return;
    }

    let scheme = scheme(QuantValue::Q4S, 1, 32);
    let pack = scheme.num_quants();
    let words = 4usize;
    let lw = words * pack; // 32-value chunks, one scale block each
    let rows_per_plane = lanes / group;
    let (m, n, k) = (rows_per_plane, 1usize, lw * group * 2);

    let seq = |edge| Cut::sequential(edge);
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(M, Cut::cube(cubek_tile::CubeAxis::X, m))
                .axis(N, seq(1))
                .axis(K, seq(k))
        })
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(M, Cut::plane(m)).axis(N, seq(1)).axis(K, seq(k))
        })
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(M, unit(1, Spread::Contiguous, rows_per_plane))
                .axis(N, seq(1))
                .axis(K, unit(lw, Spread::Interleaved, group))
        })
        .build();

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .packed(&scheme, DequantAt::Read)
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    let (i_dtype, e_dtype) = (u32::elem_type_native(), f32::elem_type_native());
    launch_matmul_quant_lhs::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        words,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
        i_dtype,
        e_dtype,
    );

    let got = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // x(p, 0) = p (arange over [K, 1]); one scale per 32-value block of each row.
    let blocks = k / 32;
    let expected: Vec<f32> = (0..m)
        .map(|i| {
            (0..k)
                .map(|p| a.q[i * k + p] as f32 * a.scale_values[i * blocks + p / 32] * p as f32)
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, cubecl::zspace::shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&got, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// The plane-fold shape with the rhs bound **k-lined** — `x` as `[N, K]`, `K` innermost, four
/// values per line — the orientation the quantized gemv's col arm launches: the chunked walk
/// then loads whole x lines and issues vector fmas per decoded group instead of a scalar pair
/// per value. Must agree with the `[K, N]` scalar orientation to the same reference.
#[test]
fn register_matmul_packed_lhs_plane_fold_klined() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let lanes = client.properties().hardware.plane_size_max as usize;
    let group = 8usize;
    if lanes == 0 || !lanes.is_multiple_of(group) {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "plane width {lanes} does not divide into {group}-lane groups"
        )))
        .enforce();
        return;
    }

    for (value, words) in [(QuantValue::Q4S, 4usize), (QuantValue::Q8S, 4)] {
        let scheme = scheme(value, 1, 32);
        let pack = scheme.num_quants();
        let lw = words * pack;
        let xw = 4usize; // x line width along K
        let rows_per_plane = lanes / group;
        let (m, n, k) = (rows_per_plane, 1usize, lw * group * 2);

        let seq = |edge| Cut::sequential(edge);
        let space = Tiling::new()
            .extents(&[(M, m), (N, n), (K, k)])
            .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
                l.axis(M, Cut::cube(cubek_tile::CubeAxis::X, m))
                    .axis(N, seq(1))
                    .axis(K, seq(k))
            })
            .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
                l.axis(M, Cut::plane(m)).axis(N, seq(1)).axis(K, seq(k))
            })
            .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
                l.axis(M, unit(1, Spread::Contiguous, rows_per_plane))
                    .axis(N, seq(1))
                    .axis(K, unit(lw, Spread::Interleaved, group))
            })
            .build();

        let a = TileInput::builder(&client, space.project(&[M, K]))
            .untiled()
            .packed(&scheme, DequantAt::Read)
            .arange();
        // `[N, K]`: the same 1×k buffer, spanned with `K` innermost so the
        // binding lines along the contraction.
        let b = TileInput::builder(&client, space.project(&[N, K]))
            .untiled()
            .arange();
        let c = TileInput::builder(&client, space.project(&[M, N]))
            .untiled()
            .zeros();

        let (i_dtype, e_dtype) = (u32::elem_type_native(), f32::elem_type_native());
        launch_matmul_quant_lhs::launch::<TestRuntime>(
            &client,
            space.cube_count(),
            space.cube_dim(&client),
            words,
            xw,
            a.arg(),
            b.arg(),
            c.arg(),
            space,
            i_dtype,
            e_dtype,
        );

        let got = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
        let blocks = k / 32;
        let expected: Vec<f32> = (0..m)
            .map(|i| {
                (0..k)
                    .map(|p| a.q[i * k + p] as f32 * a.scale_values[i * blocks + p / 32] * p as f32)
                    .sum()
            })
            .collect();
        let (_, expected) = TestInput::builder(client.clone(), cubecl::zspace::shape![m, n])
            .custom(expected)
            .generate_with_f32_host_data();
        assert_equals_approx(&got, &expected, 1e-3)
            .as_test_outcome()
            .enforce()
    }
}

/// The plane-fold shape through the **resident** bracket: the accumulator promoted to its
/// register block before the walk, the packed contraction accumulating into it region after
/// region, one lane-share fold and one store at the drain — where the memory-backed walk pays a
/// butterfly and a read-modify-write per chunk. Pins two things at once: `promote` deriving the
/// `Group` share from the whole partitioner (the top level alone says `Whole`), and the register
/// block's own chunked packed walk. The output is poisoned, not zeroed: the drain overwrites.
#[test]
fn register_matmul_packed_lhs_plane_fold_resident() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let lanes = client.properties().hardware.plane_size_max as usize;
    let group = 8usize;
    if lanes == 0 || !lanes.is_multiple_of(group) {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "plane width {lanes} does not divide into {group}-lane groups"
        )))
        .enforce();
        return;
    }

    for (value, words, xw) in [
        (QuantValue::Q4S, 4usize, 4usize), // vectorized decode
        (QuantValue::Q8S, 4, 4),
        (QuantValue::Q4S, 4, 1), // scalar decode under the resident block
    ] {
        let scheme = scheme(value, 1, 32);
        let pack = scheme.num_quants();
        let lw = words * pack;
        let rows_per_plane = lanes / group;
        let (m, n, k) = (rows_per_plane, 1usize, lw * group * 2);

        let seq = |edge| Cut::sequential(edge);
        let space = Tiling::new()
            .extents(&[(M, m), (N, n), (K, k)])
            .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
                l.axis(M, Cut::cube(cubek_tile::CubeAxis::X, m))
                    .axis(N, seq(1))
                    .axis(K, seq(k))
            })
            .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
                l.axis(M, Cut::plane(m)).axis(N, seq(1)).axis(K, seq(k))
            })
            .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
                l.axis(M, unit(1, Spread::Contiguous, rows_per_plane))
                    .axis(N, seq(1))
                    .axis(K, unit(lw, Spread::Interleaved, group))
            })
            .build();

        let a = TileInput::builder(&client, space.project(&[M, K]))
            .untiled()
            .packed(&scheme, DequantAt::Read)
            .arange();
        let b = if xw == 1 {
            TileInput::builder(&client, space.project(&[K, N]))
                .untiled()
                .arange()
        } else {
            TileInput::builder(&client, space.project(&[N, K]))
                .untiled()
                .arange()
        };
        // Poisoned: the resident drain must overwrite, not accumulate.
        let c = TileInput::builder(&client, space.project(&[M, N]))
            .untiled()
            .uniform(4242, 10., 100.);

        let (i_dtype, e_dtype) = (u32::elem_type_native(), f32::elem_type_native());
        launch_matmul_quant_lhs_resident::launch::<TestRuntime>(
            &client,
            space.cube_count(),
            space.cube_dim(&client),
            words,
            xw,
            a.arg(),
            b.arg(),
            c.arg(),
            space,
            i_dtype,
            e_dtype,
        );

        let got = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
        let blocks = k / 32;
        let expected: Vec<f32> = (0..m)
            .map(|i| {
                (0..k)
                    .map(|p| a.q[i * k + p] as f32 * a.scale_values[i * blocks + p / 32] * p as f32)
                    .sum()
            })
            .collect();
        let (_, expected) = TestInput::builder(client.clone(), cubecl::zspace::shape![m, n])
            .custom(expected)
            .generate_with_f32_host_data();
        assert_equals_approx(&got, &expected, 1e-3)
            .as_test_outcome()
            .enforce()
    }
}

/// `I` names the binding's stored element (`u32` words); the arg's scheme recovers the served
/// values, so the body is the plain matmul kernel. `V` is the binding width in words; `VX` the
/// rhs's line width (1 for the plain `[K, N]` orientation, the group width for a k-lined rhs).
#[cube(launch)]
fn launch_matmul_quant_lhs<I: Numeric, E: Numeric, V: Size, VX: Size>(
    a: &QuantTileArg<'_, I, V>,
    b: &TileArg<'_, E, VX>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(I)] _idtype: ElemType,
    #[define(E)] _edtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.mma(&a, &b);
}

/// The resident twin of [`launch_matmul_quant_lhs`]: promote, zero, contract, drain — the
/// production col gemv's bracket.
#[cube(launch)]
fn launch_matmul_quant_lhs_resident<I: Numeric, E: Numeric, V: Size, VX: Size>(
    a: &QuantTileArg<'_, I, V>,
    b: &TileArg<'_, E, VX>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(I)] _idtype: ElemType,
    #[define(E)] _edtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    let mut acc = c.promote::<E, E>(&a);
    acc.zero();
    acc.mma(&a, &b);
    acc.drain_cast_into(&mut c);
}
