//! A contracted axis spelled as two: `k = kb · B + ki`, the axis partitioned into blocks.
//!
//! Nothing here is quantized. The point is that a *partition* of an axis is not a gather — the
//! windows tile instead of overlapping, so a logical position still determines a cell and every
//! window is still a dense box. An operand spanning `(KB, KI)` must therefore read exactly as one
//! spanning `K` does, and cost the same.
//!
//! Why an axis would be split at all: an axis exists when an operand *distinguishes* it. A
//! quantized operand's scales vary over the block index and not over the position inside the
//! block, so those are two axes. The first tests here pin the split with no scales at all, and
//! the last one adds them: a scales operand that simply omits the axis inside the block.

use cubecl::{Runtime, TestRuntime, prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
/// The contracted axis, unsplit: what the reference kernel walks.
const K: Axis = Axis(2);
/// Which block, and where inside it. Together they are `K`.
const KB: Axis = Axis(3);
const KI: Axis = Axis(4);

const ROWS: usize = 4;
const COLS: usize = 4;
const BLOCK: usize = 8;
const BLOCKS: usize = 4;
const DEPTH: usize = BLOCK * BLOCKS;

/// `k = kb · BLOCK + ki`: the physical contracted axis as the two logical ones' digits.
fn blocked() -> PhysicalAxisMap {
    PhysicalAxisMap::disjoint(&[(KB, BLOCK), (KI, 1)])
}

fn lhs_data() -> Vec<f32> {
    (0..ROWS * DEPTH).map(|i| (i % 5) as f32 - 2.0).collect()
}

fn rhs_data() -> Vec<f32> {
    (0..DEPTH * COLS).map(|i| (i % 7) as f32 - 3.0).collect()
}

/// `C = A · B`, whichever axes the operands spell their contraction over.
#[cube(launch)]
fn matmul<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.mm(&a, &b, Semiring::SUM_PROD);
}

/// Launch [`matmul`] with the operands' specs and the space the caller states.
fn run(space: Space, a_spec: TileSpec, b_spec: TileSpec) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();

    let (a, _) = TestInput::builder(client.clone(), shape![ROWS, DEPTH])
        .dtype(dtype)
        .custom(lhs_data())
        .generate_with_f32_host_data();
    let (b, _) = TestInput::builder(client.clone(), shape![DEPTH, COLS])
        .dtype(dtype)
        .custom(rhs_data())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(a.binding().into_tensor_arg(), a_spec),
        TileArgLaunch::new(b.binding().into_tensor_arg(), b_spec),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        space,
        dtype,
    );

    HostData::from_tensor_handle(&client, c, HostDataType::F32)
}

/// `C[m,n] = Σ_k A[m,k] · B[k,n]`.
fn assert_matmul(got: &HostData) {
    let (a, b) = (lhs_data(), rhs_data());
    for m in 0..ROWS {
        for n in 0..COLS {
            let want: f32 = (0..DEPTH).map(|k| a[m * DEPTH + k] * b[k * COLS + n]).sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The reference: one contracted axis, cut at the block. What the split has to reproduce.
#[test]
fn one_contracted_axis_is_the_reference() {
    let space = Tiling::new()
        .extents(&[(M, ROWS), (N, COLS), (K, DEPTH)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(ROWS))
                .axis(N, Cut::sequential(COLS))
                .axis(K, Cut::sequential(BLOCK))
        })
        .build()
        .with_instruction(Instruction::registers(16));

    assert_matmul(&run(
        space,
        TileSpec::direct(&[M, K]),
        TileSpec::direct(&[K, N]),
    ));
}

/// The same contraction with `K` spelled `(KB, KI)`, the walk stepping one block per region so
/// the leaf spans one. Same buffers, same numbers.
#[test]
fn a_partitioned_axis_contracts_the_same() {
    let space = Tiling::new()
        .extents(&[(M, ROWS), (N, COLS), (KB, BLOCKS), (KI, BLOCK)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(ROWS))
                .axis(N, Cut::sequential(COLS))
                .axis(KB, Cut::sequential(1))
                .axis(KI, Cut::sequential(BLOCK))
        })
        .build()
        .with_instruction(Instruction::registers(16));

    let a_spec = TileSpec::new(Projection::new(
        &[M, KB, KI],
        &[PhysicalAxisMap::of(M), blocked()],
    ));
    let b_spec = TileSpec::new(Projection::new(
        &[KB, KI, N],
        &[blocked(), PhysicalAxisMap::of(N)],
    ));

    assert_matmul(&run(space, a_spec, b_spec));
}

// A partition whose stride misses its block is refused where the projection and the space first
// meet (`Projection::validate_composition`). That refusal is unit-tested host-side in
// `physical::projection::base`, not here: this one would fire inside the kernel, on a worker
// thread, where `#[should_panic]` never sees it and the launch just returns zeros.

/// `c = (a ⊗ s) · b` with the scales spanning the block axis and *not* the axis inside it.
#[cube(launch)]
fn scaled_matmul<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    scales: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let scales = scales.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.mm_scaled(&a, &b, &scales, Semiring::SUM_PROD);
}

/// One scale per `(row, block)`, halves so the reference is exact.
fn scale_data() -> Vec<f32> {
    (0..ROWS * BLOCKS).map(|i| (i as f32 + 1.0) / 2.0).collect()
}

/// **The split, with scales.** `S` spans `(M, KB)` and omits `KI`, so it cannot vary inside a
/// block. One scale per block stops being an arithmetic claim (`⌊k / 32⌋`) and becomes a fact
/// about which axes the operand spans, which is what already makes a broadcast a broadcast.
#[test]
fn scales_omit_the_axis_inside_the_block() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let (a, b, s) = (lhs_data(), rhs_data(), scale_data());

    let space = Tiling::new()
        .extents(&[(M, ROWS), (N, COLS), (KB, BLOCKS), (KI, BLOCK)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(ROWS))
                .axis(N, Cut::sequential(COLS))
                .axis(KB, Cut::sequential(1))
                .axis(KI, Cut::sequential(BLOCK))
        })
        .build()
        .with_instruction(Instruction::registers(16));

    let (a_t, _) = TestInput::builder(client.clone(), shape![ROWS, DEPTH])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![DEPTH, COLS])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![ROWS, BLOCKS])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // Every axis of the problem, and `KI` addressing nothing: that omission is the whole
    // statement that one scale covers the block.
    let scales_spec = TileSpec::new(Projection::new(
        &[M, KB, KI],
        &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
    ));

    scaled_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[PhysicalAxisMap::of(M), blocked()],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[blocked(), PhysicalAxisMap::of(N)],
            )),
        ),
        TileArgLaunch::new(s_t.binding().into_tensor_arg(), scales_spec),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        space,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..ROWS {
        for n in 0..COLS {
            let want: f32 = (0..DEPTH)
                .map(|k| a[m * DEPTH + k] * s[m * BLOCKS + k / BLOCK] * b[k * COLS + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}
