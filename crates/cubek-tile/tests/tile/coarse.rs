//! A **coarse** operand: one value per block of an axis, rather than one per element.
//!
//! This is the shape a per-block quantization scale has — `s[m, k / block]` beside a
//! `v[m, k]` — and it is the one capability the explicit-scales design needs, since there the
//! scales are a real operand of the kernel instead of a buffer hidden on the values' binding.
//! Proven here with plain floats, deliberately: if the mechanism only works inside the quant
//! machinery it is not a mechanism.
//!
//! The spelling is a rational [`Projection`] — `⌊k / BLOCK⌋`, the same floor the resample
//! mapping already rides — so a coarse operand is a gather like any other, and nothing about it
//! is quantization's.
//!
//! The probe is a contraction, not a copy, because the read is what the design needs: a scale
//! is consumed where the values are, never staged into the shape of its own expansion.
//! ([`Tile::copy`] refuses this outright — a compacted stage fill requires source and
//! destination to share a projection, which a coarse source by definition does not.)

use cubecl::{Runtime, TestRuntime, prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

const ROWS: usize = 4;
const COLS: usize = 4;
const DEPTH: usize = 32;
/// Contracted values per coarse value: the quantization block, in the shape this stands in for.
const BLOCK: usize = 8;
const BLOCKS: usize = DEPTH / BLOCK;

/// `c = a · b` with `a` coarse along the contracted axis: one value per block of `K`, read by
/// every `k` the block covers.
#[cube(launch)]
fn coarse_lhs_matmul<E: Numeric>(
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

/// `⌊k / BLOCK⌋` on the contracted axis, the row addressed as it stands: the coarse operand's
/// whole declaration.
fn coarse_spec() -> TileSpec {
    TileSpec::new(Projection::new(
        &[M, K],
        &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(K).over(BLOCK)],
    ))
}

/// One level, cutting `K` at `cut` so a walk that cuts *at* the block, finer, and coarser are
/// all expressible.
fn space(cut: usize) -> Space {
    Tiling::new()
        .extents(&[(M, ROWS), (N, COLS), (K, DEPTH)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(ROWS))
                .axis(N, Cut::sequential(COLS))
                .axis(K, Cut::sequential(cut))
        })
        .build()
        .with_instruction(Instruction::registers(16))
}

/// Distinct per `(m, block)` and not integers, so an off-by-one block index cannot pass.
fn coarse_data() -> Vec<f32> {
    (0..ROWS * BLOCKS).map(|i| i as f32 + 0.5).collect()
}

/// Distinct per `(k, n)`, and small enough that the sum stays exact in `f32`.
fn rhs_data() -> Vec<f32> {
    (0..DEPTH * COLS).map(|i| (i % 7) as f32 - 3.0).collect()
}

/// Launch [`coarse_lhs_matmul`] over `space` and return `c`.
fn run(space: Space) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();

    let (a, _) = TestInput::builder(client.clone(), shape![ROWS, BLOCKS])
        .dtype(dtype)
        .custom(coarse_data())
        .generate_with_f32_host_data();
    let (b, _) = TestInput::builder(client.clone(), shape![DEPTH, COLS])
        .dtype(dtype)
        .custom(rhs_data())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    coarse_lhs_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(a.binding().into_tensor_arg(), coarse_spec()),
        TileArgLaunch::new(b.binding().into_tensor_arg(), TileSpec::direct(&[K, N])),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        space,
        dtype,
    );

    HostData::from_tensor_handle(&client, c, HostDataType::F32)
}

/// Every `k` of a block contracted against that block's one coarse value.
fn assert_contracted(got: &HostData) {
    let (a, b) = (coarse_data(), rhs_data());
    for m in 0..ROWS {
        for n in 0..COLS {
            let want: f32 = (0..DEPTH)
                .map(|k| a[m * BLOCKS + k / BLOCK] * b[k * COLS + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-4,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The walk cuts `K` exactly at the block, which is the alignment the explicit-scales design
/// asks a routine for: one coarse value per region.
#[test]
fn a_coarse_operand_contracts_over_its_block() {
    assert_contracted(&run(space(BLOCK)));
}

/// A cut finer than the block: several regions in a row read the same coarse value, which is
/// omission-is-invariance arriving through the floor rather than through a missing axis.
#[test]
fn a_cut_finer_than_the_block_reads_the_same_value_across_regions() {
    assert_contracted(&run(space(BLOCK / 2)));
}

/// A cut coarser than the block: one region spans several coarse values, so the read addresses
/// them within the region rather than once per region.
#[test]
fn a_cut_coarser_than_the_block_addresses_each_value_in_the_region() {
    assert_contracted(&run(space(BLOCK * 2)));
}
