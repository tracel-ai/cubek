//! Mixture-of-experts routing: one contraction whose weights operand is placed by an index
//! tensor rather than by a coordinate.
//!
//! The weights are `[EXPERT, K, N]` with `EXPERT` held at `Extent::Static(1)` in the kernel
//! space, so the operand spans the axis without walking it and the projection stays direct. What
//! selects the expert is `ids`, addressed by the token tile `M`, an axis the weights do not span
//! at all. Every test here turns on that asymmetry: the routing varies over a walk the operand's
//! own space says nothing about.

use cubecl::{
    Runtime, TestRuntime, client::ComputeClient, prelude::*, std::tensor::TensorHandle,
    zspace::shape,
};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
/// The weights' leading axis, `Static(1)` in the kernel space: the operand holds one expert at a
/// time and the lookup says which.
const EXPERT: Axis = Axis(3);

/// `C = A · W[ids[m_tile]]`: an ordinary `mm`, the routing carried by the operand.
#[cube(launch)]
fn moe_matmul<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    w: &IndexedTileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let w = w.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.mm(&a, &w, Semiring::SUM_PROD);
}

/// One `[experts, k, n]` weight tensor, one `[m, k]` activation tensor, and a routing that hands
/// every `tile_m` block of rows to `ids[block]`. Ragged only in which expert each block reads.
struct Moe {
    m: usize,
    n: usize,
    k: usize,
    tile_m: usize,
    experts: usize,
    ids: Vec<u32>,
}

impl Moe {
    fn a(&self) -> Vec<f32> {
        (0..self.m * self.k).map(|i| (i % 7) as f32 - 3.0).collect()
    }

    /// Each expert's slab is scaled by its own index, so contracting against the wrong expert
    /// cannot land on the right numbers by accident.
    fn w(&self) -> Vec<f32> {
        (0..self.experts * self.k * self.n)
            .map(|i| {
                let expert = i / (self.k * self.n);
                ((i % 5) as f32 - 2.0) * (expert as f32 + 1.0)
            })
            .collect()
    }

    /// `c[i][j] = Σ_l a[i][l] · w[ids[i / tile_m]][l][j]`, and zero where the entry names no
    /// expert: that is what [`IndexPolicy::Checked`] masking produces, and under
    /// [`IndexPolicy::Trusted`] no test names one.
    fn reference(&self) -> Vec<f32> {
        let (a, w) = (self.a(), self.w());
        let mut out = vec![0.0f32; self.m * self.n];
        for i in 0..self.m {
            let expert = self.ids[i / self.tile_m] as usize;
            if expert >= self.experts {
                continue;
            }
            for j in 0..self.n {
                out[i * self.n + j] = (0..self.k)
                    .map(|l| a[i * self.k + l] * w[(expert * self.k + l) * self.n + j])
                    .sum();
            }
        }
        out
    }
}

/// Run `moe` with the weights living at `residence` and the routing read under `policy`, and
/// compare against the per-tile reference.
fn check_moe(moe: &Moe, residence: Residence, policy: IndexPolicy) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();

    let (a_t, _) = TestInput::builder(client.clone(), shape![moe.m, moe.k])
        .dtype(dtype)
        .custom(moe.a())
        .generate_with_f32_host_data();
    let (w_t, _) = TestInput::builder(client.clone(), shape![moe.experts, moe.k, moe.n])
        .dtype(dtype)
        .custom(moe.w())
        .generate_with_f32_host_data();
    let ids_t = TestInput::builder(client.clone(), shape![moe.ids.len()])
        .dtype(u32::elem_type_native())
        .custom(moe.ids.iter().map(|&e| e as f32).collect())
        .generate_without_host_data();
    let c_t = TestInput::builder(client.clone(), shape![moe.m, moe.n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // One level: it steps `M` and cuts nothing else, which is exactly the level that stages the
    // weights. `EXPERT` is cut at its whole extent of 1, so the lookup resolves here.
    let space = Tiling::new()
        .extents(&[(M, moe.m), (N, moe.n), (K, moe.k), (EXPERT, 1)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(moe.tile_m))
                .axis(N, Cut::sequential(moe.n))
                .axis(K, Cut::sequential(moe.k))
                .axis(EXPERT, Cut::sequential(1))
        })
        .build()
        .with_instruction(Instruction::registers(16));

    let mut weights = Operand::new(&[EXPERT, K, N], dtype);
    weights.stage(residence);
    let w_operand = StridedOperand::source(w_t.binding())
        .space(&space)
        .subspace(&[EXPERT, K, N])
        .operand(&weights)
        .checked(policy == IndexPolicy::Checked)
        .indexed(ids_t.binding(), M, EXPERT, policy)
        .build();

    moe_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(a_t.binding().into_tensor_arg(), TileSpec::direct(&[M, K])),
        w_operand.arg(),
        TileArgLaunch::new(
            c_t.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        space,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c_t, HostDataType::F32);
    let want = moe.reference();
    for i in 0..moe.m {
        for j in 0..moe.n {
            let have = got.get_f32(&[i, j]);
            let wanted = want[i * moe.n + j];
            assert!(
                (have - wanted).abs() < 1e-3,
                "at ({i}, {j}): got {have}, want {wanted} (expert {})",
                moe.ids[i / moe.tile_m]
            );
        }
    }
}

/// Two token tiles routed to two different experts, weights staged in shared memory.
///
/// This is the regression test for the staging rule: `w` spans `{EXPERT, K, N}` and the level
/// steps `M`, so the old `walk_invariant` reported it invariant, filled the stage once above the
/// loop, and contracted the second tile against the first tile's expert. Nothing asserted; the
/// numbers were simply wrong.
#[test]
fn a_staged_moe_refills_its_weights_per_token_tile() {
    check_moe(
        &Moe {
            m: 8,
            n: 4,
            k: 4,
            tile_m: 4,
            experts: 2,
            ids: vec![1, 0],
        },
        Residence::Smem,
        IndexPolicy::Trusted,
    );
}

/// The same shape read where it lies: nothing is staged, so every read re-resolves the lookup
/// and this passes with or without the staging fix. Here to localize a failure of the test above
/// to the staging plan rather than to the addressing.
#[test]
fn an_in_place_moe_reads_the_routed_expert() {
    check_moe(
        &Moe {
            m: 8,
            n: 4,
            k: 4,
            tile_m: 4,
            experts: 2,
            ids: vec![1, 0],
        },
        Residence::InPlace,
        IndexPolicy::Trusted,
    );
}

/// A routing that is neither the identity nor a reversal, over more tiles than experts: an
/// off-by-one in the table base, or a base read off the window instead of the region, lands on a
/// neighbouring expert and shows up here rather than cancelling.
#[test]
fn a_shuffled_routing_reads_every_expert_it_names() {
    check_moe(
        &Moe {
            m: 16,
            n: 4,
            k: 4,
            tile_m: 4,
            experts: 3,
            ids: vec![2, 0, 2, 1],
        },
        Residence::Smem,
        IndexPolicy::Trusted,
    );
}

/// Several token tiles sharing one expert, which is what a sorted MoE dispatch actually produces.
/// The window must land on the same slab each time without the repeats folding into one fill.
#[test]
fn consecutive_tiles_may_share_an_expert() {
    check_moe(
        &Moe {
            m: 12,
            n: 4,
            k: 4,
            tile_m: 4,
            experts: 2,
            ids: vec![1, 1, 0],
        },
        Residence::Smem,
        IndexPolicy::Trusted,
    );
}

/// `IndexPolicy::Checked` masks an entry past the target axis's bound: the displaced window sits
/// outside the weights, so the reads zero and the tile contracts to nothing rather than reading a
/// neighbouring expert's memory.
#[test]
fn a_checked_entry_past_the_bound_reads_as_zero() {
    check_moe(
        &Moe {
            m: 8,
            n: 4,
            k: 4,
            tile_m: 4,
            experts: 2,
            // The second tile names an expert the weights do not hold.
            ids: vec![0, 7],
        },
        Residence::Smem,
        IndexPolicy::Checked,
    );
}

// ---- construction-time refusals --------------------------------------------

/// The `[experts, k, n]` weights, the `[m_tiles]` routing table, and the space they meet in: what
/// every refusal below varies one thing against.
fn refusal_fixture() -> (
    ComputeClient<TestRuntime>,
    TensorHandle<TestRuntime>,
    TensorHandle<TestRuntime>,
) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let w = TestInput::builder(client.clone(), shape![2, 4, 4])
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();
    let ids = TestInput::builder(client.clone(), shape![2])
        .dtype(u32::elem_type_native())
        .zeros()
        .generate_without_host_data();
    (client, w, ids)
}

fn refusal_space(expert_edge: usize) -> Space {
    Tiling::new()
        .extents(&[(M, 8), (N, 4), (K, 4), (EXPERT, 1)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(4))
                .axis(N, Cut::sequential(4))
                .axis(K, Cut::sequential(4))
                .axis(EXPERT, Cut::sequential(expert_edge))
        })
        .build()
}

fn build_indexed(subspace: &[Axis], index: Axis, target: Axis, expert_edge: usize) {
    let (_, w, ids) = refusal_fixture();
    let space = refusal_space(expert_edge);
    let _ = StridedOperand::source(w.binding())
        .space(&space)
        .subspace(subspace)
        .checked(false)
        .indexed(ids.binding(), index, target, IndexPolicy::Trusted)
        .build();
}

/// The accepting plan, so the refusals below are refusing what they name and not the shape.
#[test]
fn a_well_formed_indirection_builds() {
    build_indexed(&[EXPERT, K, N], M, EXPERT, 1);
}

/// An operand that does not span its target axis has no window origin for the entry to displace.
#[test]
#[should_panic(expected = "does not span its target axis")]
fn a_target_axis_the_operand_omits_is_refused() {
    build_indexed(&[EXPERT, K, N], M, Axis(9), 1);
}

/// The innermost axis's window is addressed in lines, not elements, so a displacement there has
/// no line to land on.
#[test]
#[should_panic(expected = "innermost axis")]
fn displacing_the_innermost_axis_is_refused() {
    build_indexed(&[EXPERT, K, N], M, N, 1);
}

/// No level cuts the target axis down to a single table entry, so the lookup would never fire and
/// the operand would silently read expert `0` forever.
#[test]
#[should_panic(expected = "never resolves")]
fn a_lookup_that_never_fires_is_refused() {
    build_indexed(&[EXPERT, K, N], M, K, 1);
}

/// An index axis the launched space does not have could never reach a region coordinate, so the
/// table would be read at a constant offset for every tile.
#[test]
#[should_panic(expected = "not an axis of the launched space")]
fn an_index_axis_outside_the_space_is_refused() {
    build_indexed(&[EXPERT, K, N], Axis(9), EXPERT, 1);
}

/// A storage-tiled operand splits its extent across dims, so no single dim carries the target
/// axis alone and the displacement has nowhere to land. Refused here rather than in the kernel,
/// where a comptime assert on a worker thread reads as zeroed output.
#[test]
#[should_panic(expected = "direct, untiled mapping")]
fn a_storage_tiled_operand_is_refused() {
    let (client, _, ids) = refusal_fixture();
    let space = refusal_space(1);
    // Rank 4: `N` splits into a grid dim and a tile dim, which is what the tiling describes.
    let w = TestInput::builder(client.clone(), shape![2, 4, 2, 2])
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();
    let _ = StridedOperand::source(w.binding())
        .space(&space)
        .subspace(&[EXPERT, K, N])
        .tiling(StorageTiling::per_axis(&[1, 1, 2]))
        .checked(false)
        .indexed(ids.binding(), M, EXPERT, IndexPolicy::Trusted)
        .build();
}

/// A gathered dim holds the receptive field several axes reach over, so it has no origin an
/// entry could displace.
#[test]
#[should_panic(expected = "rides beside a direct mapping")]
fn a_gathered_operand_is_refused() {
    let (_, w, ids) = refusal_fixture();
    let space = refusal_space(1);
    let _ = StridedOperand::source(w.binding())
        .space(&space)
        .gathered(Projection::new(
            &[EXPERT, K, N],
            &[
                PhysicalAxisMap::of(EXPERT),
                PhysicalAxisMap::affine(&[(K, 1), (N, 1)]),
                PhysicalAxisMap::of(N),
            ],
        ))
        .checked(false)
        .indexed(ids.binding(), M, EXPERT, IndexPolicy::Trusted)
        .build();
}

// Quantization and indirection both read from the window origin, and the lookup moves it, so the
// two may not meet. That refusal has no test because it is not a runtime one: `quantized` and
// `indexed` both live on the builder's `Q = Unset` impl, so either call consumes the typestate
// the other needs and neither order compiles.

// The last rule of `IndirectionSpec::validate` — that the *operation*'s region spans each index
// axis — has no test here. Which operands meet at a level is only known once the walk builds the
// merged space, so the check lives in `Indirection::advanced_base` as a comptime assert, and a
// comptime assert inside a kernel fires on a worker thread, where `#[should_panic]` never sees it
// and the launch merely returns zeros (the same reason `blocked.rs` and `packed.rs` unit-test
// their kernel-side refusals host-side instead).
