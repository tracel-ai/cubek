//! Mixture-of-experts routing: one contraction whose weights operand is placed by an index
//! tensor rather than by a coordinate.
//!
//! The weights are `[EXPERT, K, N]` with `EXPERT` held at `Extent::Static(1)` in the kernel
//! space, so the operand spans the axis without walking it and the projection stays direct. What
//! selects the expert is `ids`, addressed by the token tile `M`, an axis the weights do not span
//! at all. Every test here turns on that asymmetry: the routing varies over a walk the operand's
//! own space says nothing about.

use cubecl::std::tensor::layout::CoordsDyn;
use cubecl::{Runtime, TestRuntime, client::ComputeClient, prelude::*, zspace::shape};
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

/// Copy a routed target slice into an output that also spans the index axis. This makes both the
/// index coordinate and the target entry observable without involving contraction semantics.
#[cube(launch)]
fn nested_indexed_copy<E: Numeric>(
    input: &IndexedTileArg<'_, E, Const<1>>,
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
) {
    let input = input.tile(comptime!(space.clone()));
    let output = output.tile(comptime!(space.clone()));
    for outer_region in Walk::over(output.runtime_space()) {
        let outer_input = input.at(&outer_region);
        let outer_output = output.at(&outer_region);
        for inner_region in Walk::over(outer_output.runtime_space()) {
            let inner_input = outer_input.at(&inner_region);
            let mut inner_output = outer_output.at(&inner_region);
            let source = inner_input.view::<Const<1>>();
            let mut destination = inner_output.view_mut::<Const<1>>();
            let source_shape = source.shape();
            let destination_shape = destination.shape();
            for m in 0..destination_shape[0] {
                for target in 0..source_shape[0] {
                    for n in 0..source_shape[1] {
                        let mut source_pos = CoordsDyn::new();
                        source_pos.push(target);
                        source_pos.push(n);
                        let mut destination_pos = CoordsDyn::new();
                        destination_pos.push(m);
                        destination_pos.push(target);
                        destination_pos.push(n);
                        destination.write(destination_pos, source.read(source_pos));
                    }
                }
            }
        }
    }
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

    let mut operands = (
        Operand::new(&[M, K], dtype),
        Operand::new(&[EXPERT, K, N], dtype),
        Operand::new(&[M, N], dtype),
    );

    let space = Tiling::over(
        &mut operands,
        &[(M, moe.m), (N, moe.n), (K, moe.k), (EXPERT, 1)],
    )
    .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, ops| {
        l.axis(M, Cut::sequential(moe.tile_m))
            .axis(N, Cut::sequential(moe.n))
            .axis(K, Cut::sequential(moe.k))
            .axis(EXPERT, Cut::sequential(1));
        ops.1.stage(residence);
    })
    .build()
    .with_instruction(Instruction::registers(16));

    let launcher = space.launcher_over(&client, &[M, N, K]);
    let a_operand = launcher.bind(&operands.0, a_t.binding()).build();
    let w_operand = launcher
        .bind(&operands.1, w_t.binding())
        .indexed(ids_t.binding(), M, EXPERT, policy)
        .build();
    let c_operand = launcher.bind(&operands.2, c_t.clone().binding()).build();

    moe_matmul::launch::<TestRuntime>(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a_operand.arg(),
        w_operand.arg(),
        c_operand.arg(),
        launcher.space().clone(),
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

/// Verify staged weights refill when token tiles advance across different experts.
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

/// In-place weights reading the routed expert directly without staging.
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

/// Arbitrary shuffled routing across more tiles than experts.
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

/// Several token tiles sharing one expert in a sorted MoE dispatch.
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

/// Checked policy masks out-of-range expert IDs to zero.
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

/// The policy is the target-axis bounds statement. No separate `.checked(true)` is needed for a
/// checked route built through `Launcher`; this also keeps policy in the generated kernel spec.
#[test]
fn checked_policy_arms_its_own_target_boundary() {
    let (client, w, ids) = refusal_fixture();
    let space = refusal_space(1);
    let operand = space
        .launcher_over(&client, &[K, N])
        .arg(w)
        .subspace(&[EXPERT, K, N])
        .indexed(ids, M, EXPERT, IndexPolicy::Checked)
        .build();
    assert_eq!(operand.spec.boundaries[0], Some(Boundary::Zero));
}

/// Conversely, `Trusted` promises that the routed target needs no test even when a generic
/// `.checked(true)` mode is requested for other axes.
#[test]
fn trusted_policy_keeps_its_target_unchecked() {
    let (_, w, ids) = refusal_fixture();
    let space = refusal_space(1);
    let operand = StridedOperand::source(w)
        .space(&space)
        .subspace(&[EXPERT, K, N])
        .checked(true)
        .indexed(ids, M, EXPERT, IndexPolicy::Trusted)
        .build();
    assert_eq!(operand.spec.boundaries[0], None);
    assert_eq!(operand.spec.boundaries[1], Some(Boundary::Zero));
}

fn lane_distribution_space(level1_m: Cut, level2_m: Cut) -> Space {
    Tiling::new()
        .extents(&[(M, 8), (N, 4), (K, 4), (EXPERT, 1)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, level1_m)
                .axis(N, Cut::sequential(4))
                .axis(K, Cut::sequential(4))
                .axis(EXPERT, Cut::sequential(1))
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, level2_m)
                .axis(N, Cut::sequential(4))
                .axis(K, Cut::sequential(4))
                .axis(EXPERT, Cut::sequential(1))
        })
        .build()
}

/// Once the lookup fires, its child carries no indirection. Distributing the index axis across
/// lanes at that child therefore cannot produce lane-divergent table reads and must be accepted.
#[test]
fn an_index_axis_may_be_distributed_across_lanes_below_the_fire_level() {
    let (client, w, ids) = refusal_fixture();
    let space = lane_distribution_space(Cut::sequential(4), Cut::unit(1));
    let _ = space
        .launcher_over(&client, &[K, N])
        .arg(w)
        .subspace(&[EXPERT, K, N])
        .indexed(ids, M, EXPERT, IndexPolicy::Trusted)
        .build();
}

/// The fire level itself still reads the table, so a lane-distributed index coordinate there
/// would resolve a different base pointer per lane and remains invalid.
#[test]
#[should_panic(expected = "distributed across lanes")]
fn an_index_axis_distributed_across_lanes_at_the_fire_level_is_refused() {
    let (client, w, ids) = refusal_fixture();
    let space = lane_distribution_space(Cut::unit(4), Cut::sequential(4));
    let _ = space
        .launcher_over(&client, &[K, N])
        .arg(w)
        .subspace(&[EXPERT, K, N])
        .indexed(ids, M, EXPERT, IndexPolicy::Trusted)
        .build();
}

/// Both `M` and the displaced target are cut at two levels. The outer `M` coordinate must be
/// scaled by its two table entries per step: without that scale, rows 2 and 3 alias rows 1 and 2.
#[test]
fn nested_tiling_addresses_absolute_index_table_entries() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let source_targets = 8;
    let m = 16;
    let logical_targets = 4;
    let n = 2;

    let values = (0..source_targets * n)
        .map(|i| (100 * (i / n) + i % n) as f32)
        .collect::<Vec<_>>();
    let routing = vec![
        6, 2, 7, 1, // M tile 0
        0, 5, 3, 4, // M tile 1
        7, 6, 1, 0, // M tile 2
        2, 4, 5, 3, // M tile 3
    ];
    let (values_t, _) = TestInput::builder(client.clone(), shape![source_targets, n])
        .dtype(dtype)
        .custom(values.clone())
        .generate_with_f32_host_data();
    let ids_t = TestInput::builder(client.clone(), shape![m / 4, logical_targets])
        .dtype(u32::elem_type_native())
        .custom(routing.iter().map(|&i| i as f32).collect())
        .generate_without_host_data();
    let output_t = TestInput::builder(client.clone(), shape![m, logical_targets, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let mut operands = (
        Operand::new(&[EXPERT, N], dtype),
        Operand::new(&[M, EXPERT, N], dtype),
    );
    let space = Tiling::over(&mut operands, &[(M, m), (EXPERT, logical_targets), (N, n)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::sequential(8))
                .axis(EXPERT, Cut::sequential(2))
                .axis(N, Cut::sequential(n));
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::sequential(4))
                .axis(EXPERT, Cut::sequential(1))
                .axis(N, Cut::sequential(n));
        })
        .build();

    let launcher = space.launcher(&client);
    let input = launcher
        .bind(&operands.0, values_t.binding())
        .indexed(ids_t.binding(), M, EXPERT, IndexPolicy::Trusted)
        .build();
    let output = launcher
        .bind(&operands.1, output_t.clone().binding())
        .build();

    nested_indexed_copy::launch::<f32, TestRuntime>(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        input.arg(),
        output.arg(),
        launcher.space().clone(),
    );

    let got = HostData::from_tensor_handle(&client, output_t, HostDataType::F32);
    for row in 0..m {
        for target in 0..logical_targets {
            let routed = routing[(row / 4) * logical_targets + target];
            for col in 0..n {
                assert_eq!(
                    got.get_f32(&[row, target, col]),
                    values[routed * n + col],
                    "wrong routed value at ({row}, {target}, {col})"
                );
            }
        }
    }
}

// ---- construction-time refusals --------------------------------------------

/// The `[experts, k, n]` weights, the `[m_tiles]` routing table, and the space they meet in: what
/// every refusal below varies one thing against.
fn refusal_fixture() -> (
    ComputeClient<TestRuntime>,
    TensorBinding<TestRuntime>,
    TensorBinding<TestRuntime>,
) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let w = TestInput::builder(client.clone(), shape![2, 4, 4])
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data()
        .binding();
    let ids = u32_tensor(&client, &[2]);
    (client, w, ids)
}

fn u32_tensor(client: &ComputeClient<TestRuntime>, shape: &[usize]) -> TensorBinding<TestRuntime> {
    TestInput::builder(client.clone(), shape.to_vec())
        .dtype(u32::elem_type_native())
        .zeros()
        .generate_without_host_data()
        .binding()
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

fn build_indexed_with_table(ids: TensorBinding<TestRuntime>) {
    let (_, w, _) = refusal_fixture();
    let space = refusal_space(1);
    let _ = StridedOperand::source(w)
        .space(&space)
        .subspace(&[EXPERT, K, N])
        .checked(false)
        .indexed(ids, M, EXPERT, IndexPolicy::Trusted)
        .build();
}

fn build_indexed(subspace: &[Axis], index: Axis, target: Axis, expert_edge: usize) {
    let (_, w, ids) = refusal_fixture();
    let space = refusal_space(expert_edge);
    let _ = StridedOperand::source(w)
        .space(&space)
        .subspace(subspace)
        .checked(false)
        .indexed(ids, index, target, IndexPolicy::Trusted)
        .build();
}

/// The accepting plan, so the refusals below are refusing what they name and not the shape.
#[test]
fn a_well_formed_indirection_builds() {
    build_indexed(&[EXPERT, K, N], M, EXPERT, 1);
}

/// The index table is read without a device-side bounds check, so the builder must reject a
/// table that cannot cover every fire-level index tile.
#[test]
#[should_panic(expected = "index table shape must have one leading dimension per index axis")]
fn a_short_index_table_is_refused() {
    let (client, _, _) = refusal_fixture();
    build_indexed_with_table(u32_tensor(&client, &[1]));
}

/// A singleton target has no trailing target-entry dimension. Refusing an extra dimension keeps
/// the table rank contract explicit instead of silently accepting a layout the kernel ignores.
#[test]
#[should_panic(expected = "index table shape must have one leading dimension per index axis")]
fn an_unexpected_index_table_rank_is_refused() {
    let (client, _, _) = refusal_fixture();
    build_indexed_with_table(u32_tensor(&client, &[2, 1]));
}

/// Shape alone cannot bound a strided tensor's largest offset. Requiring a dense row-major table
/// prevents a short backing allocation with padded strides from becoming an out-of-bounds read.
#[test]
#[should_panic(expected = "index table must be dense row-major")]
fn an_index_table_with_padded_strides_is_refused() {
    let (client, _, _) = refusal_fixture();
    let mut ids = u32_tensor(&client, &[2]);
    ids.strides = vec![2].into();
    build_indexed_with_table(ids);
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
    let w = TestInput::builder(client, shape![2, 4, 2, 2])
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();
    let _ = StridedOperand::source(w.binding())
        .space(&space)
        .subspace(&[EXPERT, K, N])
        .tiling(StorageTiling::per_axis(&[1, 1, 2]))
        .checked(false)
        .indexed(ids, M, EXPERT, IndexPolicy::Trusted)
        .build();
}

/// A gathered dim holds the receptive field several axes reach over, so it has no origin an
/// entry could displace.
#[test]
#[should_panic(expected = "rides beside a direct mapping")]
fn a_gathered_operand_is_refused() {
    let (_, w, ids) = refusal_fixture();
    let space = refusal_space(1);
    let _ = StridedOperand::source(w)
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
        .indexed(ids, M, EXPERT, IndexPolicy::Trusted)
        .build();
}
