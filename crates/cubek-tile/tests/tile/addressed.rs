//! A register block contracting operands stored along the contraction, the case the block reads
//! through their window addresses rather than their views (`block::contract_by_address`), under
//! every kernel form a memory-bound matmul wants.
//!
//! The product is `out = x · w`, an activation `[m, k]` and a weight stored `[n, k]` read as
//! `[k, n]` (contiguous along the contraction, as a model stores it), each step reading four
//! lines of `K`, and cubes dealt along `N` only. Two walks of it:
//!
//! - **whole `K` per lane**: every lane owns one column and walks all of `K` a step at a time, in
//!   registers;
//! - **`K` split across lanes**: the lanes of a column group take the steps of `K` in turns, one
//!   step each per chunk of the walk, and the drain folds their partials, as `matmul_mb` does
//!   where the weight is contiguous along `K`.
//!
//! Each is checked against the host static, with `N` dynamic (the axis a dynamic form can free
//! for the cube count alone), with `K` dynamic, and with both. The whole-`K` walk also carries a
//! scale on the rhs, one per column, applied at every line's own position; and a leading batch
//! axis of extent one on the activation, whose rows are then not one axis, so the block falls
//! back to the views over the same data.
//!
//! A `k` whose last step reaches past the end is not a case: `K` is the weight's vectorized axis,
//! and an operand served in lines along an axis it overhangs is refused whatever the form.

use cubecl::{prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
const B: Axis = Axis(3);

/// The weight's line width, the widest an `f32` read takes on every runtime.
const VECTOR: usize = 4;
/// Lines of the contraction one step reads.
const LINES_PER_STEP: usize = 4;
const STEP: usize = VECTOR * LINES_PER_STEP;
const LANES: usize = 32;
const PLANES: usize = 4;
/// Lanes sharing one column when `K` is split across them.
const LANES_PER_COLUMN: usize = 4;

/// Every lane sums its column over the whole of `K` in its own register block, the rhs under `s`,
/// one scale per column, where `scaled`.
#[cube(launch)]
fn whole_k_per_lane<E: Numeric, VA: Size, VB: Size, VC: Size>(
    x: &TileArg<'_, E, VA>,
    w: &TileArg<'_, E, VB>,
    s: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, VC>,
    partitioning: Partitioning,
    #[comptime] block: RegisterBlock,
    #[comptime] scaled: bool,
    #[define(E)] _dtype: ElemType,
) {
    let a = x.tile(comptime!(partitioning.clone()));
    let b = w.tile(comptime!(partitioning.clone()));
    let s = s.tile(comptime!(partitioning.clone()));
    let c = out.tile(comptime!(partitioning.clone()));
    for cube in &partitioning {
        for plane in cube {
            for unit in plane {
                let (a_unit, b_unit, c_unit) = (a.at(&unit), b.at(&unit), c.at(&unit));
                let s_unit = s.at(&unit);
                let mut sum = c_unit.block_accumulator::<E, E, E>(
                    &a_unit,
                    &b_unit,
                    comptime!(Fragments::below(&c_unit, &a_unit)),
                    block,
                    Monoid::Sum,
                );
                sum.zero();
                for step in &unit {
                    let mut sum_step = sum.at(&step);
                    if comptime!(scaled) {
                        sum_step.mma_scaled(
                            &a_unit.at(&step).plain(),
                            &b_unit.at(&step).scaled_by(s_unit.at(&step)),
                            Semiring::SUM_PROD,
                        );
                    } else {
                        sum_step.mma(&a_unit.at(&step), &b_unit.at(&step), Semiring::SUM_PROD);
                    }
                }
                sum.drained_into(&c_unit, comptime!(None));
            }
        }
    }
}

/// A plane's block accumulator summed by lanes taking the steps of `K` in turns under a walk over
/// the chunks, and drained through `cells`, the lane level, which folds their partials.
#[cube(launch)]
fn k_split_across_lanes<E: Numeric, VA: Size, VB: Size, VC: Size>(
    x: &TileArg<'_, E, VA>,
    w: &TileArg<'_, E, VB>,
    out: &TileArg<'_, E, VC>,
    partitioning: Partitioning,
    #[comptime] block: RegisterBlock,
    #[comptime] cells: Level,
    #[define(E)] _dtype: ElemType,
) {
    let a = x.tile(comptime!(partitioning.clone()));
    let b = w.tile(comptime!(partitioning.clone()));
    let c = out.tile(comptime!(partitioning.clone()));
    for cube in &partitioning {
        for plane in cube {
            let (a_plane, b_plane, c_plane) = (a.at(&plane), b.at(&plane), c.at(&plane));
            let mut sum = c_plane.block_accumulator::<E, E, E>(
                &a_plane,
                &b_plane,
                comptime!(Fragments::below(&c_plane, &a_plane)),
                block,
                Monoid::Sum,
            );
            sum.zero();
            for chunk in plane {
                for leaf in chunk {
                    let mut sum_leaf = sum.at(&leaf);
                    sum_leaf.mma(&a.at(&leaf), &b.at(&leaf), Semiring::SUM_PROD);
                }
            }
            sum.drained_into(&c_plane, comptime!(Some(cells.clone())));
        }
    }
}

/// How the lanes take the contraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Walk {
    WholeKPerLane,
    KSplitAcrossLanes,
}

fn x_value(r: usize, i: usize, k: usize) -> f32 {
    ((r * k + i) % 7) as f32 - 3.0
}

fn w_value(j: usize, i: usize, k: usize) -> f32 {
    ((j * k + i) % 5) as f32 - 2.0
}

fn s_value(j: usize) -> f32 {
    (j % 3) as f32 * 0.5 + 0.5
}

/// One shape of the product, and how its operands are opened.
#[derive(Clone, Copy, Debug)]
struct Case {
    walk: Walk,
    m: usize,
    k: usize,
    n: usize,
    /// The rhs carries its column scales (whole `K` per lane only).
    scaled: bool,
    /// The activation and the output carry a leading batch axis of extent one (whole `K` per
    /// lane only).
    batched: bool,
}

/// `x · w` under `form`, read back as `m × n`.
fn run(case: Case, form: KernelForm<'_>) -> Vec<f32> {
    let Case {
        walk,
        m,
        k,
        n,
        scaled,
        batched,
    } = case;
    assert!(
        walk == Walk::WholeKPerLane || !(scaled || batched),
        "a scale or a batch axis is a whole-K case"
    );
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();

    let x: Vec<f32> = (0..m * k).map(|idx| x_value(idx / k, idx % k, k)).collect();
    let w: Vec<f32> = (0..n * k).map(|idx| w_value(idx / k, idx % k, k)).collect();
    let s: Vec<f32> = (0..n).map(s_value).collect();
    let x_shape = if batched {
        shape![1, m, k]
    } else {
        shape![m, k]
    };
    let out_shape = if batched {
        shape![1, m, n]
    } else {
        shape![m, n]
    };
    let (x_handle, _) = TestInput::builder(client.clone(), x_shape)
        .dtype(dtype)
        .custom(x)
        .generate_with_f32_host_data();
    let (w_handle, _) = TestInput::builder(client.clone(), shape![n, k])
        .dtype(dtype)
        .custom(w)
        .generate_with_f32_host_data();
    let (s_handle, _) = TestInput::builder(client.clone(), shape![n])
        .dtype(dtype)
        .custom(s)
        .generate_with_f32_host_data();
    // A value no product here reaches, so a column no unit wrote shows, and so does a kernel
    // that failed to expand and left the buffer as it was.
    let out = TestInput::builder(client.clone(), out_shape)
        .dtype(dtype)
        .custom(vec![-1e6; m * n])
        .generate_without_host_data();

    let (space, leaf) = if batched {
        (
            Space::new(&[(B, 1), (M, m), (N, n), (K, k)]),
            Tiling::leaf(&[(B, 1), (M, m), (N, 1), (K, STEP)]),
        )
    } else {
        (
            Space::new(&[(M, m), (N, n), (K, k)]),
            Tiling::leaf(&[(M, m), (N, 1), (K, STEP)]),
        )
    };
    let tiling = match walk {
        Walk::WholeKPerLane => leaf.walk_every(&[K]).lanes(&[(N, LANES)]),
        Walk::KSplitAcrossLanes => leaf
            .lanes(&[(N, LANES / LANES_PER_COLUMN), (K, LANES_PER_COLUMN)])
            .interleaved(K)
            .walk_every(&[K]),
    };
    let levels = tiling.planes(&[(N, PLANES)]).cubes(&[N]).levels();
    // The lane level, the last one when the lanes sit under the walk over the chunks.
    let cells = levels[levels.len() - 1].clone();
    let partitioning = Partitioning::new(space, levels);
    let grid = (
        partitioning.cube_count(),
        partitioning.cube_dim(LANES as u32),
    );
    let launcher = Launcher::partitioned(&client, partitioning, grid, form);

    let x_axes: &[Axis] = if batched { &[B, M, K] } else { &[M, K] };
    let out_axes: &[Axis] = if batched { &[B, M, N] } else { &[M, N] };
    let a = launcher
        .arg(x_handle.clone().binding())
        .subspace(x_axes)
        .vectorize(VECTOR)
        .build();
    // The `[n, k]` buffer read as `[k, n]`: the weight's contiguous axis is the contraction.
    let mut weight = w_handle.clone().binding();
    weight.shape = [k, n].into();
    weight.strides = [1, k].into();
    let b = launcher
        .arg(weight)
        .subspace(&[K, N])
        .stored()
        .vectorize(VECTOR)
        .build();
    let s = launcher
        .arg(s_handle.clone().binding())
        .subspace(&[N])
        .build();
    let c = launcher
        .arg(out.clone().binding())
        .subspace(out_axes)
        .vectorize(1)
        .build();

    let block = RegisterBlock::new(m * VECTOR);
    match walk {
        Walk::WholeKPerLane => whole_k_per_lane::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            a.vector_size,
            b.vector_size,
            c.vector_size,
            a.arg(),
            b.arg(),
            s.arg(),
            c.arg(),
            launcher.partitioning_arg(),
            block,
            scaled,
            dtype,
        ),
        Walk::KSplitAcrossLanes => k_split_across_lanes::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            a.vector_size,
            b.vector_size,
            c.vector_size,
            a.arg(),
            b.arg(),
            c.arg(),
            launcher.partitioning_arg(),
            block,
            cells,
            dtype,
        ),
    }

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    (0..m * n)
        .map(|idx| {
            let (r, j) = (idx / n, idx % n);
            if batched {
                got.get_f32(&[0, r, j])
            } else {
                got.get_f32(&[r, j])
            }
        })
        .collect()
}

fn reference(case: Case) -> Vec<f32> {
    let Case {
        m, k, n, scaled, ..
    } = case;
    (0..m * n)
        .map(|idx| {
            let (r, j) = (idx / n, idx % n);
            let scale = if scaled { s_value(j) } else { 1.0 };
            (0..k)
                .map(|i| x_value(r, i, k) * w_value(j, i, k) * scale)
                .sum()
        })
        .collect()
}

/// Every case under `form`, reporting each that differs from the host rather than stopping at
/// the first, so one run says which shape a form fails at.
fn check(cases: impl IntoIterator<Item = Case>, form: KernelForm<'_>, what: &str) {
    let mut failures = Vec::new();
    for case in cases {
        let want = reference(case);
        let got = run(case, form);
        let wrong = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| (*g - *w).abs() > 1e-3)
            .count();
        if wrong > 0 {
            failures.push(format!("{case:?}: {wrong} of {} cells", want.len()));
        }
    }
    assert!(
        failures.is_empty(),
        "{what}: differs from the host at {failures:#?}"
    );
}

/// At one row and eight, at a `k` the steps tile and one they tile that is no power of two, on
/// an `n` the cubes overhang and one they tile exactly. The split walk takes only the exact `n`:
/// its columns per plane must divide it.
fn cases(walk: Walk, scaled: bool, batched: bool) -> impl Iterator<Item = Case> {
    let ns: &[usize] = match walk {
        Walk::WholeKPerLane => &[2352, 2048],
        Walk::KSplitAcrossLanes => &[2048],
    };
    [1, 8].into_iter().flat_map(move |m| {
        [256, 320].into_iter().flat_map(move |k| {
            ns.iter().map(move |&n| Case {
                walk,
                m,
                k,
                n,
                scaled,
                batched,
            })
        })
    })
}

#[test]
fn whole_k_per_lane_static() {
    check(
        cases(Walk::WholeKPerLane, false, false),
        KernelForm::Static,
        "whole K per lane, static",
    );
}

#[test]
fn whole_k_per_lane_dynamic_n() {
    check(
        cases(Walk::WholeKPerLane, false, false),
        KernelForm::DynamicAlong(&[N]),
        "whole K per lane, dynamic N",
    );
}

#[test]
fn whole_k_per_lane_dynamic_k() {
    check(
        cases(Walk::WholeKPerLane, false, false),
        KernelForm::DynamicAlong(&[K]),
        "whole K per lane, dynamic K",
    );
}

#[test]
fn whole_k_per_lane_dynamic_n_and_k() {
    check(
        cases(Walk::WholeKPerLane, false, false),
        KernelForm::DynamicAlong(&[N, K]),
        "whole K per lane, dynamic N and K",
    );
}

#[test]
fn a_column_scale_is_applied_at_the_address() {
    check(
        cases(Walk::WholeKPerLane, true, false),
        KernelForm::Static,
        "whole K per lane, scaled",
    );
}

#[test]
fn a_batched_activation_reads_through_its_views() {
    check(
        cases(Walk::WholeKPerLane, false, true),
        KernelForm::Static,
        "whole K per lane, batched",
    );
}

#[test]
fn k_split_across_lanes_static() {
    check(
        cases(Walk::KSplitAcrossLanes, false, false),
        KernelForm::Static,
        "K split across lanes, static",
    );
}

#[test]
fn k_split_across_lanes_dynamic_k() {
    check(
        cases(Walk::KSplitAcrossLanes, false, false),
        KernelForm::DynamicAlong(&[K]),
        "K split across lanes, dynamic K",
    );
}

#[test]
fn k_split_across_lanes_dynamic_n_and_k() {
    check(
        cases(Walk::KSplitAcrossLanes, false, false),
        KernelForm::DynamicAlong(&[N, K]),
        "K split across lanes, dynamic N and K",
    );
}
