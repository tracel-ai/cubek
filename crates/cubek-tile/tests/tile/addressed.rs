//! A register block contracting a weight stored along the contraction, the case it reads through
//! window addresses rather than views (`block::contract_by_address`), checked against the host
//! under every kernel form.
//!
//! `out = x · w`, the weight stored `[n, k]` and read as `[k, n]`, each step four lines of `K`.
//! Two walks of it: every lane taking a column's whole `K`, and a column's lanes taking its steps
//! in turns, their partials folded by the drain, as `matmul_mb` does. A `k` whose last step
//! reaches past the end is not a case: an operand lined along an axis it overhangs is refused.

use cubecl::{prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

const VECTOR: usize = 4;
const STEP: usize = VECTOR * 4;
const LANES: usize = 32;
const PLANES: usize = 4;

const FORMS: [KernelForm<'static>; 4] = [
    KernelForm::Static,
    KernelForm::DynamicAlong(&[N]),
    KernelForm::DynamicAlong(&[K]),
    KernelForm::DynamicAlong(&[N, K]),
];

/// Every lane sums its column over the whole of `K`, the rhs under `s`, one scale per column,
/// where `scaled`.
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

/// A plane's block summed by its lanes taking a column's steps in turns under a walk over the
/// chunks, drained through `cells`, the lane level, which folds their partials.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Walk {
    WholeKPerLane,
    KSplitAcrossLanes,
}

fn x_value(r: usize, i: usize) -> f32 {
    ((r * 5 + i) % 7) as f32 - 3.0
}

fn w_value(j: usize, i: usize) -> f32 {
    ((j * 3 + i) % 5) as f32 - 2.0
}

fn s_value(j: usize) -> f32 {
    (j % 3) as f32 * 0.5 + 0.5
}

/// `x · w` under `form`, read back as `m × n`, the rhs under its column scales where `scaled`.
fn run(walk: Walk, m: usize, k: usize, n: usize, scaled: bool, form: KernelForm<'_>) -> Vec<f32> {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let input = |shape, values: Vec<f32>| {
        TestInput::builder(client.clone(), shape)
            .dtype(dtype)
            .custom(values)
            .generate_with_f32_host_data()
            .0
    };
    let x = input(
        shape![m, k],
        (0..m * k).map(|idx| x_value(idx / k, idx % k)).collect(),
    );
    let w = input(
        shape![n, k],
        (0..n * k).map(|idx| w_value(idx / k, idx % k)).collect(),
    );
    let s = input(shape![n], (0..n).map(s_value).collect());
    // A value no product reaches, so a cell no lane wrote shows.
    let out = input(shape![m, n], vec![-1e6; m * n]);

    let leaf = Tiling::leaf(&[(M, m), (N, 1), (K, STEP)]);
    let tiling = match walk {
        Walk::WholeKPerLane => leaf.walk_every(&[K]).lanes(&[(N, LANES)]),
        Walk::KSplitAcrossLanes => leaf.lanes(&[(K, LANES)]).interleaved(K).walk_every(&[K]),
    };
    let levels = tiling.planes(&[(N, PLANES)]).cubes(&[N]).levels();
    let cells = levels[levels.len() - 1].clone();
    let partitioning = Partitioning::new(Space::new(&[(M, m), (N, n), (K, k)]), levels);
    let grid = (
        partitioning.cube_count(),
        partitioning.cube_dim(LANES as u32),
    );
    let launcher = Launcher::partitioned(&client, partitioning, grid, form);

    let a = launcher
        .arg(x.binding())
        .subspace(&[M, K])
        .vectorize(VECTOR)
        .build();
    let mut weight = w.binding();
    weight.shape = [k, n].into();
    weight.strides = [1, k].into();
    let b = launcher
        .arg(weight)
        .subspace(&[K, N])
        .stored()
        .vectorize(VECTOR)
        .build();
    let s = launcher.arg(s.binding()).subspace(&[N]).build();
    let c = launcher
        .arg(out.clone().binding())
        .subspace(&[M, N])
        .vectorize(1)
        .build();

    let block = RegisterBlock::new(m * VECTOR);
    let (count, dim) = (launcher.cube_count(), launcher.cube_dim());
    let (va, vb, vc) = (a.vector_size, b.vector_size, c.vector_size);
    let partitioning = launcher.partitioning_arg();
    match walk {
        Walk::WholeKPerLane => whole_k_per_lane::launch(
            &client,
            count,
            dim,
            va,
            vb,
            vc,
            a.arg(),
            b.arg(),
            s.arg(),
            c.arg(),
            partitioning,
            block,
            scaled,
            dtype,
        ),
        Walk::KSplitAcrossLanes => k_split_across_lanes::launch(
            &client,
            count,
            dim,
            va,
            vb,
            vc,
            a.arg(),
            b.arg(),
            c.arg(),
            partitioning,
            block,
            cells,
            dtype,
        ),
    }

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    (0..m * n)
        .map(|idx| got.get_f32(&[idx / n, idx % n]))
        .collect()
}

/// Every form at one row and eight, at a `k` of two chunks of the split walk and one of three,
/// which is no power of two, on an `n` the whole-`K` walk's cubes overhang and one they tile.
/// Reports every case that differs from the host rather than stopping at the first.
fn check(walk: Walk, scaled: bool, forms: &[KernelForm<'_>]) {
    let ns: &[usize] = match walk {
        Walk::WholeKPerLane => &[2352, 2048],
        Walk::KSplitAcrossLanes => &[2048],
    };
    let mut failures = Vec::new();
    for &form in forms {
        for m in [1, 8] {
            for k in [1024, 1536] {
                for &n in ns {
                    let scale = |j| if scaled { s_value(j) } else { 1.0 };
                    let want = (0..m * n).map(|idx| {
                        let (r, j) = (idx / n, idx % n);
                        (0..k).map(|i| x_value(r, i) * w_value(j, i)).sum::<f32>() * scale(j)
                    });
                    let got = run(walk, m, k, n, scaled, form);
                    let wrong = got
                        .iter()
                        .zip(want)
                        .filter(|(g, w)| (*g - w).abs() > 1e-3)
                        .count();
                    if wrong > 0 {
                        failures.push(format!("{form:?} m={m} k={k} n={n}: {wrong} cells"));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{walk:?}: differs from the host at {failures:#?}"
    );
}

#[test]
fn whole_k_per_lane_in_every_form() {
    check(Walk::WholeKPerLane, false, &FORMS);
}

#[test]
fn k_split_across_lanes_in_every_form() {
    check(Walk::KSplitAcrossLanes, false, &FORMS);
}

#[test]
fn a_column_scale_is_applied_at_the_address() {
    check(Walk::WholeKPerLane, true, &[KernelForm::Static]);
}
