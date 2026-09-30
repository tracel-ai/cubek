//! An attention folded by [`OnlineSoftmax`] where its holder keeps the score: each unit owning
//! its rows in registers, or each plane owning its rows in its own window, from loading to
//! writing.
//!
//! A unit reads its query, keys and values where they lie, holds its score in a register block,
//! and contracts the probabilities it just computed with the values without leaving registers.
//!
//! One partitioning spans both contractions: `score = q · kᵀ` contracts `D` and `out = p · v`
//! contracts `S`, and each walks its own axes of the fragment level, the other's routed to one
//! step. The plane stages its query once and its keys and values a block at a time, in stages of
//! its own met on `sync_plane`; the score lands in the plane's own window, the softmax folds it
//! there, and the output's fragments are corrected through the plane's own scratch.

use cubecl::{client::Client, prelude::*, zspace::Shape};
use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult};
use cubek_tile::{
    Accumulate, AccumulateExpand, Axis, Levels, Monoid, Partitioning, RegisterBlock, Scratch,
    Semiring, Space, StageStorage, Stages, Tile, TileArg, TileArgLaunch, TileSpec,
    ops::softmax::OnlineSoftmax,
    procedural::{Procedural, Reads, Recipe, RecipeCoords, RecipeExpand},
};

use super::{Form, implied};

const Q: Axis = Axis(0);
const S: Axis = Axis(1);
const D: Axis = Axis(2);
const V: Axis = Axis(3);

/// The bias a query's scores are summed with: zero at a key it sees, the minimum value past the
/// attended keys and, under `causal`, past its own position aligned at the bottom right.
#[derive(CubeType, Clone)]
struct Attended {
    keys: u32,
    queries: u32,
    #[cube(comptime)]
    causal: bool,
}

#[cube]
impl Recipe<f32> for Attended {
    fn evaluate(&self, coordinates: &RecipeCoords) -> f32 {
        let s = coordinates.along(S);
        let mut seen = s < self.keys;
        if self.causal {
            seen = seen && s + self.queries <= coordinates.along(Q) + self.keys;
        }
        select(seen, 0.0, f32::min_value())
    }
}

impl Reads for AttendedExpand {
    fn reads(&self, _scope: &cubecl::ir::Scope, axis: Axis) -> bool {
        axis == S || (self.causal && axis == Q)
    }
}

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn plane_attention<E: Float>(
    q: &TileArg<'_, E, Const<1>>,
    k: &TileArg<'_, E, Const<1>>,
    v: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, f32, Const<1>>,
    keys: u32,
    queries: u32,
    scale: f32,
    space: Partitioning,
    #[comptime] causal: bool,
    #[comptime] block_keys: usize,
    #[define(E)] _dtype: ElemType,
) {
    let q = q.tile(comptime!(space.clone()));
    // Past the attended keys a read is zero: a masked probability never meets a stale value.
    let k = k.tile(comptime!(space.clone())).within(S, 0, keys as usize);
    let v = v.tile(comptime!(space.clone())).within(S, 0, keys as usize);
    let out = out.tile(comptime!(space.clone()));
    let bias = Procedural::<f32>::new::<Attended>(
        comptime!(space.space().subspace(&[Q, S])),
        Attended {
            keys,
            queries,
            causal,
        },
    )
    .tile_in(&space);
    for cube in space {
        for plane in cube {
            let (q_p, k_p, v_p, out_p) = (q.at(&plane), k.at(&plane), v.at(&plane), out.at(&plane));
            let bias_p = bias.at(&plane);
            // No block past the attended keys is stepped to.
            let walk = plane.walk().range(0, (keys as usize).div_ceil(block_keys));
            let score = Tile::<f32>::scratch(&walk, comptime!(vec![Q, S]), StageStorage::Strided);
            let mut p = Tile::<E>::scratch(&walk, comptime!(vec![Q, S]), StageStorage::Strided);
            let mut acc = out_p
                .cmma_accumulator::<f32, f32>(&bias_p, Monoid::Sum)
                .with_scratch(Scratch::OneTile);
            acc.zero();
            let mut softmax = OnlineSoftmax::<f32>::along(&score, S);
            // The query spans no key: staged once, before the walk, where the keys are staged.
            let mut q_s = q_p.stage_for(&walk, StageStorage::Strided);
            q_s.copy_from(&q_p);
            sync_plane();
            let mut stages = Stages::smem(&walk, &k_p, &v_p, StageStorage::Strided, 1usize);
            stages.pipelined(walk, |slot, stage| {
                let bias_s = bias_p.at(stage);
                slot.consume(|k_s, v_s| {
                    let mut logits = score.cmma_accumulator::<f32, E>(&q_s, Monoid::Sum);
                    logits.zero();
                    for fragment in stage.walk().routed(V, 0).unrolled() {
                        let mut cell = logits.at(&fragment);
                        cell.mma(&q_s.at(&fragment), &k_s.at(&fragment), Semiring::SUM_PROD);
                    }
                    logits.drained_into(&score);
                    // The score's cells were stored in the fragments' layout; the softmax reads
                    // them slice by slice.
                    sync_plane();
                    let correction = softmax.step(&score, &bias_s, scale);
                    p.copy_cast_from(&score);
                    // The probabilities land before the fragments load them.
                    sync_plane();
                    acc.along(V).mul(&correction);
                    for fragment in stage.walk().routed(D, 0).unrolled() {
                        let mut cell = acc.at(&fragment);
                        cell.mma(&p.at(&fragment), &v_s.at(&fragment), Semiring::SUM_PROD);
                    }
                });
            });
            acc.along(V).mul(&softmax.recip_l());
            acc.drained_into(&out_p);
        }
    }
}

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn unit_attention<E: Float>(
    q: &TileArg<'_, E, Const<1>>,
    k: &TileArg<'_, E, Const<1>>,
    v: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, f32, Const<1>>,
    keys: u32,
    queries: u32,
    scale: f32,
    space: Partitioning,
    #[comptime] causal: bool,
    #[comptime] block_keys: usize,
    #[comptime] rows: usize,
    #[define(E)] _dtype: ElemType,
) {
    let q = q.tile(comptime!(space.clone()));
    let k = k.tile(comptime!(space.clone())).within(S, 0, keys as usize);
    let v = v.tile(comptime!(space.clone())).within(S, 0, keys as usize);
    let out = out.tile(comptime!(space.clone()));
    let bias = Procedural::<f32>::new::<Attended>(
        comptime!(space.space().subspace(&[Q, S])),
        Attended {
            keys,
            queries,
            causal,
        },
    )
    .tile_in(&space);
    let value = comptime!(space.space().extent(V));
    for cube in space {
        for unit in cube {
            let (q_u, k_u, v_u, out_u, bias_u) = (
                q.at(&unit),
                k.at(&unit),
                v.at(&unit),
                out.at(&unit),
                bias.at(&unit),
            );
            let mut acc = out_u.block_accumulator::<f32, f32, E>(
                &bias_u,
                &v_u,
                comptime!(RegisterBlock::new(rows * value)),
                Monoid::Sum,
            );
            acc.zero();
            let mut softmax = OnlineSoftmax::<f32>::along(&bias_u, S);
            for step in unit.walk().range(0, (keys as usize).div_ceil(block_keys)) {
                let (q_s, k_s, v_s, bias_s) = (
                    q_u.at(&step),
                    k_u.at(&step),
                    v_u.at(&step),
                    bias_u.at(&step),
                );
                let mut score = bias_s.block_accumulator::<f32, E, E>(
                    &q_s,
                    &k_s,
                    comptime!(RegisterBlock::new(rows * block_keys)),
                    Monoid::Sum,
                );
                score.zero();
                score.mma(&q_s, &k_s, Semiring::SUM_PROD);
                let correction = softmax.step(&score, &bias_s, scale);
                acc.along(V).mul(&correction);
                acc.mma(&score, &v_s, Semiring::SUM_PROD);
            }
            acc.along(V).mul(&softmax.recip_l());
            acc.drained_into(&out_u);
        }
    }
}

/// The problem's shape and how the kernel cuts it.
#[derive(Clone, Copy)]
struct Case {
    queries: usize,
    keys: usize,
    /// Keys attended, at most `keys`: the rest is poisoned.
    attended: usize,
    head: usize,
    value: usize,
    /// Fragment edge: the instruction is `edge × edge × edge`.
    edge: usize,
    /// Fragments a plane holds along the queries, and along a block of keys.
    rows: usize,
    block: usize,
    planes: usize,
    causal: bool,
}

fn run<E: Float + CubeElement>(case: Case) {
    let client: Client = cubecl::test_device().client();
    let e_ty = E::elem_type_native();
    let f32_ty = f32::elem_type_native();
    let offered = client.properties().features.matmul.cmma.iter().any(|cfg| {
        cfg.a_type == e_ty
            && cfg.b_type == e_ty
            && cfg.cd_type == f32_ty
            && [cfg.m, cfg.n, cfg.k] == [case.edge as u32; 3]
    });
    if !offered {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "no {0}x{0}x{0} {e_ty:?} fragment summing in f32",
            case.edge
        )))
        .enforce();
        return;
    }
    let Case {
        queries,
        keys,
        attended,
        head,
        value,
        edge,
        rows,
        block,
        planes,
        causal,
    } = case;
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(Q, queries), (S, keys), (D, head), (V, value)]),
            Levels::leaf(&[(Q, edge), (S, edge), (D, edge), (V, edge)])
                .walk(&[(Q, rows), (S, block), (D, head / edge), (V, value / edge)])
                .walk_every(&[S])
                .planes(&[(Q, planes)])
                .cubes(&[Q])
                .build(),
        ),
        Form::Static,
    );
    let (q_data, k_data, v_data, q_handle, k_handle, v_handle) =
        inputs::<E>(&client, queries, keys, attended, head, value, false);
    let out_handle = TestInput::builder(client.clone(), Shape::new([queries, value]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();
    let scale = 1. / (head as f32).sqrt();

    plane_attention::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            q_handle.binding().into_tensor_arg(),
            TileSpec::direct(&[Q, D]),
        ),
        TileArgLaunch::new(
            k_handle.binding().into_tensor_arg(),
            TileSpec::direct(&[S, D]),
        ),
        TileArgLaunch::new(
            v_handle.binding().into_tensor_arg(),
            TileSpec::direct(&[S, V]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[Q, V]),
        ),
        attended as u32,
        queries as u32,
        scale,
        launcher.partitioning_arg(),
        causal,
        edge * block,
        e_ty,
    );
    let out = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    check(
        &out, &q_data, &k_data, &v_data, queries, keys, attended, head, value, causal, false,
    );
}

/// The query, keys and values of a case and their host copies. Every key past the attended ones
/// is NaN, so reading one shows; the keys are stored `[head, keys]` where `transposed_keys`.
#[allow(clippy::type_complexity)]
fn inputs<E: Float + CubeElement>(
    client: &Client,
    queries: usize,
    keys: usize,
    attended: usize,
    head: usize,
    value: usize,
    transposed_keys: bool,
) -> (
    HostData,
    HostData,
    HostData,
    cubecl::std::tensor::TensorHandle,
    cubecl::std::tensor::TensorHandle,
    cubecl::std::tensor::TensorHandle,
) {
    let e_ty = E::elem_type_native();
    let wobble =
        |i: usize, salt: usize| ((i * 2654435761 + salt * 40503) % 2048) as f32 / 512. - 2.;
    let key_cell = |j: usize, d: usize| match j < attended {
        true => wobble(j * head + d, 2),
        false => f32::NAN,
    };
    let key_values: Vec<f32> = match transposed_keys {
        true => (0..head * keys)
            .map(|i| key_cell(i % keys, i / keys))
            .collect(),
        false => (0..keys * head)
            .map(|i| key_cell(i / head, i % head))
            .collect(),
    };
    let key_shape = match transposed_keys {
        true => Shape::new([head, keys]),
        false => Shape::new([keys, head]),
    };
    let (q_handle, q_data) = TestInput::builder(client.clone(), Shape::new([queries, head]))
        .dtype(e_ty)
        .custom((0..queries * head).map(|i| wobble(i, 1)).collect())
        .generate_with_f32_host_data();
    let (k_handle, k_data) = TestInput::builder(client.clone(), key_shape)
        .dtype(e_ty)
        .custom(key_values)
        .generate_with_f32_host_data();
    let (v_handle, v_data) = TestInput::builder(client.clone(), Shape::new([keys, value]))
        .dtype(e_ty)
        .custom(
            (0..keys * value)
                .map(|i| match i / value < attended {
                    true => wobble(i, 3),
                    false => f32::NAN,
                })
                .collect(),
        )
        .generate_with_f32_host_data();
    (q_data, k_data, v_data, q_handle, k_handle, v_handle)
}

/// `out` against the attention computed on the host in `f64`: a query sees the attended keys,
/// and under `causal` none past its own position aligned at the bottom right.
#[allow(clippy::too_many_arguments)]
fn check(
    out: &HostData,
    q_data: &HostData,
    k_data: &HostData,
    v_data: &HostData,
    queries: usize,
    keys: usize,
    attended: usize,
    head: usize,
    value: usize,
    causal: bool,
    transposed_keys: bool,
) {
    let scale = 1. / (head as f64).sqrt();
    let key = |j: usize, d: usize| match transposed_keys {
        true => k_data.get_f32(&[d, j]),
        false => k_data.get_f32(&[j, d]),
    };
    for i in 0..queries {
        let seen = |j: usize| j < attended && (!causal || j + queries <= i + attended);
        let logits: Vec<(usize, f64)> = (0..keys)
            .filter(|&j| seen(j))
            .map(|j| {
                let dot: f64 = (0..head)
                    .map(|d| q_data.get_f32(&[i, d]) as f64 * key(j, d) as f64)
                    .sum();
                (j, dot * scale)
            })
            .collect();
        let max = logits.iter().fold(f64::NEG_INFINITY, |m, &(_, s)| m.max(s));
        let sum: f64 = logits.iter().map(|&(_, s)| (s - max).exp()).sum();
        for c in 0..value {
            let expected = match logits.is_empty() {
                true => 0.,
                false => {
                    logits
                        .iter()
                        .map(|&(j, s)| (s - max).exp() * v_data.get_f32(&[j, c]) as f64)
                        .sum::<f64>()
                        / sum
                }
            };
            let actual = out.get_f32(&[i, c]) as f64;
            // f16 operands and probabilities: a few ulps of each through two contractions.
            let tolerance = 2e-2 * (1. + expected.abs());
            assert!(
                (actual - expected).abs() <= tolerance,
                "out[{i}, {c}] = {actual}, expected {expected}"
            );
        }
    }
}

/// A unit-owned case: `rows` queries a unit, `units` units a cube, `block` keys a step.
#[derive(Clone, Copy)]
struct UnitCase {
    queries: usize,
    keys: usize,
    attended: usize,
    head: usize,
    value: usize,
    rows: usize,
    units: usize,
    block: usize,
    causal: bool,
}

fn run_units<E: Float + CubeElement>(case: UnitCase) {
    let client: Client = cubecl::test_device().client();
    let e_ty = E::elem_type_native();
    let f32_ty = f32::elem_type_native();
    let UnitCase {
        queries,
        keys,
        attended,
        head,
        value,
        rows,
        units,
        block,
        causal,
    } = case;
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(Q, queries), (S, keys), (D, head), (V, value)]),
            Levels::leaf(&[(Q, rows), (S, block), (D, head), (V, value)])
                .walk_every(&[S])
                .units_distributed(Q, units)
                .cubes(&[Q])
                .build(),
        ),
        Form::Static,
    );
    let (q_data, k_data, v_data, q_handle, k_handle, v_handle) =
        inputs::<E>(&client, queries, keys, attended, head, value, true);
    let out_handle = TestInput::builder(client.clone(), Shape::new([queries, value]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();
    let scale = 1. / (head as f32).sqrt();
    unit_attention::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            q_handle.binding().into_tensor_arg(),
            TileSpec::direct(&[Q, D]),
        ),
        // The keys are stored transposed, so a step's rhs runs along the keys it scores.
        TileArgLaunch::new(
            k_handle.binding().into_tensor_arg(),
            TileSpec::direct(&[D, S]),
        ),
        TileArgLaunch::new(
            v_handle.binding().into_tensor_arg(),
            TileSpec::direct(&[S, V]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[Q, V]),
        ),
        attended as u32,
        queries as u32,
        scale,
        launcher.partitioning_arg(),
        causal,
        block,
        rows,
        e_ty,
    );
    let out = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    check(
        &out, &q_data, &k_data, &v_data, queries, keys, attended, head, value, causal, true,
    );
}

const PREFILL: Case = Case {
    queries: 32,
    keys: 64,
    attended: 64,
    head: 16,
    value: 16,
    edge: 8,
    rows: 2,
    block: 2,
    planes: 2,
    causal: false,
};

#[test]
fn plane_attention_matches_the_reference() {
    run::<half::f16>(PREFILL);
}

#[test]
fn plane_attention_reads_nothing_past_the_attended_keys() {
    run::<half::f16>(Case {
        attended: 40,
        ..PREFILL
    });
}

#[test]
fn plane_attention_masks_causally_from_the_bottom_right() {
    run::<half::f16>(Case {
        attended: 50,
        causal: true,
        ..PREFILL
    });
}

#[test]
fn plane_attention_in_f32() {
    run::<f32>(Case {
        value: 24,
        ..PREFILL
    });
}

const UNIT: UnitCase = UnitCase {
    queries: 24,
    keys: 40,
    attended: 40,
    head: 16,
    value: 8,
    rows: 2,
    units: 4,
    block: 4,
    causal: false,
};

#[test]
fn unit_attention_matches_the_reference() {
    run_units::<f32>(UNIT);
}

#[test]
fn unit_attention_reads_nothing_past_the_attended_keys() {
    run_units::<f32>(UnitCase {
        attended: 27,
        ..UNIT
    });
}

#[test]
fn unit_attention_masks_causally_from_the_bottom_right() {
    run_units::<f32>(UnitCase {
        attended: 30,
        causal: true,
        ..UNIT
    });
}

#[test]
fn unit_attention_in_f16() {
    run_units::<half::f16>(UNIT);
}
