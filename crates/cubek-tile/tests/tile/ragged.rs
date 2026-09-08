//! Variable-length sequences as a plain windowed walk.
//!
//! Sequences of different lengths packed end to end on one axis. The kernel walks that axis with
//! [`Walk::window`], whose base and length are runtime values, so a sequence steps its own tokens
//! and nothing else: no operand carries a side channel, no window origin is displaced behind a
//! read, and `at` is the same call it is everywhere else.
//!
//! The reduce is the shape that isolates it: the output omits the packed axis, so nothing but the
//! walk's bounds decides which tokens a sequence folds.
#![allow(non_snake_case)]

use cubecl::{prelude::*, zspace::Shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;

/// Which sequence.
const B: Axis = Axis(0);
/// The packed position: every sequence's tokens, end to end in one buffer.
const P: Axis = Axis(1);
/// The feature a token carries.
const D: Axis = Axis(2);

/// `out[b] = Σ packed[p]` over the tokens of sequence `b`, whose packed range the cumulative
/// lengths give as `[ends[b], ends[b + 1])`.
#[cube(launch)]
fn ragged_sum_kernel<E: Numeric>(
    packed: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    ends: &Tensor<u32>,
    space: Space,
    #[comptime] seq: Level,
    #[comptime] token: Level,
    #[define(E)] _dtype: ElemType,
) {
    let packed = packed.tile(comptime!(space.clone()));
    let out = out.tile(comptime!(space.clone()));

    for sequence in space.level(comptime!(seq.clone())) {
        let b = sequence.coord(B);
        let start = ends[b] as usize;
        let end = ends[b + 1] as usize;

        let mut total = out.at(&sequence);
        total.init(Monoid::identity::<E>(comptime!(Monoid::Sum)));

        // The whole of it: the walk over the packed axis takes `end - start` steps from `start`,
        // so an empty sequence takes none and a short one never reads its neighbour's tokens.
        let tokens = sequence.level(comptime!(token.clone())).window(start, end - start);
        for pos in tokens {
            let mut cell = out.at(&pos);
            cell.reduce_axis_accumulate(&packed.at(&pos), comptime!(Monoid::Sum));
        }
    }
}

/// The four sequences every case here packs, as cumulative lengths: 2, 0, 5 and 3 tokens.
/// The zero-length one is the edge that a walk gets right by taking no steps at all.
const ENDS: [u32; 5] = [0, 2, 2, 7, 10];
const TOKENS: usize = 10;
const SEQS: usize = 4;
const FEATURES: usize = 4;

/// Every packed cell distinct, so folding one token of a neighbouring sequence cannot land on the
/// right answer by accident.
fn packed_values() -> Vec<f32> {
    (0..TOKENS * FEATURES)
        .map(|i| {
            let (p, d) = (i / FEATURES, i % FEATURES);
            (p * 10 + d) as f32
        })
        .collect()
}

fn run() -> HostData {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let u32_ty = u32::elem_type_native();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(B, SEQS), (P, TOKENS), (D, FEATURES)]),
            vec![Level::walk(&[(B, 1)]), Level::walk(&[(P, 1)])],
        ),
        KernelForm::Static,
    );

    let (packed_handle, _) = TestInput::builder(client.clone(), Shape::new([TOKENS, FEATURES]))
        .dtype(f32_ty)
        .custom(packed_values())
        .generate_with_f32_host_data();

    let ends_handle = TestInput::builder(client.clone(), Shape::new([ENDS.len()]))
        .dtype(u32_ty)
        .custom(ENDS.iter().map(|&e| e as f32).collect())
        .generate_without_host_data();

    // Poisoned, not zeroed: the kernel owes every cell a value, including the empty sequence's.
    let out_handle = TestInput::builder(client.clone(), Shape::new([SEQS, FEATURES]))
        .dtype(f32_ty)
        .custom(vec![-1.0; SEQS * FEATURES])
        .generate_without_host_data();

    ragged_sum_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            packed_handle.binding().into_tensor_arg(),
            TileSpec::direct(&[P, D]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[B, D]),
        ),
        ends_handle.binding().into_tensor_arg(),
        launcher.space_arg(),
        launcher.level(0),
        launcher.level(1),
        f32_ty,
    );

    HostData::from_tensor_handle(&client, out_handle, HostDataType::F32)
}

/// The reference, folded on the host from the same cumulative lengths.
fn expected() -> Vec<Vec<f32>> {
    let values = packed_values();
    (0..SEQS)
        .map(|b| {
            (0..FEATURES)
                .map(|d| {
                    (ENDS[b] as usize..ENDS[b + 1] as usize)
                        .map(|p| values[p * FEATURES + d])
                        .sum()
                })
                .collect()
        })
        .collect()
}

#[test]
fn a_windowed_walk_folds_each_sequence_over_its_own_tokens() {
    let got = run();
    let want = expected();
    for (b, row) in want.iter().enumerate() {
        for (d, cell) in row.iter().enumerate() {
            assert_eq!(
                got.get_f32(&[b, d]),
                *cell,
                "sequence {b} feature {d}: tokens {}..{}",
                ENDS[b],
                ENDS[b + 1]
            );
        }
    }
}
