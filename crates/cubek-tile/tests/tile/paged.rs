//! A paged KV cache, which is expert routing with the axis split.
//!
//! The physical pool is `[PAGE, OFFSET]` over one buffer, spelled with
//! [`PhysicalAxisMap::disjoint`] exactly as a quantization block splits `K`. A page table names
//! which physical page holds a sequence's logical one, and the walk routes `PAGE` to it, so the
//! pool needs no ordering of its own and the page size is the split's radix rather than a number
//! carried beside a lookup.
#![allow(non_snake_case)]

use cubecl::{prelude::*, zspace::Shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;

/// Which sequence.
const B: Axis = Axis(0);
/// Which of the sequence's own pages: what the table is addressed by.
const LP: Axis = Axis(1);
/// Which physical page of the pool: what the table names, and what the walk routes.
const PAGE: Axis = Axis(2);
/// Where inside a page.
const OFFSET: Axis = Axis(3);
/// The feature a slot carries.
const D: Axis = Axis(4);

const SEQS: usize = 2;
const PAGES_PER_SEQ: usize = 2;
const POOL_PAGES: usize = 4;
const PAGE_SIZE: usize = 2;
const FEATURES: usize = 2;
const SLOTS: usize = POOL_PAGES * PAGE_SIZE;

/// Page 2 is shared: sequence 0 holds it first, sequence 1 second. Nothing about the pool's order
/// says so, which is the point of a table.
const TABLE: [u32; SEQS * PAGES_PER_SEQ] = [2, 0, 3, 2];

/// `out[b] = Σ kv[slot]` over the slots of the pages sequence `b` holds.
#[cube(launch)]
fn paged_sum_kernel<E: Numeric>(
    kv: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    table: &Tensor<u32>,
    space: Space,
    #[comptime] seq: Level,
    #[comptime] logical: Level,
    #[comptime] physical: Level,
    #[define(E)] _dtype: ElemType,
) {
    let kv = kv.tile(comptime!(space.clone()));
    let out = out.tile(comptime!(space.clone()));

    for sequence in space.over(&seq) {
        let b = sequence.coord(B);

        let mut total = out.at(&sequence);
        total.init(Monoid::identity::<E>(comptime!(Monoid::Sum)));

        for lp in sequence.over(&logical) {
            let phys = table[b * PAGES_PER_SEQ + lp.coord(LP)] as usize;

            // The same call MoE makes: the axis keeps the pool's four pages, the walk visits the
            // one the table named.
            for page in lp.over(&physical).routed(PAGE, phys) {
                let mut cell = out.at(&page);
                cell.reduce_axis_accumulate(&kv.at(&page), comptime!(Monoid::Sum));
            }
        }
    }
}

/// Every slot distinct, so reading a neighbouring page cannot land on the right answer.
fn kv_values() -> Vec<f32> {
    (0..SLOTS * FEATURES)
        .map(|i| {
            let (slot, d) = (i / FEATURES, i % FEATURES);
            (slot * 10 + d) as f32
        })
        .collect()
}

fn run(table: &[u32]) -> HostData {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let u32_ty = u32::elem_type_native();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (B, SEQS),
                (LP, PAGES_PER_SEQ),
                (PAGE, POOL_PAGES),
                (OFFSET, PAGE_SIZE),
                (D, FEATURES),
            ]),
            vec![
                Level::walk(&[(B, 1)]),
                Level::walk(&[(LP, 1)]),
                Level::walk(&[(PAGE, 1)]),
            ],
        ),
        KernelForm::Static,
    );

    let (kv_handle, _) = TestInput::builder(client.clone(), Shape::new([SLOTS, FEATURES]))
        .dtype(f32_ty)
        .custom(kv_values())
        .generate_with_f32_host_data();
    let table_handle = TestInput::builder(client.clone(), Shape::new([TABLE.len()]))
        .dtype(u32_ty)
        .custom(table.iter().map(|&p| p as f32).collect())
        .generate_without_host_data();
    let out_handle = TestInput::builder(client.clone(), Shape::new([SEQS, FEATURES]))
        .dtype(f32_ty)
        .custom(vec![-1.0; SEQS * FEATURES])
        .generate_without_host_data();

    paged_sum_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            kv_handle.binding().into_tensor_arg(),
            // The pool's one dimension read as two axes: `slot = page * PAGE_SIZE + offset`.
            TileSpec::new(Projection::new(
                &[PAGE, OFFSET, D],
                &[
                    PhysicalAxisMap::disjoint(&[(PAGE, PAGE_SIZE), (OFFSET, 1)]),
                    PhysicalAxisMap::of(D),
                ],
            )),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[B, D]),
        ),
        table_handle.binding().into_tensor_arg(),
        launcher.space_arg(),
        launcher.level(0),
        launcher.level(1),
        launcher.level(2),
        f32_ty,
    );

    HostData::from_tensor_handle(&client, out_handle, HostDataType::F32)
}

/// The reference: each sequence over the slots of the pages its table names.
fn expected(table: &[u32]) -> Vec<Vec<f32>> {
    let values = kv_values();
    (0..SEQS)
        .map(|b| {
            (0..FEATURES)
                .map(|d| {
                    (0..PAGES_PER_SEQ)
                        .flat_map(|lp| {
                            let page = table[b * PAGES_PER_SEQ + lp] as usize;
                            (0..PAGE_SIZE).map(move |o| (page * PAGE_SIZE + o) * FEATURES + d)
                        })
                        .map(|i| values[i])
                        .sum()
                })
                .collect()
        })
        .collect()
}

/// Two page tables over the same pool: the answer follows the table, which it could not if the
/// route did nothing, since the walk would fold the whole pool under both.
#[test]
fn a_routed_page_axis_folds_each_sequence_over_the_pages_it_holds() {
    for table in [TABLE, [0, 1, 1, 0]] {
        let got = run(&table);
        let want = expected(&table);
        for (b, row) in want.iter().enumerate() {
            for (d, cell) in row.iter().enumerate() {
                assert_eq!(
                    got.get_f32(&[b, d]),
                    *cell,
                    "sequence {b} feature {d}: pages {:?}",
                    &table[b * PAGES_PER_SEQ..(b + 1) * PAGES_PER_SEQ]
                );
            }
        }
    }
}
