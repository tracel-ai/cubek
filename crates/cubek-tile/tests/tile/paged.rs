//! Paged KV reads: an operand whose token axis is placed a whole page at a time, so the physical
//! cache pool needs no ordering of its own.
//!
//! The shape every test here shares is a pool of `pages` physical pages, a `[batch, max_pages]`
//! table naming which physical page holds each logical one, and a kernel reading the logical
//! sequence out of it. What paging owes over an ordinary index is that the position *inside* a
//! page carries through untouched while the page itself moves, and that a window never straddles
//! two pages. A shuffled table is what proves the first; the level cuts prove the second.

use cubecl::{Runtime, TestRuntime, client::ComputeClient, prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;

const BATCH: Axis = Axis(0);
/// The absolute logical token position, which the page table displaces a page at a time. The
/// cache spans it at the pool's extent; the space states the sequence's.
const TOKEN: Axis = Axis(1);
const DIM: Axis = Axis(2);

/// `out[b, t, d] = cache[t, d]` with `cache`'s token axis paged. Two levels, so a page boundary
/// falls between the outer tiles and the inner walk reads inside one.
#[cube(launch)]
fn paged_copy<E: Numeric>(
    cache: &IndexedTileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
) {
    let cache = cache.tile(comptime!(space.clone()));
    let out = out.tile(comptime!(space.clone()));
    for outer_region in Walk::over(out.runtime_space()) {
        let outer_cache = cache.at(&outer_region);
        let outer_out = out.at(&outer_region);
        for inner_region in Walk::over(outer_out.runtime_space()) {
            let mut inner_out = outer_out.at(&inner_region);
            inner_out.copy_from(&outer_cache.at(&inner_region));
        }
    }
}

/// A pool of `pages` pages of `page_size` rows, and the `[batch, max_pages]` table describing
/// which of them each sequence reads. `pages` is deliberately larger than what the table names,
/// so a pool row no sequence claims is never read.
struct Paged {
    batch: usize,
    tokens: usize,
    dim: usize,
    page_size: usize,
    pages: usize,
    table: Vec<u32>,
}

impl Paged {
    fn max_pages(&self) -> usize {
        self.tokens / self.page_size
    }

    /// Every pool row distinct, so a wrong page cannot read as a right one.
    fn pool(&self) -> Vec<f32> {
        (0..self.pages * self.page_size * self.dim)
            .map(|i| i as f32)
            .collect()
    }

    /// `table[b, t / page_size] · page_size + t % page_size`, the whole contract in one line.
    fn expected(&self, b: usize, t: usize, d: usize) -> f32 {
        let page = self.table[b * self.max_pages() + t / self.page_size] as usize;
        ((page * self.page_size + t % self.page_size) * self.dim + d) as f32
    }
}

/// Run `paged_copy` over `paged` and check every element against the reference. `outer` and
/// `inner` are the two levels' cuts of `TOKEN`, stated rather than derived because which level
/// resolves the lookup is exactly what varies between these tests: the lookup fires at the first
/// level whose cut is at most one page.
fn check_paged(paged: &Paged, outer: usize, inner: usize) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let (batch, tokens, dim, page_size) = (paged.batch, paged.tokens, paged.dim, paged.page_size);

    let (pool_t, _) = TestInput::builder(client.clone(), shape![paged.pages * page_size, dim])
        .dtype(dtype)
        .custom(paged.pool())
        .generate_with_f32_host_data();
    let table_t = TestInput::builder(client.clone(), shape![batch, paged.max_pages()])
        .dtype(u32::elem_type_native())
        .custom(paged.table.iter().map(|&p| p as f32).collect())
        .generate_without_host_data();
    let out_t = TestInput::builder(client.clone(), shape![batch, tokens, dim])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let mut operands = (
        Operand::new(&[TOKEN, DIM], dtype),
        Operand::new(&[BATCH, TOKEN, DIM], dtype),
    );
    let space = Tiling::over(
        &mut operands,
        &[(BATCH, batch), (TOKEN, tokens), (DIM, dim)],
    )
    .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
        l.axis(BATCH, Cut::sequential(1))
            .axis(TOKEN, Cut::sequential(outer))
            .axis(DIM, Cut::sequential(dim));
    })
    .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
        l.axis(BATCH, Cut::sequential(1))
            .axis(TOKEN, Cut::sequential(inner))
            .axis(DIM, Cut::sequential(dim));
    })
    .build();

    let launcher = space.launcher(&client);
    let cache = launcher
        .bind(&operands.0, pool_t.binding())
        .paged(
            table_t.binding(),
            BATCH,
            TOKEN,
            page_size,
            IndexPolicy::Trusted,
        )
        .build();
    let out = launcher.bind(&operands.1, out_t.clone().binding()).build();

    paged_copy::launch::<f32, TestRuntime>(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        cache.arg(),
        out.arg(),
        launcher.space().clone(),
    );

    let got = HostData::from_tensor_handle(&client, out_t, HostDataType::F32);
    for b in 0..batch {
        for t in 0..tokens {
            for d in 0..dim {
                assert_eq!(
                    got.get_f32(&[b, t, d]),
                    paged.expected(b, t, d),
                    "wrong paged value at ({b}, {t}, {d})"
                );
            }
        }
    }
}

/// Two sequences whose pages are scattered through the pool in no order, and reversed against
/// each other: the case an identity or a shifted table would let pass.
#[test]
fn a_shuffled_page_table_reads_every_page_it_names() {
    check_paged(
        &Paged {
            batch: 2,
            tokens: 8,
            dim: 4,
            page_size: 2,
            pages: 8,
            table: vec![5, 1, 6, 0, 3, 7, 2, 4],
        },
        4,
        2,
    );
}

/// Prefix sharing: two sequences naming the same physical page, and one naming a page twice. The
/// table is a map, not a permutation, and nothing may assume otherwise.
#[test]
fn sequences_may_share_a_physical_page() {
    check_paged(
        &Paged {
            batch: 2,
            tokens: 6,
            dim: 4,
            page_size: 2,
            pages: 5,
            table: vec![4, 4, 1, 4, 0, 1],
        },
        6,
        2,
    );
}

/// The lookup resolving at the first level rather than the second: the outer cut is already one
/// page, so nothing above it steps whole pages and the inner walk stays inside one.
#[test]
fn the_lookup_may_resolve_at_the_first_level() {
    check_paged(
        &Paged {
            batch: 2,
            tokens: 8,
            dim: 4,
            page_size: 4,
            pages: 4,
            table: vec![2, 0, 1, 3],
        },
        4,
        2,
    );
}

/// A page holding one element is an ordinary per-element index, and must read the same as one.
#[test]
fn a_single_element_page_reads_as_a_plain_index() {
    check_paged(
        &Paged {
            batch: 2,
            tokens: 4,
            dim: 4,
            page_size: 1,
            pages: 6,
            table: vec![5, 0, 3, 1, 2, 4, 0, 5],
        },
        2,
        1,
    );
}

// ---- construction-time refusals --------------------------------------------

/// The pool, the page table and the operands every refusal below varies one thing against.
fn refusal_fixture() -> (
    ComputeClient<TestRuntime>,
    TensorBinding<TestRuntime>,
    TensorBinding<TestRuntime>,
) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let pool = TestInput::builder(client.clone(), shape![16, 4])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data()
        .binding();
    let table = TestInput::builder(client.clone(), shape![2, 4])
        .dtype(u32::elem_type_native())
        .zeros()
        .generate_without_host_data()
        .binding();
    (client, pool, table)
}

/// Build a paged cache operand over a space whose two levels cut `TOKEN` at `outer` and `inner`.
fn build_paged(page_size: usize, outer: usize, inner: usize) {
    let (client, pool, table) = refusal_fixture();
    let dtype = f32::elem_type_native();
    let mut operands = (
        Operand::new(&[TOKEN, DIM], dtype),
        Operand::new(&[BATCH, TOKEN, DIM], dtype),
    );
    let space = Tiling::over(&mut operands, &[(BATCH, 2), (TOKEN, 8), (DIM, 4)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(BATCH, Cut::sequential(1))
                .axis(TOKEN, Cut::sequential(outer))
                .axis(DIM, Cut::sequential(4));
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(BATCH, Cut::sequential(1))
                .axis(TOKEN, Cut::sequential(inner))
                .axis(DIM, Cut::sequential(4));
        })
        .build();
    let launcher = space.launcher(&client);
    let _ = launcher
        .bind(&operands.0, pool)
        .paged(table, BATCH, TOKEN, page_size, IndexPolicy::Trusted)
        .build();
}

/// A level above the lookup cutting `TOKEN` into four-element windows over three-element pages:
/// its children begin mid-page, and no displacement below could put them back.
#[test]
#[should_panic(expected = "are not whole")]
fn a_level_above_the_lookup_that_splits_a_page_is_refused() {
    build_paged(3, 4, 1);
}

/// A fire level cutting `TOKEN` finer than a page, but not into a divisor of one: its windows
/// start mid-page even though each lies inside one.
#[test]
#[should_panic(expected = "start mid-entry")]
fn a_fire_level_that_does_not_divide_a_page_is_refused() {
    build_paged(4, 8, 3);
}

/// A zero-element page has no entry to land on and would divide by zero resolving one.
#[test]
#[should_panic(expected = "a page holds at least one element")]
fn a_zero_page_size_is_refused() {
    build_paged(0, 4, 2);
}

/// The table's trailing dimension enumerates the target's pages, so a table sized for a different
/// page count cannot bound the offsets this space computes.
#[test]
#[should_panic(expected = "index table shape must have one leading dimension per index axis")]
fn a_table_whose_page_count_disagrees_with_the_space_is_refused() {
    // `TOKEN` is 8 long at a page size of 4, so the table owes 2 pages per sequence, not the 4
    // the fixture holds.
    build_paged(4, 4, 4);
}
