//! Distributing a level's work as one index, which is stream-K.
//!
//! Dealing each axis on its own gives a cube the product of its per-axis runs, which is a box of
//! the grid. A share of the work is not a box: it is a range of the index the axes make together,
//! and it may start inside one output tile and end inside another. No box of a four by two grid
//! holds three regions.
//!
//! [`Walk::window`] is that range. The axes of the distributed work stay `Sequential`, so the
//! walk's counts are the whole grid and its flat index already carries every coordinate; an
//! instance's share is `base` and `steps` into it, both runtime. The first tests here are the
//! assignment on a copy, with no contraction and nothing partial: they prove the shares cover the
//! grid exactly once, and that a share starting late reads the regions it was given.

use cubecl::{
    features::AtomicUsage,
    ir::{ElemType, FloatKind, Type},
    prelude::*,
    std::tensor::TensorHandle,
    zspace::shape,
};
use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult};
use cubek_tile::*;

const ROW: Axis = Axis(0);
const COL: Axis = Axis(1);
const ROWS: usize = 12;
const COLS: usize = 10;
/// Tile edges: a 4 by 2 grid, so 8 regions in the flat step space.
const TILE_ROWS: usize = 3;
const TILE_COLS: usize = 5;
const REGIONS: usize = (ROWS / TILE_ROWS) * (COLS / TILE_COLS);

/// Each cube copies the contiguous run of regions `[pos · total / cubes, (pos + 1) · total /
/// cubes)`. Written as two divisions rather than a per-cube share so the ragged case needs no
/// branch: the runs abut, cover the grid once, and differ in length by at most one.
#[cube(launch)]
fn copy_run<E: Numeric>(
    src: &TileArg<'_, E, Const<1>>,
    dst: &TileArg<'_, E, Const<1>>,
    #[comptime] nest: Nest,
    #[comptime] cubes: usize,
    #[define(E)] _dtype: ElemType,
) {
    let src = src.tile(comptime!(nest.space.clone()));
    let dst = dst.tile(comptime!(nest.space.clone()));
    let walk = dst.runtime_space().level(comptime!(nest.at(0)));
    let total = walk.total();
    let pos = CUBE_POS_X as usize;

    let start = pos * total / cubes;
    let end = (pos + 1) * total / cubes;

    for region in walk.window(start, end - start) {
        dst.at(&region).copy_from(&src.at(&region));
    }
}

/// One cube copying the stated run and nothing else, so a `base` that is dropped copies the wrong
/// regions rather than the same ones in another order.
#[cube(launch)]
fn copy_one_run<E: Numeric>(
    src: &TileArg<'_, E, Const<1>>,
    dst: &TileArg<'_, E, Const<1>>,
    #[comptime] start: usize,
    #[comptime] steps: usize,
    #[comptime] nest: Nest,
    #[define(E)] _dtype: ElemType,
) {
    let src = src.tile(comptime!(nest.space.clone()));
    let dst = dst.tile(comptime!(nest.space.clone()));
    let walk = dst.runtime_space().level(comptime!(nest.at(0)));

    // Stated at launch but taken as runtime values: a window whose bounds fold to constants
    // would prove the decode only for the case the compiler could have unrolled.
    for region in walk.window(comptime!(start).runtime(), comptime!(steps).runtime()) {
        dst.at(&region).copy_from(&src.at(&region));
    }
}

struct Harness {
    client: Client,
    dtype: ElemType,
    nest: Nest,
}

impl Harness {
    fn new() -> Self {
        Self {
            client: cubecl::test_device().client(),
            dtype: f32::elem_type_native(),
            nest: Nest::over(&[(ROW, ROWS), (COL, COLS)]).level(|level| {
                level.walk(&[(ROW, TILE_ROWS), (COL, TILE_COLS)]);
            }),
        }
    }

    fn source(&self) -> (TensorHandle, HostData) {
        TestInput::builder(self.client.clone(), shape![ROWS, COLS])
            .dtype(self.dtype)
            .arange()
            .generate_with_f32_host_data()
    }

    fn destination(&self) -> TensorHandle {
        TestInput::builder(self.client.clone(), shape![ROWS, COLS])
            .dtype(self.dtype)
            .zeros()
            .generate_without_host_data()
    }

    fn read(&self, output: TensorHandle) -> HostData {
        HostData::from_tensor_handle(&self.client, output, HostDataType::F32)
    }
}

/// The launch arguments. Each kernel gets its own opaque element type, so this cannot be a
/// function.
macro_rules! tile_args {
    ($h:expr, $src:expr, $dst:expr) => {
        (
            TileArgLaunch::new(
                $src.clone().binding().into_tensor_arg(),
                TileSpec::direct(&[ROW, COL]),
            ),
            TileArgLaunch::new(
                $dst.clone().binding().into_tensor_arg(),
                TileSpec::direct(&[ROW, COL]),
            ),
        )
    };
}

/// `cubes` runs between them copy every cell exactly once. A run that overlapped its neighbour
/// would still pass on a copy, so the case that matters is the cell no run claimed: it stays at
/// the destination's zero.
fn runs_cover_the_grid(cubes: usize) {
    let h = Harness::new();
    let (src, want) = h.source();
    let dst = h.destination();
    let (src_arg, dst_arg) = tile_args!(h, src, dst);

    copy_run::launch(
        &h.client,
        CubeCount::Static(cubes as u32, 1, 1),
        h.nest.cube_dim(&h.client),
        src_arg,
        dst_arg,
        h.nest.clone(),
        cubes,
        h.dtype,
    );

    let got = h.read(dst);
    for row in 0..ROWS {
        for col in 0..COLS {
            assert_eq!(
                got.get_f32(&[row, col]),
                want.get_f32(&[row, col]),
                "{cubes} cubes: cell ({row}, {col}) was not copied exactly once"
            );
        }
    }
}

#[test]
fn runs_dividing_the_grid_cover_it_once() {
    // 8 regions over 4 cubes: two each.
    runs_cover_the_grid(4);
}

#[test]
fn runs_that_do_not_divide_the_grid_still_cover_it_once() {
    // 8 regions over 3 cubes: 2, 3, 3. The case a per-axis deal cannot express, since no
    // rectangle of a 4 by 2 grid has three regions in it.
    runs_cover_the_grid(3);
    // And one region each for five of eight, which leaves three cubes with an empty run.
    runs_cover_the_grid(REGIONS + 2);
}

#[test]
fn a_run_starting_late_copies_the_regions_it_was_given() {
    let h = Harness::new();
    let (src, want) = h.source();
    let dst = h.destination();
    let (src_arg, dst_arg) = tile_args!(h, src, dst);

    // Regions 3 and 4 of the row-major walk: the second half of row band 1, and the first half of
    // row band 2. A rectangle cannot name that pair either.
    let (start, steps) = (3usize, 2usize);
    copy_one_run::launch(
        &h.client,
        CubeCount::Static(1, 1, 1),
        h.nest.cube_dim(&h.client),
        src_arg,
        dst_arg,
        start,
        steps,
        h.nest.clone(),
        h.dtype,
    );

    let got = h.read(dst);
    let cols_per_band = COLS / TILE_COLS;
    for row in 0..ROWS {
        for col in 0..COLS {
            let region = (row / TILE_ROWS) * cols_per_band + col / TILE_COLS;
            let copied = region >= start && region < start + steps;
            let expect = match copied {
                true => want.get_f32(&[row, col]),
                false => 0.0,
            };
            assert_eq!(
                got.get_f32(&[row, col]),
                expect,
                "cell ({row}, {col}) in region {region}"
            );
        }
    }
}

// -- The contraction ---------------------------------------------------------
//
// The assignment above, over a matmul. `K` is not cut at cube scope and no axis is: the line runs
// over the output's tiles and each tile's `K` blocks together, and a cube takes a run of it. A run
// covers whole tiles in the middle and part of the contraction of the two at either end, so
// several cubes hold slices of the same cell and the destination folds them.

const MM: Axis = Axis(2);
const NN: Axis = Axis(3);
const KK: Axis = Axis(4);

/// The leaf's register block, held fixed across every kernel here.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// This cube's share of the work: the run of output regions it touches and, for the first and
/// last, how much of their contraction is its own. `stride` is how many steps of the level below
/// one region costs a share (the steps its instances take *together*), so the share is counted
/// in the same unit whatever the level below cuts across the plane.
#[derive(CubeType)]
struct Share {
    /// The first region touched, and how many are.
    first: usize,
    touched: usize,
    /// The share's bounds on the joint step index.
    start: usize,
    end: usize,
}

/// The share `instances` cubes cut `regions` regions of `stride` steps into. Two divisions rather
/// than a size each: the shares abut, cover the work once, and differ in length by at most one.
#[cube]
fn share(regions: &Walk, #[comptime] instances: usize, #[comptime] stride: usize) -> Share {
    let work = regions.total() * stride;
    let pos = CUBE_POS_X as usize;
    let start = pos * work / instances;
    let end = (pos + 1) * work / instances;
    let first = start / stride;
    // Through the region the share's last step falls in. `end` is exclusive, so the step before
    // it is the one to find; an empty share (more cubes than work) touches nothing.
    let touched = select(start < end, (end - 1) / stride + 1 - first, 0);
    Share {
        first,
        touched,
        start,
        end,
    }
}

/// The steps of region `step` of the share that are the share's own: whole in the middle of the
/// share, part of a region at either end of it.
#[cube]
fn own_steps(share: &Share, step: usize, #[comptime] stride: usize) -> (usize, usize) {
    let base = (share.first + step) * stride;
    let from = select(base < share.start, share.start - base, 0);
    let to = select(share.end < base + stride, share.end - base, stride);
    (from, to - from)
}

/// The streamed contraction: each cube takes its share of the joint index over the output's
/// tiles and their contraction, opens a register accumulator per output region it touches, folds
/// that region's part of the share, and drains once through the atomic sink.
#[cube(launch)]
fn stream_matmul<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    out: &AccumulateArg<'_, E>,
    #[comptime] nest: Nest,
    #[comptime] instances: usize,
    #[comptime] stride: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(nest.space.clone()));
    let b = b.tile(comptime!(nest.space.clone()));
    let c = out.tile(comptime!(nest.space.clone()));
    let regions = c.op_space(&a, &b).level(comptime!(nest.at(0)));
    let share = share(&regions, instances, stride);
    let walk = regions.window(share.first, share.touched);
    for step in 0..share.touched {
        let region = walk.region(step);
        let (from, steps) = own_steps(&share, step, stride);
        let mut c_region = c.at(&region);
        let a_region = a.at(&region);
        let b_region = b.at(&region);
        let mut acc = c_region.block_accumulator::<E, E>(
            &a_region,
            comptime!(Fragments::of(
                &c_region.space,
                &a_region.space,
                nest.below(1)
            )),
            REGISTER_BLOCK,
            Monoid::Sum,
        );
        acc.zero();
        for cell in acc
            .op_space(&a_region, &b_region)
            .level(comptime!(nest.at(1)))
            .window(from, steps)
        {
            let mut acc_cell = acc.at(&cell);
            acc_cell.mma(&a_region.at(&cell), &b_region.at(&cell), Semiring::SUM_PROD);
        }
        acc.drain_cast_into(&mut c_region);
    }
}

/// [`stream_matmul`] with the right operand staged into shared memory under the share: each
/// region runs the level below through its own ring, so what a share does inside a region is
/// what any walk does.
#[cube(launch)]
fn stream_matmul_staged_rhs<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    out: &AccumulateArg<'_, E>,
    #[comptime] nest: Nest,
    #[comptime] instances: usize,
    #[comptime] stride: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(nest.space.clone()));
    let b = b.tile(comptime!(nest.space.clone()));
    let c = out.tile(comptime!(nest.space.clone()));
    let regions = c.op_space(&a, &b).level(comptime!(nest.at(0)));
    let share = share(&regions, instances, stride);
    let walk = regions.window(share.first, share.touched);
    for step in 0..share.touched {
        let region = walk.region(step);
        let (from, steps) = own_steps(&share, step, stride);
        let mut c_region = c.at(&region);
        let a_region = a.at(&region);
        let b_region = b.at(&region);
        let mut acc = c_region.block_accumulator::<E, E>(
            &a_region,
            comptime!(Fragments::of(
                &c_region.space,
                &a_region.space,
                nest.below(1)
            )),
            REGISTER_BLOCK,
            Monoid::Sum,
        );
        acc.zero();
        let cells = acc
            .op_space(&a_region, &b_region)
            .level(comptime!(nest.at(1)))
            .window(from, steps);
        let mut ring = Ring::smem_single(&cells, &b_region, StageStorage::Strided, 1usize);
        pipelined(cells, &mut ring, |slot, cell| {
            let mut acc_cell = acc.at(cell);
            let a_cell = a_region.at(cell);
            slot.consume(|b_s| {
                acc_cell.mma(&a_cell, b_s, Semiring::SUM_PROD);
            });
        });
        acc.drain_cast_into(&mut c_region);
    }
}

/// The tile the contraction accumulates in, and the block it walks `K` in.
const TILE_M: usize = 4;
const TILE_N: usize = 4;
const BLOCK_K: usize = 4;

/// Where the right operand is read from at the level *below* the distribution, which is the one
/// that walks a share step by step and the only one that can stage anything.
#[derive(Clone, Copy)]
enum RhsStage {
    InPlace,
    Smem,
}

/// `a · b` with the work shared between `runs` cubes, folded atomically into a zeroed output.
fn run_stream_k(m: usize, n: usize, k: usize, runs: usize, rhs: RhsStage) -> HostData {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();

    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();
    let (a_handle, _) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(dtype)
        .custom(a)
        .generate_with_f32_host_data();
    let (b_handle, _) = TestInput::builder(client.clone(), shape![k, n])
        .dtype(dtype)
        .custom(b)
        .generate_with_f32_host_data();
    // Zeroed by the launch: no cube owns a cell, so none of them may seed one.
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let nest = Nest::over(&[(MM, m), (NN, n), (KK, k)])
        // The output's tiles and their contraction, distributed as one. `K` is uncut here, so a
        // region of this level is one output tile and the index reaches through the level below.
        .level(|l| {
            l.distribute(
                cubes(CubeAxis::X).instances(runs),
                &[(MM, TILE_M), (NN, TILE_N), (KK, k)],
            );
        })
        // One tile's contraction, which is what a run counts in.
        .level(|l| {
            l.walk(&[(MM, TILE_M), (NN, TILE_N), (KK, BLOCK_K)]);
        });

    // One region of the distribution costs a share every `K` block of its contraction.
    let stride = k / BLOCK_K;
    match rhs {
        RhsStage::InPlace => stream_matmul::launch(
            &client,
            nest.cube_count(),
            nest.cube_dim(&client),
            TileArgLaunch::new(
                a_handle.clone().binding().into_tensor_arg(),
                TileSpec::direct(&[MM, KK]),
            ),
            TileArgLaunch::new(
                b_handle.clone().binding().into_tensor_arg(),
                TileSpec::direct(&[KK, NN]),
            ),
            AccumulateArgLaunch::new(
                out.clone().binding().into_tensor_arg(),
                TileSpec::direct(&[MM, NN]),
            ),
            nest.clone(),
            runs,
            stride,
            dtype,
        ),
        RhsStage::Smem => stream_matmul_staged_rhs::launch(
            &client,
            nest.cube_count(),
            nest.cube_dim(&client),
            TileArgLaunch::new(
                a_handle.clone().binding().into_tensor_arg(),
                TileSpec::direct(&[MM, KK]),
            ),
            TileArgLaunch::new(
                b_handle.clone().binding().into_tensor_arg(),
                TileSpec::direct(&[KK, NN]),
            ),
            AccumulateArgLaunch::new(
                out.clone().binding().into_tensor_arg(),
                TileSpec::direct(&[MM, NN]),
            ),
            nest.clone(),
            runs,
            stride,
            dtype,
        ),
    }

    HostData::from_tensor_handle(&client, out, HostDataType::F32)
}

/// The same contraction on the host, from the same seeds.
fn reference(m: usize, n: usize, k: usize) -> Vec<f32> {
    let a = |i: usize, p: usize| ((i * k + p) % 7) as f32 - 3.0;
    let b = |p: usize, j: usize| ((p * n + j) % 5) as f32 - 2.0;
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..k).map(|p| a(i, p) * b(p, j)).sum();
        }
    }
    out
}

/// Every run count computes the same contraction. The interesting ones are those that do not
/// divide the line: their runs start and end inside a tile's contraction, which is what makes
/// this stream-K rather than a split of `K`.
fn stream_k_agrees_with_the_whole(cubes: usize) {
    let (m, n, k) = (8usize, 8usize, 16usize);
    let got = run_stream_k(m, n, k, cubes, RhsStage::InPlace);
    let want = reference(m, n, k);
    for i in 0..m {
        for j in 0..n {
            let have = got.get_f32(&[i, j]);
            let want = want[i * n + j];
            assert!(
                (have - want).abs() < 1e-3,
                "{cubes} runs: at ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

/// The control: one run is the whole line, so nothing is partial and nothing is folded across
/// cubes. A stream that only works when it is split is not working.
#[test]
fn a_stream_of_one_run_is_the_whole_contraction() {
    if !folds_atomically() {
        return;
    }
    stream_k_agrees_with_the_whole(1);
}

/// Runs that end on a tile boundary: the same work a split of `K` would do, reached by dealing a
/// line rather than by cutting an axis.
#[test]
fn runs_that_end_on_a_tile_do_the_split_a_cut_would() {
    if !folds_atomically() {
        return;
    }
    // 4 tiles of 4 K blocks: 16 steps of line over 4 and 8 runs.
    stream_k_agrees_with_the_whole(4);
    stream_k_agrees_with_the_whole(8);
}

/// Runs that do not: 16 steps over 3, 5 and 7 leaves every run straddling a tile boundary, so a
/// cube seeds a fresh accumulator part way through a tile's contraction and folds a slice that no
/// cut of `K` could hand it.
#[test]
fn runs_that_straddle_a_tile_boundary_still_sum_to_the_whole() {
    if !folds_atomically() {
        return;
    }
    stream_k_agrees_with_the_whole(3);
    stream_k_agrees_with_the_whole(5);
    stream_k_agrees_with_the_whole(7);
    // More runs than the line has steps, so some cubes get nothing at all.
    stream_k_agrees_with_the_whole(24);
}

/// The drain folds, so a device without a float atomic add cannot run any of this.
fn folds_atomically() -> bool {
    let client = cubecl::test_device().client();
    let folds = client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add);
    if !folds {
        TestOutcome::Validated(ValidationResult::Skipped(
            "device has no f32 atomic add".to_string(),
        ))
        .enforce();
    }
    folds
}

/// An operand staged under the distribution. A share is walked region by region, and each region
/// runs the level below through its own ring, so what a share does inside a region is what any
/// walk does: nothing about staging changes because the regions arrived as a share.
#[test]
fn an_operand_stages_under_a_share_as_it_does_under_a_walk() {
    if !folds_atomically() {
        return;
    }
    let (m, n, k) = (8usize, 8usize, 16usize);
    let want = reference(m, n, k);
    // Shares that straddle a tile boundary, so the staged operand is read from part way through
    // a tile's contraction as well as from its start.
    for runs in [1usize, 3, 5] {
        let got = run_stream_k(m, n, k, runs, RhsStage::Smem);
        for i in 0..m {
            for j in 0..n {
                let have = got.get_f32(&[i, j]);
                let want = want[i * n + j];
                assert!(
                    (have - want).abs() < 1e-3,
                    "{runs} shares, rhs staged: at ({i}, {j}): got {have}, want {want}"
                );
            }
        }
    }
}

/// Two scopes sharing one contraction: the cubes take shares of the work, and inside a cube the
/// plane's lanes cut `K` between them and meet in registers. The share is counted in the steps
/// the lanes take *together*, so a cube's slice of the work is the same size however many lanes
/// cover one step of it.
#[test]
fn cubes_take_shares_while_the_lanes_cut_k_between_them() {
    let client = cubecl::test_device().client();
    if !folds_atomically() {
        return;
    }
    let dtype = f32::elem_type_native();
    let plane_size = client.properties().hardware.plane_size_max as usize;
    // Two steps of `K` per lane, so a cube's share is counted in something longer than one.
    let (m, n, k) = (8usize, 8usize, 2 * plane_size);
    let want = reference(m, n, k);

    // 4 output tiles of 2 steps each: 8 steps of work, and 3 shares of it straddle tiles.
    for runs in [1usize, 3, 5] {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();
        let (a_handle, _) = TestInput::builder(client.clone(), shape![m, k])
            .dtype(dtype)
            .custom(a)
            .generate_with_f32_host_data();
        let (b_handle, _) = TestInput::builder(client.clone(), shape![k, n])
            .dtype(dtype)
            .custom(b)
            .generate_with_f32_host_data();
        let out = TestInput::builder(client.clone(), shape![m, n])
            .dtype(dtype)
            .zeros()
            .generate_without_host_data();

        let nest = Nest::over(&[(MM, m), (NN, n), (KK, k)])
            .level(|l| {
                l.distribute(
                    cubes(CubeAxis::X).instances(runs),
                    &[(MM, TILE_M), (NN, TILE_N), (KK, k)],
                );
            })
            .level(|l| {
                l.distribute(lanes(plane_size), &[(KK, 1)])
                    .walk(&[(MM, TILE_M), (NN, TILE_N)]);
            });

        // The lanes cover `K` between them, one step each together: a region costs the share
        // `k / plane_size` steps, not `k`.
        let stride = k / plane_size;
        stream_matmul::launch(
            &client,
            nest.cube_count(),
            nest.cube_dim(&client),
            TileArgLaunch::new(
                a_handle.clone().binding().into_tensor_arg(),
                TileSpec::direct(&[MM, KK]),
            ),
            TileArgLaunch::new(
                b_handle.clone().binding().into_tensor_arg(),
                TileSpec::direct(&[KK, NN]),
            ),
            AccumulateArgLaunch::new(
                out.clone().binding().into_tensor_arg(),
                TileSpec::direct(&[MM, NN]),
            ),
            nest.clone(),
            runs,
            stride,
            dtype,
        );

        let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
        for i in 0..m {
            for j in 0..n {
                let have = got.get_f32(&[i, j]);
                let want = want[i * n + j];
                assert!(
                    (have - want).abs() < 1e-2,
                    "{runs} shares over cubes, K over {plane_size} plane_size: at ({i}, {j}): got {have}, \
                     want {want}"
                );
            }
        }
    }
}
