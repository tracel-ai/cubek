//! Dealing a level's grid out as contiguous runs of its flat step space.
//!
//! The split that landed gives each cube a rectangular block: every axis is dealt on its own, so
//! a cube's share is a product of per-axis runs. Stream-K's share is not a rectangle. It is a
//! range of the flattened grid, which may start in the middle of one output tile and end in the
//! middle of another, and the assignment is a division rather than a decode per axis.
//!
//! [`Walk::window`] is that range. The axes stay `Sequential`, so the walk's counts are the whole
//! grid and its flat index already carries every coordinate; an instance's share is `base` and
//! `steps` into it, both runtime. These tests are the assignment on a copy, with no contraction
//! and nothing partial: what they prove is that the runs tile the grid exactly once, and that a
//! run starting late reads the regions it was given rather than the first ones.

use cubecl::{
    Runtime, TestRuntime, ir::ElemType, prelude::*, std::tensor::TensorHandle, zspace::shape,
};
use cubek_test_utils::{HostData, HostDataType, TestInput};
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
    #[comptime] space: Space,
    #[comptime] cubes: usize,
    #[define(E)] _dtype: ElemType,
) {
    let src = src.tile(comptime!(space.clone()));
    let dst = dst.tile(space);
    let walk = Walk::over(dst.runtime_space());
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
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let src = src.tile(comptime!(space.clone()));
    let dst = dst.tile(space);
    let walk = Walk::over(dst.runtime_space());

    // Stated at launch but taken as runtime values: a window whose bounds fold to constants
    // would prove the decode only for the case the compiler could have unrolled.
    for region in walk.window(comptime!(start).runtime(), comptime!(steps).runtime()) {
        dst.at(&region).copy_from(&src.at(&region));
    }
}

struct Harness {
    client: ComputeClient<TestRuntime>,
    dtype: ElemType,
    space: Space,
}

impl Harness {
    fn new() -> Self {
        Self {
            client: <TestRuntime as Runtime>::client(&Default::default()),
            dtype: f32::elem_type_native(),
            space: Tiling::new()
                .extents(&[(ROW, ROWS), (COL, COLS)])
                .level(WalkOrder::RowMajor, Buffering::SINGLE, |level| {
                    level
                        .axis(ROW, Cut::sequential(TILE_ROWS))
                        .axis(COL, Cut::sequential(TILE_COLS))
                })
                .build(),
        }
    }

    fn source(&self) -> (TensorHandle<TestRuntime>, HostData) {
        TestInput::builder(self.client.clone(), shape![ROWS, COLS])
            .dtype(self.dtype)
            .arange()
            .generate_with_f32_host_data()
    }

    fn destination(&self) -> TensorHandle<TestRuntime> {
        TestInput::builder(self.client.clone(), shape![ROWS, COLS])
            .dtype(self.dtype)
            .zeros()
            .generate_without_host_data()
    }

    fn read(&self, output: TensorHandle<TestRuntime>) -> HostData {
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

    copy_run::launch::<TestRuntime>(
        &h.client,
        CubeCount::Static(cubes as u32, 1, 1),
        h.space.cube_dim(&h.client),
        src_arg,
        dst_arg,
        h.space.clone(),
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
    copy_one_run::launch::<TestRuntime>(
        &h.client,
        CubeCount::Static(1, 1, 1),
        h.space.cube_dim(&h.client),
        src_arg,
        dst_arg,
        start,
        steps,
        h.space.clone(),
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
