//! Destinations that add: one written a writer at a time ([`Write::Fold`]) and one written
//! atomically ([`Write::Accumulate`]). Launches that each add their values into one buffer, taken
//! in turn by the stream, leave the sum of what they wrote on top of what the buffer held, a
//! pitched buffer included; and a window of memory lands in a narrower output through a cast.

use super::{Form, implied};
use cubecl::{prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, StridedLayout, TestInput};
use cubek_tile::kind::Write;
use cubek_tile::launch::{Buffered, Destination, DestinationLaunch};
use cubek_tile::*;

const ROW: Axis = Axis(0);
const COL: Axis = Axis(1);

/// Rows and columns every case holds, cut by tiles of [`TILE`] that overhang both.
const ROWS: usize = 5;
const COLS: usize = 29;
const TILE: (usize, usize) = (2, 8);

/// The row stride of a pitched output: rows 29 wide padded to 32, so the buffer holds lines past
/// the shape's product.
const PITCH: usize = 32;

/// `out` written with `input`, as the destination's write says: replaced, added atomically, or
/// folded.
#[cube(launch)]
fn store<E: Float, V: Size, O: Destination>(
    input: &TileArg<'_, E, V>,
    out: &O::Arg<E, V>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let src = input.tile(comptime!(space.clone()));
    let mut dst = O::tile::<E, V>(out, comptime!(space.clone()));
    dst.copy_from(&src);
}

/// `out` written with `input` cast to the output's element.
#[cube(launch)]
fn store_cast<E: Float, F: Float>(
    input: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, F, Const<1>>,
    space: Partitioning,
    #[define(E, F)] _dtypes: [ElemType; 2],
) {
    let src = input.tile(comptime!(space.clone()));
    let mut dst = out.tile(comptime!(space.clone()));
    dst.copy_cast_from(&src);
}

fn launcher() -> Launcher {
    implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(ROW, ROWS), (COL, COLS)]),
            Levels::leaf(&[(ROW, TILE.0), (COL, TILE.1)])
                .walk_every(&[ROW, COL])
                .build(),
        ),
        Form::Static,
    )
}

/// What the output held before the first launch: not zero, so a write that replaced the cell
/// rather than adding into it shows.
fn start(row: usize, col: usize) -> f32 {
    ((row * 7 + col) % 11) as f32 - 5.0
}

/// `launches` launches, each adding `row · COLS + col` into an output that started at [`start`],
/// laid out `layout`; the output read back.
fn add_launches(write: Write, layout: StridedLayout, launches: usize) -> HostData {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let launcher = launcher();
    let input = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .arange()
        .generate_without_host_data();
    let initial = (0..ROWS * COLS)
        .map(|i| start(i / COLS, i % COLS))
        .collect();
    let output = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .layout(layout)
        .custom(initial)
        .generate_without_host_data();
    for _ in 0..launches {
        let src = launcher
            .arg(input.clone().binding())
            .axes(&[ROW, COL])
            .build();
        let bound = launcher
            .arg(output.clone().binding())
            .axes(&[ROW, COL])
            .build();
        store::launch::<Buffered>(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            1,
            src.arg(),
            Buffered::arg((bound, write)),
            launcher.partitioning_arg(),
            dtype,
        );
    }
    HostData::from_tensor_handle(&client, output, HostDataType::F32)
}

fn assert_added(got: &HostData, launches: usize, what: &str) {
    for row in 0..ROWS {
        for col in 0..COLS {
            let expected = start(row, col) + (launches * (row * COLS + col)) as f32;
            assert_eq!(
                got.get_f32(&[row, col]),
                expected,
                "{what}: [{row}, {col}] is not what it started at plus {launches} launches' values"
            );
        }
    }
}

/// Three launches folding into one buffer, the stream taking them in turn: the buffer's own values
/// plus each launch's, the tiles past the problem's edge written only where they lie inside.
#[test]
fn launches_folding_into_one_buffer_sum() {
    let got = add_launches(Write::Fold, StridedLayout::RowMajor, 3);
    assert_added(&got, 3, "folded");
}

/// The same through the atomic destination.
#[test]
fn launches_adding_atomically_into_one_buffer_sum() {
    let got = add_launches(Write::Accumulate, StridedLayout::RowMajor, 3);
    assert_added(&got, 3, "added atomically");
}

/// A pitched buffer holds lines past its shape's product; a destination that adds writes the
/// rows there like any other, rather than dropping every write past `rows · cols`.
#[test]
fn a_pitched_buffer_takes_every_row_it_is_folded_into() {
    let pitched = || StridedLayout::Explicit(vec![PITCH, 1]);
    assert_added(
        &add_launches(Write::Fold, pitched(), 2),
        2,
        "folded, pitched",
    );
    assert_added(
        &add_launches(Write::Accumulate, pitched(), 2),
        2,
        "added atomically, pitched",
    );
}

/// A window of `f32` memory copied into an `f16` output of the same box by the whole cube, the
/// tiles past the problem's edge copied only where they lie inside.
#[test]
fn a_memory_window_lands_in_a_narrower_output_through_a_cast() {
    let client = cubecl::test_device().client();
    let (wide, narrow) = (f32::elem_type_native(), half::f16::elem_type_native());
    let launcher = launcher();
    let input = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(wide)
        .arange()
        .generate_without_host_data();
    let output = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(narrow)
        .zeros()
        .generate_without_host_data();
    let src = launcher.arg(input.binding()).axes(&[ROW, COL]).build();
    let dst = launcher
        .arg(output.clone().binding())
        .axes(&[ROW, COL])
        .build();
    store_cast::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        src.arg(),
        dst.arg(),
        launcher.partitioning_arg(),
        [wide, narrow],
    );
    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for row in 0..ROWS {
        for col in 0..COLS {
            // Every value here is an integer under 2048, which f16 holds exactly.
            assert_eq!(
                got.get_f32(&[row, col]),
                (row * COLS + col) as f32,
                "the cast copy is wrong at [{row}, {col}]"
            );
        }
    }
}
