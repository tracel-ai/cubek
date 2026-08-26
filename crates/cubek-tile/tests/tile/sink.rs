//! A tile whose destination is an [`ErasedSink`] stores where a buffer-backed
//! one stores.
//!
//! The point of a sink is that the address is never formed: the tile walks its
//! layout exactly as it would over memory, and what happens at the end of the
//! walk is a call rather than a store. So the property worth pinning is that the
//! walk is *unchanged* — the same kernel, over the same space and the same spec,
//! must put the same value in the same place whichever destination it was given.
//!
//! The values are `row * COLS + col` rather than anything smooth, so a store
//! that lands one element over shows up as another cell's value instead of as a
//! near miss.

use cubecl::{Runtime, TestRuntime, prelude::*, std::tensor::ErasedSink, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;

const ROW: Axis = Axis(0);
const COL: Axis = Axis(1);
const ROWS: usize = 4;
const COLS: usize = 6;

/// `row * COLS + col`.
#[derive(CubeType, Clone)]
struct Position;

#[cube]
impl<T: Float> Recipe<T> for Position {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        T::cast_from(coordinates.along(ROW)) * T::cast_from(COLS as u32)
            + T::cast_from(coordinates.along(COL))
    }
}

/// The store as it has always been: the operand's own buffer.
#[cube(launch)]
fn buffer_kernel<E: Float>(
    out: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let mut dst = out.tile(comptime!(space.clone()));
    let src = Tile::<E>::procedural::<Position>(comptime!(space.clone()), Position {});
    dst.copy_from(&src);
}

/// The same store, reached through a sink.
///
/// The sink here is backed by the very tensor the buffer kernel writes, which is
/// what makes the two comparable: nothing about the destination differs except
/// that this one is addressed through [`Tile::of_sink`] and lands in a call.
#[cube(launch)]
fn sink_kernel<E: Float>(
    out: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    // The geometry a sink cannot be asked for, taken off the tensor behind it.
    let mut shape = Coords::<u32>::new();
    let mut strides = Coords::<u32>::new();
    #[unroll]
    for i in 0..2usize {
        shape.push(out.tensor.shape(i) as u32);
        strides.push(out.tensor.stride(i) as u32);
    }
    let sink = ErasedSink::<E>::of_tensor::<Const<1>>(out.tensor);
    let mut dst = Tile::<E>::of_sink(
        sink,
        shape,
        strides,
        1usize,
        comptime!(space.clone()),
        comptime!(out.spec.clone()),
    );
    let src = Tile::<E>::procedural::<Position>(comptime!(space.clone()), Position {});
    dst.copy_from(&src);
}

/// The output launch argument, as in the sibling suites: each kernel gets its own
/// opaque element type, so this cannot be a function.
macro_rules! output_arg {
    ($output:expr) => {
        TileArgLaunch::new(
            $output.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        )
    };
}

/// The space both kernels walk, cut so the store is not one contiguous run —
/// a sink that only happened to work on a dense window would pass a flatter one.
fn space() -> Space {
    Tiling::new()
        .extents(&[(ROW, ROWS), (COL, COLS)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |level| {
            level
                .axis(ROW, Cut::sequential(2))
                .axis(COL, Cut::sequential(3))
        })
        .build()
}

fn run(sink: bool) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let space = space();
    let output = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    match sink {
        true => sink_kernel::launch::<TestRuntime>(
            &client,
            space.cube_count(),
            space.cube_dim(&client),
            output_arg!(output),
            space.clone(),
            dtype,
        ),
        false => buffer_kernel::launch::<TestRuntime>(
            &client,
            space.cube_count(),
            space.cube_dim(&client),
            output_arg!(output),
            space.clone(),
            dtype,
        ),
    }
    HostData::from_tensor_handle(&client, output, HostDataType::F32)
}

/// The whole contract: the two destinations are the same walk.
#[test]
fn a_sink_stores_where_a_buffer_stores() {
    let through_sink = run(true);
    let through_buffer = run(false);
    for row in 0..ROWS {
        for col in 0..COLS {
            let expected = (row * COLS + col) as f32;
            assert_eq!(
                through_buffer.get_f32(&[row, col]),
                expected,
                "the buffer store is wrong at [{row}, {col}], so the comparison means nothing"
            );
            assert_eq!(
                through_sink.get_f32(&[row, col]),
                expected,
                "the sink stored the wrong value at [{row}, {col}]"
            );
        }
    }
}
