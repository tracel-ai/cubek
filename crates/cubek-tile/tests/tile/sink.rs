//! A tile whose destination is an [`ErasedTensor`] stores where a buffer-backed
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

use cubecl::{
    Runtime, TestRuntime,
    prelude::*,
    std::tensor::{ErasedTensor, WriteOnly},
    zspace::shape,
};
use cubek_test_utils::{HostData, HostDataType, TestInput, TileInput};
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
    let sink = ErasedTensor::<E, WriteOnly>::of_tensor::<Const<1>>(out.tensor);
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

// ===========================================================================
// The host half: `Launcher::spec`
// ===========================================================================

/// The same sink store, with the destination's [`TileSpec`] and geometry derived on the host by
/// [`Launcher::spec`] instead of stated at the call site.
///
/// This is the pair a fused store actually reaches for. The kernels above take their spec off a
/// `TileArg` — but a destination written through a call has no `TileArg` to take it off, which is
/// the whole reason `spec` exists: it runs the derivation a bound operand runs and hands back
/// both halves, the spec *and* the geometry it settled on, so neither is restated here.
#[cube(launch)]
fn derived_sink_kernel<E: Float>(
    out: &Tensor<Vector<E, Const<1>>>,
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    #[comptime] space: Space,
    #[comptime] spec: TileSpec,
    #[define(E)] _dtype: ElemType,
) {
    // The settled geometry, arriving as scalars because the host is where it was settled.
    let mut shape = Coords::<u32>::new();
    shape.push(rows);
    shape.push(cols);
    let mut strides = Coords::<u32>::new();
    strides.push(row_stride);
    strides.push(col_stride);

    let sink = ErasedTensor::<E, WriteOnly>::of_tensor::<Const<1>>(out);
    let mut dst = Tile::<E>::of_sink(sink, shape, strides, 1usize, comptime!(space.clone()), spec);
    let src = Tile::<E>::procedural::<Position>(space, Position {});
    dst.copy_from(&src);
}

/// A sink whose spec and geometry both come from [`Launcher::spec`] stores what the buffer kernel
/// stores: the host-derived pair addresses the destination the same way the binding-derived one
/// does, which is the only thing that makes a fused store a drop-in for the kernel it replaces.
#[test]
fn a_launcher_derived_spec_addresses_the_sink() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let launcher = space().launcher_over(&client, &[]);
    let output = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // What the destination would have been, had it been a tensor to bind.
    let derived = launcher.spec(
        &Operand::new(&[ROW, COL], dtype),
        &[ROWS, COLS],
        &[COLS, 1],
        1,
    );
    assert_eq!(derived.shape, vec![ROWS, COLS]);
    assert_eq!(derived.strides, vec![COLS, 1]);

    derived_sink_kernel::launch::<TestRuntime>(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        output.clone().binding().into_tensor_arg(),
        derived.shape[0] as u32,
        derived.shape[1] as u32,
        derived.strides[0] as u32,
        derived.strides[1] as u32,
        launcher.space().clone(),
        derived.spec,
        dtype,
    );

    let stored = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for row in 0..ROWS {
        for col in 0..COLS {
            assert_eq!(
                stored.get_f32(&[row, col]),
                (row * COLS + col) as f32,
                "the launcher-derived sink stored the wrong value at [{row}, {col}]"
            );
        }
    }
}

// ===========================================================================
// A contraction into a sink
// ===========================================================================

const M: Axis = Axis(2);
const N: Axis = Axis(3);
const K: Axis = Axis(4);

/// The promoted matmul, storing to a buffer: `launch_promoted_matmul`'s twin, kept here so the
/// comparison is between two kernels in one file rather than across suites.
#[cube(launch)]
fn buffer_matmul<E: Numeric, EA: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(space);
    let mut acc = c.accumulate::<EA, _>(&a, Monoid::Sum);
    acc.mm(&a, &b, Semiring::SUM_PROD);
}

/// The same contraction, draining into a sink.
///
/// The accumulator is register-resident, which is the whole point: the contraction folds `K` in
/// the block and the scope's drain is the *only* touch of the destination. A drain writes each
/// cell once and never reads it back, so it asks a sink for nothing a sink does not have.
#[cube(launch)]
fn sink_matmul<E: Numeric, EA: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    // The geometry a sink cannot be asked for, taken off the tensor behind it.
    let mut shape = Coords::<u32>::new();
    let mut strides = Coords::<u32>::new();
    #[unroll]
    for i in 0..2usize {
        shape.push(c.tensor.shape(i) as u32);
        strides.push(c.tensor.stride(i) as u32);
    }
    let sink = ErasedTensor::<E, WriteOnly>::of_tensor::<Const<1>>(c.tensor);
    let c = Tile::<E>::of_sink(
        sink,
        shape,
        strides,
        1usize,
        comptime!(space.clone()),
        comptime!(c.spec.clone()),
    );
    let mut acc = c.accumulate::<EA, _>(&a, Monoid::Sum);
    acc.mm(&a, &b, Semiring::SUM_PROD);
}

/// The output operand: register-resident at the level the accumulator opens, in place below it.
fn accumulator_in_registers(space: &Space) -> Operand {
    let mut out = Operand::new(&[M, N], f32::elem_type_native());
    out.stage(Residence::Register);
    for _ in 1..space.partitioner().depth() {
        out.stage(Residence::InPlace);
    }
    out
}

/// `K` walked in four steps above a one-block leaf: every step returns to the same promoted
/// accumulator, so the destination is touched exactly once, on the drain.
fn matmul_space() -> Space {
    let (m, n, k, edge) = (4usize, 4usize, 16usize, 4usize);
    let partitioner = Partitioner::row_major(
        ByAxis::new(&[(M, edge), (N, edge), (K, edge)]),
        ByAxis::new(&[
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    )
    .buffered(Buffering::SINGLE);
    Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(partitioner)
}

fn run_matmul(sink: bool) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let space = matmul_space();

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel owns `out = A·B` whatever the buffer held, and a drain
    // that folded the destination in instead of writing it would show up as the poison.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .operand(&accumulator_in_registers(&space))
        .untiled()
        .uniform(4242, 10., 100.);

    let instructed = space.clone().with_instruction(Instruction::registers(16));
    match sink {
        true => sink_matmul::launch::<TestRuntime>(
            &client,
            space.cube_count(),
            space.cube_dim(&client),
            a.arg(),
            b.arg(),
            c.arg(),
            instructed,
            dtype,
            dtype,
        ),
        false => buffer_matmul::launch::<TestRuntime>(
            &client,
            space.cube_count(),
            space.cube_dim(&client),
            a.arg(),
            b.arg(),
            c.arg(),
            instructed,
            dtype,
            dtype,
        ),
    }
    HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32)
}

/// A contraction whose accumulator is promoted stores its result through a sink exactly as it
/// stores it through a buffer.
#[test]
fn a_register_accumulator_drains_into_a_sink() {
    let (m, n, k) = (4usize, 4usize, 16usize);
    let through_sink = run_matmul(true);
    let through_buffer = run_matmul(false);
    // Row-major arange operands: lhs(i, p) = i·k + p, rhs(p, j) = p·n + j.
    for i in 0..m {
        for j in 0..n {
            let expected: f32 = (0..k).map(|p| ((i * k + p) * (p * n + j)) as f32).sum();
            assert!(
                (through_buffer.get_f32(&[i, j]) - expected).abs() < 1e-3,
                "the buffer contraction is wrong at [{i}, {j}], so the comparison means nothing"
            );
            assert!(
                (through_sink.get_f32(&[i, j]) - expected).abs() < 1e-3,
                "the sink contraction is wrong at [{i}, {j}]: got {}, want {expected}",
                through_sink.get_f32(&[i, j])
            );
        }
    }
}
