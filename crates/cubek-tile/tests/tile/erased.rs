//! A tile backed by an [`ErasedTensor`] reads and stores where a buffer-backed
//! one reads and stores.
//!
//! The point of an erased backing is that the address is never formed: the tile
//! walks its layout exactly as it would over memory, and what happens at the end
//! of the walk is a call rather than a load or a store. So the property worth
//! pinning is that the walk is *unchanged* — the same kernel, over the same
//! space and the same spec, must touch the same place whichever backing it was
//! given. That is what makes a fused operand a drop-in for the kernel it
//! replaces, on either side: [`WriteOnly`] for fuse-on-write, [`ReadOnly`] for
//! fuse-on-read.
//!
//! The values are `row * COLS + col` rather than anything smooth, so a touch
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
    let geometry = RuntimeGeometry::of_tensor::<Vector<E, Const<1>>>(out.tensor, 2usize);
    let sink = ErasedTensor::<E, WriteOnly>::of_tensor::<Const<1>>(out.tensor);
    let mut dst = Tile::<E>::of_sink(
        sink,
        geometry,
        1usize,
        comptime!(space.clone()),
        comptime!(out.spec.clone()),
        Write::Replace,
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
// The host half: `Launcher::bind_geometry`
// ===========================================================================

/// The same sink store, with the destination's [`TileSpec`] and geometry derived on the host by
/// [`Launcher::bind_geometry`] instead of stated at the call site.
///
/// This is the pair a fused store actually reaches for. The kernels above take their spec off a
/// `TileArg` — but a destination written through a call has no `TileArg` to take it off, which is
/// the whole reason `bind_geometry` exists: it runs the derivation a bound operand runs and hands
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
    let mut geometry = RuntimeGeometry::new();
    geometry.push(rows, row_stride);
    geometry.push(cols, col_stride);

    let sink = ErasedTensor::<E, WriteOnly>::of_tensor::<Const<1>>(out);
    let mut dst = Tile::<E>::of_sink(
        sink,
        geometry,
        1usize,
        comptime!(space.clone()),
        spec,
        Write::Replace,
    );
    let src = Tile::<E>::procedural::<Position>(space, Position {});
    dst.copy_from(&src);
}

/// A sink whose spec and geometry both come from [`Launcher::bind_geometry`] stores what the buffer
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
    let derived = launcher
        .bind_geometry(
            &Operand::new(&[ROW, COL], dtype),
            &Geometry::of_dims(&[(ROWS, COLS), (COLS, 1)]),
        )
        .vectorize(1)
        .build_spec();
    assert_eq!(derived.geometry.shape(), [ROWS, COLS]);
    assert_eq!(derived.geometry.strides(), [COLS, 1]);

    derived_sink_kernel::launch::<TestRuntime>(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        output.clone().binding().into_tensor_arg(),
        derived.geometry.shape()[0] as u32,
        derived.geometry.shape()[1] as u32,
        derived.geometry.strides()[0] as u32,
        derived.geometry.strides()[1] as u32,
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
    let geometry = RuntimeGeometry::of_tensor::<Vector<E, Const<1>>>(c.tensor, 2usize);
    let sink = ErasedTensor::<E, WriteOnly>::of_tensor::<Const<1>>(c.tensor);
    let c = Tile::<E>::of_sink(
        sink,
        geometry,
        1usize,
        comptime!(space.clone()),
        comptime!(c.spec.clone()),
        Write::Replace,
    );
    let mut acc = c.accumulate::<EA, _>(&a, Monoid::Sum);
    acc.mm(&a, &b, Semiring::SUM_PROD);
}

/// The same contraction again, this time reading its **lhs** through an erased source.
///
/// The mirror of [`sink_matmul`], and the reason the read path had to become a view: an operand
/// tile reads through `matrix_transparent`, which composes onto `MemData::read_view` exactly as
/// the drain composes onto `write_view`. Nothing about the leaf changes — it asks the same layout
/// for the same coordinates, and what answers is a call instead of a load.
#[cube(launch)]
fn source_matmul<E: Numeric, EA: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    // The geometry a source cannot be asked for, taken off the tensor behind it.
    let geometry = RuntimeGeometry::of_tensor::<Vector<E, Const<1>>>(a.tensor, 2usize);
    let source = ErasedTensor::<E, ReadOnly>::of_tensor::<Const<1>>(a.tensor);
    let a = Tile::<E>::of_source(
        source,
        geometry,
        1usize,
        comptime!(space.clone()),
        comptime!(a.spec.clone()),
    );
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(space);
    let mut acc = c.accumulate::<EA, _>(&a, Monoid::Sum);
    acc.mm(&a, &b, Semiring::SUM_PROD);
}

/// Which backing the contraction under test is given.
#[derive(Clone, Copy)]
enum Backed {
    /// Every operand and the destination in memory: the control.
    Buffer,
    /// The destination erased — fuse-on-write.
    Sink,
    /// The lhs erased — fuse-on-read.
    Source,
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

fn run_matmul(backed: Backed) -> HostData {
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
    let (count, dim) = (space.cube_count(), space.cube_dim(&client));
    match backed {
        Backed::Sink => sink_matmul::launch::<TestRuntime>(
            &client,
            count,
            dim,
            a.arg(),
            b.arg(),
            c.arg(),
            instructed,
            dtype,
            dtype,
        ),
        Backed::Source => source_matmul::launch::<TestRuntime>(
            &client,
            count,
            dim,
            a.arg(),
            b.arg(),
            c.arg(),
            instructed,
            dtype,
            dtype,
        ),
        Backed::Buffer => buffer_matmul::launch::<TestRuntime>(
            &client,
            count,
            dim,
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
    let through_sink = run_matmul(Backed::Sink);
    let through_buffer = run_matmul(Backed::Buffer);
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

/// The read half of the same contract: an operand read through an erased source contracts to
/// what the same operand read out of memory contracts to.
///
/// The other direction of fusion. A sink lets a kernel hand its result to a generated epilogue;
/// a source lets it take its input from a generated producer, and the leaf must not be able to
/// tell — same space, same spec, same coordinates, a call where the load was.
#[test]
fn a_contraction_reads_its_lhs_through_a_source() {
    let (m, n, k) = (4usize, 4usize, 16usize);
    let through_source = run_matmul(Backed::Source);
    let through_buffer = run_matmul(Backed::Buffer);
    // Row-major arange operands: lhs(i, p) = i·k + p, rhs(p, j) = p·n + j.
    for i in 0..m {
        for j in 0..n {
            let expected: f32 = (0..k).map(|p| ((i * k + p) * (p * n + j)) as f32).sum();
            assert!(
                (through_buffer.get_f32(&[i, j]) - expected).abs() < 1e-3,
                "the buffer contraction is wrong at [{i}, {j}], so the comparison means nothing"
            );
            assert!(
                (through_source.get_f32(&[i, j]) - expected).abs() < 1e-3,
                "the source contraction is wrong at [{i}, {j}]: got {}, want {expected}",
                through_source.get_f32(&[i, j])
            );
        }
    }
}

// ===========================================================================
// A masked, vectorized store
// ===========================================================================

/// Five rows over a leaf of two: the last block hangs off the end, so the store is masked.
const MASKED_ROWS: usize = 5;

/// The rows overhang their leaf and the columns are served two at a time.
///
/// The two properties the aligned scalar spaces above cannot reach. Masking puts a guard between
/// the walk and the write, and a served width of two makes the tile count its innermost extent in
/// lines and re-express every coarser stride as `stride / 2` — arithmetic a stated geometry runs
/// on numbers nobody read off a tensor. The columns stay exact and in bounds, since a vectorized
/// innermost axis that can leave the buffer is refused outright.
fn masked_space() -> Space {
    Tiling::new()
        .extents(&[(ROW, MASKED_ROWS), (COL, COLS)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |level| {
            level
                .axis(ROW, Cut::sequential(2))
                .axis(COL, Cut::sequential(2))
        })
        .build()
}

/// [`buffer_kernel`] at a served width of two.
///
/// The source is a bound operand rather than [`Position`]: a procedural recipe is evaluated once
/// per *line*, so at a width of two both lanes of a line would carry one value and the test could
/// not tell a masked store from a store one lane wide.
#[cube(launch)]
fn wide_buffer_kernel<E: Float>(
    input: &TileArg<'_, E, Const<2>>,
    out: &TileArg<'_, E, Const<2>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let src = input.tile(comptime!(space.clone()));
    let mut dst = out.tile(comptime!(space.clone()));
    dst.copy_from(&src);
}

/// [`sink_kernel`] at a served width of two, over the same masked space.
#[cube(launch)]
fn wide_sink_kernel<E: Float>(
    input: &TileArg<'_, E, Const<2>>,
    out: &TileArg<'_, E, Const<2>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let src = input.tile(comptime!(space.clone()));
    let geometry = RuntimeGeometry::of_tensor::<Vector<E, Const<2>>>(out.tensor, 2usize);
    let sink = ErasedTensor::<E, WriteOnly>::of_tensor::<Const<2>>(out.tensor);
    let mut dst = Tile::<E>::of_sink(
        sink,
        geometry,
        2usize,
        comptime!(space.clone()),
        comptime!(out.spec.clone()),
        Write::Replace,
    );
    dst.copy_from(&src);
}

/// [`wide_buffer_kernel`] with the *input* reached through an erased source, over the same
/// masked space: the read-side twin of [`wide_sink_kernel`].
#[cube(launch)]
fn wide_source_kernel<E: Float>(
    input: &TileArg<'_, E, Const<2>>,
    out: &TileArg<'_, E, Const<2>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let geometry = RuntimeGeometry::of_tensor::<Vector<E, Const<2>>>(input.tensor, 2usize);
    let source = ErasedTensor::<E, ReadOnly>::of_tensor::<Const<2>>(input.tensor);
    let src = Tile::<E>::of_source(
        source,
        geometry,
        2usize,
        comptime!(space.clone()),
        comptime!(input.spec.clone()),
    );
    let mut dst = out.tile(comptime!(space.clone()));
    dst.copy_from(&src);
}

/// Which side of the masked, vectorized copy is erased.
#[derive(Clone, Copy)]
enum Erased {
    /// Neither: the control.
    Neither,
    /// The destination.
    Sink,
    /// The input.
    Source,
}

fn run_masked(erased: Erased) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let launcher = masked_space().launcher(&client);
    let operand = Operand::new(&[ROW, COL], dtype);
    let input = TestInput::builder(client.clone(), shape![MASKED_ROWS, COLS])
        .dtype(dtype)
        .arange()
        .generate_without_host_data();
    let output = TestInput::builder(client.clone(), shape![MASKED_ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    // Bound both ways: the width and the armed check are the launcher's derivation, so the sink
    // walks the tile the buffer kernel walks rather than one this test talked it into.
    let src = launcher
        .bind(&operand, input.binding())
        .vectorize(2)
        .build();
    let out = launcher
        .bind(&operand, output.clone().binding())
        .vectorize(2)
        .build();
    let (count, dim) = (launcher.cube_count(), launcher.cube_dim());
    match erased {
        Erased::Sink => wide_sink_kernel::launch::<TestRuntime>(
            &client,
            count,
            dim,
            src.arg(),
            out.arg(),
            launcher.space().clone(),
            dtype,
        ),
        Erased::Source => wide_source_kernel::launch::<TestRuntime>(
            &client,
            count,
            dim,
            src.arg(),
            out.arg(),
            launcher.space().clone(),
            dtype,
        ),
        Erased::Neither => wide_buffer_kernel::launch::<TestRuntime>(
            &client,
            count,
            dim,
            src.arg(),
            out.arg(),
            launcher.space().clone(),
            dtype,
        ),
    }
    HostData::from_tensor_handle(&client, output, HostDataType::F32)
}

/// A masked store through a sink writes the cells a masked store through a buffer writes.
///
/// The guard is the whole question. A sink's write ends in a call, so an overhanging lane a
/// buffer would have clipped has nothing to clip it: the call happens or it does not, and a mask
/// dropped between the walk and `write_view` hands the epilogue coordinates off the end of the
/// product. The width is the other half — the tile counts its innermost extent in lines and
/// re-expresses every coarser stride as `stride / 2`, on numbers nobody read off a tensor.
#[test]
fn a_masked_vectorized_sink_stores_where_a_buffer_stores() {
    let through_sink = run_masked(Erased::Sink);
    let through_buffer = run_masked(Erased::Neither);
    for row in 0..MASKED_ROWS {
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
                "the masked sink stored the wrong value at [{row}, {col}]"
            );
        }
    }
}

/// A masked, vectorized read through a source reads the cells a buffer read reads.
///
/// The read half of the same question, and not the same code: a buffer's masked read is the
/// slice's, while an erased one is [`ErasedTensor`]'s own — it folds an out-of-bounds index to
/// zero, reads *that* cell, and selects the mask value after. So a guard dropped between the walk
/// and the call does not fault here either; it returns the wrong cell's value, which the
/// `row * COLS + col` fill makes visible as another cell rather than as a near miss. The width is
/// the other half, on a geometry stated rather than read.
#[test]
fn a_masked_vectorized_source_reads_where_a_buffer_reads() {
    let through_source = run_masked(Erased::Source);
    let through_buffer = run_masked(Erased::Neither);
    for row in 0..MASKED_ROWS {
        for col in 0..COLS {
            let expected = (row * COLS + col) as f32;
            assert_eq!(
                through_buffer.get_f32(&[row, col]),
                expected,
                "the buffer read is wrong at [{row}, {col}], so the comparison means nothing"
            );
            assert_eq!(
                through_source.get_f32(&[row, col]),
                expected,
                "the masked source read the wrong value at [{row}, {col}]"
            );
        }
    }
}
