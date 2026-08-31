//! Variable-length sequences stored contiguously on one packed token axis.
//!
//! Each sequence is presented to the kernel as a rectangular `max_tokens` window. Adjacent
//! cumulative lengths translate that logical window to its packed physical start and bound it at
//! its exclusive end, so a short sequence cannot read into the next one.

use cubecl::std::tensor::layout::CoordsDyn;
use cubecl::{Runtime, TestRuntime, prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;

const BATCH: Axis = Axis(0);
const TOKEN: Axis = Axis(1);
const DIM: Axis = Axis(2);

#[cube(launch)]
fn unpack_sequences<E: Numeric>(
    packed: &IndexedTileArg<'_, E, Const<1>>,
    padded: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
) {
    let packed = packed.tile(comptime!(space.clone()));
    let padded = padded.tile(comptime!(space.clone()));
    for outer_region in Walk::over(padded.runtime_space()) {
        let outer_packed = packed.at(&outer_region);
        let outer_padded = padded.at(&outer_region);
        for inner_region in Walk::over(outer_padded.runtime_space()) {
            let inner_packed = outer_packed.at(&inner_region);
            let mut inner_padded = outer_padded.at(&inner_region);
            let source = inner_packed.nd::<E, Const<1>, Const<1>>(comptime!(Guard::Checked));
            let mut destination = inner_padded.view_mut::<Const<1>>();
            let shape = source.shape();
            for token in 0..shape[0] {
                for dim in 0..shape[1] {
                    let mut source_pos = CoordsDyn::new();
                    source_pos.push(token);
                    source_pos.push(dim);
                    let mut destination_pos = CoordsDyn::new();
                    destination_pos.push(0u32);
                    destination_pos.push(token);
                    destination_pos.push(dim);
                    destination.write(destination_pos, source.read(source_pos));
                }
            }
        }
    }
}

#[cube(launch)]
fn pack_sequences<E: Numeric>(
    rectangular: &TileArg<'_, E, Const<1>>,
    packed: &IndexedTileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
) {
    let rectangular = rectangular.tile(comptime!(space.clone()));
    let packed = packed.tile(comptime!(space.clone()));
    for outer_region in Walk::over(rectangular.runtime_space()) {
        let outer_rectangular = rectangular.at(&outer_region);
        let outer_packed = packed.at(&outer_region);
        for inner_region in Walk::over(outer_rectangular.runtime_space()) {
            let inner_rectangular = outer_rectangular.at(&inner_region);
            let mut inner_packed = outer_packed.at(&inner_region);
            let source = inner_rectangular.matrix::<Const<1>>(0usize);
            let mut destination = inner_packed.matrix_mut::<Const<1>>(0usize);
            let shape = source.shape();
            for token in 0..shape.0 {
                for dim in 0..shape.1 {
                    let pos = (token, dim);
                    destination.write(pos, source.read(pos));
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum StageAt {
    Never,
    Outer,
    Inner,
}

fn check_unpack(outer_batch: usize, stage_at: StageAt) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let lengths = [2usize, 0, 5, 3];
    let cumulative = [0u32, 2, 2, 7, 10];
    let batch = lengths.len();
    let max_tokens = 6;
    let dim = 3;

    // Every physical row is distinct. In particular, the first row of the next sequence cannot
    // pass for a zero at the preceding sequence's tail.
    let packed_values = (0..10 * dim).map(|i| i as f32 + 1.0).collect::<Vec<_>>();
    let (packed_t, _) = TestInput::builder(client.clone(), shape![10, dim])
        .dtype(dtype)
        .custom(packed_values.clone())
        .generate_with_f32_host_data();
    let lengths_t = TestInput::builder(client.clone(), shape![batch + 1])
        .dtype(u32::elem_type_native())
        .custom(cumulative.iter().map(|&x| x as f32).collect())
        .generate_without_host_data();
    let padded_t = TestInput::builder(client.clone(), shape![batch, max_tokens, dim])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let mut operands = (
        Operand::new(&[TOKEN, DIM], dtype),
        Operand::new(&[BATCH, TOKEN, DIM], dtype),
    );
    let space = Tiling::over(
        &mut operands,
        &[(BATCH, batch), (TOKEN, max_tokens), (DIM, dim)],
    )
    .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, ops| {
        l.axis(BATCH, Cut::sequential(outer_batch))
            .axis(TOKEN, Cut::sequential(3))
            .axis(DIM, Cut::sequential(dim));
        if matches!(stage_at, StageAt::Outer) {
            ops.0.stage(Residence::Smem);
        }
    })
    .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, ops| {
        l.axis(BATCH, Cut::sequential(1))
            .axis(TOKEN, Cut::sequential(1))
            .axis(DIM, Cut::sequential(dim));
        if matches!(stage_at, StageAt::Inner) {
            ops.0.stage(Residence::Smem);
        }
    })
    .build();

    let launcher = space.launcher(&client);
    let packed = launcher
        .bind(&operands.0, packed_t.binding())
        .variable_length(lengths_t.binding(), BATCH, TOKEN)
        .build();
    let padded = launcher
        .bind(&operands.1, padded_t.clone().binding())
        .build();

    unpack_sequences::launch::<f32, TestRuntime>(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        packed.arg(),
        padded.arg(),
        launcher.space().clone(),
    );

    let got = HostData::from_tensor_handle(&client, padded_t, HostDataType::F32);
    for b in 0..batch {
        for token in 0..max_tokens {
            for d in 0..dim {
                let expected = if token < lengths[b] {
                    packed_values[(cumulative[b] as usize + token) * dim + d]
                } else {
                    0.0
                };
                assert_eq!(
                    got.get_f32(&[b, token, d]),
                    expected,
                    "wrong value at sequence {b}, token {token}, dim {d}"
                );
            }
        }
    }
}

#[test]
fn cumulative_lengths_translate_and_bound_each_sequence() {
    check_unpack(1, StageAt::Never);
}

#[test]
fn the_lookup_may_resolve_after_an_outer_sequence_tile() {
    check_unpack(2, StageAt::Inner);
}

#[test]
fn a_resolved_sequence_may_be_staged() {
    check_unpack(1, StageAt::Outer);
}

#[test]
fn writes_past_each_sequence_end_are_skipped() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let lengths = [2usize, 0, 5, 3];
    let cumulative = [0u32, 2, 2, 7, 10];
    let (batch, max_tokens, dim) = (lengths.len(), 6, 3);
    // Two rows past the last sequence's end that no sequence owns. Sequences are walked in
    // ascending order, so every earlier overflow is overwritten by the next sequence's correct
    // writes; only a tail that runs off the *last* sequence has nothing behind it to repair it,
    // and only if the buffer is long enough to accept the write.
    let packed_rows = *cumulative.last().unwrap() as usize + 2;
    let rectangular_values = (0..batch * max_tokens * dim)
        .map(|i| i as f32 + 1.0)
        .collect::<Vec<_>>();
    let rectangular_t = TestInput::builder(client.clone(), shape![batch, max_tokens, dim])
        .dtype(dtype)
        .custom(rectangular_values.clone())
        .generate_without_host_data();
    let cumulative_t = TestInput::builder(client.clone(), shape![batch + 1])
        .dtype(u32::elem_type_native())
        .custom(cumulative.iter().map(|&x| x as f32).collect())
        .generate_without_host_data();
    let packed_t = TestInput::builder(client.clone(), shape![packed_rows, dim])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let mut operands = (
        Operand::new(&[BATCH, TOKEN, DIM], dtype),
        Operand::new(&[TOKEN, DIM], dtype),
    );
    let space = Tiling::over(
        &mut operands,
        &[(BATCH, batch), (TOKEN, max_tokens), (DIM, dim)],
    )
    .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
        l.axis(BATCH, Cut::sequential(1))
            .axis(TOKEN, Cut::sequential(3))
            .axis(DIM, Cut::sequential(dim));
    })
    .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
        l.axis(BATCH, Cut::sequential(1))
            .axis(TOKEN, Cut::sequential(1))
            .axis(DIM, Cut::sequential(dim));
    })
    .build();
    let launcher = space.launcher(&client);
    let rectangular = launcher.bind(&operands.0, rectangular_t.binding()).build();
    let packed = launcher
        .bind(&operands.1, packed_t.clone().binding())
        .variable_length(cumulative_t.binding(), BATCH, TOKEN)
        .build();
    pack_sequences::launch::<f32, TestRuntime>(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        rectangular.arg(),
        packed.arg(),
        launcher.space().clone(),
    );

    // One expected value per physical row, so the rows past the last sequence are asserted to be
    // untouched rather than merely left out of the walk.
    let mut expected = vec![0.0f32; packed_rows * dim];
    for b in 0..batch {
        for token in 0..lengths[b] {
            for d in 0..dim {
                expected[(cumulative[b] as usize + token) * dim + d] =
                    rectangular_values[(b * max_tokens + token) * dim + d];
            }
        }
    }
    let got = HostData::from_tensor_handle(&client, packed_t, HostDataType::F32);
    for row in 0..packed_rows {
        for d in 0..dim {
            assert_eq!(
                got.get_f32(&[row, d]),
                expected[row * dim + d],
                "wrong packed value at row {row}, dim {d}"
            );
        }
    }
}

fn build_with_lengths_layout(shape: &[usize], strides: Option<&[usize]>) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let packed = TestInput::builder(client.clone(), shape![10, 3])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let cumulative = TestInput::builder(client.clone(), shape.to_vec())
        .dtype(u32::elem_type_native())
        .zeros()
        .generate_without_host_data();
    let mut cumulative = cumulative.binding();
    if let Some(strides) = strides {
        cumulative.strides = strides.to_vec().into();
    }
    let mut operands = (Operand::new(&[TOKEN, DIM], dtype),);
    let space = Tiling::over(&mut operands, &[(BATCH, 4), (TOKEN, 5), (DIM, 3)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(BATCH, Cut::sequential(1))
                .axis(TOKEN, Cut::sequential(2))
                .axis(DIM, Cut::sequential(3));
        })
        .build();
    let launcher = space.launcher(&client);
    let _ = launcher
        .bind(&operands.0, packed.binding())
        .variable_length(cumulative, BATCH, TOKEN)
        .build();
}

#[test]
#[should_panic(expected = "cumulative sequence lengths have shape [sequence_count + 1]")]
fn cumulative_lengths_owe_one_more_entry_than_sequences() {
    build_with_lengths_layout(&[4], None);
}

#[test]
#[should_panic(expected = "cumulative sequence lengths have shape [sequence_count + 1]")]
fn cumulative_lengths_are_rank_one() {
    build_with_lengths_layout(&[1, 5], None);
}

#[test]
#[should_panic(expected = "cumulative sequence lengths must be dense row-major")]
fn cumulative_lengths_are_dense() {
    build_with_lengths_layout(&[5], Some(&[2]));
}

/// Build a variable-length operand over a well-formed table, optionally stating a boundary mode
/// the sequence range has to answer for.
fn build_with_boundary(boundary: Option<Option<Boundary>>) -> IndexedOperand<TestRuntime> {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let packed = TestInput::builder(client.clone(), shape![10, 3])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let cumulative = TestInput::builder(client.clone(), shape![5])
        .dtype(u32::elem_type_native())
        .zeros()
        .generate_without_host_data();
    let mut operands = (Operand::new(&[TOKEN, DIM], dtype),);
    let space = Tiling::over(&mut operands, &[(BATCH, 4), (TOKEN, 5), (DIM, 3)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(BATCH, Cut::sequential(1))
                .axis(TOKEN, Cut::sequential(2))
                .axis(DIM, Cut::sequential(3));
        })
        .build();
    let launcher = space.launcher(&client);
    let source = launcher.bind(&operands.0, packed.binding());
    let source = match boundary {
        Some(boundary) => source.with_boundary(boundary),
        None => source,
    };
    source
        .variable_length(cumulative.binding(), BATCH, TOKEN)
        .build()
}

#[test]
fn a_variable_length_target_derives_a_zero_boundary() {
    assert_eq!(
        build_with_boundary(None).spec.boundaries[0],
        Some(Boundary::Zero)
    );
}

/// The exclusive end is the mask, so clamping onto the sequence's last token would read a live
/// value where the tail owes a zero. Refused rather than silently corrected.
#[test]
#[should_panic(expected = "owes its target Boundary::Zero, but this operand states Clamp")]
fn a_variable_length_target_refuses_a_stated_clamp() {
    build_with_boundary(Some(Some(Boundary::Clamp)));
}

/// Likewise for an operand declared unchecked: without the mask a short sequence reads straight
/// into the next one.
#[test]
#[should_panic(expected = "owes its target Boundary::Zero, but this operand states unchecked")]
fn a_variable_length_target_refuses_an_unchecked_declaration() {
    build_with_boundary(Some(None));
}

#[test]
#[should_panic(expected = "no level isolates the variable-length sequence axis")]
fn a_sequence_axis_that_is_never_isolated_is_refused() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let packed = TestInput::builder(client.clone(), shape![10, 3])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let cumulative = TestInput::builder(client.clone(), shape![5])
        .dtype(u32::elem_type_native())
        .zeros()
        .generate_without_host_data();
    let mut operands = (Operand::new(&[TOKEN, DIM], dtype),);
    let space = Tiling::over(&mut operands, &[(BATCH, 4), (TOKEN, 6), (DIM, 3)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(BATCH, Cut::sequential(2))
                .axis(TOKEN, Cut::sequential(3))
                .axis(DIM, Cut::sequential(3));
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(BATCH, Cut::sequential(2))
                .axis(TOKEN, Cut::sequential(1))
                .axis(DIM, Cut::sequential(3));
        })
        .build();
    let launcher = space.launcher(&client);
    let _ = launcher
        .bind(&operands.0, packed.binding())
        .variable_length(cumulative.binding(), BATCH, TOKEN)
        .build();
}
