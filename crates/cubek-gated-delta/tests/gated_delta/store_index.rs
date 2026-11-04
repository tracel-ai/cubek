//! Whether a store bound whole, with a table of slots placing each row's window, is read and
//! written in place: strided, past a gap between slots, and starting off its buffer's start.

use cubecl::bytes::Bytes;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Grid as LaunchGrid;
use cubecl::tile::{Indexing, Launcher, Levels, Partitioning, Space, TileTensor};
use cubecl::zspace::metadata::Metadata;
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_gated_delta::kernels::walk::GATED_DELTA_WALK_ELEMENT;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::gated_delta::{BATCH, KEY_FEATURE, STORE_SLOT, VALUE_FEATURE};
use cubek_std::routine::ADMITTED_PLAN_RUNS;

/// Every cube doubles its row's slot of the store and adds one, each element read, then written
/// where it lies.
#[cube(launch_unchecked)]
fn indexed_write_probe(store: &TileTensor<'_>, partitioning: Partitioning, #[comptime] cells: usize) {
    let store = store.tile(&partitioning);
    for cube in &partitioning {
        let mut slab = store.at(&cube);
        for cell in 0..cells {
            let old = slab.flat::<f32, Const<1>>().read(cell).extract(0usize);
            slab.flat_mut::<f32, Const<1>>().write(
                cell,
                Vector::<f32, Const<1>>::cast_from(old * 2.0f32 + 1.0f32),
            );
        }
    }
}

/// Rows listing shuffled slots of a store with a gap after each slot and a start two floats past
/// its buffer's: the listed slots are updated, the unlisted slot and the gaps are untouched.
#[test]
fn an_indexed_store_is_read_and_written_in_place() {
    for (start, gap) in [(0, 0), (0, 3), (2, 0), (2, 3)] {
        println!("start {start} gap {gap}");
        indexed_store(start, gap);
    }
}

fn indexed_store(start: usize, gap: usize) {
    let client = cubecl::test_device().client();
    let (slots, keys, values) = (5usize, 3usize, 4usize);
    let cells = keys * values;
    let pitch = cells + gap;
    let listed = [3u32, 0, 4];

    let mut buffer = vec![f32::NAN; start + (slots - 1) * pitch + cells];
    let at = |slot: usize, cell: usize| start + slot * pitch + cell;
    for slot in 0..slots {
        for cell in 0..cells {
            buffer[at(slot, cell)] = (slot * 100 + cell) as f32;
        }
    }
    let raw = client.create(Bytes::from_elems(buffer.clone()));
    let handle = raw.clone().offset_start((start * size_of::<f32>()) as u64);
    let store = PlainTensor::of_handle(TensorHandle::from_metadata(
        handle.clone(),
        Metadata::new([slots, keys, values], [pitch, values, 1]),
        GATED_DELTA_WALK_ELEMENT,
    ));
    let table = PlainTensor::of_u32s(&client, &[listed.len()], &listed);

    let space = Space::new(&[
        (BATCH, listed.len()),
        (STORE_SLOT, slots),
        (KEY_FEATURE, keys),
        (VALUE_FEATURE, values),
    ]);
    let partitioning = Partitioning::new(
        space.clone().with_dynamic(&[STORE_SLOT]),
        Levels::leaf(&[(BATCH, 1), (KEY_FEATURE, keys), (VALUE_FEATURE, values)])
            .cubes(&[])
            .batches(&[BATCH])
            .build(),
    );
    let launcher = Launcher::new(
        &client,
        partitioning.clone(),
        &space,
        LaunchGrid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(1),
        },
    )
    .expect(ADMITTED_PLAN_RUNS);
    let entries = launcher
        .arg(table.binding.clone(), ElemType::UInt(UIntKind::U32))
        .axes(&[BATCH])
        .vectorize(1)
        .build()
        .expect(ADMITTED_PLAN_RUNS)
        .arg();
    let store_arg = launcher
        .arg(store.binding.clone(), GATED_DELTA_WALK_ELEMENT)
        .axes(&[STORE_SLOT, KEY_FEATURE, VALUE_FEATURE])
        .vectorize(1)
        .build()
        .expect(ADMITTED_PLAN_RUNS)
        .arg()
        .indexed(entries, Indexing::Coordinate(STORE_SLOT))
        .expect(ADMITTED_PLAN_RUNS);
    // SAFETY: the store is bound through the launcher, which places each window by the table.
    unsafe {
        indexed_write_probe::launch_unchecked(
            &client,
            partitioning.cube_count(),
            partitioning.cube_dim(1),
            store_arg,
            launcher.partitioning_arg(),
            cells,
        );
    }

    let bytes = client.read_one(raw).expect("the store reads back");
    let got = f32::from_bytes(&bytes);
    for slot in 0..slots {
        for cell in 0..cells {
            let old = buffer[at(slot, cell)];
            let expected = if listed.contains(&(slot as u32)) {
                old * 2.0 + 1.0
            } else {
                old
            };
            assert_eq!(
                got[at(slot, cell)],
                expected,
                "start {start} gap {gap} slot {slot} cell {cell}: {got:?}"
            );
        }
    }
    for (i, (&got, &before)) in got.iter().zip(&buffer).enumerate() {
        let in_a_slot = (0..slots).any(|s| (at(s, 0)..at(s, cells)).contains(&i));
        if !in_a_slot {
            assert!(
                got.to_bits() == before.to_bits(),
                "the gap at {i} was written: {got}"
            );
        }
    }
}
