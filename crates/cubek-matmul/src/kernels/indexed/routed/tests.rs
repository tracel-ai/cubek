//! The routed product's key, listings and floors, the operands it is measured on, and its launch
//! of no rows.

use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::prelude::*;
use cubecl::server::CopyDescriptor;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Decline;
use cubek_std::{Hardware, devices};
use cubek_test_utils::client;

use super::{ROUTED_SHAPES_DISAGREE, ROUTES_NOT_U32, RoutedFloors, RoutedMatmulTensors};
use crate::kernels::indexed::stack::{MATMUL_STACK_NOT_WHOLE_SLABS, SlabDimension};
use crate::kernels::tensor::{PlainTensor, StoredTensor};
use crate::kernels::{
    CubeStagedMatmulRoutineSettings, PlaneShuffleMatmulRoutineSettings, UnitRegistersMatmulRoutine,
    UnitRegistersMatmulRoutineSettings,
};
use crate::kernels::{Listing, Product};

const F32: ElemType = ElemType::Float(FloatKind::F32);

/// `values` uploaded as an `f32` tensor of `shape` laid at `strides`.
fn f32s(client: &Client, shape: [usize; 2], strides: [usize; 2], values: &[f32]) -> PlainTensor {
    let handle = client.create_from_slice(f32::as_bytes(values));
    PlainTensor::of_handle(TensorHandle::new(handle, shape, strides, F32))
}

/// `values` uploaded as a `u32` tensor of `shape`, row-major.
fn u32s(client: &Client, shape: [usize; 2], values: &[u32]) -> PlainTensor {
    let handle = client.create_from_slice(u32::as_bytes(values));
    let strides = [shape[1], 1];
    let elem = ElemType::UInt(UIntKind::U32);
    PlainTensor::of_handle(TensorHandle::new(handle, shape, strides, elem))
}

/// `rows` rows of `k`, each routed to `slots` of `experts` col slabs of `[k, n]`.
fn routed(client: &Client, rows: usize, slots: usize) -> RoutedMatmulTensors {
    routed_by(client, rows, slots, &vec![0; rows * slots])
}

/// [`routed`] with each row's routes `routes`.
fn routed_by(client: &Client, rows: usize, slots: usize, routes: &[u32]) -> RoutedMatmulTensors {
    let (k, n, experts) = (8, 4, 4);
    RoutedMatmulTensors::new(
        client,
        f32s(client, [rows, k], [k, 1], &vec![1.0; rows * k]),
        StoredTensor::Plain(f32s(
            client,
            [k, experts * n],
            [1, k],
            &vec![1.0; experts * n * k],
        )),
        u32s(client, [rows, slots], routes),
        experts,
        SlabDimension::Output,
    )
    .expect("float operands the routines read")
}

/// The rows are keyed by the power of two above them, so a bucket is the same power of two at
/// every autotune level, which the derive's anchor attribute does not give (it scales its base by
/// the level).
#[test]
fn the_key_anchors_its_rows_to_a_power_of_two() {
    let client = client();
    let key = |rows| routed(&client, rows, 2).key();
    assert_eq!(key(65), key(128));
    assert_eq!(key(100), key(128));
    assert_ne!(key(100), key(129));
    assert_ne!(key(5), key(100));
}

/// The slots a row selects tell two products apart: the launch writes a row of output for each.
#[test]
fn the_key_tells_slots_apart() {
    let client = client();
    assert_ne!(routed(&client, 8, 2).key(), routed(&client, 8, 3).key());
}

/// The entry asks the floors listed for its stack's kind and no others: a float stack served only
/// by a packed floor passes the entry and then every floor of the round declines it.
#[test]
fn a_float_stack_is_asked_every_plane_shuffle_floor_the_float_cube_staged_one_and_the_unit_one() {
    let floors = RoutedFloors::LISTED;
    let plane_shuffle: Vec<_> = Listing::admitted(&floors.plane_shuffle, Product::Float)
        .map(|floor| floor.settings)
        .collect();
    let cube_staged: Vec<_> = Listing::admitted(&floors.cube_staged, Product::Float)
        .map(|floor| floor.settings)
        .collect();
    let unit_registers: Vec<_> = Listing::admitted(&floors.unit_registers, Product::Float)
        .map(|floor| floor.settings)
        .collect();
    assert_eq!(
        plane_shuffle,
        [
            PlaneShuffleMatmulRoutineSettings::DEFAULT,
            PlaneShuffleMatmulRoutineSettings::PACKED
        ]
    );
    assert_eq!(cube_staged, [CubeStagedMatmulRoutineSettings::DEFAULT]);
    assert_eq!(
        unit_registers,
        [UnitRegistersMatmulRoutineSettings::DEFAULT]
    );
}

#[test]
fn a_packed_stack_is_asked_every_plane_shuffle_floor_and_the_packed_cube_staged_one() {
    let floors = RoutedFloors::LISTED;
    let plane_shuffle: Vec<_> = Listing::admitted(&floors.plane_shuffle, Product::Packed)
        .map(|floor| floor.settings)
        .collect();
    let cube_staged: Vec<_> = Listing::admitted(&floors.cube_staged, Product::Packed)
        .map(|floor| floor.settings)
        .collect();
    assert_eq!(
        plane_shuffle,
        [
            PlaneShuffleMatmulRoutineSettings::DEFAULT,
            PlaneShuffleMatmulRoutineSettings::PACKED,
            PlaneShuffleMatmulRoutineSettings::PACKED_BY_BLOCK
        ]
    );
    assert_eq!(cube_staged, [CubeStagedMatmulRoutineSettings::PACKED]);
    assert_eq!(
        Listing::admitted(&floors.unit_registers, Product::Packed).count(),
        0
    );
}

/// Routes far out of range, as a discarded launch leaves them, are replaced by routes naming every
/// expert in turn.
#[test]
fn spread_routes_replace_garbage_from_a_discarded_launch() {
    let client = client();
    let (rows, slots, experts) = (5, 3, 4);
    let tensors = routed_by(&client, rows, slots, &vec![0xF000_0000; rows * slots]);
    let spread = tensors.with_spread_routes();
    let routes = spread.routes();
    let bytes = client.read_one_unchecked_tensor(CopyDescriptor::new(
        routes.binding.handle.clone(),
        routes.binding.shape.clone(),
        routes.binding.strides.clone(),
        routes.elem.size(),
    ));
    let routes = u32::from_bytes(&bytes);
    assert_eq!(routes.len(), rows * slots);
    assert!(routes.iter().all(|&route| (route as usize) < experts));
    assert_eq!(routes[..5], [0, 1, 2, 3, 0]);
}

/// On a CPU runtime a float stack is served by a floor, the unit registers asking nothing of the
/// device. A packed stack is the tuned integration tests' to ask, with a real one.
#[test]
fn on_a_cpu_runtime_a_float_stack_has_a_floor() {
    let client = client();
    let tensors = routed(&client, 8, 2);
    let floors = RoutedFloors::LISTED;
    let served = floors.served(&devices::cpu_12(), Product::Float, &tensors);
    assert!(served.is_ok(), "{served:?}");
}

/// No row means no launch: the output is empty rather than refused, whatever the routine.
#[test]
fn no_rows_return_an_empty_output() {
    let client = client();
    let (k, n, experts, slots) = (4usize, 4usize, 2usize, 2usize);
    // Zero-row views over one-value buffers: a device buffer cannot be empty.
    let tensors = RoutedMatmulTensors::new(
        &client,
        f32s(&client, [0, k], [k, 1], &[0.0]),
        StoredTensor::Plain(f32s(
            &client,
            [k, experts * n],
            [experts * n, 1],
            &vec![1.0; k * experts * n],
        )),
        PlainTensor::of_handle(TensorHandle::new(
            client.create_from_slice(u32::as_bytes(&[0])),
            [0, slots],
            [slots, 1],
            ElemType::UInt(UIntKind::U32),
        )),
        experts,
        SlabDimension::Output,
    )
    .expect("no row declines nothing");
    let out = tensors
        .launch(
            &Hardware::from_client(&client),
            &UnitRegistersMatmulRoutine,
            &UnitRegistersMatmulRoutineSettings::DEFAULT,
        )
        .expect("no row declines nothing");
    assert_eq!(out.shape().to_vec(), vec![0, slots, n]);
}

/// What two rows of `k` against a stack of `experts` slabs of four columns, eight deep, decline
/// under `routes`.
fn declined(k: usize, experts: usize, routes: PlainTensor) -> Option<Decline> {
    let client = client();
    RoutedMatmulTensors::new(
        &client,
        f32s(&client, [2, k], [k, 1], &vec![1.0; 2 * k]),
        StoredTensor::Plain(f32s(&client, [8, 16], [16, 1], &[1.0; 128])),
        routes,
        experts,
        SlabDimension::Output,
    )
    .err()
}

#[test]
fn rows_of_another_contraction_decline() {
    let routes = u32s(&client(), [2, 1], &[0; 2]);

    assert_eq!(declined(6, 4, routes), Some(ROUTED_SHAPES_DISAGREE));
}

#[test]
fn routes_of_other_rows_decline() {
    let routes = u32s(&client(), [3, 1], &[0; 3]);

    assert_eq!(declined(8, 4, routes), Some(ROUTED_SHAPES_DISAGREE));
}

#[test]
fn float_routes_decline() {
    let routes = f32s(&client(), [2, 1], [1, 1], &[0.0; 2]);

    assert_eq!(declined(8, 4, routes), Some(ROUTES_NOT_U32));
}

#[test]
fn a_stack_of_parts_of_slabs_declines() {
    let routes = u32s(&client(), [2, 1], &[0; 2]);

    assert_eq!(declined(8, 3, routes), Some(MATMUL_STACK_NOT_WHOLE_SLABS));
}

#[test]
fn a_stack_of_no_slab_declines() {
    let routes = u32s(&client(), [2, 1], &[0; 2]);

    assert_eq!(declined(8, 0, routes), Some(MATMUL_STACK_NOT_WHOLE_SLABS));
}
