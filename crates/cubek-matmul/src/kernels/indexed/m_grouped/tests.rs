//! The M-grouped launch against a host reference.

use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::Axis;
use cubecl::tile::launch::Refusal;
use cubek_std::autotune::GroupPriority;
use cubek_test_utils::client;

use super::{MGroupedMatmulTensors, launch_tuned};
use crate::kernels::MATMUL_SLABS_TAKE_NO_STORED_TILES;
use crate::kernels::indexed::m_grouped::upload;
use crate::kernels::indexed::stack::SlabAxis;
use crate::kernels::tensor::{PlainTensor, StoredTensor};
use crate::tiled::storage::{StorageLevels, tile};

const F32: ElemType = ElemType::Float(FloatKind::F32);

/// `values` uploaded as an `f32` tensor of `shape` laid at `strides`.
fn f32s(client: &Client, shape: [usize; 2], strides: [usize; 2], values: &[f32]) -> PlainTensor {
    let handle = client.create_from_slice(f32::as_bytes(values));
    PlainTensor::of_handle(TensorHandle::new(handle, shape, strides, F32))
}

/// `values` uploaded as a `u32` vector.
fn u32s(client: &Client, values: &[u32]) -> PlainTensor {
    PlainTensor::of_handle(upload(client, values))
}

/// The grouped product of `lhs` against the slabs of `rhs` laid `along`, grouped by `ends`,
/// through the grouped autotune.
fn launch(
    client: &Client,
    lhs: PlainTensor,
    rhs: PlainTensor,
    ends: PlainTensor,
    along: SlabAxis,
) -> Result<TensorHandle, Refusal> {
    let tensors = MGroupedMatmulTensors::new(client, lhs, StoredTensor::Plain(rhs), ends, along)?;
    launch_tuned(&cubecl::test_device(), tensors, GroupPriority::DEFAULT)
}

/// Every row against a host reference: ragged groups, one of them empty
/// and one wider than a tile, so the padding, the skipped group and a
/// group spanning tiles are all on the path.
#[test]
fn each_row_contracts_against_its_groups_slab() {
    let client = client();
    let (k, n) = (64usize, 48usize);
    let sizes = [5usize, 0, 11, 2];
    let groups = sizes.len();
    let rows: usize = sizes.iter().sum();
    let ends: Vec<u32> = sizes
        .iter()
        .scan(0, |end, size| {
            *end += size;
            Some(*end as u32)
        })
        .collect();
    let group_of = |m: usize| ends.iter().position(|&end| m < end as usize).unwrap();
    let x: Vec<f32> = (0..rows * k).map(|i| ((i % 11) as f32) - 5.0).collect();
    let weight = |g: usize, i: usize, j: usize| ((g + 1) * ((i * 3 + j) % 5 + 1)) as f32;
    // A col stack: an `[groups · n, k]` buffer behind the `[k, groups · n]` view.
    let buffer: Vec<f32> = (0..groups * n * k)
        .map(|p| weight(p / k / n, p % k, p / k % n))
        .collect();

    let out = launch(
        &client,
        f32s(&client, [rows, k], [k, 1], &x),
        f32s(&client, [k, groups * n], [1, k], &buffer),
        u32s(&client, &ends),
        SlabAxis::Output,
    )
    .expect("a col stack derives a grouped launch");
    assert_eq!(out.shape().to_vec(), vec![rows, n]);
    let bytes = client.read_one(out.handle).expect("the output reads back");
    let got = f32::from_bytes(&bytes);
    for m in 0..rows {
        let g = group_of(m);
        for j in 0..n {
            let want: f32 = (0..k).map(|i| x[m * k + i] * weight(g, i, j)).sum();
            assert_eq!(got[m * n + j], want, "row {m} column {j}, group {g}");
        }
    }
}

/// No row means no launch: the output is empty rather than refused.
#[test]
fn no_rows_return_an_empty_output() {
    let client = client();
    let (k, n, groups) = (4usize, 4usize, 2usize);
    // A zero-row view over a one-value buffer: a device buffer cannot be empty.
    let out = launch(
        &client,
        f32s(&client, [0, k], [k, 1], &[0.0]),
        f32s(
            &client,
            [k, groups * n],
            [groups * n, 1],
            &vec![1.0; k * groups * n],
        ),
        u32s(&client, &[0, 0]),
        SlabAxis::Output,
    )
    .expect("no row declines nothing");
    assert_eq!(out.shape().to_vec(), vec![0, n]);
}

/// A stack stored in tiles is refused rather than read as rows: a grouped product reads its
/// stack a slab at a time through a table, which reads no stored tile.
#[test]
fn a_stack_stored_in_tiles_is_refused() {
    let client = client();
    let (rows, k, n, groups) = (8, 32, 32, 2);
    let (row_axis, col_axis) = (Axis(0), Axis(1));
    let stack = f32s(
        &client,
        [k, groups * n],
        [groups * n, 1],
        &vec![1.0; k * groups * n],
    );
    let storage = StorageLevels::new(&[(col_axis, 16), (row_axis, 16)]).grid(&[col_axis, row_axis]);
    let tiled = tile(
        &client,
        stack.binding,
        &[row_axis, col_axis],
        F32,
        None,
        storage,
    )
    .expect("the stack tiles");
    let declined = launch(
        &client,
        f32s(&client, [rows, k], [k, 1], &vec![1.0; rows * k]),
        PlainTensor::of_handle(tiled),
        u32s(&client, &[4, 8]),
        SlabAxis::Output,
    );
    assert!(
        matches!(declined, Err(MATMUL_SLABS_TAKE_NO_STORED_TILES)),
        "{declined:?}"
    );
}
