//! Storing a matrix in storage tiles and back: the tiled buffer holds the tiles it states,
//! the round trip is the identity, and what cannot be tiled is refused on the host.

use cubecl::{ir::ElemType, prelude::*, zspace::Shape};
use cubek_matmul::{
    definition::MatmulSetupError,
    tiled::storage::{COLS, ROWS, tile, untile},
};
use cubek_test_utils::{HostData, HostDataType, TestInput, client};
use cubek_tile::{Axis, Layout};

/// Tile, check every cell sits in its tile, untile, check every cell is back. `grid` orders the
/// tiles, finest first.
fn round_trip(
    batches: &[usize],
    rows: usize,
    cols: usize,
    (tr, tc): (usize, usize),
    grid: &[Axis],
) {
    let client = client();
    let dtype: ElemType = f32::elem_type_native();
    let shape: Vec<usize> = batches.iter().copied().chain([rows, cols]).collect();
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let (src, _) = TestInput::builder(client.clone(), Shape::from(shape.clone()))
        .dtype(dtype)
        .custom(data.clone())
        .generate_with_f32_host_data();

    let layout = Layout::storage(&[(COLS, tc), (ROWS, tr)]).grid(grid);
    let tiled = tile(&client, src.binding(), dtype, layout).unwrap();
    let physical: Vec<usize> = batches
        .iter()
        .copied()
        .chain([rows / tr, cols / tc, tr, tc])
        .collect();
    assert_eq!(tiled.shape().as_slice(), &physical[..]);
    assert!(tiled.metadata.is_tiled());
    assert_eq!(
        tiled.metadata.logical_shape().unwrap().as_slice(),
        &shape[..]
    );

    let got = HostData::from_tensor_handle(&client, tiled.clone(), HostDataType::F32);
    let nb: usize = batches.iter().product();
    for b in 0..nb {
        // The batch coordinate, outermost first.
        let mut coord = Vec::with_capacity(batches.len() + 4);
        let mut rest = b;
        for &extent in batches.iter().rev() {
            coord.push(rest % extent);
            rest /= extent;
        }
        coord.reverse();
        for r in 0..rows {
            for c in 0..cols {
                let want = data[(b * rows + r) * cols + c];
                let mut at = coord.clone();
                at.extend([r / tr, c / tc, r % tr, c % tc]);
                let have = got.get_f32(&at);
                assert_eq!(have, want, "batch {b} cell ({r}, {c}) tiled at {at:?}");
            }
        }
    }

    let back = untile(&client, tiled.binding(), dtype).unwrap();
    assert_eq!(back.shape().as_slice(), &shape[..]);
    assert!(!back.metadata.is_tiled());
    let got = HostData::from_tensor_handle(&client, back, HostDataType::F32);
    for b in 0..nb {
        let mut coord = Vec::with_capacity(batches.len() + 2);
        let mut rest = b;
        for &extent in batches.iter().rev() {
            coord.push(rest % extent);
            rest /= extent;
        }
        coord.reverse();
        for r in 0..rows {
            for c in 0..cols {
                let want = data[(b * rows + r) * cols + c];
                let mut at = coord.clone();
                at.extend([r, c]);
                assert_eq!(got.get_f32(&at), want, "batch {b} cell ({r}, {c}) untiled");
            }
        }
    }
}

#[test]
fn tiling_stores_one_tile_at_a_time() {
    round_trip(&[], 64, 96, (16, 32), &[COLS, ROWS]);
}

/// A grid ordered rows first puts the next tile down the rows right after this one: the dims are
/// the same, the grid's strides are swapped, and the round trip still holds.
#[test]
fn tiling_orders_the_grid_as_the_layout_states() {
    round_trip(&[], 64, 96, (16, 32), &[ROWS, COLS]);
    let client = client();
    let dtype: ElemType = f32::elem_type_native();
    let (src, _) = TestInput::builder(client.clone(), Shape::from(vec![64, 96]))
        .dtype(dtype)
        .custom((0..64 * 96).map(|i| i as f32).collect())
        .generate_with_f32_host_data();
    let layout = Layout::storage(&[(COLS, 32), (ROWS, 16)]).grid(&[ROWS, COLS]);
    let tiled = tile(&client, src.binding(), dtype, layout).unwrap();
    // [64/16, 96/32, 16, 32]: a tile is 512 values, the next down the rows 512 on, across 2048.
    assert_eq!(tiled.metadata.strides().to_vec(), vec![512, 2048, 32, 1]);
}

#[test]
fn tiling_takes_a_narrow_column_run() {
    round_trip(&[], 128, 64, (32, 8), &[COLS, ROWS]);
}

#[test]
fn tiling_keeps_batches_plain() {
    round_trip(&[3], 32, 64, (16, 16), &[COLS, ROWS]);
}

#[test]
fn tiling_keeps_two_batch_dims_plain() {
    round_trip(&[2, 3], 32, 32, (8, 16), &[COLS, ROWS]);
}

#[test]
fn tiling_refuses_a_tile_that_does_not_divide() {
    let client = client();
    let dtype: ElemType = f32::elem_type_native();
    let (src, _) = TestInput::builder(client.clone(), Shape::from(vec![48, 64]))
        .dtype(dtype)
        .custom((0..48 * 64).map(|i| i as f32).collect())
        .generate_with_f32_host_data();
    assert!(matches!(
        tile(
            &client,
            src.binding(),
            dtype,
            Layout::storage(&[(COLS, 32), (ROWS, 32)]).grid(&[COLS, ROWS])
        ),
        Err(MatmulSetupError::InvalidConfig(_))
    ));
}

#[test]
fn tiling_refuses_a_tiled_source_and_untiling_a_plain_one() {
    let client = client();
    let dtype: ElemType = f32::elem_type_native();
    let (src, _) = TestInput::builder(client.clone(), Shape::from(vec![32, 32]))
        .dtype(dtype)
        .custom((0..32 * 32).map(|i| i as f32).collect())
        .generate_with_f32_host_data();
    assert!(matches!(
        untile(&client, src.clone().binding(), dtype),
        Err(MatmulSetupError::InvalidConfig(_))
    ));
    let tiled = tile(
        &client,
        src.binding(),
        dtype,
        Layout::storage(&[(COLS, 16), (ROWS, 16)]).grid(&[COLS, ROWS]),
    )
    .unwrap();
    assert!(matches!(
        tile(
            &client,
            tiled.binding(),
            dtype,
            Layout::storage(&[(COLS, 16), (ROWS, 16)]).grid(&[COLS, ROWS])
        ),
        Err(MatmulSetupError::InvalidConfig(_))
    ));
}
