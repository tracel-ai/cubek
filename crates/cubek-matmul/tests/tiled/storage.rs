//! Storing a matrix in storage tiles and back: the tiled buffer holds the tiles it states,
//! the round trip is the identity, and what cannot be tiled is refused on the host.

use cubecl::{ir::ElemType, prelude::*, zspace::Shape};
use cubek_matmul::{
    definition::MatmulSetupError,
    tiled::storage::{LayoutBuilder, tile, untile},
};
use cubek_test_utils::{HostData, HostDataType, TestInput, client};
use cubek_tile::Axis;

const ROWS: Axis = Axis(0);
const COLS: Axis = Axis(1);

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

    let layout = LayoutBuilder::new(&[(COLS, tc), (ROWS, tr)]).grid(grid);
    let tiled = tile(&client, src.binding(), [ROWS, COLS], dtype, layout).unwrap();
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
    let layout = LayoutBuilder::new(&[(COLS, 32), (ROWS, 16)]).grid(&[ROWS, COLS]);
    let tiled = tile(&client, src.binding(), [ROWS, COLS], dtype, layout).unwrap();
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
            [ROWS, COLS],
            dtype,
            LayoutBuilder::new(&[(COLS, 32), (ROWS, 32)]).grid(&[COLS, ROWS])
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
        [ROWS, COLS],
        dtype,
        LayoutBuilder::new(&[(COLS, 16), (ROWS, 16)]).grid(&[COLS, ROWS]),
    )
    .unwrap();
    assert!(matches!(
        tile(
            &client,
            tiled.binding(),
            [ROWS, COLS],
            dtype,
            LayoutBuilder::new(&[(COLS, 16), (ROWS, 16)]).grid(&[COLS, ROWS])
        ),
        Err(MatmulSetupError::InvalidConfig(_))
    ));
}

/// Tile as `layout` states, find every cell where `stored_at` says the layout put it, untile, and
/// read back the matrix it started as; the stored buffer's dims and strides are returned for the
/// caller to check.
fn tiles_and_comes_back(
    layout: cubek_tile::layout::GridLayout,
    rows: usize,
    cols: usize,
    stored_at: impl Fn(usize, usize) -> Vec<usize>,
) -> (Vec<usize>, Vec<usize>) {
    let client = client();
    let dtype: ElemType = f32::elem_type_native();
    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    let (src, _) = TestInput::builder(client.clone(), Shape::from(vec![rows, cols]))
        .dtype(dtype)
        .custom(data.clone())
        .generate_with_f32_host_data();
    let tiled = tile(&client, src.binding(), [ROWS, COLS], dtype, layout).unwrap();
    let stored = (
        tiled.shape().as_slice().to_vec(),
        tiled.metadata.strides().to_vec(),
    );
    let raw = HostData::from_tensor_handle(&client, tiled.clone(), HostDataType::F32);
    for r in 0..rows {
        for c in 0..cols {
            let at = stored_at(r, c);
            assert_eq!(
                raw.get_f32(&at),
                data[r * cols + c],
                "cell ({r}, {c}) at {at:?}"
            );
        }
    }
    let back = untile(&client, tiled.binding(), dtype).unwrap();
    let got = HostData::from_tensor_handle(&client, back, HostDataType::F32);
    for r in 0..rows {
        for c in 0..cols {
            assert_eq!(got.get_f32(&[r, c]), data[r * cols + c], "cell ({r}, {c})");
        }
    }
    stored
}

/// Two storage levels: 2 x 4 blocks, 4 x 2 of them to a tile, the tiles down the rows first.
/// Each dim is stored in three pieces, listed coarsest first, and the relayout walks one outer
/// tile per cube whatever lies inside it.
#[test]
fn tiling_nests_two_levels() {
    let layout = LayoutBuilder::new(&[(COLS, 4), (ROWS, 2)])
        .tile(&[(COLS, 2), (ROWS, 4)])
        .grid(&[ROWS, COLS]);
    // rows = grid 4 x 4 x 2, cols = grid 6 x 2 x 4, each dim's pieces listed coarsest first.
    let (shape, strides) = tiles_and_comes_back(layout, 32, 48, |r, c| {
        vec![r / 8, c / 8, r / 2 % 4, c / 4 % 2, r % 2, c % 4]
    });
    assert_eq!(shape, vec![4, 6, 4, 2, 2, 4]);
    // Finest first: cols 4 (1), rows 2 (4), cols 2 (8), rows 4 (16), grid rows 4 (64), cols 6
    // (256), listed back in the tiling's coarsest-first order.
    assert_eq!(strides, vec![64, 256, 16, 8, 4, 1]);
}

/// A tile stored a column at a time: its rows are the finest entry, and only the strides say so.
#[test]
fn tiling_stores_a_tile_column_first() {
    let layout = LayoutBuilder::new(&[(ROWS, 16), (COLS, 32)]).grid(&[COLS, ROWS]);
    let (shape, strides) =
        tiles_and_comes_back(layout, 64, 96, |r, c| vec![r / 16, c / 32, r % 16, c % 32]);
    assert_eq!(shape, vec![4, 3, 16, 32]);
    assert_eq!(strides, vec![1536, 512, 1, 16]);
}
