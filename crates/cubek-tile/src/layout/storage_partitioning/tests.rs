//! Tests for [`StoragePartitioning`](super::StoragePartitioning).

use super::base::*;
use crate::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// A buffer reads back as the statement it was written from, whatever order its tiling lists.
#[test]
fn a_written_buffer_reads_back_as_its_statement() {
    let statements = [
        StorageLevels::new(&[(N, 32), (K, 16)]).grid(&[N, K]),
        StorageLevels::new(&[(N, 16), (K, 32)]).grid(&[K, N]),
        StorageLevels::new(&[(N, 4), (N, 8)])
            .tile(&[(K, 2), (K, 16)])
            .grid(&[K, N]),
        StorageLevels::new(&[(N, 4)])
            .tile(&[(N, 1), (K, 2)])
            .grid(&[N, K]),
    ];
    for storage in statements {
        let geometry = storage.physical(&[(K, 64), (N, 64)]).unwrap();
        let read = StoragePartitioning::new(&geometry, &geometry.labels(&[K, N])).unwrap();
        assert_eq!(read, storage);
    }
}

/// An untiled buffer states no tiles: its order is its dims by stride.
#[test]
fn an_untiled_buffer_is_its_order_alone() {
    let rows = Geometry::new(&[(64, 4096), (4096, 1)]);
    let read = StoragePartitioning::new(&rows, &[K, N]).unwrap();
    assert!(read.tiles().is_empty());
    assert_eq!(read.order(), &[N, K]);

    let transposed = Geometry::new(&[(64, 1), (4096, 64)]);
    let read = StoragePartitioning::new(&transposed, &[K, N]).unwrap();
    assert_eq!(read.order(), &[K, N]);

    let broadcast = Geometry::new(&[(3, 0), (1, 5), (4096, 1)]);
    let read = StoragePartitioning::new(&broadcast, &[M, K, N]).unwrap();
    assert_eq!(read.order(), &[N]);
}

#[test]
fn a_stored_read_and_its_whole_tile_are_both_held() {
    let with_read = StorageLevels::new(&[(N, 4)])
        .tile(&[(N, 8), (K, 32)])
        .grid(&[N, K]);
    let extents = [(K, 64), (N, 64)];
    assert_eq!(with_read.holds(&[(N, 4)], &extents), Ok(()));
    assert_eq!(with_read.holds(&[(N, 32), (K, 32)], &extents), Ok(()));
}

#[test]
fn a_stated_tile_is_never_split() {
    let extents = [(K, 64), (N, 64)];
    let tiles = StorageLevels::new(&[(N, 32), (K, 32)]).grid(&[N, K]);
    assert_eq!(
        tiles.holds(&[(N, 4)], &extents),
        Err(TileMisfit::Overshoots {
            axis: N,
            wanted: 4,
            piece: 32
        })
    );
    let column_first = StorageLevels::new(&[(K, 32), (N, 32)]).grid(&[N, K]);
    assert_eq!(
        column_first.holds(&[(N, 32), (K, 32)], &extents),
        Err(TileMisfit::Interleaved {
            wanted: N,
            found: K
        })
    );
}

#[test]
fn a_read_cuts_itself_out_of_the_rest_of_an_axis() {
    let rows = StorageLevels::new(&[]).grid(&[N, K]);
    let extents = [(K, 64), (N, 4096)];
    assert_eq!(rows.holds(&[(N, 8)], &extents), Ok(()));
    assert!(rows.holds(&[(N, 3)], &extents).is_err());
}

/// NVFP4: a word of eight along `K` and a two-by-two-word read are whole stated tiles.
#[test]
fn a_word_and_a_two_dimensional_read_are_whole_tiles() {
    let values = StorageLevels::new(&[(K, 8)])
        .tile(&[(K, 2), (N, 2)])
        .tile(&[(K, 8), (N, 8)])
        .grid(&[K, N]);
    let extents = [(K, 256), (N, 64)];
    assert_eq!(values.holds(&[(K, 8)], &extents), Ok(()));
    assert_eq!(values.holds(&[(K, 16), (N, 2)], &extents), Ok(()));
    assert!(values.holds(&[(K, 32)], &extents).is_err());
}

#[test]
fn a_k_first_grid_moves_the_strides_not_the_dims() {
    let storage = StorageLevels::new(&[(N, 16), (K, 32)]).grid(&[K, N]);
    let geometry = storage.physical(&[(K, 128), (N, 64)]).unwrap();
    // [k/32, n/16, 32, 16]: a tile is 512 values, the next along K 512 on, along N 4 tiles on.
    assert_eq!(
        geometry,
        Geometry::new(&[(4, 512), (4, 2048), (32, 16), (16, 1)])
            .with_tiling(cubecl::zspace::Tiling::new(&[2, 2]).unwrap())
    );
}

#[test]
fn a_statement_that_does_not_fit_the_tensor_is_refused_when_written() {
    let eight_rows = StorageLevels::new(&[(K, 8)]).grid(&[K, N]);
    assert_eq!(
        eight_rows.physical(&[(K, 12), (N, 4)]),
        Err(StorageMisfit::PartialTile {
            axis: K,
            extent: 12,
            tile: 8
        })
    );
    let one_axis = StorageLevels::new(&[(K, 4)]).grid(&[K]);
    assert!(matches!(
        one_axis.physical(&[(K, 12), (N, 4)]),
        Err(StorageMisfit::Order { .. })
    ));
}

/// A tile's piece of one is kept as a dim of the buffer, and a reader is held the same either way.
#[test]
fn a_piece_of_one_is_a_dim_kept_as_stated() {
    let with_one = StorageLevels::new(&[(N, 4)])
        .tile(&[(N, 1), (K, 2)])
        .grid(&[N, K]);
    let without = StorageLevels::new(&[(N, 4), (K, 2)]).grid(&[N, K]);
    assert_ne!(with_one, without);
    assert_eq!(with_one.labels(&[K, N]), vec![K, N, K, N, N]);
    assert_eq!(without.labels(&[K, N]), vec![K, N, K, N]);
    let (read, extents) = ([(N, 4), (K, 2)], [(K, 8), (N, 8)]);
    assert_eq!(with_one.holds(&read, &extents), Ok(()));
    assert_eq!(without.holds(&read, &extents), Ok(()));
}

/// The level-major order cubecl's tiling lists dims in: every axis's coarsest piece in the
/// tensor's order, then the next of every axis still split, so an untiled axis drops out after
/// the first level and a deeper axis appears alone at the finest.
#[test]
fn labels_name_the_dims_level_major_coarsest_first() {
    let untiled = StorageLevels::new(&[]).grid(&[N, K]);
    assert_eq!(untiled.labels(&[K, N]), vec![K, N]);
    let one_level = StorageLevels::new(&[(N, 32), (K, 32)]).grid(&[N, K]);
    assert_eq!(one_level.labels(&[K, N]), vec![K, N, K, N]);
    let deeper_n = StorageLevels::new(&[(N, 4), (N, 8), (K, 32)]).grid(&[N, K]);
    assert_eq!(deeper_n.labels(&[K, N]), vec![K, N, K, N, N]);
    // A batch axis ahead of the tiled block is its own single dim.
    let batched = StorageLevels::new(&[(N, 32), (K, 32)]).grid(&[N, K, M]);
    assert_eq!(batched.labels(&[M, K, N]), vec![M, K, N, K, N]);
}

#[test]
fn a_window_is_one_run_inside_the_tiles_whose_axes_do_not_interleave() {
    let with_read = StorageLevels::new(&[(N, 4)])
        .tile(&[(N, 8), (K, 32)])
        .grid(&[N, K]);
    assert_eq!(
        with_read.contiguous_tiles(&[(K, 64), (N, 64)]),
        vec![vec![(N, 4)], vec![(N, 32)], vec![(N, 32), (K, 32)]]
    );

    let down_k = StorageLevels::new(&[(N, 16), (K, 32)]).grid(&[K, N]);
    assert_eq!(
        down_k.contiguous_tiles(&[(K, 128), (N, 64)]),
        vec![
            vec![(N, 16)],
            vec![(N, 16), (K, 32)],
            vec![(N, 16), (K, 128)]
        ]
    );

    let interleaved = StorageLevels::new(&[(N, 16), (K, 2)])
        .tile(&[(N, 2), (K, 16)])
        .grid(&[N, K]);
    assert_eq!(
        interleaved.contiguous_tiles(&[(K, 64), (N, 64)]),
        vec![vec![(N, 16)], vec![(N, 16), (K, 2)]]
    );
}
