//! How a [`Layout`](super::Layout) reads a buffer back, is stated, and is refined.

use super::base::*;
use crate::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// The rule a read's refinement replaces, kept here as the gate that the replacement accepts
/// what it accepted on an untiled buffer: the innermost dim steps by one and holds whole lines, and every coarser stride
/// is a whole number of lines.
fn served_before(geometry: &Geometry, width: usize) -> bool {
    if width == 1 {
        return true;
    }
    let Some(last) = geometry.rank().checked_sub(1) else {
        return false;
    };
    geometry.strides()[last] == 1
        && geometry.shape()[last].is_multiple_of(width)
        && geometry.strides()[..last]
            .iter()
            .all(|s| s.is_multiple_of(width))
}

/// Over plain, padded, transposed and broadcast geometries, and a tiled one read as if it were
/// not, at every width a device reads, a read serves exactly what the rule it replaces did.
#[test]
fn a_read_serves_what_the_line_rule_served() {
    let geometries: Vec<(Geometry, Vec<Axis>)> = vec![
        (Geometry::new(&[(64, 128), (128, 1)]), vec![M, N]),
        (Geometry::new(&[(64, 136), (128, 1)]), vec![M, N]),
        (Geometry::new(&[(64, 130), (130, 1)]), vec![M, N]),
        (Geometry::new(&[(64, 1), (128, 64)]), vec![M, N]),
        (Geometry::new(&[(3, 0), (64, 128), (128, 1)]), vec![M, N]),
        (Geometry::new(&[(1, 5), (64, 128), (128, 1)]), vec![M, N]),
        (Geometry::new(&[(2, 4096), (32, 1)]), vec![K, N]),
        (
            Geometry::new(&[(4, 2048), (4, 512), (16, 32), (32, 1)]),
            vec![K, N, K, N],
        ),
        (Geometry::new(&[(6, 1)]), vec![N]),
        (Geometry::new(&[]), vec![]),
    ];
    for (geometry, labels) in &geometries {
        let layout = Layout::of(geometry, labels);
        for width in [1, 2, 3, 4, 8, 16, 32] {
            assert_eq!(
                layout.serves(width).is_ok(),
                served_before(geometry, width),
                "{geometry:?} at {width}"
            );
        }
    }
}

/// The buffer a stated layout writes, with the tiling it carries: what a reader reads back.
fn stored(layout: &Layout, axes: &[Axis]) -> Geometry {
    let (geometry, tiling) = layout.physical(axes);
    let fragments: Vec<usize> = (0..tiling.rank()).map(|i| tiling.fragments(i)).collect();
    geometry.with_tiling(cubecl::zspace::Tiling::new(&fragments).unwrap())
}

/// A storage-tiled `[k, n]` in a row-major grid read back: the tile's dims are the finest
/// pieces and the grid's the coarsest, though the tiling lists the grid first.
#[test]
fn a_tiled_binding_reads_as_its_pieces() {
    // 64 x 64 in 16 x 32 tiles, row-major tiles: [k/16, n/32, 16, 32].
    let geometry = Geometry::new(&[(4, 1024), (2, 512), (16, 32), (32, 1)])
        .with_tiling(cubecl::zspace::Tiling::new(&[2, 2]).unwrap());
    let layout = Layout::of(&geometry, &[K, N, K, N]);
    assert_eq!(layout.pieces(), vec![(N, 32), (K, 16), (N, 2), (K, 4)]);
}

/// Stated leaf-up and settled against the tensor, a layout describes the buffer cubek has
/// always written for a row-major grid of tiles, and reads back as itself.
#[test]
fn a_row_major_grid_is_the_buffer_cubek_writes() {
    let layout = Layout::tile(&[(N, 32), (M, 16)])
        .grid(&[N, M])
        .over(&[(M, 64), (N, 64)])
        .unwrap();
    let (geometry, tiling) = layout.physical(&[M, N]);
    assert_eq!(
        geometry,
        Geometry::new(&[(4, 1024), (2, 512), (16, 32), (32, 1)])
    );
    assert_eq!(tiling, StorageTiling::per_axis(&[2, 2]));
    assert_eq!(
        Layout::of(&stored(&layout, &[M, N]), &tiling.order(&[M, N])),
        layout
    );
}

/// A grid ordered `K` first puts the next tile along `K` right after this one: the physical
/// dims keep cubecl's order and only the strides move.
#[test]
fn a_k_first_grid_moves_the_strides_not_the_dims() {
    let layout = Layout::tile(&[(N, 16), (K, 32)])
        .grid(&[K, N])
        .over(&[(K, 128), (N, 64)])
        .unwrap();
    let (geometry, _) = layout.physical(&[K, N]);
    // [k/32, n/16, 32, 16]: a tile is 512 values, the next along K 512 on, along N 4 tiles on.
    assert_eq!(
        geometry,
        Geometry::new(&[(4, 512), (4, 2048), (32, 16), (16, 1)])
    );
}

/// Nested tiles whose pieces the tiling lists in another order than memory holds them read
/// back as stated: a tiled buffer is walked by stride.
#[test]
fn nested_tiles_read_back_in_memory_order() {
    let layout = Layout::tile(&[(N, 4), (N, 8)])
        .tile(&[(K, 2), (K, 16)])
        .grid(&[K, N])
        .over(&[(K, 64), (N, 64)])
        .unwrap();
    let (_, tiling) = layout.physical(&[K, N]);
    assert_eq!(
        Layout::of(&stored(&layout, &[K, N]), &tiling.order(&[K, N])),
        layout
    );
}

/// Two tiles and a grid: every size a product, the extents divided once, at the top.
#[test]
fn levels_multiply_and_only_the_grid_divides() {
    let layout = Layout::tile(&[(K, 8)])
        .tile(&[(K, 4), (N, 16)])
        .grid(&[K, N])
        .over(&[(K, 256), (N, 32)])
        .unwrap();
    assert_eq!(
        layout.pieces(),
        vec![(K, 8), (K, 4), (N, 16), (K, 8), (N, 2)]
    );
    let refused = Layout::tile(&[(K, 8)])
        .grid(&[K, N])
        .over(&[(K, 12), (N, 4)]);
    assert_eq!(
        refused,
        Err(LayoutMisfit::PartialTile {
            axis: K,
            extent: 12,
            tile: 8
        })
    );
}

/// A 32 x 32 tile stored for a four-wide read: the read is its own finest tile, and a stage
/// reader that wants the whole tile sees it fused back, exactly the tile without the read.
#[test]
fn a_stage_reader_sees_a_stored_read_fused_away() {
    let with_read = Layout::tile(&[(N, 4)])
        .tile(&[(N, 8), (K, 32)])
        .grid(&[N, K])
        .over(&[(K, 64), (N, 64)])
        .unwrap();
    let without = Layout::tile(&[(N, 32), (K, 32)])
        .grid(&[N, K])
        .over(&[(K, 64), (N, 64)])
        .unwrap();
    let stage = Layout::wanted(&[(N, 32), (K, 32)]);
    assert_eq!(
        with_read.refines(&stage).unwrap(),
        without.refines(&stage).unwrap()
    );
    assert!(with_read.refines(&Layout::wanted(&[(N, 4)])).is_ok());
}

/// A stored tile is taken whole or not at all: a reader wanting less of it, or a tile stored
/// a column at a time for one wanting rows first, is refused and says where.
#[test]
fn a_stored_tile_is_not_cut() {
    let tiles = Layout::tile(&[(N, 32), (K, 32)])
        .grid(&[N, K])
        .over(&[(K, 64), (N, 64)])
        .unwrap();
    assert_eq!(
        tiles.refines(&Layout::wanted(&[(N, 4)])),
        Err(Unrefined::Overshoots {
            axis: N,
            wanted: 4,
            piece: 32
        })
    );
    let column_first = Layout::tile(&[(K, 32), (N, 32)])
        .grid(&[N, K])
        .over(&[(K, 64), (N, 64)])
        .unwrap();
    assert_eq!(
        column_first.refines(&Layout::wanted(&[(N, 32), (K, 32)])),
        Err(Unrefined::Interleaved {
            wanted: N,
            found: K
        })
    );
}

/// An untiled buffer's dims are extents, which a read may cut its own tile out of: the one
/// division, against a count the tensor's size decided.
#[test]
fn a_read_cuts_itself_out_of_an_extent() {
    let rows = Layout::of(&Geometry::new(&[(64, 4096), (4096, 1)]), &[K, N]);
    let read = rows.refines(&Layout::wanted(&[(N, 8)])).unwrap();
    assert_eq!(read.pieces(), vec![(N, 8), (N, 512), (K, 64)]);
    assert!(rows.refines(&Layout::wanted(&[(N, 3)])).is_err());
}

/// NVFP4 values stored for a word of eight along `K` and a four-word read of two words along
/// `K` by two columns: the word and the read are whole tiles, a two-dimensional read included.
#[test]
fn a_word_and_a_two_dimensional_read_are_whole_tiles() {
    let values = Layout::tile(&[(K, 8)])
        .tile(&[(K, 2), (N, 2)])
        .tile(&[(K, 8), (N, 8)])
        .grid(&[K, N])
        .over(&[(K, 256), (N, 64)])
        .unwrap();
    assert!(values.refines(&Layout::wanted(&[(K, 8)])).is_ok());
    assert!(
        values
            .refines(&Layout::wanted(&[(K, 8), (K, 2), (N, 2)]))
            .is_ok()
    );
    assert!(values.refines(&Layout::wanted(&[(K, 16), (N, 2)])).is_ok());
    assert!(values.refines(&Layout::wanted(&[(K, 32)])).is_err());
}
