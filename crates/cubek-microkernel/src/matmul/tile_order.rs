//! Tile order: an NVFP4 matrix's values stored for the kernels that read a whole tile in one load
//! — a tile of `k` by some lines of the matrix's other axis, each line's words side by side —
//! beside its block scales in the rows they are packed in, a line's blocks one run, four to a word.
//! The matrix is either input of a product: its axes are `K` and the one it is `across`, `N` for
//! a `[k, n]` input and `M` for an `[m, k]` one.
//!
//! The tile is the writer's: a load picks it for the device from the step of the product
//! ([`MatmulStep`]), a whole number of blocks along `k`. Under the block-scaled
//! instruction, its stages filled by the TMA engine, that step is `128 × 8`: two of the
//! instruction's `16 × 8 × 64` calls deep, so a column's row in a tile is the 64 bytes a box copies
//! at full speed, and the instruction's eight columns wide. A descriptor over the tiles lands a
//! stage of them as it lands rows ([`words_in_tiles`]). A reader takes the tile off the matrix
//! ([`tile`](cubek_std::layout::tile)), never a constant.
//!
//! One memory, stated two ways. The matrix states its values over its own `[k, n]`, as every
//! tensor states its storage; the kernel states them over `[n, k/16, 16]` (`NI`, `KB`, `KI`), or
//! `[m, k/16, 16]` (`MI`, `KB`, `KI`).
//! Both are tilings of the same bytes, and a reader takes the order off the strides, so a
//! restatement moves nothing.
//!
//! [`MatmulStep`]: crate::matmul::MatmulStep

use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubecl::tile::Axis;
use cubecl::tile::layout::{StorageLevels, StoragePartitioning};
use cubecl::zspace::metadata::Metadata;
use cubecl::zspace::{Shape, Strides};
use cubek_std::layout::{Fragments, assemble, fragments_of};
use cubek_std::matmul_axes::{KB, KI, blocked};

/// NVFP4's block: sixteen values along `k` under one scale.
pub const BLOCK: usize = 16;

/// The axes the kernel names the values of a matrix `across` an axis by: `[n, k/16, 16]` across
/// `N`, `[m, k/16, 16]` across `M`.
pub fn values_axes(across: Axis) -> [Axis; 3] {
    [blocked(across).1, KB, KI]
}

/// e2m1 values in a `u32` word.
const VALUES_PER_WORD: usize = 8;

/// The one scheme tile order stores: e2m1 values eight to a word along `k`, under ue4m3 scales
/// for blocks of sixteen along `k` and an f32 per-tensor scale.
pub fn scheme() -> QuantScheme {
    QuantScheme::default()
        .with_value(QuantValue::E2M1)
        .per_block([BLOCK as u8, 1], ScaleDtype::UE4M3)
        .with_store(QuantStore::PackedU32(1))
        .per_tensor(ScaleDtype::F32)
}

/// How the values of a matrix `across` an axis are stored in `tile`, `(k, lines)`, over
/// [`values_axes`], for a device whose load holds `words_per_load` words: eight to a word along
/// `KI`, a line's words down a tile's `k`, then the tile's lines, the tiles ordered `KB` within
/// the lines.
///
/// The read is a level of its own because a reader takes a stated tile whole or not at all:
/// stated as one level, the same bytes are read a word a load.
pub fn values_storage(
    across: Axis,
    tile: (usize, usize),
    words_per_load: usize,
) -> StoragePartitioning {
    let (tile_k, tile_lines) = tile;
    let lines = blocked(across).1;
    // A tile's pieces above the word, finest first, each a count of the piece below it.
    let pieces = [
        (KI, BLOCK / VALUES_PER_WORD),
        (KB, tile_k / BLOCK),
        (lines, tile_lines),
    ];
    let (read, rest) = split_at_load(&pieces, words_per_load);
    // A read of one word or of the whole tile is no level of its own.
    let levels = match (read.is_empty(), rest.is_empty()) {
        (false, false) => vec![read, rest],
        _ => vec![pieces.to_vec()],
    };
    levels
        .iter()
        .fold(
            StorageLevels::new(&[(KI, VALUES_PER_WORD)]),
            |storage, level| storage.tile(level),
        )
        .grid(&[KI, KB, lines])
}

/// A tile's pieces above the word, finest first, each `(axis, count)` of the piece below it.
type Pieces = Vec<(Axis, usize)>;

/// `pieces`, finest first, split where they fill `words` words: the pieces a load takes whole,
/// a piece cut where the load ends inside it, then the rest.
fn split_at_load(pieces: &[(Axis, usize)], words: usize) -> (Pieces, Pieces) {
    let (mut read, mut rest) = (Vec::new(), Vec::new());
    let mut taken = 1;
    for &(axis, count) in pieces {
        let fits = (words / taken).min(count);
        match rest.is_empty() && fits > 1 && count.is_multiple_of(fits) {
            true => {
                read.push((axis, fits));
                taken *= fits;
                if fits < count {
                    rest.push((axis, count / fits));
                }
            }
            false => rest.push((axis, count)),
        }
    }
    (read, rest)
}

/// The matrix's `[k, lines]` statement of values the kernel states over [`values_axes`]: `k` is a
/// block's index then its offset inside it, so `KB`'s fragments come before `KI`'s, coarsest
/// first.
///
/// # Panics
///
/// When `kernel` is not a statement of three logical dims.
pub fn matrix_values(kernel: &Metadata) -> Metadata {
    let [lines, kb, ki] = fragments_of::<3>(kernel)
        .unwrap_or_else(|rank| panic!("tile order states values over three dims, not {rank}"));
    assemble(&[&[kb.as_slice(), ki.as_slice()].concat(), &lines])
}

/// A matrix's values as the rows of `k` a product reads, `[lines, k]`: the same bytes as the
/// matrix's own `[k, lines]` statement, the two logical dims swapped. What a reader whose kernel
/// names no scale blocks binds, `k` one axis over all of its fragments.
///
/// # Panics
///
/// When `matrix` is not a statement of two logical dims.
pub fn values_in_rows(matrix: &Metadata) -> Metadata {
    let [k, n] = fragments_of::<2>(matrix)
        .unwrap_or_else(|rank| panic!("a matrix states its values over two dims, not {rank}"));
    // A fragment of one below an axis's rest is where `KI`'s rest sat as a dim of its own; inside
    // `k` it is no piece. The rest stays, whatever its extent: a reader takes it for the rest.
    let pieces = |fragments: Fragments| {
        let (rest, finer) = fragments.split_first().expect("an axis has a rest");
        core::iter::once(*rest)
            .chain(finer.iter().copied().filter(|&(extent, _)| extent > 1))
            .collect::<Vec<_>>()
    };
    assemble(&[&pieces(n), &pieces(k)])
}

/// The kernel's statement of the values of a matrix `k` deep and `across` an axis of some extent,
/// stored in `tile`, over [`values_axes`], read `words_per_load` words a load
/// ([`values_storage`]). The bytes are the same at every read, so a reader states the one it
/// takes over a matrix stored for another.
pub fn kernel_values_read(
    k: usize,
    (across, lines): (Axis, usize),
    tile: (usize, usize),
    words_per_load: usize,
) -> Metadata {
    let extents = [(blocked(across).1, lines), (KB, k / BLOCK), (KI, BLOCK)];
    let stored = values_storage(across, tile, words_per_load)
        .physical(&extents)
        .expect("tile order's extents close whole tiles");
    Metadata::new(
        Shape::from(stored.shape().to_vec()),
        Strides::from(stored.strides().to_vec()),
    )
    .with_tiling(stored.tiling())
    .expect("cubek states its own tiling")
}

/// A tile-order matrix `k` deep and `lines` across stored in `tile`'s words as the tiles a
/// descriptor reads: the strips of lines, the tiles along `k` in a strip, then a tile's lines, each
/// line's words in a tile one run. A descriptor landing rows (`TmaLanding::Rows`) lands a box of
/// whole tiles of it a line's words after another, as a box over rows of `k` would.
pub fn words_in_tiles(k: usize, lines: usize, tile: (usize, usize)) -> Metadata {
    let (tile_k, tile_lines) = tile;
    let words_per_line = tile_k / VALUES_PER_WORD;
    let tile_words = words_per_line * tile_lines;
    let k_tiles = k / tile_k;
    Metadata::new(
        Shape::from(vec![lines / tile_lines, k_tiles, tile_lines, words_per_line]),
        Strides::from(vec![k_tiles * tile_words, tile_words, words_per_line, 1]),
    )
}

/// A tile-order matrix `k` deep and `across` an axis stored in `tile`'s values as the rows of `k`
/// a product reads, `[lines, k]`, stated for a read of `words_per_load` words:
/// [`kernel_values_read`] restated as [`values_in_rows`] does.
pub fn rows_read(
    k: usize,
    across: (Axis, usize),
    tile: (usize, usize),
    words_per_load: usize,
) -> Metadata {
    values_in_rows(&matrix_values(&kernel_values_read(
        k,
        across,
        tile,
        words_per_load,
    )))
}

/// The kernel's statement of a matrix's values over [`values_axes`]: `k`'s finest fragments, as
/// many as make a block, become `KI`, and the rest `KB`. The statement the tile DSL wrote, exactly:
/// a reader matches the tiles it is stored in, not only the addresses. `None` for a matrix in rows, or one
/// whose `k` fragments do not close a block.
pub fn kernel_values(matrix: &Metadata) -> Option<Metadata> {
    let [k, n] = matrix_fragments(matrix)?;
    let (kb, ki) = k.split_at(block_split(&k)?);
    Some(assemble(&[&n, kb, ki]))
}

/// The tile a matrix in tile order is stored in, `(k, lines)` values: every fragment of each of its
/// dims but the coarsest. `None` for a matrix in rows, or one whose `k` fragments do not close a
/// block, as [`kernel_values`] declines it.
pub fn tile(matrix: &Metadata) -> Option<(usize, usize)> {
    let [k, n] = matrix_fragments(matrix)?;
    block_split(&k)?;
    let inside = |dim: &[(usize, usize)]| dim.iter().skip(1).map(|&(extent, _)| extent).product();
    Some((inside(&k), inside(&n)))
}

/// A tiled matrix's `[k, lines]` fragments, coarsest first; `None` for one in rows.
fn matrix_fragments(matrix: &Metadata) -> Option<[Fragments; 2]> {
    if !matrix.is_tiled() {
        return None;
    }
    fragments_of::<2>(matrix).ok()
}

/// Where `k`'s fragments split into the blocks and the values inside one: its finest fragments,
/// as many as make a block, and any fragment of one coarser than them, where the tile DSL leaves
/// it. `None` where they do not close a block.
fn block_split(k: &[(usize, usize)]) -> Option<usize> {
    let mut split = k.len();
    let mut inside = 1;
    while inside < BLOCK && split > 0 {
        split -= 1;
        inside *= k[split].0;
    }
    if inside != BLOCK {
        return None;
    }
    while split > 0 && k[split - 1].0 == 1 {
        split -= 1;
    }
    Some(split)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubek_std::layout::{self, fragments};
    use cubek_std::matmul_axes::N;

    /// The address `meta` stores logical `coords` at: each coordinate spread over its dim's
    /// fragments, coarsest first.
    fn offset(meta: &Metadata, coords: &[usize]) -> usize {
        fragments(meta)
            .iter()
            .zip(coords)
            .map(|(fragments, &coord)| {
                let mut finer: usize = fragments.iter().map(|&(extent, _)| extent).product();
                fragments
                    .iter()
                    .map(|&(extent, stride)| {
                        finer /= extent;
                        coord / finer % extent * stride
                    })
                    .sum::<usize>()
            })
            .sum()
    }

    /// Words one load holds: a single word, a column's block, half a column's tile of `128` (a
    /// 128-bit load), a column's tile of `128`, and more than a tile holds.
    const LOADS: [usize; 5] = [1, 2, 4, 8, 64];

    /// The tiles a load stores in, `(k, n)`: the step under the block-scaled instruction with
    /// the TMA engine filling its stages, under an `8 × 8 × 8` one, and under none.
    const TILES: [(usize, usize); 3] = [(128, 8), (64, 8), (64, 1)];

    /// The restatement moves nothing: both statements send every value to one address, and no
    /// two values share one. A restatement that misplaced a fragment would have the kernel read
    /// another column's block.
    #[test]
    fn the_rhs_and_the_kernel_state_one_memory() {
        let (k, n) = (128, 48);
        for (tile, words_per_load) in TILES.iter().flat_map(|&tile| LOADS.map(|w| (tile, w))) {
            let kernel = kernel_values_read(k, (N, n), tile, words_per_load);
            let weight = matrix_values(&kernel);
            assert_eq!(weight.logical_shape().unwrap().as_slice(), &[k, n]);
            assert_eq!(
                layout::tile(&weight),
                tile,
                "{tile:?}, {words_per_load} words"
            );

            let mut seen = vec![false; k * n];
            for ki in 0..k {
                for ni in 0..n {
                    let at = offset(&weight, &[ki, ni]);
                    let block = [ni, ki / BLOCK, ki % BLOCK];
                    assert_eq!(at, offset(&kernel, &block), "value ({ki}, {ni})");
                    assert!(
                        !std::mem::replace(&mut seen[at], true),
                        "two values at {at}"
                    );
                }
            }
        }
    }

    /// The kernel recovers its own statement from the one the rhs carries, fragment for
    /// fragment, so the rhs is the only place the layout travels: a reader matches the tiles
    /// it is stored in, and two statements of one memory need not match.
    #[test]
    fn the_kernel_reads_back_what_the_rhs_states() {
        let (k, n) = (128, 48);
        for (tile, words_per_load) in TILES.iter().flat_map(|&tile| LOADS.map(|w| (tile, w))) {
            let kernel = kernel_values_read(k, (N, n), tile, words_per_load);
            let restated = kernel_values(&matrix_values(&kernel))
                .expect("a tile-order rhs closes its blocks");
            assert_eq!(restated, kernel, "{tile:?}, {words_per_load} words a load");
        }
    }

    /// Every load width stores the same bytes: the read is a statement, not a layout. A writer
    /// that moved a value for a wider load would hand a reader built for a narrower one another
    /// rhs.
    #[test]
    fn every_load_width_stores_the_same_bytes() {
        let (k, n) = (128, 48);
        let address = |tile, words_per_load| {
            let kernel = kernel_values_read(k, (N, n), tile, words_per_load);
            (0..n)
                .flat_map(|ni| (0..k).map(move |ki| (ni, ki)))
                .map(|(ni, ki)| offset(&kernel, &[ni, ki / BLOCK, ki % BLOCK]))
                .collect::<Vec<_>>()
        };
        for (tile, words_per_load) in TILES.iter().flat_map(|&tile| LOADS.map(|w| (tile, w))) {
            assert_eq!(
                address(tile, words_per_load),
                address(tile, 1),
                "{tile:?}, {words_per_load} words a load"
            );
        }
    }

    /// The words a descriptor reads in tiles lie where the kernel's statement puts them: word
    /// `w` of column `ni`'s run of `k` is the word holding value `8w` of that column.
    #[test]
    fn a_descriptor_reads_the_words_where_the_tiles_hold_them() {
        let (k, n) = (384, 24);
        let tile = (128, 8);
        let tiles = words_in_tiles(k, n, tile);
        let kernel = kernel_values_read(k, (N, n), tile, 4);
        for ni in 0..n {
            for w in 0..k / VALUES_PER_WORD {
                let (tile_k, tile_n) = tile;
                let per_column = tile_k / VALUES_PER_WORD;
                let coords = [ni / tile_n, w / per_column, ni % tile_n, w % per_column];
                let at: usize = coords
                    .iter()
                    .zip(tiles.strides().iter())
                    .map(|(c, s)| c * s)
                    .sum();
                let ki = w * VALUES_PER_WORD;
                let value = offset(&kernel, &[ni, ki / BLOCK, ki % BLOCK]);
                assert_eq!(at * VALUES_PER_WORD, value, "column {ni}, word {w}");
            }
        }
    }

    /// The values restated as rows of `k` send every value where the rhs's statement does.
    #[test]
    fn the_rows_of_k_state_the_same_memory() {
        let (k, n) = (128, 48);
        let kernel = kernel_values_read(k, (N, n), (128, 8), 4);
        let weight = matrix_values(&kernel);
        let rows = values_in_rows(&weight);
        assert_eq!(layout::tile(&rows), (8, 128));
        for ki in 0..k {
            for ni in 0..n {
                assert_eq!(offset(&rows, &[ni, ki]), offset(&weight, &[ki, ni]));
            }
        }
    }

    /// An rhs in rows is not mistaken for tile order.
    #[test]
    fn an_rhs_in_rows_has_no_kernel_statement() {
        let rows = Metadata::new(Shape::new([64, 48]), vec![48, 1]);
        assert!(kernel_values(&rows).is_none());
        assert!(tile(&rows).is_none());
    }

    /// The tile an rhs states is the one it was stored in, at every load width, read off it in
    /// one pass as the kernel's statement would give it.
    #[test]
    fn an_rhs_states_the_tile_it_was_stored_in() {
        let (k, n) = (128, 48);
        for (stored, words_per_load) in TILES.iter().flat_map(|&tile| LOADS.map(|w| (tile, w))) {
            let weight = matrix_values(&kernel_values_read(k, (N, n), stored, words_per_load));
            assert_eq!(tile(&weight), Some(stored), "{words_per_load} words a load");
        }
    }
}
