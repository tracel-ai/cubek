//! Tile order: a block-scaled matrix's values ([`TileOrder`]: NVFP4's, MXFP4's, MXFP8's) stored for
//! the kernels that read a whole tile in one load — a tile of `k` by some lines of the matrix's
//! other axis — beside its block scales, as many to a word as the scale type packs.
//! The matrix is either input of a product: its axes are `K` and the one it is `across`, `N` for
//! a `[k, n]` input and `M` for an `[m, k]` one.
//!
//! A tile's words lie in one of two orders ([`TileWords`]), and the device decides which: each
//! line's words down the tile's `k`, where the TMA engine copies rows of a tile and a tensor-core
//! plane reads a line's run, its scales in the rows they are packed in; or each block's words of
//! every line side by side, where software reads the tile a load at a time, so a load is one block
//! of several lines, its scales stored in tiles of their own beside the values.
//!
//! The tile is the writer's: a load picks it for the device from the step of the product
//! (`MatmulStep`), a whole number of blocks along `k`. Under the block-scaled
//! instruction, its stages filled by the TMA engine, NVFP4's step is `128 × 8`: two of the
//! instruction's `16 × 8 × 64` calls deep, so a column's row in a tile is the 64 bytes a box copies
//! at full speed, and the instruction's eight columns wide. A descriptor over the tiles lands a
//! stage of them as it lands rows ([`TileOrder::words_in_tiles`]). A reader takes the tile off the matrix
//! ([`tile`](crate::layout::tile)), never a constant.
//!
//! One memory, stated two ways. The matrix states its values over its own `[k, n]`, as every
//! tensor states its storage; the kernel states them over `[n, k/block, block]` (`NI`, `KB`,
//! `KI`), or `[m, k/block, block]` (`MI`, `KB`, `KI`).
//! Both are tilings of the same bytes, and a reader takes the order off the strides, so a
//! restatement moves nothing.
//!

use crate::NVFP4_BLOCK;
use crate::layout::stored::Operand;
use crate::layout::{Fragments, assemble, fragments_of};
use crate::matmul_axes::{K, KB, KI, blocked};
use cubecl::ir::ElemType;
use cubecl::quant::scheme::{QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubecl::tile::Axis;
use cubecl::tile::layout::{StorageLevels, StoragePartitioning};
use cubecl::zspace::metadata::Metadata;
use cubecl::zspace::{Shape, Strides};

/// The axes the kernel names the values of a matrix `across` an axis by: `[n, k/block, block]`
/// across `N`, `[m, k/block, block]` across `M`.
pub fn values_axes(across: Axis) -> [Axis; 3] {
    [blocked(across).1, KB, KI]
}

/// The axes the kernel names the block scales of a matrix `across` an axis by: `[n, k/block]`
/// across `N`, `[m, k/block]` across `M`.
pub fn scales_axes(across: Axis) -> [Axis; 2] {
    [blocked(across).1, KB]
}

/// The order a tile's words lie in, past a block's words of one line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TileWords {
    /// Each line's words down the tile's `k` one run, the lines one after another: a row of the
    /// tile is a line's run, which the TMA engine copies and a tensor-core plane reads whole. The
    /// block scales stay in the rows they are packed in, a line's blocks one run.
    LineByLine,
    /// Each block's words of every line of the tile side by side, the blocks one after another: a
    /// load is one block of as many lines as it holds, each under its own scale. The block scales
    /// are stored in tiles of their own ([`TileOrder::scales_storage`]), a tile's lines' words
    /// side by side, so a load of them holds the scales of every line a load of values spans.
    BlockByBlock,
}

/// How a matrix packed under one scheme lies in tile order: the values of `k` one block scale
/// covers, the values one `u32` word holds, and the order its tiles' words lie in. Every scheme
/// tile order stores has one ([`of`](Self::of)): codes packed down `k`, under a scale every whole
/// number of words of `k`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileOrder {
    /// Values of `k` one block scale covers.
    pub block: usize,
    /// Values one stored word holds.
    pub values_per_word: usize,
    /// The order a tile's words lie in.
    pub words: TileWords,
}

impl TileOrder {
    /// Block scales one stored word holds: tile order stores byte scales, four to a `u32`, and
    /// [`of`](Self::of) and [`of_operand`](Self::of_operand) refuse any other.
    pub const SCALES_PER_WORD: usize = 4;

    /// NVFP4's tile order: a scale every sixteen values, eight to a word, its tiles line by line.
    pub const NVFP4: Self = Self {
        block: NVFP4_BLOCK,
        values_per_word: 8,
        words: TileWords::LineByLine,
    };

    /// This order with its tiles' words lying as `words` says.
    pub const fn with_words(self, words: TileWords) -> Self {
        Self { words, ..self }
    }

    /// This order laid as `words` asks in `tile`, `(k, lines)`: block by block only where the
    /// tile's `k` holds whole words of block scales, which its scales' tiles are made of, and the
    /// tile holds more than one line, the only tile the two orders lay out differently; line by
    /// line otherwise, which lays the values out in any tile. A writer lays its tiles so, and
    /// [`words_of`](Self::words_of) reads the same order back.
    pub fn laid_in(self, words: TileWords, (tile_k, tile_lines): (usize, usize)) -> Self {
        let scale_word = self.block * Self::SCALES_PER_WORD;
        match words {
            TileWords::BlockByBlock if tile_k.is_multiple_of(scale_word) && tile_lines > 1 => {
                self.with_words(words)
            }
            TileWords::BlockByBlock | TileWords::LineByLine => {
                self.with_words(TileWords::LineByLine)
            }
        }
    }

    /// The order `matrix`'s tiles lie in, read off its own fragments: block by block where a
    /// block's words of one line are followed by the next line's, line by line where by that
    /// line's next block. A tile one block deep or one line wide lies the same either way, and a
    /// tile too shallow for whole words of block scales is never laid block by block
    /// ([`laid_in`](Self::laid_in)): both read as line by line. `None` for a matrix in rows, or one
    /// whose `k` fragments do not close a block.
    ///
    /// What every reader asks: the matrix carries its order, and a reader that worked it out again
    /// from its own device could disagree with the writer that laid it.
    pub fn words_of(self, matrix: &Metadata) -> Option<TileWords> {
        self.laid_out(matrix).map(|(order, _)| order.words)
    }

    /// This order as `matrix` lays it out, and the tile it is stored in, `(k, lines)` values:
    /// [`words_of`](Self::words_of) and [`tile`](Self::tile) read in one pass over its fragments,
    /// what a reader binding the matrix asks on every launch. `None` for a matrix in rows, or one
    /// whose `k` fragments do not close a block.
    pub fn laid_out(self, matrix: &Metadata) -> Option<(Self, (usize, usize))> {
        let [k, lines] = matrix_fragments(matrix)?;
        let (blocks, _) = k.split_at(self.block_split(&k)?);
        let inside =
            |dim: &[(usize, usize)]| dim.iter().skip(1).map(|&(extent, _)| extent).product();
        let tile = (inside(&k), inside(&lines));
        let finest = |dim: &[(usize, usize)]| {
            dim.iter()
                .filter(|&&(extent, _)| extent > 1)
                .map(|&(_, stride)| stride)
                .min()
        };
        let stated = match (finest(&lines), finest(blocks)) {
            (Some(line), Some(block)) if line < block => TileWords::BlockByBlock,
            _ => TileWords::LineByLine,
        };
        Some((self.laid_in(stated, tile), tile))
    }

    /// The tile order of a matrix packed under `scheme`: a symmetric scheme whose words run down
    /// `k` (`PackedU32(1)` of a `[k, lines]` matrix), under block scales of one byte along `k`
    /// alone, four to a word as a block-scaled instruction reads them, each block a whole number
    /// of words. NVFP4's is sixteen values of eight a word, MXFP4's thirty-two of eight, MXFP8's
    /// thirty-two of four. `None` for a scheme tile order does not store.
    ///
    /// The scheme says nothing of the order a matrix's tiles lie in: these tiles are line by line.
    /// A writer chooses its order ([`laid_in`](Self::laid_in)), and a reader reads the matrix's
    /// ([`of_matrix`](Self::of_matrix)).
    pub fn of(scheme: &QuantScheme) -> Option<Self> {
        let QuantStore::PackedU32(1) = scheme.store else {
            return None;
        };
        let dims = scheme.block_size()?.to_dim_vec(2);
        let byte_scales = ElemType::from_scale_dtype(scheme.block_scale()?.dtype).size_bits() == 8;
        let symmetric = scheme.mode == QuantMode::Symmetric;
        if !(symmetric && byte_scales && dims[1] == 1) {
            return None;
        }
        Self::in_whole_words(dims[0] as usize, scheme.num_quants())
    }

    /// The tile order `matrix` lies in, packed under `scheme`: the scheme's block and word, its tiles'
    /// words as the matrix's own fragments say ([`words_of`](Self::words_of)). `None` for a scheme
    /// tile order does not store, or a matrix in rows.
    pub fn of_matrix(scheme: &QuantScheme, matrix: &Metadata) -> Option<Self> {
        Self::of(scheme)?.laid_out(matrix).map(|(order, _)| order)
    }

    /// The tile order of a matrix read as `operand`: its words down `K`, each a whole read, under a
    /// scale every whole number of them along `K`, byte scales
    /// [`SCALES_PER_WORD`](Self::SCALES_PER_WORD) to a word, its tiles line by line as
    /// [`of`](Self::of)'s are. `None` for an operand tile order does not store.
    pub fn of_operand(operand: &Operand) -> Option<Self> {
        if operand.lined() != K || operand.scales_per_word() != Self::SCALES_PER_WORD {
            return None;
        }
        Self::in_whole_words(operand.scale(K)?, operand.read(K))
    }

    /// A scale every `block` values, `values_per_word` to a word, its tiles line by line: `None`
    /// unless the words pack several values and a block is a whole number of them.
    fn in_whole_words(block: usize, values_per_word: usize) -> Option<Self> {
        (values_per_word > 1 && block.is_multiple_of(values_per_word)).then_some(Self {
            block,
            values_per_word,
            words: TileWords::LineByLine,
        })
    }

    /// How the values of a matrix `across` an axis are stored in `tile`, `(k, lines)`, over
    /// [`values_axes`], for a device whose load holds `words_per_load` words: a word's values along
    /// `KI`, a block's words, then as [`words`](Self::words) orders them a line's blocks down the
    /// tile's `k` and the tile's lines, or a block's lines and the tile's blocks, the tiles ordered
    /// `KB` within the lines.
    ///
    /// The read is a level of its own because a reader takes a stated tile whole or not at all:
    /// stated as one level, the same bytes are read a word a load.
    pub fn values_storage(
        self,
        across: Axis,
        tile: (usize, usize),
        words_per_load: usize,
    ) -> StoragePartitioning {
        let lines = blocked(across).1;
        stored_in_levels(
            (KI, self.values_per_word),
            &self.values_pieces(across, tile),
            words_per_load,
            &[KI, KB, lines],
        )
    }

    /// A tile's pieces of values above the word, finest first, each a count of the piece below
    /// it: a block's words, then a line's blocks and the lines, or a block's lines and the blocks,
    /// as [`words`](Self::words) orders them where the tile holds that order
    /// ([`laid_in`](Self::laid_in)), so values are never laid in an order the matrix reads back
    /// as another.
    fn values_pieces(self, across: Axis, tile: (usize, usize)) -> [(Axis, usize); 3] {
        let (tile_k, tile_lines) = tile;
        let block = (KI, self.block / self.values_per_word);
        let down_k = (KB, tile_k / self.block);
        let across_lines = (blocked(across).1, tile_lines);
        match self.laid_in(self.words, tile).words {
            TileWords::LineByLine => [block, down_k, across_lines],
            TileWords::BlockByBlock => [block, across_lines, down_k],
        }
    }

    /// The lines one load of values `words_per_load` words wide spans in `tile`: one a load line by
    /// line, and block by block as many as a block's words leave room for. The lines a read of
    /// their scales spans, so it brings every scale that load needs and no other reader's.
    pub fn lines_a_load(self, across: Axis, tile: (usize, usize), words_per_load: usize) -> usize {
        let lines = blocked(across).1;
        let (read, _) = split_at_load(&self.values_pieces(across, tile), words_per_load);
        read.iter()
            .find(|&&(axis, _)| axis == lines)
            .map_or(1, |&(_, count)| count)
    }

    /// How the block scales of a matrix `across` an axis are stored beside values stored in `tile`,
    /// `(k, lines)`, over [`scales_axes`], read `lines_a_read` lines a read: a word's
    /// [`SCALES_PER_WORD`](Self::SCALES_PER_WORD) blocks of one line along `KB`, the tile's lines'
    /// words side by side, then the tile's further words down `k`, the tiles ordered as the
    /// values' are. A line's scales of a tile's blocks are one word, so a read of `lines_a_read`
    /// lines is that many words; the lines a load of values spans ([`lines_a_load`](Self::lines_a_load))
    /// make it the read that serves one load of values. Every read stores the same bytes.
    ///
    /// `None` for tiles laid line by line, whose scales stay in the rows they are packed in
    /// ([`TileWords::LineByLine`]), and for a tile too shallow for whole words of them
    /// ([`laid_in`](Self::laid_in)).
    pub fn scales_storage(
        self,
        across: Axis,
        tile: (usize, usize),
        lines_a_read: usize,
    ) -> Option<StoragePartitioning> {
        if self.laid_in(self.words, tile).words == TileWords::LineByLine {
            return None;
        }
        let (tile_k, tile_lines) = tile;
        let lines = blocked(across).1;
        let pieces = [
            (lines, tile_lines),
            (KB, tile_k / (self.block * Self::SCALES_PER_WORD)),
        ];
        Some(stored_in_levels(
            (KB, Self::SCALES_PER_WORD),
            &pieces,
            lines_a_read,
            &[KB, lines],
        ))
    }

    /// The kernel's statement of the block scales of a matrix `k` deep and `across` an axis of some
    /// extent, beside values stored in `tile`, over [`scales_axes`], read `lines_a_read` lines a
    /// read ([`scales_storage`](Self::scales_storage)), counted in scales. `None` where they are
    /// not stored in tiles of their own.
    pub fn kernel_scales_read(
        self,
        k: usize,
        (across, lines): (Axis, usize),
        tile: (usize, usize),
        lines_a_read: usize,
    ) -> Option<Metadata> {
        let storage = self.scales_storage(across, tile, lines_a_read)?;
        Some(stated(
            &storage,
            &[(blocked(across).1, lines), (KB, k / self.block)],
        ))
    }

    /// The kernel's statement of the values of a matrix `k` deep and `across` an axis of some extent,
    /// stored in `tile`, over [`values_axes`], read `words_per_load` words a load
    /// ([`values_storage`](Self::values_storage)). The bytes are the same at every read, so a reader
    /// states the one it takes over a matrix stored for another.
    pub fn kernel_values_read(
        self,
        k: usize,
        (across, lines): (Axis, usize),
        tile: (usize, usize),
        words_per_load: usize,
    ) -> Metadata {
        stated(
            &self.values_storage(across, tile, words_per_load),
            &[
                (blocked(across).1, lines),
                (KB, k / self.block),
                (KI, self.block),
            ],
        )
    }

    /// A tile-order matrix `k` deep and `lines` across stored in `tile`'s words as the tiles a
    /// descriptor reads: the strips of lines, the tiles along `k` in a strip, then a tile's lines, each
    /// line's words in a tile one run. A descriptor landing rows (`TmaLanding::Rows`) lands a box of
    /// whole tiles of it a line's words after another, as a box over rows of `k` would.
    pub fn words_in_tiles(self, k: usize, lines: usize, tile: (usize, usize)) -> Metadata {
        let (tile_k, tile_lines) = tile;
        let words_per_line = tile_k / self.values_per_word;
        let tile_words = words_per_line * tile_lines;
        let k_tiles = k / tile_k;
        Metadata::new(
            Shape::from(vec![
                lines / tile_lines,
                k_tiles,
                tile_lines,
                words_per_line,
            ]),
            Strides::from(vec![k_tiles * tile_words, tile_words, words_per_line, 1]),
        )
    }

    /// The kernel's statement of a matrix's values over [`values_axes`]: `k`'s finest fragments, as
    /// many as make a block, become `KI`, and the rest `KB`. The statement the tile DSL wrote, exactly:
    /// a reader matches the tiles it is stored in, not only the addresses. `None` for a matrix in rows, or one
    /// whose `k` fragments do not close a block.
    pub fn kernel_values(self, matrix: &Metadata) -> Option<Metadata> {
        let [k, n] = matrix_fragments(matrix)?;
        let (kb, ki) = k.split_at(self.block_split(&k)?);
        Some(assemble(&[&n, kb, ki]))
    }

    /// The tile a matrix in tile order is stored in, `(k, lines)` values: every fragment of each of its
    /// dims but the coarsest. `None` for a matrix in rows, or one whose `k` fragments do not close a
    /// block, as [`kernel_values`](Self::kernel_values) declines it.
    pub fn tile(self, matrix: &Metadata) -> Option<(usize, usize)> {
        let [k, n] = matrix_fragments(matrix)?;
        self.block_split(&k)?;
        let inside =
            |dim: &[(usize, usize)]| dim.iter().skip(1).map(|&(extent, _)| extent).product();
        Some((inside(&k), inside(&n)))
    }

    /// Where `k`'s fragments split into the blocks and the values inside one: its finest fragments,
    /// as many as make a block, and any fragment of one coarser than them, where the tile DSL leaves
    /// it. `None` where they do not close a block.
    fn block_split(self, k: &[(usize, usize)]) -> Option<usize> {
        let mut split = k.len();
        let mut inside = 1;
        while inside < self.block && split > 0 {
            split -= 1;
            inside *= k[split].0;
        }
        if inside != self.block {
            return None;
        }
        while split > 0 && k[split - 1].0 == 1 {
            split -= 1;
        }
        Some(split)
    }
}

/// NVFP4 as tile order stores it: `e2m1` values eight to a word down `k`, under a `ue4m3` scale
/// every sixteen of `k` and an `f32` per-tensor scale. A preset: tile order stores any scheme
/// [`TileOrder::of`] reads.
pub fn nvfp4_scheme() -> QuantScheme {
    QuantScheme::default()
        .with_value(QuantValue::E2M1)
        .per_block([TileOrder::NVFP4.block as u8, 1], ScaleDtype::UE4M3)
        .with_store(QuantStore::PackedU32(1))
        .per_tensor(ScaleDtype::F32)
}

/// A tile's pieces above the word, finest first, each `(axis, count)` of the piece below it.
type Pieces = Vec<(Axis, usize)>;

/// A tile of `pieces` above the word `word`, read `words` words a load, the tiles in `grid`'s
/// order: the read a level of its own, because a reader takes a stated tile whole or not at all,
/// and stated as one level the same bytes are read a word a load.
fn stored_in_levels(
    word: (Axis, usize),
    pieces: &[(Axis, usize)],
    words: usize,
    grid: &[Axis],
) -> StoragePartitioning {
    let (read, rest) = split_at_load(pieces, words);
    // A read of one word or of the whole tile is no level of its own.
    let levels = match (read.is_empty(), rest.is_empty()) {
        (false, false) => vec![read, rest],
        _ => vec![pieces.to_vec()],
    };
    levels
        .iter()
        .fold(StorageLevels::new(&[word]), |storage, level| {
            storage.tile(level)
        })
        .grid(grid)
}

/// The kernel's statement of a matrix of `extents` stored as `storage` states.
fn stated(storage: &StoragePartitioning, extents: &[(Axis, usize)]) -> Metadata {
    let stored = storage
        .physical(extents)
        .expect("tile order's extents close whole tiles");
    Metadata::new(
        Shape::from(stored.shape().to_vec()),
        Strides::from(stored.strides().to_vec()),
    )
    .with_tiling(stored.tiling())
    .expect("cubek states its own tiling")
}

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

/// The matrix's `[k/block, lines]` statement of the block scales the kernel states over
/// [`scales_axes`]: the blocks, then the lines, each dim's fragments coarsest first.
///
/// # Panics
///
/// When `kernel` is not a statement of two logical dims.
pub fn matrix_scales(kernel: &Metadata) -> Metadata {
    let [lines, kb] = fragments_of::<2>(kernel)
        .unwrap_or_else(|rank| panic!("tile order states scales over two dims, not {rank}"));
    assemble(&[kb.as_slice(), lines.as_slice()])
}

/// A tiled matrix's `[k, lines]` fragments, coarsest first; `None` for one in rows.
fn matrix_fragments(matrix: &Metadata) -> Option<[Fragments; 2]> {
    if !matrix.is_tiled() {
        return None;
    }
    fragments_of::<2>(matrix).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{self, fragments};
    use crate::matmul_axes::N;

    const NVFP4: TileOrder = TileOrder::NVFP4;
    const BLOCK: usize = NVFP4.block;
    const VALUES_PER_WORD: usize = NVFP4.values_per_word;

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

    /// NVFP4's tile order in both orders of its words.
    const ORDERS: [TileOrder; 2] = [NVFP4, NVFP4.with_words(TileWords::BlockByBlock)];

    /// Every order, at every tile and load width.
    fn reads() -> impl Iterator<Item = (TileOrder, (usize, usize), usize)> {
        ORDERS.into_iter().flat_map(|order| {
            TILES
                .into_iter()
                .flat_map(move |tile| LOADS.map(|words| (order, tile, words)))
        })
    }

    /// The restatement moves nothing: both statements send every value to one address, and no
    /// two values share one. A restatement that misplaced a fragment would have the kernel read
    /// another column's block.
    #[test]
    fn the_rhs_and_the_kernel_state_one_memory() {
        let (k, n) = (128, 48);
        for (order, tile, words_per_load) in reads() {
            let kernel = order.kernel_values_read(k, (N, n), tile, words_per_load);
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
        for (order, tile, words_per_load) in reads() {
            let kernel = order.kernel_values_read(k, (N, n), tile, words_per_load);
            let restated = order
                .kernel_values(&matrix_values(&kernel))
                .expect("a tile-order rhs closes its blocks");
            assert_eq!(
                restated, kernel,
                "{order:?}, {tile:?}, {words_per_load} words a load"
            );
        }
    }

    /// Every load width stores the same bytes: the read is a statement, not a layout. A writer
    /// that moved a value for a wider load would hand a reader built for a narrower one another
    /// rhs.
    #[test]
    fn every_load_width_stores_the_same_bytes() {
        let (k, n) = (128, 48);
        let address = |order: TileOrder, tile, words_per_load| {
            let kernel = order.kernel_values_read(k, (N, n), tile, words_per_load);
            (0..n)
                .flat_map(|ni| (0..k).map(move |ki| (ni, ki)))
                .map(|(ni, ki)| offset(&kernel, &[ni, ki / BLOCK, ki % BLOCK]))
                .collect::<Vec<_>>()
        };
        for (order, tile, words_per_load) in reads() {
            assert_eq!(
                address(order, tile, words_per_load),
                address(order, tile, 1),
                "{order:?}, {tile:?}, {words_per_load} words a load"
            );
        }
    }

    /// A matrix carries the order it was laid in: read off its own fragments, it is the order its
    /// tiles were laid in, at every tile and load width, where the tile holds that order whole.
    #[test]
    fn a_matrix_states_the_order_it_was_laid_in() {
        let (k, n) = (128, 48);
        for (order, tile, words_per_load) in reads() {
            let matrix = matrix_values(&order.kernel_values_read(k, (N, n), tile, words_per_load));
            assert_eq!(
                order.words_of(&matrix),
                Some(order.laid_in(order.words, tile).words),
                "{order:?}, {tile:?}, {words_per_load} words a load"
            );
        }
    }

    /// A tile too shallow for whole words of block scales, or one block deep, is laid line by line
    /// whatever was asked, keeps its scales in rows, and reads back as line by line: NVFP4 in a
    /// tile of two blocks, and MXFP4 and MXFP8, blocks of thirty-two, in a tile of sixty-four.
    #[test]
    fn a_tile_too_shallow_for_a_scale_word_is_laid_line_by_line() {
        let mxfp4 = TileOrder {
            block: 32,
            values_per_word: 8,
            words: TileWords::BlockByBlock,
        };
        let mxfp8 = TileOrder {
            values_per_word: 4,
            ..mxfp4
        };
        let block_by_block = NVFP4.with_words(TileWords::BlockByBlock);
        for (order, tile) in [
            (block_by_block, (32, 8)),
            (block_by_block, (16, 8)),
            (mxfp4, (64, 8)),
            (mxfp8, (64, 8)),
        ] {
            let laid = order.laid_in(order.words, tile);
            assert_eq!(laid.words, TileWords::LineByLine, "{order:?} in {tile:?}");
            assert!(laid.scales_storage(N, tile, 2).is_none(), "{order:?}");
            assert!(order.scales_storage(N, tile, 2).is_none(), "{order:?}");
            let matrix = matrix_values(&laid.kernel_values_read(256, (N, 48), tile, 4));
            assert_eq!(laid.words_of(&matrix), Some(TileWords::LineByLine));
        }
        // A tile of whole scale words keeps the order asked.
        assert_eq!(
            block_by_block
                .laid_in(TileWords::BlockByBlock, (64, 8))
                .words,
            TileWords::BlockByBlock
        );
    }

    /// Values asked block by block in a tile that holds no block-by-block order are laid line by
    /// line, the order the matrix then reads back: a writer that skips
    /// [`laid_in`](TileOrder::laid_in) still writes what readers read.
    #[test]
    fn values_asked_block_by_block_in_a_shallow_tile_are_laid_line_by_line() {
        let (k, n) = (256, 48);
        let asked = NVFP4.with_words(TileWords::BlockByBlock);
        for tile in [(32, 8), (64, 1)] {
            let laid = asked.kernel_values_read(k, (N, n), tile, 4);
            assert_eq!(
                laid,
                NVFP4.kernel_values_read(k, (N, n), tile, 4),
                "{tile:?}"
            );
            assert_eq!(
                asked.words_of(&matrix_values(&laid)),
                Some(TileWords::LineByLine),
                "{tile:?}"
            );
            assert_eq!(asked.lines_a_load(N, tile, 4), 1, "{tile:?}");
        }
    }

    /// Block by block, every read stores the block scales at the same bytes: the read is a
    /// statement, not a layout, as it is for the values. A writer and a reader stating different
    /// reads of them address the same scales.
    #[test]
    fn every_read_stores_the_scales_at_the_same_bytes() {
        let (k, n) = (256, 48);
        let order = NVFP4.with_words(TileWords::BlockByBlock);
        for tile in WIDE_TILES {
            let address = |lines_a_read| {
                let scales = order
                    .kernel_scales_read(k, (N, n), tile, lines_a_read)
                    .expect("block by block, the scales are in tiles");
                (0..n)
                    .flat_map(|ni| (0..k / BLOCK).map(move |kb| (ni, kb)))
                    .map(|(ni, kb)| offset(&scales, &[ni, kb]))
                    .collect::<Vec<_>>()
            };
            for lines_a_read in LOADS {
                assert_eq!(
                    address(lines_a_read),
                    address(1),
                    "{tile:?}, {lines_a_read} lines a read"
                );
            }
        }
    }

    /// A read of scales spans the lines a load of values spans: one block by block where a load
    /// holds a block, two where it holds a block of two lines, and one line by line.
    #[test]
    fn a_read_of_scales_spans_the_lines_a_load_of_values_spans() {
        let block_by_block = NVFP4.with_words(TileWords::BlockByBlock);
        assert_eq!(block_by_block.lines_a_load(N, (64, 8), 2), 1);
        assert_eq!(block_by_block.lines_a_load(N, (64, 8), 4), 2);
        assert_eq!(block_by_block.lines_a_load(N, (64, 8), 8), 4);
        assert_eq!(NVFP4.lines_a_load(N, (64, 8), 4), 1);
    }

    /// Block by block, a load of four words holds one block of two neighbouring lines: the words
    /// a reader takes for a block of two columns, each under its own scale.
    #[test]
    fn block_by_block_a_load_holds_a_block_of_two_lines() {
        let (k, n) = (128, 48);
        let order = NVFP4.with_words(TileWords::BlockByBlock);
        let kernel = order.kernel_values_read(k, (N, n), (64, 8), 4);
        let word = |ni, ki| offset(&kernel, &[ni, ki / BLOCK, ki % BLOCK]) / VALUES_PER_WORD;
        let first = word(0, 0);
        let load: Vec<_> = [(0, 0), (0, 8), (1, 0), (1, 8)]
            .into_iter()
            .map(|(ni, ki)| word(ni, ki) - first)
            .collect();
        assert_eq!(load, [0, 1, 2, 3]);
    }

    /// The words a descriptor reads in tiles lie where the kernel's statement puts them: word
    /// `w` of column `ni`'s run of `k` is the word holding value `8w` of that column.
    #[test]
    fn a_descriptor_reads_the_words_where_the_tiles_hold_them() {
        let (k, n) = (384, 24);
        let tile = (128, 8);
        let tiles = NVFP4.words_in_tiles(k, n, tile);
        let kernel = NVFP4.kernel_values_read(k, (N, n), tile, 4);
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

    /// Tiles of more than one line, the only ones laid block by block ([`TileOrder::laid_in`]).
    const WIDE_TILES: [(usize, usize); 2] = [(128, 8), (64, 8)];

    /// Block by block, every scale lies at an address of its own, inside the matrix's scales, at
    /// every tile and load width; and where a load holds four words, the scales of four
    /// neighbouring lines' blocks of one word lie side by side, a read of them those a load of
    /// values spans. Line by line, the scales are no tiles at all: they stay in rows.
    #[test]
    fn a_load_of_scales_holds_neighbouring_lines() {
        let (k, n) = (256, 48);
        let blocks = k / BLOCK;
        let order = NVFP4.with_words(TileWords::BlockByBlock);
        for (tile, words_per_load) in WIDE_TILES.iter().flat_map(|&tile| LOADS.map(|w| (tile, w))) {
            let lines_a_read = order.lines_a_load(N, tile, words_per_load);
            assert!(
                NVFP4
                    .kernel_scales_read(k, (N, n), tile, lines_a_read)
                    .is_none()
            );
            let scales = order
                .kernel_scales_read(k, (N, n), tile, lines_a_read)
                .expect("block by block, the scales are in tiles");
            let mut seen = vec![false; n * blocks];
            for ni in 0..n {
                for kb in 0..blocks {
                    let at = offset(&scales, &[ni, kb]);
                    assert!(at < n * blocks, "{tile:?}: scale ({ni}, {kb}) at {at}");
                    assert!(
                        !std::mem::replace(&mut seen[at], true),
                        "{tile:?}: two scales at {at}"
                    );
                }
            }
            let (_, tile_lines) = tile;
            if words_per_load == 4 && tile_lines >= 4 {
                let per_word = TileOrder::SCALES_PER_WORD;
                let first = offset(&scales, &[0, 0]);
                let lines: Vec<_> = (0..4).map(|ni| offset(&scales, &[ni, 0]) - first).collect();
                assert_eq!(
                    lines,
                    [0, per_word, 2 * per_word, 3 * per_word],
                    "{tile:?}: four lines' first words side by side"
                );
                for kb in 0..per_word {
                    assert_eq!(
                        offset(&scales, &[0, kb]) - first,
                        kb,
                        "{tile:?}: block {kb}"
                    );
                }
            }
        }
    }

    /// An rhs in rows is not mistaken for tile order.
    #[test]
    fn an_rhs_in_rows_has_no_kernel_statement() {
        let rows = Metadata::new(Shape::new([64, 48]), vec![48, 1]);
        assert!(NVFP4.kernel_values(&rows).is_none());
        assert!(NVFP4.tile(&rows).is_none());
    }

    /// The tile an rhs states is the one it was stored in, at every load width, read off it in
    /// one pass as the kernel's statement would give it.
    #[test]
    fn an_rhs_states_the_tile_it_was_stored_in() {
        let (k, n) = (128, 48);
        for (order, stored, words_per_load) in reads() {
            let weight =
                matrix_values(&order.kernel_values_read(k, (N, n), stored, words_per_load));
            assert_eq!(
                order.tile(&weight),
                Some(stored),
                "{order:?}, {words_per_load} words a load"
            );
        }
    }

    /// Tile order stores any symmetric scheme whose words run down `k` under blocks of whole
    /// words along it, and reads its block and word off the scheme: NVFP4's, MXFP4's and MXFP8's,
    /// none of a scheme packed along the lines, scaled across them, or indexing a table.
    #[test]
    fn each_scheme_down_k_has_its_tile_order() {
        use cubecl::quant::scheme::QuantMode;
        let down_k = |value, block: u8, scale| {
            QuantScheme::default()
                .with_value(value)
                .per_block([block, 1], scale)
                .with_store(QuantStore::PackedU32(1))
        };
        let order = |block, values_per_word| {
            Some(TileOrder {
                block,
                values_per_word,
                words: TileWords::LineByLine,
            })
        };
        assert_eq!(TileOrder::of(&nvfp4_scheme()), Some(TileOrder::NVFP4));
        assert_eq!(
            TileOrder::of(&down_k(QuantValue::E2M1, 32, ScaleDtype::UE8M0)),
            order(32, 8)
        );
        assert_eq!(
            TileOrder::of(&down_k(QuantValue::E4M3, 32, ScaleDtype::UE8M0)),
            order(32, 4)
        );
        let along_lines =
            down_k(QuantValue::E2M1, 16, ScaleDtype::UE4M3).with_store(QuantStore::PackedU32(0));
        let across = QuantScheme::default()
            .with_value(QuantValue::E2M1)
            .per_block([1, 16], ScaleDtype::UE4M3)
            .with_store(QuantStore::PackedU32(1));
        let lookup = down_k(QuantValue::Q4S, 32, ScaleDtype::F16).with_mode(QuantMode::Lookup);
        for refused in [along_lines, across, lookup] {
            assert_eq!(TileOrder::of(&refused), None, "{refused:?}");
        }
    }

    /// An MXFP8 matrix, a block of thirty-two four to a word, round-trips through the kernel's
    /// statement as NVFP4's does.
    #[test]
    fn an_mxfp8_rhs_reads_back_what_it_states() {
        let mxfp8 = TileOrder {
            block: 32,
            values_per_word: 4,
            words: TileWords::LineByLine,
        };
        let (k, n) = (256, 16);
        for words_per_load in LOADS {
            let kernel = mxfp8.kernel_values_read(k, (N, n), (128, 8), words_per_load);
            let matrix = matrix_values(&kernel);
            assert_eq!(mxfp8.kernel_values(&matrix).as_ref(), Some(&kernel));
            assert_eq!(mxfp8.tile(&matrix), Some((128, 8)));
        }
    }
}
