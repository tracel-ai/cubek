//! Tile order: a block-scaled matrix's values ([`TileOrder`]: NVFP4's, MXFP4's, MXFP8's) stored for
//! the kernels that read a whole tile in one load — a tile of `k` by some rows of the matrix's
//! other axis — beside its block scales, as many to a word as the scale type packs.
//! The matrix is either input of a product: its axes are `K` and the one it is `across`, `N` for
//! a `[k, n]` input and `M` for an `[m, k]` one. A row is one index of the axis it is across, its
//! values running down `k`, as the kernel states it ([`values_axes`]); a line, the vector a buffer
//! is read in, is a different thing.
//!
//! A tile's words lie in one of two orders ([`TileWords`]), and the device decides which: each
//! row's words down the tile's `k` one run, which the TMA engine copies and a tensor-core plane
//! reads whole, its scales in the rows they are packed in; or each block's words of
//! every row side by side, where software reads the tile a load at a time, so a load is one block
//! of several rows, its scales stored in tiles of their own beside the values.
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
use cubecl::tile::layout::{Stagger, StorageLevels, StoragePartitioning};
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

/// The order a tile's words lie in, past a block's words of one row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TileWords {
    /// Each row's words down the tile's `k` one run, the rows one after another: the run the TMA
    /// engine copies and a tensor-core plane reads whole. The
    /// block scales stay in the rows they are packed in, a row's blocks one run.
    RowByRow,
    /// Each block's words of every row of the tile side by side, the blocks one after another: a
    /// load is one block of as many rows as it holds, each under its own scale. The block scales
    /// are stored in tiles of their own ([`TileOrder::scales_storage`]), a tile's rows' words
    /// side by side, so a load of them holds the scales of every row a load of values spans.
    BlockByBlock,
}

/// How a matrix packed under one scheme lies in tile order: the values of `k` one block scale
/// covers, the values one `u32` word holds, the tile it is stored in, `(k, rows)` values, the
/// order that tile's words lie in, and how far apart its strips of tiles start. Fixed when it is
/// built, by the writer laying a matrix out ([`laid`](Self::laid)) or by a reader taking it off
/// one ([`of_matrix`](Self::of_matrix), [`of_operand`](Self::of_operand)), and the same either
/// way: no tile holds an order it cannot lay out, so every statement of it reads back as the
/// order that made it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileOrder {
    block: usize,
    values_per_word: usize,
    tile: (usize, usize),
    words: TileWords,
    /// The transactions its strips of tiles, values and scales each, start an odd number of
    /// apart, stated over its values ([`Stagger`]); back to back where `None`.
    stagger: Option<Stagger>,
}

impl TileOrder {
    /// Block scales one stored word holds: tile order stores byte scales, four to a `u32`, and
    /// refuses any other.
    pub const SCALES_PER_WORD: usize = 4;

    /// A matrix packed under `scheme` laid out in `tile`, `(k, rows)` values, its words as `words`
    /// asks where the tile holds that order: block by block only where the tile's `k` holds whole
    /// words of block scales, which its scales' tiles are made of, and the tile holds more than
    /// one row, the only tile the two orders lay out differently; row by row otherwise, which
    /// lays the values out in any tile. `None` for a scheme tile order does not store
    /// ([`stores`](Self::stores)), or a tile not whole blocks of `k`.
    pub fn laid(scheme: &QuantScheme, words: TileWords, tile: (usize, usize)) -> Option<Self> {
        let (block, values_per_word) = Self::form(scheme)?;
        Self::in_tile(block, values_per_word, tile, words)
    }

    /// The order `matrix`, packed under `scheme`, lies in, read off its own fragments in one
    /// pass: its tile, every fragment of each dim but the coarsest, and its words block by block
    /// where a block's words of one row are followed by the next row's, row by row otherwise.
    /// `None` for a scheme tile order does not store, a matrix in rows, or one whose `k`
    /// fragments do not close a block.
    ///
    /// What every reader asks: the matrix carries its order, and a reader that worked it out again
    /// from its own device could disagree with the writer that laid it.
    pub fn of_matrix(scheme: &QuantScheme, matrix: &Metadata) -> Option<Self> {
        let (block, values_per_word) = Self::form(scheme)?;
        Self::read_off(block, values_per_word, matrix)
    }

    /// The order `matrix`, read in rows as `operand`, lies in: the operand's words down `K`, each a
    /// whole read, under a scale every whole number of them along `K`, byte scales
    /// [`SCALES_PER_WORD`](Self::SCALES_PER_WORD) to a word, its tiles read off the matrix as
    /// [`of_matrix`](Self::of_matrix) reads them. `None` for an operand tile order does not store,
    /// a matrix in rows, or one whose `k` fragments do not close a block.
    pub fn of_operand(operand: &Operand, matrix: &Metadata) -> Option<Self> {
        if operand.lined() != K || operand.scales_per_word() != Self::SCALES_PER_WORD {
            return None;
        }
        let (block, values_per_word) = Self::in_whole_words(operand.scale(K)?, operand.read(K))?;
        Self::read_off(block, values_per_word, matrix)
    }

    /// Whether tile order stores a matrix packed under `scheme`: a symmetric scheme whose words run
    /// down `k` (`PackedU32(1)` of a `[k, rows]` matrix), under block scales of one byte along `k`
    /// alone, four to a word as a block-scaled instruction reads them, each block a whole number
    /// of words. NVFP4's is sixteen values of eight a word, MXFP4's thirty-two of eight, MXFP8's
    /// thirty-two of four.
    pub fn stores(scheme: &QuantScheme) -> bool {
        Self::form(scheme).is_some()
    }

    /// Values of `k` one block scale covers.
    pub fn block(self) -> usize {
        self.block
    }

    /// Values one stored word holds.
    pub fn values_per_word(self) -> usize {
        self.values_per_word
    }

    /// The tile it is stored in, `(k, rows)` values.
    pub fn tile(self) -> (usize, usize) {
        self.tile
    }

    /// The order its tile's words lie in.
    pub fn words(self) -> TileWords {
        self.words
    }

    /// The transactions its strips of tiles start an odd number of apart, stated over its values:
    /// `None` where they lie back to back.
    pub fn stagger(self) -> Option<Stagger> {
        self.stagger
    }

    /// This order with its strips of tiles, of values and of scales alike, starting an odd number
    /// of `stagger`'s transactions apart: what a writer lays a matrix in on a device whose memory
    /// picks its channel from low address bits, where strips a power of two bytes apart, as they
    /// lie whenever `k` is a power of two, would all wait on one channel.
    ///
    /// # Panics
    ///
    /// When a value does not divide a transaction, which no memory moves.
    pub fn staggered(self, stagger: Stagger) -> Self {
        let over_values = stagger
            .over(self.value_bits())
            .expect("a value of four or eight bits divides a transaction");
        Self {
            stagger: Some(over_values),
            ..self
        }
    }

    /// This order as a matrix of `k` values by `rows` is laid in: its stagger kept only where it
    /// pads the values' strips. The values decide for every statement of the matrix — the words
    /// and the scales lie as they do — so a reader, which reads the stagger off the values, reads
    /// the matrix back as this order ([`of_matrix`](Self::of_matrix)). One strip, or a strip
    /// already an odd number of transactions, pads nothing: such a matrix is laid back to back.
    pub fn for_matrix(self, k: usize, rows: usize) -> Self {
        let (_, tile_rows) = self.tile;
        let run = k * tile_rows;
        let pads = self
            .stagger
            .is_some_and(|stagger| rows / tile_rows > 1 && stagger.pitch(run) != run);
        match pads {
            true => self,
            false => Self {
                stagger: None,
                ..self
            },
        }
    }

    /// Bits one value holds, stored `values_per_word` to a `u32`.
    fn value_bits(self) -> usize {
        u32::BITS as usize / self.values_per_word
    }

    /// `storage` with its strips laid as [`stagger`](Self::stagger) says, its elements
    /// `element_bits` wide.
    fn strips(self, storage: StoragePartitioning, element_bits: usize) -> StoragePartitioning {
        match self.stagger {
            Some(stagger) => storage.staggered(
                stagger
                    .over(element_bits)
                    .expect("a value, a byte scale and a word each divide a transaction"),
            ),
            None => storage,
        }
    }

    /// The block and word `scheme` packs a matrix tile order stores in ([`stores`](Self::stores)).
    fn form(scheme: &QuantScheme) -> Option<(usize, usize)> {
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

    /// A scale every `block` values, `values_per_word` to a word: `None` unless the words pack
    /// several values and a block is a whole number of them.
    fn in_whole_words(block: usize, values_per_word: usize) -> Option<(usize, usize)> {
        (values_per_word > 1 && block.is_multiple_of(values_per_word))
            .then_some((block, values_per_word))
    }

    /// A block of `block` values, `values_per_word` to a word, in `tile`, its words as `words` asks
    /// where the tile holds that order ([`laid`](Self::laid)): `None` for a tile not whole blocks
    /// of `k`.
    fn in_tile(
        block: usize,
        values_per_word: usize,
        (tile_k, tile_rows): (usize, usize),
        words: TileWords,
    ) -> Option<Self> {
        if tile_k == 0 || tile_rows == 0 || !tile_k.is_multiple_of(block) {
            return None;
        }
        let scale_word = block * Self::SCALES_PER_WORD;
        let words = match words {
            TileWords::BlockByBlock if tile_k.is_multiple_of(scale_word) && tile_rows > 1 => words,
            TileWords::BlockByBlock | TileWords::RowByRow => TileWords::RowByRow,
        };
        Some(Self {
            block,
            values_per_word,
            tile: (tile_k, tile_rows),
            words,
            stagger: None,
        })
    }

    /// The order `matrix` lies in, of `block` values `values_per_word` to a word, read off its own
    /// fragments in one pass: its tile, every fragment of each dim but the coarsest, and its words
    /// block by block where a block's words of one row are followed by the next row's, row by row
    /// where by that row's next block. A tile one block deep or one row wide lies the same either
    /// way, and a tile too shallow for whole words of block scales is never laid block by block:
    /// both read as row by row. Its stagger is read off its coarsest stride ([`strips_of`]).
    fn read_off(block: usize, values_per_word: usize, matrix: &Metadata) -> Option<Self> {
        let [k, rows] = matrix_fragments(matrix)?;
        let (blocks, _) = k.split_at(block_split(block, &k)?);
        let inside =
            |dim: &[(usize, usize)]| dim.iter().skip(1).map(|&(extent, _)| extent).product();
        let finest = |dim: &[(usize, usize)]| {
            dim.iter()
                .filter(|&&(extent, _)| extent > 1)
                .map(|&(_, stride)| stride)
                .min()
        };
        let stated = match (finest(&rows), finest(blocks)) {
            (Some(row), Some(block)) if row < block => TileWords::BlockByBlock,
            _ => TileWords::RowByRow,
        };
        let order = Self::in_tile(block, values_per_word, (inside(&k), inside(&rows)), stated)?;
        match StripLayout::of(&[k, rows], order.value_bits()) {
            StripLayout::Dense => Some(order),
            StripLayout::Staggered(stagger) => Some(order.staggered(stagger)),
            StripLayout::Foreign => None,
        }
    }

    /// How the values of a matrix `across` an axis are stored in its tile, over [`values_axes`],
    /// for a device whose load holds `words_per_load` words: a word's values along `KI`, a block's
    /// words, then as its [`words`](Self::words) order them a row's blocks down the tile's `k` and
    /// the tile's rows, or a block's rows and the tile's blocks, the tiles ordered `KB` within the
    /// rows.
    ///
    /// The read is a level of its own because a reader takes a stated tile whole or not at all:
    /// stated as one level, the same bytes are read a word a load.
    pub fn values_storage(self, across: Axis, words_per_load: usize) -> StoragePartitioning {
        let rows = blocked(across).1;
        let storage = stored_in_levels(
            (KI, self.values_per_word),
            &self.values_pieces(across),
            words_per_load,
            &[KI, KB, rows],
        );
        self.strips(storage, self.value_bits())
    }

    /// A tile's pieces of values above the word, finest first, each a count of the piece below
    /// it: a block's words, then a row's blocks and the rows, or a block's rows and the blocks,
    /// as its [`words`](Self::words) order them.
    fn values_pieces(self, across: Axis) -> [(Axis, usize); 3] {
        let (tile_k, tile_rows) = self.tile;
        let block = (KI, self.block / self.values_per_word);
        let down_k = (KB, tile_k / self.block);
        let across_rows = (blocked(across).1, tile_rows);
        match self.words {
            TileWords::RowByRow => [block, down_k, across_rows],
            TileWords::BlockByBlock => [block, across_rows, down_k],
        }
    }

    /// The rows one load of values `words_per_load` words wide spans in its tile: one a load row
    /// by row, and block by block as many as a block's words leave room for. The rows a read of
    /// their scales spans, so it brings every scale that load needs and no other reader's.
    pub fn rows_a_load(self, across: Axis, words_per_load: usize) -> usize {
        let rows = blocked(across).1;
        let (read, _) = split_at_load(&self.values_pieces(across), words_per_load);
        read.iter()
            .find(|&&(axis, _)| axis == rows)
            .map_or(1, |&(_, count)| count)
    }

    /// How the block scales of a matrix `across` an axis are stored beside its values, over
    /// [`scales_axes`], read `rows_a_read` rows a read: in tiles of their own, a word holding
    /// [`SCALES_PER_WORD`](Self::SCALES_PER_WORD) blocks of one row along `KB`, the tiles ordered
    /// as the values' are. Block by block, the tile's rows' words side by side, then the tile's
    /// further words down `k`: a row's scales of a tile's blocks are one word, so a read of
    /// `rows_a_read` rows is that many words, and the rows a load of values spans
    /// ([`rows_a_load`](Self::rows_a_load)) make it the read that serves one load of values. Row by
    /// row, each row's words one run and the tile's rows one after another: a tile's scales are
    /// one dense run, which a TMA box lands as the rows of scales a stage holds
    /// ([`scale_words_in_tiles`](Self::scale_words_in_tiles)). Every read stores the same bytes.
    ///
    /// `None` where a tile's `k` is not whole words of block scales, or the tile is one row: its
    /// scales stay in the rows they are packed in.
    pub fn scales_storage(self, across: Axis, rows_a_read: usize) -> Option<StoragePartitioning> {
        if !self.scales_in_tiles() {
            return None;
        }
        let (tile_k, tile_rows) = self.tile;
        let scale_word = self.block * Self::SCALES_PER_WORD;
        let rows = (blocked(across).1, tile_rows);
        let words = (KB, tile_k / scale_word);
        let pieces = match self.words {
            TileWords::BlockByBlock => [rows, words],
            TileWords::RowByRow => [words, rows],
        };
        let storage = stored_in_levels(
            (KB, Self::SCALES_PER_WORD),
            &pieces,
            rows_a_read,
            &[KB, blocked(across).1],
        );
        Some(self.strips(storage, u8::BITS as usize))
    }

    /// Whether its block scales lie in tiles of their own ([`scales_storage`](Self::scales_storage))
    /// rather than in the rows they are packed in.
    pub fn scales_in_tiles(self) -> bool {
        let (tile_k, tile_rows) = self.tile;
        tile_k.is_multiple_of(self.block * Self::SCALES_PER_WORD) && tile_rows > 1
    }

    /// The block scales of a matrix `k` deep and `rows` across, laid row by row in tiles
    /// ([`scales_storage`](Self::scales_storage)), as the `u32` words a descriptor reads: the
    /// strips of rows, the tiles along `k` in a strip, a tile's words, its rows' runs one after
    /// another. A box of whole strips one tile deep lands the rows of scales a stage of one tile
    /// along `k` holds, each row's words one run. Counted in words.
    ///
    /// # Panics
    ///
    /// On an order laid block by block, whose tiles hold their rows' words side by side, or whose
    /// scales are not in tiles.
    pub fn scale_words_in_tiles(self, k: usize, rows: usize) -> Metadata {
        assert!(
            self.words == TileWords::RowByRow && self.scales_in_tiles(),
            "TileOrder::scale_words_in_tiles: scales in tiles laid row by row, not {self:?}"
        );
        let (tile_k, tile_rows) = self.tile;
        let tile_words = tile_rows * tile_k / (self.block * Self::SCALES_PER_WORD);
        let k_tiles = k / tile_k;
        let strip = match self.stagger {
            Some(stagger) => {
                let bytes = stagger
                    .over(u8::BITS as usize)
                    .expect("a byte divides a transaction")
                    .pitch(k_tiles * tile_words * size_of::<u32>());
                bytes / size_of::<u32>()
            }
            None => k_tiles * tile_words,
        };
        Metadata::new(
            Shape::from(vec![rows / tile_rows, k_tiles, tile_words]),
            Strides::from(vec![strip, tile_words, 1]),
        )
    }

    /// The kernel's statement of the block scales of a matrix `k` deep and `across` an axis of some
    /// extent, over [`scales_axes`], read `rows_a_read` rows a read
    /// ([`scales_storage`](Self::scales_storage)), counted in scales. `None` where they are not
    /// stored in tiles of their own.
    pub fn kernel_scales_read(
        self,
        k: usize,
        (across, rows): (Axis, usize),
        rows_a_read: usize,
    ) -> Option<Metadata> {
        let storage = self.scales_storage(across, rows_a_read)?;
        Some(stated(
            &storage,
            &[(blocked(across).1, rows), (KB, k / self.block)],
        ))
    }

    /// The kernel's statement of the values of a matrix `k` deep and `across` an axis of some
    /// extent, over [`values_axes`], read `words_per_load` words a load
    /// ([`values_storage`](Self::values_storage)). The bytes are the same at every read, so a reader
    /// states the one it takes over a matrix stored for another.
    pub fn kernel_values_read(
        self,
        k: usize,
        (across, rows): (Axis, usize),
        words_per_load: usize,
    ) -> Metadata {
        stated(
            &self.values_storage(across, words_per_load),
            &[
                (blocked(across).1, rows),
                (KB, k / self.block),
                (KI, self.block),
            ],
        )
    }

    /// A tile-order matrix `k` deep and `rows` across stored in its tiles' words as the tiles a
    /// descriptor reads: the strips of rows, the tiles along `k` in a strip, then a tile's rows, each
    /// row's words in a tile one run. A descriptor landing rows (`TmaLanding::Rows`) lands a box of
    /// whole tiles of it a row's words after another, as a box over rows of `k` would.
    pub fn words_in_tiles(self, k: usize, rows: usize) -> Metadata {
        let (tile_k, tile_rows) = self.tile;
        let words_per_row = tile_k / self.values_per_word;
        let tile_words = words_per_row * tile_rows;
        let k_tiles = k / tile_k;
        let strip = match self.stagger {
            Some(stagger) => stagger
                .over(u32::BITS as usize)
                .expect("a word divides a transaction")
                .pitch(k_tiles * tile_words),
            None => k_tiles * tile_words,
        };
        Metadata::new(
            Shape::from(vec![rows / tile_rows, k_tiles, tile_rows, words_per_row]),
            Strides::from(vec![strip, tile_words, words_per_row, 1]),
        )
    }

    /// The kernel's statement of a matrix's values over [`values_axes`]: `k`'s finest fragments, as
    /// many as make a block, become `KI`, and the rest `KB`. The statement the tile DSL wrote, exactly:
    /// a reader matches the tiles it is stored in, not only the addresses. `None` for a matrix in rows, or one
    /// whose `k` fragments do not close a block.
    pub fn kernel_values(self, matrix: &Metadata) -> Option<Metadata> {
        let [k, n] = matrix_fragments(matrix)?;
        let (kb, ki) = k.split_at(block_split(self.block, &k)?);
        Some(assemble(&[&n, kb, ki]))
    }
}

/// Where `k`'s fragments split into the blocks of `block` values and the values inside one: its
/// finest fragments, as many as make a block, and any fragment of one coarser than them, where the
/// tile DSL leaves it. `None` where they do not close a block.
fn block_split(block: usize, k: &[(usize, usize)]) -> Option<usize> {
    let mut split = k.len();
    let mut inside = 1;
    while inside < block && split > 0 {
        split -= 1;
        inside *= k[split].0;
    }
    if inside != block {
        return None;
    }
    while split > 0 && k[split - 1].0 == 1 {
        split -= 1;
    }
    Some(split)
}

/// NVFP4 as tile order stores it: `e2m1` values eight to a word down `k`, under a `ue4m3` scale
/// every sixteen of `k` and an `f32` per-tensor scale. A preset: tile order stores any scheme
/// [`TileOrder::stores`] accepts.
pub fn nvfp4_scheme() -> QuantScheme {
    QuantScheme::default()
        .with_value(QuantValue::E2M1)
        .per_block([NVFP4_BLOCK as u8, 1], ScaleDtype::UE4M3)
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

/// How a matrix's strips of tiles lie, as its values' fragments, `[k, rows]` each finest first,
/// state it: the cases [`TileOrder::of_matrix`] reads an order's stagger back by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StripLayout {
    /// Back to back: the coarsest stride is the run of what is finer.
    Dense,
    /// An odd number of transactions apart, as the [`Stagger`] lays them.
    Staggered(Stagger),
    /// Apart by a step no stagger lays: no order tile order writes.
    Foreign,
}

impl StripLayout {
    /// The layout of the strips `fragments` state, of `value_bits`-bit values. A pitch is an odd
    /// number of transactions of a power of two bits, so its lowest set bit is the transaction: the
    /// stagger is read off exactly, held to whole words, and checked by laying the run out again.
    fn of(fragments: &[Fragments], value_bits: usize) -> Self {
        let pieces: Vec<(usize, usize)> = fragments
            .iter()
            .flatten()
            .copied()
            .filter(|&(extent, _)| extent > 1)
            .collect();
        let Some(&(strips, stride)) = pieces.iter().max_by_key(|&&(_, stride)| stride) else {
            return Self::Dense;
        };
        let run = pieces.iter().map(|&(extent, _)| extent).product::<usize>() / strips;
        if stride == run {
            return Self::Dense;
        }
        let bits = stride * value_bits;
        let transaction_bits = bits & bits.wrapping_neg();
        let stagger = Stagger::new(transaction_bits, value_bits)
            .filter(|_| transaction_bits.is_multiple_of(u32::BITS as usize))
            .filter(|stagger| stagger.pitch(run) == stride);
        match stagger {
            Some(stagger) => Self::Staggered(stagger),
            None => Self::Foreign,
        }
    }
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

/// The matrix's `[k, rows]` statement of values the kernel states over [`values_axes`]: `k` is a
/// block's index then its offset inside it, so `KB`'s fragments come before `KI`'s, coarsest
/// first.
///
/// # Panics
///
/// When `kernel` is not a statement of three logical dims.
pub fn matrix_values(kernel: &Metadata) -> Metadata {
    let [rows, kb, ki] = fragments_of::<3>(kernel)
        .unwrap_or_else(|rank| panic!("tile order states values over three dims, not {rank}"));
    assemble(&[&[kb.as_slice(), ki.as_slice()].concat(), &rows])
}

/// The matrix's `[k/block, rows]` statement of the block scales the kernel states over
/// [`scales_axes`]: the blocks, then the rows, each dim's fragments coarsest first.
///
/// # Panics
///
/// When `kernel` is not a statement of two logical dims.
pub fn matrix_scales(kernel: &Metadata) -> Metadata {
    let [rows, kb] = fragments_of::<2>(kernel)
        .unwrap_or_else(|rank| panic!("tile order states scales over two dims, not {rank}"));
    assemble(&[kb.as_slice(), rows.as_slice()])
}

/// The same tiled matrix stated with its two dims swapped: an `[m, k]` input's statement of the
/// `[k, m]` one tile order reads ([`matrix_values`]), and back. Nothing moves: both are the
/// same fragments, each dim's coarsest first.
///
/// # Panics
///
/// When `matrix` is not a statement of two logical dims.
pub fn swapped(matrix: &Metadata) -> Metadata {
    let [first, second] = fragments_of::<2>(matrix)
        .unwrap_or_else(|rank| panic!("a matrix states two dims, not {rank}"));
    assemble(&[second.as_slice(), first.as_slice()])
}

/// A tiled matrix's `[k, rows]` fragments, coarsest first; `None` for one in rows.
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

    const BLOCK: usize = NVFP4_BLOCK;
    const VALUES_PER_WORD: usize = 8;

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

    /// Words one load holds: a single word, a row's block, half a row's tile of `128` (a
    /// 128-bit load), a row's tile of `128`, and more than a tile holds.
    const LOADS: [usize; 5] = [1, 2, 4, 8, 64];

    /// The tiles a load stores in, `(k, n)`: the step under the block-scaled instruction with
    /// the TMA engine filling its stages, under an `8 × 8 × 8` one, and under none.
    const TILES: [(usize, usize); 3] = [(128, 8), (64, 8), (64, 1)];

    /// Tiles of more than one row, the only ones laid block by block.
    const WIDE_TILES: [(usize, usize); 2] = [(128, 8), (64, 8)];

    /// A symmetric scheme of `value` packed down `k`, a `scale` every `block` values.
    fn down_k(value: QuantValue, block: u8, scale: ScaleDtype) -> QuantScheme {
        QuantScheme::default()
            .with_value(value)
            .per_block([block, 1], scale)
            .with_store(QuantStore::PackedU32(1))
    }

    /// NVFP4 laid out in `tile` as `words` asks.
    fn nvfp4(words: TileWords, tile: (usize, usize)) -> TileOrder {
        TileOrder::laid(&nvfp4_scheme(), words, tile).expect("NVFP4 is tile order")
    }

    /// NVFP4 laid out every way asked, in every tile, read at every load width.
    fn reads() -> impl Iterator<Item = (TileOrder, usize)> {
        [TileWords::RowByRow, TileWords::BlockByBlock]
            .into_iter()
            .flat_map(|words| TILES.map(|tile| nvfp4(words, tile)))
            .flat_map(|order| LOADS.map(|words_per_load| (order, words_per_load)))
    }

    /// The restatement moves nothing: both statements send every value to one address, and no
    /// two values share one. A restatement that misplaced a fragment would have the kernel read
    /// another row's block.
    #[test]
    fn the_rhs_and_the_kernel_state_one_memory() {
        let (k, n) = (128, 48);
        for (order, words_per_load) in reads() {
            let kernel = order.kernel_values_read(k, (N, n), words_per_load);
            let weight = matrix_values(&kernel);
            assert_eq!(weight.logical_shape().unwrap().as_slice(), &[k, n]);
            assert_eq!(
                layout::tile(&weight),
                order.tile(),
                "{order:?}, {words_per_load} words"
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
        for (order, words_per_load) in reads() {
            let kernel = order.kernel_values_read(k, (N, n), words_per_load);
            let restated = order
                .kernel_values(&matrix_values(&kernel))
                .expect("a tile-order rhs closes its blocks");
            assert_eq!(restated, kernel, "{order:?}, {words_per_load} words a load");
        }
    }

    /// Every load width stores the same bytes: the read is a statement, not a layout. A writer
    /// that moved a value for a wider load would hand a reader built for a narrower one another
    /// rhs.
    #[test]
    fn every_load_width_stores_the_same_bytes() {
        let (k, n) = (128, 48);
        let address = |order: TileOrder, words_per_load| {
            let kernel = order.kernel_values_read(k, (N, n), words_per_load);
            (0..n)
                .flat_map(|ni| (0..k).map(move |ki| (ni, ki)))
                .map(|(ni, ki)| offset(&kernel, &[ni, ki / BLOCK, ki % BLOCK]))
                .collect::<Vec<_>>()
        };
        for (order, words_per_load) in reads() {
            assert_eq!(
                address(order, words_per_load),
                address(order, 1),
                "{order:?}, {words_per_load} words a load"
            );
        }
    }

    /// A matrix carries the order it was laid in: read off its own fragments, it is the order it
    /// was laid in, its tile and its words, at every tile and load width.
    #[test]
    fn a_matrix_states_the_order_it_was_laid_in() {
        let (k, n) = (128, 48);
        for (order, words_per_load) in reads() {
            let matrix = matrix_values(&order.kernel_values_read(k, (N, n), words_per_load));
            assert_eq!(
                TileOrder::of_matrix(&nvfp4_scheme(), &matrix),
                Some(order),
                "{words_per_load} words a load"
            );
        }
    }

    /// An activation states its matrix `[m, k]`, the swap of the `[k, m]` tile order reads: every
    /// value stays where it is stored, and swapped back the matrix reads as the order it was laid
    /// in.
    #[test]
    fn an_activation_states_its_matrix_swapped_and_reads_back_as_laid() {
        use crate::matmul_axes::M;

        let (k, m) = (256, 64);
        for (order, words_per_load) in reads() {
            let matrix = matrix_values(&order.kernel_values_read(k, (M, m), words_per_load));
            let activation = swapped(&matrix);
            let extents: Vec<usize> = fragments(&activation)
                .iter()
                .map(|dim| dim.iter().map(|&(extent, _)| extent).product())
                .collect();
            assert_eq!(extents, [m, k]);
            for (row, at) in [(0, 0), (5, 17), (63, 255)] {
                assert_eq!(offset(&activation, &[row, at]), offset(&matrix, &[at, row]));
            }
            assert_eq!(
                TileOrder::of_matrix(&nvfp4_scheme(), &swapped(&activation)),
                Some(order),
                "{words_per_load} words a load"
            );
        }
    }

    /// A tile too shallow for whole words of block scales, or one row wide, is laid row by row
    /// whatever was asked, keeps its scales in rows, and reads back as row by row: NVFP4 in a tile
    /// of two blocks, of one block, and of one row, and MXFP4 and MXFP8, blocks of thirty-two, in
    /// a tile of sixty-four. A tile of whole scale words keeps the order asked.
    #[test]
    fn a_tile_that_holds_no_block_by_block_order_is_laid_row_by_row() {
        let mxfp4 = down_k(QuantValue::E2M1, 32, ScaleDtype::UE8M0);
        let mxfp8 = down_k(QuantValue::E4M3, 32, ScaleDtype::UE8M0);
        for (scheme, tile) in [
            (nvfp4_scheme(), (32, 8)),
            (nvfp4_scheme(), (16, 8)),
            (nvfp4_scheme(), (64, 1)),
            (mxfp4, (64, 8)),
            (mxfp8, (64, 8)),
        ] {
            let order = TileOrder::laid(&scheme, TileWords::BlockByBlock, tile)
                .expect("tile order stores it");
            assert_eq!(order.words(), TileWords::RowByRow, "{order:?}");
            assert!(order.scales_storage(N, 2).is_none(), "{order:?}");
            let matrix = matrix_values(&order.kernel_values_read(256, (N, 48), 4));
            assert_eq!(TileOrder::of_matrix(&scheme, &matrix), Some(order));
        }
        assert_eq!(
            nvfp4(TileWords::BlockByBlock, (64, 8)).words(),
            TileWords::BlockByBlock
        );
    }

    /// A tile not whole blocks of `k`, or of no rows, lays no tile order.
    #[test]
    fn a_tile_not_whole_blocks_lays_no_order() {
        for tile in [(8, 8), (24, 8), (64, 0), (0, 8)] {
            assert_eq!(
                TileOrder::laid(&nvfp4_scheme(), TileWords::RowByRow, tile),
                None,
                "{tile:?}"
            );
        }
    }

    /// Block by block, every read stores the block scales at the same bytes: the read is a
    /// statement, not a layout, as it is for the values. A writer and a reader stating different
    /// reads of them address the same scales.
    #[test]
    fn every_read_stores_the_scales_at_the_same_bytes() {
        let (k, n) = (256, 48);
        for tile in WIDE_TILES {
            let order = nvfp4(TileWords::BlockByBlock, tile);
            let address = |rows_a_read| {
                let scales = order
                    .kernel_scales_read(k, (N, n), rows_a_read)
                    .expect("block by block, the scales are in tiles");
                (0..n)
                    .flat_map(|ni| (0..k / BLOCK).map(move |kb| (ni, kb)))
                    .map(|(ni, kb)| offset(&scales, &[ni, kb]))
                    .collect::<Vec<_>>()
            };
            for rows_a_read in LOADS {
                assert_eq!(
                    address(rows_a_read),
                    address(1),
                    "{tile:?}, {rows_a_read} rows a read"
                );
            }
        }
    }

    /// A read of scales spans the rows a load of values spans: one block by block where a load
    /// holds a block, two where it holds a block of two rows, and one row by row.
    #[test]
    fn a_read_of_scales_spans_the_rows_a_load_of_values_spans() {
        let block_by_block = nvfp4(TileWords::BlockByBlock, (64, 8));
        assert_eq!(block_by_block.rows_a_load(N, 2), 1);
        assert_eq!(block_by_block.rows_a_load(N, 4), 2);
        assert_eq!(block_by_block.rows_a_load(N, 8), 4);
        assert_eq!(nvfp4(TileWords::RowByRow, (64, 8)).rows_a_load(N, 4), 1);
    }

    /// Block by block, a load of four words holds one block of two neighbouring rows: the words
    /// a reader takes for a block of two rows, each under its own scale.
    #[test]
    fn block_by_block_a_load_holds_a_block_of_two_rows() {
        let (k, n) = (128, 48);
        let kernel = nvfp4(TileWords::BlockByBlock, (64, 8)).kernel_values_read(k, (N, n), 4);
        let word = |ni, ki| offset(&kernel, &[ni, ki / BLOCK, ki % BLOCK]) / VALUES_PER_WORD;
        let first = word(0, 0);
        let load: Vec<_> = [(0, 0), (0, 8), (1, 0), (1, 8)]
            .into_iter()
            .map(|(ni, ki)| word(ni, ki) - first)
            .collect();
        assert_eq!(load, [0, 1, 2, 3]);
    }

    /// The words a descriptor reads in tiles lie where the kernel's statement puts them: word
    /// `w` of row `ni`'s run of `k` is the word holding value `8w` of that row.
    #[test]
    fn a_descriptor_reads_the_words_where_the_tiles_hold_them() {
        let (k, n) = (384, 24);
        let laid = nvfp4(TileWords::RowByRow, (128, 8));
        for order in [laid, laid.staggered(transaction())] {
            descriptor_reads_the_kernel_words(order, k, n);
        }
    }

    /// Every word a descriptor over `order`'s tiles reads lies where its kernel statement puts it.
    fn descriptor_reads_the_kernel_words(order: TileOrder, k: usize, n: usize) {
        let tiles = order.words_in_tiles(k, n);
        let kernel = order.kernel_values_read(k, (N, n), 4);
        let (tile_k, tile_n) = order.tile();
        let per_row = tile_k / VALUES_PER_WORD;
        for ni in 0..n {
            for w in 0..k / VALUES_PER_WORD {
                let coords = [ni / tile_n, w / per_row, ni % tile_n, w % per_row];
                let at: usize = coords
                    .iter()
                    .zip(tiles.strides().iter())
                    .map(|(c, s)| c * s)
                    .sum();
                let ki = w * VALUES_PER_WORD;
                let value = offset(&kernel, &[ni, ki / BLOCK, ki % BLOCK]);
                assert_eq!(at * VALUES_PER_WORD, value, "row {ni}, word {w}");
            }
        }
    }

    /// Scales laid row by row in tiles lie where a descriptor over their words reads them: each
    /// row's word of each tile, as the kernel states the scales and as
    /// [`scale_words_in_tiles`](TileOrder::scale_words_in_tiles) states them, at the same byte,
    /// strips staggered or not.
    #[test]
    fn scales_laid_row_by_row_lie_where_a_descriptor_reads_them() {
        let (k, n) = (5120, 64);
        for stagger in [None, Some(transaction())] {
            let laid = nvfp4(TileWords::RowByRow, (128, 8));
            let order = match stagger {
                Some(stagger) => laid.staggered(stagger).for_matrix(k, n),
                None => laid,
            };
            let kernel = order
                .kernel_scales_read(k, (N, n), 1)
                .expect("a tile of 128 is whole words of scales");
            let words = order.scale_words_in_tiles(k, n);
            let (shape, strides) = (words.shape(), words.strides());
            let words_per_row = 128 / (BLOCK * TileOrder::SCALES_PER_WORD);
            for ni in 0..n {
                for w in 0..k / (BLOCK * TileOrder::SCALES_PER_WORD) {
                    let (strip, row) = (ni / 8, ni % 8);
                    let (tile, in_row) = (w / words_per_row, w % words_per_row);
                    let in_tile = row * words_per_row + in_row;
                    assert!(strip < shape[0] && tile < shape[1] && in_tile < shape[2]);
                    let at = strip * strides[0] + tile * strides[1] + in_tile * strides[2];
                    let scale = offset(&kernel, &[ni, w * TileOrder::SCALES_PER_WORD]);
                    assert_eq!(
                        at * TileOrder::SCALES_PER_WORD,
                        scale,
                        "{stagger:?}: row {ni}, word {w}"
                    );
                }
            }
        }
    }

    /// A 128-byte memory transaction, over bytes.
    fn transaction() -> Stagger {
        Stagger::new(1024, 8).expect("a byte divides a transaction")
    }

    /// A matrix whose values' strips need no padding is laid with no stagger, its scales
    /// included, so it reads back as the order that laid it: a strip already an odd number of
    /// transactions (`k` 384 at two rows a tile is three), and a single strip. Laid apart from the
    /// values, its scales' shorter strips would pad where the values' do not, and a reader taking
    /// the stagger off the values would read every scale strip past the first at the wrong place.
    #[test]
    fn a_matrix_whose_values_need_no_padding_reads_back_as_laid() {
        for words in [TileWords::BlockByBlock, TileWords::RowByRow] {
            for (k, n, tile) in [(384, 8, (64, 2)), (4096, 8, (64, 8))] {
                let order = nvfp4(words, tile).staggered(transaction()).for_matrix(k, n);
                assert_eq!(order.stagger(), None, "{words:?} at k {k}, n {n}");
                let kernel = order.kernel_values_read(k, (N, n), 1);
                assert_eq!(
                    TileOrder::of_matrix(&nvfp4_scheme(), &matrix_values(&kernel)),
                    Some(order),
                    "{words:?} at k {k}, n {n}"
                );
            }
        }
    }

    /// Whether `bytes` are an odd number of 128-byte transactions.
    fn odd_transactions(bytes: usize) -> bool {
        bytes.is_multiple_of(128) && (bytes / 128) % 2 == 1
    }

    /// A staggered order lays its strips of tiles an odd number of transactions apart, its
    /// values' and its scales' alike, though `k` makes each dense strip a power of two bytes; and
    /// a matrix laid in it reads back as that order, its stagger included.
    #[test]
    fn a_staggered_order_lays_its_strips_apart_and_reads_back() {
        let (k, n) = (4096, 48);
        for words in [TileWords::BlockByBlock, TileWords::RowByRow] {
            let order = nvfp4(words, (64, 8))
                .staggered(transaction())
                .for_matrix(k, n);
            let (_, tile_rows) = order.tile();
            let kernel = order.kernel_values_read(k, (N, n), 4);
            let strip = offset(&kernel, &[tile_rows, 0, 0]) - offset(&kernel, &[0, 0, 0]);
            let bytes = strip * 4 / 8;
            assert!(bytes > tile_rows * k / 2, "{words:?}: the strip is padded");
            assert!(
                odd_transactions(bytes),
                "{words:?}: a strip of {bytes} bytes"
            );
            let matrix = matrix_values(&kernel);
            assert_eq!(
                TileOrder::of_matrix(&nvfp4_scheme(), &matrix),
                Some(order),
                "{words:?}"
            );
            if let Some(scales) = order.kernel_scales_read(k, (N, n), 1) {
                let strip = offset(&scales, &[tile_rows, 0]) - offset(&scales, &[0, 0]);
                assert!(
                    odd_transactions(strip),
                    "{words:?}: a strip of {strip} scales"
                );
            }
        }
    }

    /// Block by block, every scale lies at an address of its own, inside the matrix's scales, at
    /// every tile and load width; and where a load holds four words, the scales of four
    /// neighbouring rows' blocks of one word lie side by side, a read of them those a load of
    /// values spans. Row by row, a tile's rows' runs lie one after another instead
    /// ([`scales_laid_row_by_row_lie_where_a_descriptor_reads_them`]).
    #[test]
    fn a_load_of_scales_holds_neighbouring_rows() {
        let (k, n) = (256, 48);
        let blocks = k / BLOCK;
        for (tile, words_per_load) in WIDE_TILES.iter().flat_map(|&tile| LOADS.map(|w| (tile, w))) {
            let order = nvfp4(TileWords::BlockByBlock, tile);
            let rows_a_read = order.rows_a_load(N, words_per_load);
            let scales = order
                .kernel_scales_read(k, (N, n), rows_a_read)
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
            let (_, tile_rows) = tile;
            if words_per_load == 4 && tile_rows >= 4 {
                let per_word = TileOrder::SCALES_PER_WORD;
                let first = offset(&scales, &[0, 0]);
                let rows: Vec<_> = (0..4).map(|ni| offset(&scales, &[ni, 0]) - first).collect();
                assert_eq!(
                    rows,
                    [0, per_word, 2 * per_word, 3 * per_word],
                    "{tile:?}: four rows' first words side by side"
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
        assert!(
            nvfp4(TileWords::RowByRow, (64, 8))
                .kernel_values(&rows)
                .is_none()
        );
        assert!(TileOrder::of_matrix(&nvfp4_scheme(), &rows).is_none());
    }

    /// Tile order stores any symmetric scheme whose words run down `k` under blocks of whole
    /// words along it, and reads its block and word off the scheme: NVFP4's, MXFP4's and MXFP8's,
    /// none of a scheme packed along the rows, scaled across them, or indexing a table.
    #[test]
    fn each_scheme_down_k_has_its_tile_order() {
        use cubecl::quant::scheme::QuantMode;
        let form = |scheme: &QuantScheme| {
            TileOrder::laid(scheme, TileWords::RowByRow, (128, 8))
                .map(|order| (order.block(), order.values_per_word()))
        };
        assert_eq!(form(&nvfp4_scheme()), Some((16, 8)));
        assert_eq!(
            form(&down_k(QuantValue::E2M1, 32, ScaleDtype::UE8M0)),
            Some((32, 8))
        );
        assert_eq!(
            form(&down_k(QuantValue::E4M3, 32, ScaleDtype::UE8M0)),
            Some((32, 4))
        );
        let along_rows =
            down_k(QuantValue::E2M1, 16, ScaleDtype::UE4M3).with_store(QuantStore::PackedU32(0));
        let across = QuantScheme::default()
            .with_value(QuantValue::E2M1)
            .per_block([1, 16], ScaleDtype::UE4M3)
            .with_store(QuantStore::PackedU32(1));
        let lookup = down_k(QuantValue::Q4S, 32, ScaleDtype::F16).with_mode(QuantMode::Lookup);
        for refused in [along_rows, across, lookup] {
            assert!(!TileOrder::stores(&refused), "{refused:?}");
            assert_eq!(form(&refused), None, "{refused:?}");
        }
    }

    /// An MXFP8 matrix, a block of thirty-two four to a word, round-trips through the kernel's
    /// statement as NVFP4's does.
    #[test]
    fn an_mxfp8_rhs_reads_back_what_it_states() {
        let scheme = down_k(QuantValue::E4M3, 32, ScaleDtype::UE8M0);
        let mxfp8 = TileOrder::laid(&scheme, TileWords::RowByRow, (128, 8)).expect("MXFP8");
        let (k, n) = (256, 16);
        for words_per_load in LOADS {
            let kernel = mxfp8.kernel_values_read(k, (N, n), words_per_load);
            let matrix = matrix_values(&kernel);
            assert_eq!(mxfp8.kernel_values(&matrix).as_ref(), Some(&kernel));
            assert_eq!(TileOrder::of_matrix(&scheme, &matrix), Some(mxfp8));
        }
    }
}
