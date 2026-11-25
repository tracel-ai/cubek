//! An operand of a contraction, as a launch derives its reads.
//!
//! Two numbers per dimension answer every question a launch asks of a tensor: the microtile one
//! read of it brings, and the microtile one scale covers. Both are read off the tensor, so a float
//! operand reads one value and is constant inside nothing, and the same reads serve it without
//! asking whether anything is quantized at all.

use crate::TMA_ROW_BYTES;
use crate::layout::tile_order::{TileOrder, scales_dimensions, values_dimensions};
use crate::matmul_dimensions::{K, blocked};
use cubecl::ir::ElemType;
use cubecl::quant::scheme::{BlockExtent, BlockScale, QuantScheme, QuantStore};
use cubecl::tile::launch::Decline;
use cubecl::tile::layout::{Run, VectorTile};
use cubecl::tile::{Dimension, Instruction, MicroTile, OperandReads, Space, kind::Field};

/// How an operand is read and how it is scaled, as tiles of its own dimensions.
///
/// Both are the same kind of statement — the smallest piece something is true
/// of — so nothing here records which side of a product is quantized, or which
/// way anything runs. An operand carrying no scales is a tile and nothing else,
/// which is what a float tensor is.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Operand {
    /// What one read brings, what one scale covers, and what one read of the scales brings.
    reads: OperandReads,
    /// The tile order its matrix is stored in, which says where its values and block scales lie;
    /// `None` for one in rows.
    tiles: Option<TileOrder>,
}

impl Operand {
    /// One value a read, no scales: what a float tensor is.
    pub fn plain() -> Self {
        Operand {
            reads: OperandReads::plain(),
            tiles: None,
        }
    }

    /// This operand with its dimension `from` named `to`: the same reads and scales, over a matrix
    /// whose dimensions another reader names otherwise.
    pub fn renamed(self, from: Dimension, to: Dimension) -> Self {
        Operand {
            reads: self.reads.renamed(from, to),
            tiles: self.tiles,
        }
    }

    /// How this operand is read and scaled, as the tile layer states it for a contraction.
    pub fn reads(&self) -> OperandReads {
        self.reads
    }

    /// The tile order its matrix is stored in ([`in_tile_order`](Self::in_tile_order)); `None` for
    /// one in rows.
    pub fn tile_order(&self) -> Option<TileOrder> {
        self.tiles
    }

    /// The operand `scheme` describes when its words run along `dimension`.
    ///
    /// **The dimension is the caller's**: a buffer's strides say which dim its words
    /// run along, and nothing here re-derives it. What a scheme states is the
    /// two lengths — how many values share a word, and how many share a scale.
    /// `dimensions` names the tensor's dims in the order the scheme's block shape
    /// counts them.
    ///
    /// `None` where the finest block spans both dimensions or a whole one, where a read would
    /// straddle two scales, or under a coarser level whose scales vary along `dimension`: none is a
    /// form this kernel streams.
    pub fn of_scheme(
        scheme: &QuantScheme,
        dimension: Dimension,
        dimensions: [Dimension; 2],
    ) -> Option<Self> {
        let finest = scheme.finest();
        let extents = finest.size.extents_in::<2>();
        let position = dimensions.iter().position(|&a| a == dimension)?;
        let (BlockExtent::VALUE, BlockExtent::Values(scale), true) = (
            extents[1 - position],
            extents[position],
            Self::outer_constant_along(scheme, position),
        ) else {
            return None;
        };
        let scale = scale.get() as usize;
        let read = scheme.num_quants();
        let scales_per_word = Field::of_scale(finest.dtype).per_word();
        scale
            .is_multiple_of(read)
            .then(|| Operand::packed(dimension, read, dimension, scale, scales_per_word))
    }

    /// Whether the scales of `scheme`'s coarser level, where it has one, are each over the whole of
    /// dimension `position`, and over one position or all of the other.
    fn outer_constant_along(scheme: &QuantScheme, position: usize) -> bool {
        scheme
            .levels()
            .nth(1)
            .is_none_or(|outer| Self::constant_along(&outer, position).is_some())
    }

    /// Where the scales of `level`, over a matrix, are each constant along the whole of its
    /// dimension `whole` and over one position or all of the other: the position of the other
    /// where they vary along it, `None` where one scale covers the matrix. `None` outside, a level
    /// that varies otherwise.
    pub fn constant_along(level: &BlockScale, whole: usize) -> Option<Option<usize>> {
        let extents = level.size.extents_in::<2>();
        let other = 1 - whole;
        match (extents[whole], extents[other]) {
            (BlockExtent::WholeAxis, BlockExtent::WholeAxis) => Some(None),
            (BlockExtent::WholeAxis, BlockExtent::VALUE) => Some(Some(other)),
            _ => None,
        }
    }

    /// This operand read a scale block at a time: one read the whole block of one row, its scale
    /// read alone. In rows, a block is a run of whole reads down its dimension; in tile order, a
    /// block's words of one row lie together in either word order ([`TileOrder::values_storage`]),
    /// so the read is one block of one row however many rows a load of the order holds. `None`
    /// without scales, across several dimensions in rows, or where a block is no whole number of
    /// reads. Whether its scales can be read one apiece is the scheme's ([`Packing::by_block`]).
    fn by_block(self) -> Option<Self> {
        let dimension = self.scaled_along()?;
        let block = self.reads.scale?.extent(dimension);
        let lined = match self.tiles {
            None => {
                self.reads.read.dimension() == Some(dimension)
                    && block.is_multiple_of(self.reads.read.extent(dimension))
            }
            Some(order) => dimension == K && order.block() == block,
        };
        lined.then(|| Operand {
            reads: OperandReads {
                read: MicroTile::along(dimension, block),
                scale_read: MicroTile::along(dimension, 1),
                ..self.reads
            },
            ..self
        })
    }

    /// This operand, stored in tile order, read `words` words to a load, as
    /// [`in_tile_order`](Self::in_tile_order) reads it from the word up, `across` the dimension its
    /// matrix lies across: a narrower load than the one it was bound for, for a reader whose run is
    /// shorter. Scales read one apiece stay so. `None` in rows, where a read is one word already, or
    /// for a load tile order does not read whole.
    pub fn reloaded(self, across: Dimension, words: usize) -> Option<Self> {
        let order = self.tiles?;
        let word = Operand::packed(
            K,
            order.values_per_word(),
            K,
            order.block(),
            TileOrder::SCALES_PER_WORD,
        );
        let reloaded = word.in_tile_order(order, across, words)?;
        match self.reads_scales_alone() {
            true => reloaded.scales_alone(),
            false => Some(reloaded),
        }
    }

    /// This operand with each scale read alone and its values read as they were: what a reader
    /// whose window holds part of a word of scales reads. `None` without scales.
    fn scales_alone(self) -> Option<Self> {
        let dimension = self.scaled_along()?;
        Some(Operand {
            reads: OperandReads {
                scale_read: MicroTile::along(dimension, 1),
                ..self.reads
            },
            ..self
        })
    }

    /// Whether one read takes a single scale on its own, not a word of them.
    fn reads_scales_alone(&self) -> bool {
        self.scaled_along()
            .is_some_and(|dimension| self.reads.scale_read.extent(dimension) == 1)
    }

    /// The dimension a scheme's own block shape implies.
    ///
    /// For the two callers holding no tensor — a tuning row, and the rule asked
    /// before a launch — where the strides that decide it everywhere else are
    /// not in hand. `PackedU32(0)` counts from the innermost dim, so it is the
    /// second of `dimensions`.
    pub fn dimension_of(scheme: &QuantScheme, dimensions: [Dimension; 2]) -> Option<Dimension> {
        match scheme.store {
            QuantStore::PackedU32(0) => Some(dimensions[1]),
            QuantStore::PackedU32(1) => Some(dimensions[0]),
            _ => None,
        }
    }

    /// What one read brings along `dimension`.
    pub fn read(&self, dimension: Dimension) -> usize {
        self.reads.read.extent(dimension)
    }

    /// Scales one read of them brings: as many as share a word, one where this
    /// operand carries none. The blocks a read of scales covers are one worker's.
    pub fn scales_per_word(&self) -> usize {
        self.scaled_along()
            .map_or(1, |dimension| self.reads.scale_read.extent(dimension))
    }

    /// The box one read of this operand covers.
    pub fn read_space(&self) -> Space {
        self.reads.read.space()
    }

    /// The box one read of this operand's scales covers over its values
    /// ([`OperandReads::scale_read_space`]). `None` where it carries no scales.
    pub fn scale_read_space(&self) -> Option<Space> {
        self.reads.scale_read_space()
    }

    /// What one read of the scales brings along `dimension`, in scales.
    pub fn scale_read(&self, dimension: Dimension) -> usize {
        self.reads.scale_read.extent(dimension)
    }

    /// Values one read brings, every dimension it spans.
    pub fn read_values(&self) -> usize {
        self.reads.read.values()
    }

    /// Scales one read of them brings, every dimension it spans.
    pub fn scale_read_values(&self) -> usize {
        self.reads.scale_read.values()
    }

    /// This operand with its reads as `order` stores its matrix, `words` words to a load: a read is
    /// the stored word and the stored tiles right above it, each taken whole, for as many as fill
    /// one load without leaving one scale. The operand keeps `order`, which says where its values
    /// and block scales lie.
    ///
    /// NVFP4 in tile order row by row, a row's eight words together and the rows side by side, is
    /// read two words along `K`, one block, though a load holds four: the next two are the next
    /// block's, under another scale. Its scales, in rows, are read a word of four blocks down one
    /// row. Block by block, a block's words of every row side by side, it is read a whole load:
    /// one block of two rows, each inside its own scale; and its scales, in tiles of their own, the
    /// rows a read of the values spans, a word of four blocks down each, side by side: every scale
    /// that read needs and no other reader's.
    ///
    /// `None` for an order this operand is not read in: one that does not start with the word
    /// this operand's scheme reads, along its own dimension or along `across`, so no read of it is
    /// whole words; or one whose load would step along a dimension that is neither, or come back to
    /// one after the other, which is no longer one run along each.
    pub fn in_tile_order(self, order: TileOrder, across: Dimension, words: usize) -> Option<Self> {
        let values = order.values_storage(across, words);
        let scales = order.scales_storage(across, order.rows_a_load(across, words));
        let (values_dimensions, scales_dimensions) =
            (values_dimensions(across), scales_dimensions(across));
        let stored = self.stored(
            across,
            words,
            Some((&values_dimensions, values.tiles())),
            scales
                .as_ref()
                .map(|scales| (scales_dimensions.as_slice(), scales.tiles())),
        )?;
        Some(Operand {
            tiles: Some(order),
            ..stored
        })
    }

    /// This operand with its reads as its values and scales are stated stored in tiles, `words`
    /// words to a load ([`in_tile_order`](Self::in_tile_order)): scales stored in tiles
    /// (`scales`) are read the rows a read of the values spans, one word of them a row. A buffer
    /// in rows (`None`) keeps the read the scheme states.
    fn stored(
        self,
        across: Dimension,
        words: usize,
        values: Option<Stored<'_>>,
        scales: Option<Stored<'_>>,
    ) -> Option<Self> {
        // A read of the values stays inside one scale, and its scales' inside one unit.
        let unit = MicroTile::unit();
        let inside = self.reads.scale.as_ref().unwrap_or(&unit);
        let read = read_off(self.reads.read, self.lined(), across, words, inside, values)?;
        // Scales in tiles a word a row: a read of them takes the rows a read of values does.
        let scale_words = match scales {
            Some(_) => read.extent(across).max(1),
            None => words,
        };
        let scale_read = match self.scaled_along() {
            Some(dimension) => read_off(
                self.reads.scale_read,
                dimension,
                across,
                scale_words,
                &unit,
                scales,
            )?,
            None => self.reads.scale_read,
        };
        Some(Operand {
            reads: OperandReads {
                read,
                scale_read,
                ..self.reads
            },
            ..self
        })
    }

    /// What one scale covers along `dimension`, where this operand carries scales.
    pub fn scale(&self, dimension: Dimension) -> Option<usize> {
        self.reads.scale.as_ref().map(|tile| tile.extent(dimension))
    }

    /// The dimension this operand's scales are grouped along, where they group one.
    ///
    /// A scales operand asks, because the groups it counts are that dimension's.
    pub fn scaled_along(&self) -> Option<Dimension> {
        self.reads.scaled_along()
    }

    /// The dimension this operand's words run along, and `K` where its reads are one
    /// value wide.
    ///
    /// A binding asks, because a buffer's dims are named in the order it steps
    /// them and this is the one it steps first.
    pub fn lined(&self) -> Dimension {
        self.reads.read.dimension().unwrap_or(K)
    }

    /// The length a scale groups `dimension` into, where it groups it at all.
    ///
    /// **A scale covering one value groups nothing**, and neither does one
    /// covering the whole dimension: a grouping is what makes a walk read the same
    /// scale twice, and both of those make it read a new one every step or
    /// never. This is the one rule the cuts and the space share.
    pub fn grouping(&self, dimension: Dimension, extent: usize) -> Option<usize> {
        self.scale(dimension)
            .filter(|&scale| scale > 1 && scale < extent)
    }

    /// Which of the dimensions `dimension` is split into a `width`-wide cut goes on, and how wide
    /// it is on each: the groups where a scale groups the dimension, the position inside one where
    /// it does not — and **both** where the width is narrower than a group, which is one group and
    /// part of it.
    ///
    /// A cut wider than a group but not a whole number of them is none of the
    /// three and states nothing; a caller reading an empty answer is holding a
    /// width this operand's scales do not admit.
    pub fn cut(
        &self,
        dimension: Dimension,
        extent: usize,
        width: usize,
    ) -> Vec<(Dimension, usize)> {
        let (groups, inside) = blocked(dimension);
        let Some(scale) = self.grouping(dimension, extent) else {
            return vec![(inside, width)];
        };
        match (width.is_multiple_of(scale), scale.is_multiple_of(width)) {
            (true, _) => vec![(groups, width / scale)],
            (false, true) => vec![(groups, 1), (inside, width)],
            (false, false) => Vec::new(),
        }
    }

    /// The dimensions `dimension` is split into that a `width`-wide box spans, each with its
    /// extent: whole groups and the whole of each where `width` covers whole groups a scale makes
    /// of the dimension, the position inside one where it does not, and the position alone where no
    /// scale groups the dimension.
    ///
    /// Where a [`cut`](Self::cut) states what a level counts, the groups alone, a box states
    /// every split dimension it covers, so whole groups bring the position inside each with them.
    pub fn span(
        &self,
        dimension: Dimension,
        extent: usize,
        width: usize,
    ) -> Vec<(Dimension, usize)> {
        let (groups, inside) = blocked(dimension);
        match self.grouping(dimension, extent) {
            Some(scale) if width.is_multiple_of(scale) => {
                vec![(groups, width / scale), (inside, scale)]
            }
            Some(_) => vec![(groups, 1), (inside, width)],
            None => vec![(inside, width)],
        }
    }

    /// How many `tile`-wide tiles of `dimension` one `span`-wide tile holds, per
    /// split dimension: the counts a level states, the level below it covering `tile`
    /// and this one `span`.
    ///
    /// Both widths are ones this operand's scales admit ([`cut`](Self::cut)).
    /// A split dimension that multiplies by one is left out, a level stating only what
    /// it multiplies.
    pub fn tiles(
        &self,
        dimension: Dimension,
        extent: usize,
        tile: usize,
        span: usize,
    ) -> Vec<(Dimension, usize)> {
        let (groups, inside) = blocked(dimension);
        let Some(scale) = self.grouping(dimension, extent) else {
            return vec![(inside, span / tile)];
        };
        // Whole blocks of the dimension, or part of one block: the two split dimensions any
        // width this operand admits covers.
        let covered = |width: usize| match width >= scale {
            true => (width / scale, scale),
            false => (1, width),
        };
        let (tile_blocks, tile_inside) = covered(tile);
        let (span_blocks, span_inside) = covered(span);
        [
            (groups, span_blocks / tile_blocks),
            (inside, span_inside / tile_inside),
        ]
        .into_iter()
        .filter(|&(_, count)| count > 1)
        .collect()
    }

    /// The two dimensions `dimension` becomes in a space at `extent`: the groups a scale
    /// makes of it, and the position inside one. A dimension no scale groups is one
    /// group over the whole of it.
    pub fn dimensions(&self, dimension: Dimension, extent: usize) -> [(Dimension, usize); 2] {
        let (groups, inside) = blocked(dimension);
        let scale = self.grouping(dimension, extent).unwrap_or(extent);
        // The one division that declares a tiling, and it is exact: a space a scale does not
        // tile would be a space short of the extent, and nothing downstream could tell.
        assert!(
            extent.is_multiple_of(scale),
            "{extent} along {dimension:?} is not whole scale tiles of {scale}"
        );
        [(groups, extent / scale), (inside, scale)]
    }

    /// The widest group of at most `units` units that distributes `dimension` evenly, and
    /// that tiles the `units` it came from.
    ///
    /// A grouped dimension is distributed a word of scales a unit per turn
    /// ([`spread_across`](Self::spread_across)), so a group distributes it evenly
    /// where its width divides the dimension's words of scales. `units` as they are
    /// where they already do, where no scale groups the dimension, and where its
    /// blocks are not whole words, which no group distributes.
    pub fn units_distributing(&self, dimension: Dimension, extent: usize, units: usize) -> usize {
        if self.grouping(dimension, extent).is_none() {
            return units;
        }
        let [(_, blocks), _] = self.dimensions(dimension, extent);
        if !blocks.is_multiple_of(self.scales_per_word()) {
            return units;
        }
        let words = blocks / self.scales_per_word();
        (1..=units)
            .rev()
            .find(|&group| units.is_multiple_of(group) && words.is_multiple_of(group))
            .unwrap_or(1)
    }

    /// The run of values the TMA engine moves at full speed into a stage of this operand's, its
    /// values `element`: [`TMA_ROW_BYTES`] along the dimension its words run along. NVFP4's `e2m1`
    /// rows are a hundred and twenty-eight values of `k`.
    pub fn tma_row(&self, element: ElemType) -> Space {
        let values = (TMA_ROW_BYTES * u8::BITS as usize / element.size_bits().max(1)).max(1);
        Space::new(&[(self.lined(), values)])
    }

    /// Reads down `K` that one `run` of a `[k, rows]` matrix read as this operand holds, its
    /// values `value_bits` wide and its rows `across` that axis: the stored pieces above one read,
    /// in the order the matrix lays them, multiplied until the run is whole
    /// ([`VectorTile::run_loads`]). In tile order a transaction takes the rows a tile lays side by
    /// side before going down `K`, and a stored tile all of its reads; in rows, a transaction is
    /// the reads down a column where its words run down `K`, and one where they run across it.
    /// `None` for a stored tile of a matrix in rows, which stores none.
    pub fn reads_down_k(
        &self,
        run: Run,
        across: Dimension,
        (k, rows): (usize, usize),
        value_bits: usize,
        transaction_bytes: usize,
    ) -> Option<usize> {
        let (read, matrix) = match self.tiles {
            Some(order) => self.read_in_tiles(order, across, (k, rows))?,
            None => self.read_in_rows(across, (k, rows)),
        };
        let (blocks, in_block) = blocked(K);
        let reads = read
            .run_loads(run, &matrix, value_bits, transaction_bytes)?
            .into_iter()
            .filter(|&(dimension, _)| [K, blocks, in_block].contains(&dimension))
            .map(|(_, reads)| reads)
            .product();
        Some(reads)
    }

    /// One read of a `[k, rows]` matrix stored in `order`, its rows `across` that axis, as the
    /// tile of stored pieces it covers, and the matrix over the dimensions the order names
    /// ([`values_dimensions`]). `None` where the read cuts a stored piece.
    fn read_in_tiles(
        &self,
        order: TileOrder,
        across: Dimension,
        (k, rows): (usize, usize),
    ) -> Option<(VectorTile, Space)> {
        let values = self.read_values();
        let [rows_dimension, blocks, in_block] = values_dimensions(across);
        let storage = order.values_storage(across, values / order.values_per_word());
        let read = VectorTile::new(storage.tiles(), in_block, values).ok()?;
        let matrix = Space::new(&[
            (rows_dimension, rows),
            (blocks, k / order.block()),
            (in_block, order.block()),
        ]);
        Some((read, matrix))
    }

    /// One read of a `[k, rows]` matrix in rows, its rows `across` that axis, as a run along the
    /// dimension its words run along, and the matrix with that dimension innermost.
    fn read_in_rows(&self, across: Dimension, (k, rows): (usize, usize)) -> (VectorTile, Space) {
        let lined = self.lined();
        let matrix = match lined == K {
            true => Space::new(&[(across, rows), (K, k)]),
            false => Space::new(&[(K, k), (lined, rows)]),
        };
        (VectorTile::run(lined, self.read_values()), matrix)
    }

    /// How `units` spread `dimension` between them, stated from the leaf up.
    ///
    /// A grouped dimension is distributed by the scales: each unit takes the blocks one
    /// read of them covers — as many as share a word — and the units take
    /// turns, so neighbouring units hold neighbouring words of scales. The
    /// position inside a block is never cut: whoever holds a block's scale
    /// holds the whole block, so the walk over what the units do not span sits
    /// *above* them and counts blocks alone. An ungrouped dimension is a contiguous
    /// run each, walked *below* the units — and no unit cut at all where they
    /// do not divide it, every unit then walking the whole of it and computing
    /// the same sum.
    ///
    /// # Errors
    ///
    /// Units that do not distribute the scales' words evenly.
    pub fn spread_across(
        &self,
        dimension: Dimension,
        extent: usize,
        units: usize,
    ) -> Result<Spread, Decline> {
        let (groups, inside) = blocked(dimension);
        let read = self.read(dimension);
        let Some(scale) = self.grouping(dimension, extent) else {
            let per_unit = match extent.is_multiple_of(units) {
                true => extent / units,
                false => extent,
            };
            return Ok(Spread {
                leaf: vec![(inside, read)],
                units: match extent.is_multiple_of(units) {
                    true => vec![(inside, units)],
                    false => Vec::new(),
                },
                interleaved: Vec::new(),
                walk: Walked::Below(inside, per_unit / read),
            });
        };
        let [(_, blocks), _] = self.dimensions(dimension, extent);
        let word = self.scales_per_word();
        // Each unit takes a word of scales a turn, so a turn is `units` words: the floor a
        // contraction has to reach for this many units, and the unit a longer one is distributed
        // in.
        if !blocks.is_multiple_of(word * units) {
            return Err(Decline::Unserved {
                why: "the units do not distribute the scales' words evenly: a turn is a word of \
                          scales a unit, and the blocks are not whole turns",
            });
        }
        Ok(Spread {
            leaf: vec![(groups, word), (inside, scale)],
            units: vec![(groups, units)],
            interleaved: vec![groups],
            // The blocks past the turn the plane's units take between them.
            // A unit holds whole blocks, so the position inside one is never
            // left to the walk.
            walk: Walked::Above(vec![groups]),
        })
    }

    /// An operand whose read brings `read` values along `lined`, whose scales
    /// each cover `scale` along `tiled`, and whose scales come `scales_per_word`
    /// to a word ([`OperandReads::packed`]).
    pub fn packed(
        lined: Dimension,
        read: usize,
        tiled: Dimension,
        scale: usize,
        scales_per_word: usize,
    ) -> Self {
        Operand {
            reads: OperandReads::packed(lined, read, tiled, scale, scales_per_word),
            tiles: None,
        }
    }
}

/// This tile, a word along `along`, as a buffer `stored` so reads it `words` words a load: the
/// word, along `along` or along `across`, and the tiles stored right above it, whole, while they
/// fit one load and stay inside `inside` along each dimension it names. The tiles past the load are
/// the buffer's to step, not the read's.
///
/// `None` where the buffer does not start with that word, or where a tile inside the load lies
/// along neither dimension or comes back to one after the other.
fn read_off(
    tile: MicroTile,
    along: Dimension,
    across: Dimension,
    words: usize,
    inside: &MicroTile,
    stored: Option<Stored<'_>>,
) -> Option<MicroTile> {
    let Some((dimensions, tiles)) = stored else {
        return Some(tile);
    };
    let per_word = tile.extent(along);
    // The word starts at the finer of the dimensions a dimension is split into; a tile above it
    // continues the run along a dimension on that same split dimension, or on the blocks once
    // `read` holds a whole one of `inside`: before then the next block's values are not the next
    // ones along the dimension.
    let dimension_of = |tile_dimension: Dimension, read: &MicroTile| {
        [along, across].into_iter().find(|&dimension| {
            let (groups, within) = blocked(dimension);
            let block = inside.extent(dimension);
            finest_dimension(dimensions, dimension) == Some(tile_dimension)
                || (tile_dimension == groups
                    && dimensions.contains(&within)
                    && block > 1
                    && read.extent(dimension) == block)
        })
    };
    let (&(tile_dimension, first), above) = tiles.split_first()?;
    let word_along =
        dimension_of(tile_dimension, &MicroTile::unit()).filter(|_| first == per_word)?;
    let mut read = MicroTile::along(word_along, per_word);
    let mut taken = 1;
    let mut last = word_along;
    for &(tile_dimension, count) in above.iter().filter(|&&(_, count)| count > 1) {
        if taken * count > words {
            break;
        }
        let dimension = dimension_of(tile_dimension, &read)?;
        let bound = inside.extent(dimension);
        if bound > 1 && read.extent(dimension) * count > bound {
            break;
        }
        if dimension != last && read.extent(dimension) > 1 {
            return None;
        }
        read = read.times(dimension, count);
        (taken, last) = (taken * count, dimension);
    }
    Some(read)
}

/// The finer of the two dimensions `dimension` is split into ([`blocked`]) a buffer over
/// `dimensions` carries: the position inside a block where it has it, the blocks where it omits it
/// (as scales do).
fn finest_dimension(dimensions: &[Dimension], dimension: Dimension) -> Option<Dimension> {
    let (groups, within) = blocked(dimension);
    [within, groups]
        .into_iter()
        .find(|split| dimensions.contains(split))
}

/// A buffer stored in tiles: the dimensions its dims carry, and its stored tiles, finest first.
type Stored<'a> = (&'a [Dimension], &'a [(Dimension, usize)]);

/// How a kernel reads a packed matrix's block scales, which they are bound as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockScales {
    /// Each scale a number, widened where its values are decoded: fields of the scheme's scale
    /// type, as many a word as fit.
    Fields,
    /// The words they are stored in, read whole: the block-scaled instruction takes a word of four
    /// `e4m3` scales, the first block's in the low byte, as its scale register.
    Words,
}

/// Where the outer scales of a packed operand, each over its whole contraction, are multiplied in,
/// which the operand is bound for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OuterScale {
    /// Into the sum, once: the kernel factors them out of the operand
    /// ([`Tile::factor`](cubecl::tile::Tile::factor)).
    Factored,
    /// Into each value as it is read, beside its block scale, which is lossless where the element
    /// it is read in holds the product.
    Kept,
}

impl OuterScale {
    /// Where it is multiplied in for a sum `instruction` contracts: into the sum where each unit
    /// holds its cells, and into the values where a sum would reach its scale only through shared
    /// memory.
    pub fn of(instruction: Instruction) -> Self {
        match instruction.scales_its_sum_in_place() {
            true => Self::Factored,
            false => Self::Kept,
        }
    }

    /// Where they are multiplied in where this is what the sum prefers and the values are read in
    /// `element`: into the sum in any case for an 8-bit one, which holds no value under them.
    pub fn read_in(self, element: ElemType) -> Self {
        match element.size() {
            1 => Self::Factored,
            _ => self,
        }
    }
}

/// A matmul input's values as they lie: elements of one type, or codes packed under a
/// quantization scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StoredValues {
    /// One element of this type a value.
    Elements(ElemType),
    /// Codes packed under this scheme.
    Packed(QuantScheme),
}

/// One input of a product as it is stored: what its values are, the float element they stand for,
/// and how a read of them and of their scales lies ([`Operand`]).
///
/// Every input is one of these, and a float input is the case with no scales: its values are
/// elements of the element it stands for, read a value at a time. Nothing reads an input by asking
/// whether it is quantized; it reads what its values and its scales are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StoredInput {
    /// What its values are.
    pub values: StoredValues,
    /// The float element its values stand for: a float input's own; a packed one's, the element
    /// its codes are decoded into, its scales applied, wherever an instruction takes no codes.
    pub element: ElemType,
    /// How a read of its values and of its scales lies.
    pub operand: Operand,
}

impl StoredInput {
    /// A float input of `element`: its values are its elements, a value a read, no scales.
    pub fn plain(element: ElemType) -> Self {
        Self {
            values: StoredValues::Elements(element),
            element,
            operand: Operand::plain(),
        }
    }

    /// An input packed as `packing` says, standing for values of `element`.
    pub fn packed(packing: Packing, element: ElemType) -> Self {
        Self {
            values: StoredValues::Packed(packing.scheme),
            element,
            operand: packing.operand,
        }
    }

    /// Bytes `values` of its values hold as stored: its words and its block scales; a float
    /// input's, its elements.
    pub fn bytes(&self, values: usize) -> usize {
        match self.packing() {
            Some(packing) => packing.bytes(values),
            None => values * self.element.size(),
        }
    }

    /// How it is packed, where its values are codes: its reads and scales, and its scheme.
    pub fn packing(&self) -> Option<Packing> {
        match self.values {
            StoredValues::Packed(scheme) => Some(Packing {
                operand: self.operand,
                scheme,
            }),
            StoredValues::Elements(_) => None,
        }
    }
}

/// How a matrix is packed: what one read of its words brings and how far one block scale covers,
/// over its two dimensions, and the scheme that stores them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Packing {
    pub operand: Operand,
    pub scheme: QuantScheme,
}

impl Packing {
    /// This packing read a scale block at a time: a unit loads one block of values and its one
    /// scale, which it loads alone ([`scales_read_alone`](Self::scales_read_alone)), in rows or in
    /// tile order. `None` where the scheme or the operand reads no block whole.
    pub fn by_block(&self) -> Option<Packing> {
        Some(Packing {
            operand: self.operand.by_block()?,
            scheme: self.scheme,
        })
        .filter(Packing::scales_read_alone)
    }

    /// The finest reads of this packing, as stored, `across` the dimension its matrix lies across:
    /// its values a word a read, narrowed from the load it was bound for where the matrix is stored
    /// in tile order, and its scales one a read. What a reader plans its step
    /// on before it widens its reads to the step ([`read_within`](Self::read_within)): every other
    /// read of the packing is a whole number of these.
    pub fn finest(&self, across: Dimension) -> Packing {
        let words = self.operand.read_values() / self.scheme.num_quants();
        let operand = match words > 1 {
            true => self.operand.reloaded(across, 1).unwrap_or(self.operand),
            false => self.operand,
        };
        let operand = match self.scales_read_alone() {
            true => operand.scales_alone().unwrap_or(operand),
            false => operand,
        };
        Packing {
            operand,
            scheme: self.scheme,
        }
    }

    /// The widest reads of this packing that a reader holds whole: its values in runs of `run`
    /// along `K`, its scales over a window of `window` along `K`, `across` the dimension its
    /// matrix lies across.
    ///
    /// Values are read as they were bound, the widest load the device takes, narrowed a load at a
    /// time where the matrix is stored in tile order; in rows a read is one word already. Scales are
    /// read a word at a time where the window holds whole words of them, and one at a time
    /// otherwise ([`scales_read_alone`](Self::scales_read_alone)). `None` where no read fits: a run
    /// shorter than one word, or a window holding part of one scale.
    pub fn read_within(&self, across: Dimension, run: usize, window: usize) -> Option<Packing> {
        let mut operand = self.operand;
        while !run.is_multiple_of(operand.read(K)) {
            let words = operand.read_values() / self.scheme.num_quants();
            operand = operand.reloaded(across, (words / 2).max(1))?;
            if words == 1 {
                return None;
            }
        }
        let scale_span = |operand: &Operand| operand.scale_read_space().map(|read| read.extent(K));
        if scale_span(&operand).is_some_and(|span| !window.is_multiple_of(span)) {
            operand = match self.scales_read_alone() {
                true => operand.scales_alone()?,
                false => return None,
            };
        }
        match scale_span(&operand).is_some_and(|span| !window.is_multiple_of(span)) {
            true => None,
            false => Some(Packing {
                operand,
                scheme: self.scheme,
            }),
        }
    }

    /// Whether its reads take each scale alone, as [`by_block`](Self::by_block) reads them: what
    /// binds its block scales one apiece.
    pub fn reads_scales_alone(&self) -> bool {
        self.operand.reads_scales_alone() && self.scales_read_alone()
    }

    /// Whether each of its block scales is whole bytes, which a unit loads alone: every scale type
    /// a scheme states, `ue4m3` and `ue8m0` a byte, `f16` two, `f32` four.
    fn scales_read_alone(&self) -> bool {
        ElemType::from_scale_dtype(self.scheme.finest().dtype)
            .size_bits()
            .is_multiple_of(u8::BITS as usize)
    }

    /// Bits one value holds, packed.
    pub fn value_bits(&self) -> usize {
        self.scheme.size_bits_stored() / self.scheme.num_quants()
    }

    /// Bytes the words of a matrix of `values` values hold, packed.
    pub fn values_bytes(&self, values: usize) -> usize {
        values * self.scheme.size_bits_stored() / self.scheme.num_quants() / 8
    }

    /// Bytes a matrix of `values` values holds: its words and its block scales. The per-tensor
    /// scale is one word and left out.
    pub fn bytes(&self, values: usize) -> usize {
        let scales_per_word = Field::of_scale(self.scheme.finest().dtype).per_word();
        let scales = self.operand.scaled_along().map_or(0, |dimension| {
            let covered = self.operand.scale(dimension).unwrap_or(1);
            (values / covered) * 4 / scales_per_word
        });
        self.values_bytes(values) + scales
    }
}

/// One dimension of an operand, stated from the leaf up for the units that spread
/// it: what one unit holds at the bottom, how many of it each split dimension distributes
/// to the units, which split dimensions they take in turns, and the walk over what they do
/// not cover between them.
///
/// The two split dimensions are the dimension's own ([`blocked`]): the scale groups, and the
/// position inside one. A dimension no scale groups names only the second.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Spread {
    /// The tile one unit holds at the bottom, per split dimension.
    pub leaf: Vec<(Dimension, usize)>,
    /// How many of the thing below each split dimension distributes to the units. Empty where
    /// the units take none of this dimension and every one of them walks it whole.
    pub units: Vec<(Dimension, usize)>,
    /// Split dimensions the units take in turns rather than in runs.
    pub interleaved: Vec<Dimension>,
    /// The walk over what the units do not hold between them.
    pub walk: Walked,
}

/// Where the walk over a dimension sits, relative to the units that spread it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Walked {
    /// Above the units, taking every tile of these split dimensions the level over it
    /// hands down: a grouped dimension, whose scales and reads the units take in
    /// turns.
    Above(Vec<Dimension>),
    /// Below the units, one unit stepping this many of the leaf: a contiguous
    /// run each.
    Below(Dimension, usize),
}

#[cfg(test)]
mod tests {
    use crate::layout::stored::{Operand, Packing};
    use crate::layout::tile_order::{self, TileOrder, TileWords};
    use crate::matmul_dimensions::{K, KB, KI, N, NI};
    use cubecl::tile::layout::Run;

    /// Words one 128-bit load holds.
    const WORDS: usize = 4;

    /// NVFP4 read along `k`: eight values a word, sixteen a scale, four `ue4m3` scales a word.
    fn nvfp4() -> Operand {
        Operand::packed(K, 8, K, 16, 4)
    }

    /// NVFP4 laid out in `tile` as `words` asks.
    fn laid(words: TileWords, tile: (usize, usize)) -> TileOrder {
        TileOrder::laid(&tile_order::nvfp4_scheme(), words, tile).expect("NVFP4 is tile order")
    }

    /// A packing in `order`, read `words` words to a load, under `scheme`.
    fn in_order(scheme: cubecl::quant::scheme::QuantScheme, words: usize) -> Packing {
        let order = TileOrder::laid(&scheme, TileWords::RowByRow, (128, 8)).expect("tile order");
        let operand = Operand::of_scheme(&scheme, K, [K, N])
            .and_then(|operand| operand.in_tile_order(order, N, words))
            .expect("tile order is read");
        Packing { operand, scheme }
    }

    /// `value` packed down `k` of a `[k, n]` matrix, a scale of `scale` every `block`.
    fn down_k(
        value: cubecl::quant::scheme::QuantValue,
        block: u8,
        scale: cubecl::quant::scheme::ScaleDtype,
    ) -> cubecl::quant::scheme::QuantScheme {
        cubecl::quant::scheme::QuantScheme::default()
            .with_value(value)
            .per_block([block, 1], scale)
            .with_store(cubecl::quant::scheme::QuantStore::PackedU32(1))
    }

    /// Reads as `(values along K, K one read of scales covers)`.
    fn reads(packing: Option<Packing>) -> Option<(usize, usize)> {
        packing.map(|packing| {
            let scales = packing.operand.scale_read_space().expect("scaled");
            (packing.operand.read(K), scales.extent(K))
        })
    }

    /// Every format a reader with a 16-value run and a 64-deep window reads whole, from properties
    /// alone: MXFP4 in tile order narrowed from its 32-value load to two words, its scale word of
    /// 128 read a scale at a time; NVFP4's 16-value load and 64-wide scale word kept; MXFP8's
    /// 16-value load kept and its 128-wide scale word read a scale at a time.
    #[test]
    fn a_reader_takes_the_widest_reads_its_run_and_window_hold() {
        use cubecl::quant::scheme::{QuantValue, ScaleDtype};
        let mxfp4 = in_order(down_k(QuantValue::E2M1, 32, ScaleDtype::UE8M0), WORDS);
        let nvfp4 = in_order(tile_order::nvfp4_scheme(), WORDS);
        let mxfp8 = in_order(down_k(QuantValue::E4M3, 32, ScaleDtype::UE8M0), WORDS);
        assert_eq!(reads(Some(mxfp4)), Some((32, 128)));
        assert_eq!(reads(mxfp4.read_within(N, 16, 64)), Some((16, 32)));
        assert_eq!(reads(nvfp4.read_within(N, 16, 64)), Some((16, 64)));
        assert_eq!(reads(mxfp8.read_within(N, 16, 64)), Some((16, 32)));
        // A window of whole words of scales keeps them a word a read.
        assert_eq!(reads(mxfp4.read_within(N, 32, 128)), Some((32, 128)));
    }

    /// The finest reads are a word of values and, where scales are bytes, one scale: MXFP4 in tile
    /// order from its 32-value load and 128-wide scale word to 8 and 32; q4's word of `f32` scales,
    /// one a word, kept.
    #[test]
    fn the_finest_reads_are_a_word_and_a_scale() {
        use cubecl::quant::scheme::{QuantValue, ScaleDtype};
        let mxfp4 = in_order(down_k(QuantValue::E2M1, 32, ScaleDtype::UE8M0), WORDS);
        assert_eq!(reads(Some(mxfp4.finest(N))), Some((8, 32)));
        let scheme = down_k(QuantValue::Q4S, 32, ScaleDtype::F32);
        let q4 = Packing {
            operand: Operand::of_scheme(&scheme, K, [K, N]).expect("one dimension a block"),
            scheme,
        };
        assert_eq!(reads(Some(q4.finest(N))), Some((8, 32)));
    }

    /// A window holding part of one scale reads none: `f32` scales, one a word, are read a scale at
    /// a time, and a window of 16 holds half of one of 32.
    #[test]
    fn a_window_holding_part_of_one_scale_reads_none() {
        use cubecl::quant::scheme::{QuantValue, ScaleDtype};
        let q4 = Packing {
            operand: Operand::of_scheme(&down_k(QuantValue::Q4S, 32, ScaleDtype::F32), K, [K, N])
                .expect("one dimension a block"),
            scheme: down_k(QuantValue::Q4S, 32, ScaleDtype::F32),
        };
        assert_eq!(reads(q4.read_within(N, 8, 32)), Some((8, 32)));
        assert_eq!(reads(q4.read_within(N, 8, 16)), None);
        assert_eq!(
            reads(q4.read_within(N, 4, 32)),
            None,
            "a run shorter than a word"
        );
    }

    /// Tile order row by row, as the writer stores it for a 128-bit load, is read two words along
    /// `k`, the one block a read stays inside, and its scales, in rows, a word of four blocks down
    /// one row. The operand keeps the order.
    #[test]
    fn tile_order_row_by_row_is_read_a_block_a_load() {
        let order = laid(TileWords::RowByRow, (128, 8));
        let stored = nvfp4()
            .in_tile_order(order, N, WORDS)
            .expect("tile order is read");
        assert_eq!((stored.read(K), stored.read(N)), (16, 1));
        assert_eq!(stored.read_values(), 16);
        assert_eq!((stored.scale_read(K), stored.scale_read(N)), (4, 1));
        assert_eq!(stored.lined(), K);
        assert_eq!(stored.tile_order(), Some(order));
    }

    /// Tile order block by block, as the writer stores it for a 128-bit load, is read a load of
    /// values and their scales at once: its values one block of two rows side by side, a read
    /// staying inside one block of each row, and its scales a word of four blocks down each of
    /// those two rows, side by side.
    #[test]
    fn tile_order_block_by_block_is_read_a_load_of_values_and_its_scales() {
        let stored = nvfp4()
            .in_tile_order(laid(TileWords::BlockByBlock, (128, 8)), N, WORDS)
            .expect("tile order is read");
        assert_eq!((stored.read(K), stored.read(N)), (16, 2));
        assert_eq!(stored.read_values(), 32);
        assert_eq!((stored.scale_read(K), stored.scale_read(N)), (4, 2));
        assert_eq!(
            stored.scales_per_word(),
            4,
            "a read of scales covers a word of blocks"
        );
        assert_eq!(stored.lined(), K);
    }

    /// Block by block, a read of the scales spans the rows a read of the values does whatever the
    /// tile: a tile of four rows, whose scales a whole load could read four rows of, and one of
    /// two rows, whose next scale tile lies down `k`.
    #[test]
    fn a_read_of_scales_in_tiles_spans_the_rows_of_a_read_of_values() {
        for tile in [(128, 4), (128, 2), (128, 8)] {
            let stored = nvfp4()
                .in_tile_order(laid(TileWords::BlockByBlock, tile), N, WORDS)
                .unwrap_or_else(|| panic!("{tile:?}: tile order is read"));
            assert_eq!(stored.read(N), 2, "{tile:?}");
            assert_eq!(stored.scale_read(N), stored.read(N), "{tile:?}");
        }
    }

    /// An operand in rows states no tile order.
    #[test]
    fn an_operand_in_rows_has_no_tile_order() {
        assert_eq!(nvfp4().tile_order(), None);
        assert_eq!(Operand::plain().tile_order(), None);
    }

    /// A stated tile is taken whole or not at all: a tile of sixteen columns stated as one level
    /// is read a column's two words, and eight columns beside a word are read a word alone, the
    /// stated tiles past one load left to the buffer.
    #[test]
    fn a_read_takes_whole_tiles_while_they_fit_one_load() {
        let one_level = [(KI, 8), (KI, 2), (NI, 16)];
        let stored = nvfp4()
            .stored(
                N,
                WORDS,
                Some((&tile_order::values_dimensions(N), &one_level)),
                None,
            )
            .expect("tile order is read");
        assert_eq!((stored.read(K), stored.read(N)), (16, 1));

        for (columns, read) in [(1, 1), (2, 2), (4, 4), (8, 1)] {
            let tiles = [(KI, 8), (NI, columns)];
            let stored = nvfp4()
                .stored(
                    N,
                    WORDS,
                    Some((&tile_order::values_dimensions(N), &tiles)),
                    None,
                )
                .expect("columns beside a word are read");
            assert_eq!((stored.read(K), stored.read(N)), (8, read), "{columns}");
        }
    }

    /// The blocks continue a run along `k` only once a read holds a whole block: a word of half a
    /// block, then the next block's word, is not sixteen values in a row, and is refused rather
    /// than read as one.
    #[test]
    fn a_read_does_not_jump_to_the_next_block_inside_one() {
        let half_a_block_then_the_next = [(KI, 8), (KB, 2), (NI, 4)];
        assert_eq!(
            nvfp4().stored(
                N,
                WORDS,
                Some((
                    &tile_order::values_dimensions(N),
                    &half_a_block_then_the_next
                )),
                None
            ),
            None
        );
    }

    /// A buffer that does not start with the scheme's word, or whose load would come back to
    /// `k` after the columns, is no run along each dimension: refused rather than misread.
    #[test]
    fn a_read_that_is_not_whole_words_in_one_run_each_is_refused() {
        let half_word = [(KI, 4), (KI, 4)];
        let interleaved = [(KI, 8), (NI, 2), (KI, 2)];
        for tiles in [&half_word[..], &interleaved[..]] {
            assert_eq!(
                nvfp4().stored(
                    N,
                    WORDS,
                    Some((&tile_order::values_dimensions(N), tiles)),
                    None
                ),
                None,
                "{tiles:?}"
            );
        }
    }

    /// The packing of [`nvfp4`] laid out in `tile` as `words` asks, or in rows where `None`.
    fn packing(tile: Option<(TileWords, (usize, usize))>) -> Packing {
        let operand = match tile {
            Some((words, tile)) => nvfp4()
                .in_tile_order(laid(words, tile), N, WORDS)
                .expect("tile order is read"),
            None => nvfp4(),
        };
        Packing {
            operand,
            scheme: tile_order::nvfp4_scheme(),
        }
    }

    /// A transaction of 128 bytes over NVFP4 64 deep by 8 rows a tile, its words block by block:
    /// a read is one block of two rows, 16 bytes; the transaction takes the tile's four pairs of
    /// rows, then two of its blocks down `K`. A stored tile is its four blocks down `K`.
    #[test]
    fn block_by_block_runs_go_across_the_rows_then_down_k() {
        let packing = packing(Some((TileWords::BlockByBlock, (64, 8))));
        let reads = |run| {
            packing
                .operand
                .reads_down_k(run, N, (1024, 64), packing.value_bits(), 128)
        };
        assert_eq!(reads(Run::Load), Some(1));
        assert_eq!(reads(Run::Transaction), Some(2));
        assert_eq!(reads(Run::StoredTile), Some(4));
    }

    /// Row by row, a read is one block of one row, 8 bytes, and a row's four blocks lie together:
    /// a transaction is a row's four reads down `K` and four rows beside them, the 128 bytes that
    /// lie together, rather than sixteen reads down one row, which would cross four tiles.
    #[test]
    fn row_by_row_runs_go_down_a_row_then_across_the_rows() {
        let packing = packing(Some((TileWords::RowByRow, (64, 8))));
        let reads = |run| {
            packing
                .operand
                .reads_down_k(run, N, (1024, 64), packing.value_bits(), 128)
        };
        assert_eq!(reads(Run::Transaction), Some(4));
        assert_eq!(reads(Run::StoredTile), Some(4));
    }

    /// In rows a column's words run down `K`: a transaction is as many reads down it as fill one,
    /// and no tile is stored.
    #[test]
    fn rows_run_down_a_column() {
        let packing = packing(None);
        let read_bytes = packing.values_bytes(packing.operand.read_values());
        let reads = |run| {
            packing
                .operand
                .reads_down_k(run, N, (1024, 64), packing.value_bits(), 128)
        };
        assert_eq!(reads(Run::Transaction), Some(128 / read_bytes));
        assert_eq!(reads(Run::StoredTile), None);
    }

    /// One scale a row of a `[rows, k]` matrix is constant along `k`, position 1, and varies
    /// along the rows, position 0.
    #[test]
    fn a_scale_a_row_is_constant_along_the_contraction() {
        let scheme = cubecl::quant::scheme::QuantScheme::fp8_per_row();
        assert_eq!(Operand::constant_along(&scheme.finest(), 1), Some(Some(0)));
    }

    #[test]
    fn one_scale_for_the_matrix_varies_along_nothing() {
        let scheme = cubecl::quant::scheme::QuantScheme::default()
            .per_tensor(cubecl::quant::scheme::ScaleDtype::F32);
        assert_eq!(Operand::constant_along(&scheme.finest(), 1), Some(None));
    }

    #[test]
    fn blocks_along_the_contraction_are_not_constant_along_it() {
        let scheme = cubecl::quant::scheme::QuantScheme::default()
            .per_block([32], cubecl::quant::scheme::ScaleDtype::UE8M0);
        assert_eq!(Operand::constant_along(&scheme.finest(), 1), None);
    }

    /// Values read in an 8-bit float hold no product with an outer scale, so it goes into the sum.
    #[test]
    fn an_outer_scale_over_8_bit_values_is_taken_in_the_sum() {
        let e4m3 = cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::E4M3);
        assert_eq!(
            super::OuterScale::Kept.read_in(e4m3),
            super::OuterScale::Factored
        );
    }
}
