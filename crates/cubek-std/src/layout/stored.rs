//! An operand of a contraction, as a launch derives its reads.
//!
//! Two numbers per axis answer every question a launch asks of a tensor: the microtile one read of
//! it brings, and the microtile one scale covers. Both are read off the tensor, so a float operand
//! reads one value and is constant inside nothing, and the same reads serve it without asking
//! whether anything is quantized at all.

use crate::layout::tile_order::{TileOrder, scales_axes, values_axes};
use crate::matmul_axes::{K, blocked};
use cubecl::quant::scheme::{QuantScheme, QuantStore};
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Axis, MicroTile, OperandReads, Space, kind::Field};

/// How an operand is read and how it is scaled, as tiles of its own axes.
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

    /// This operand with its axis `from` named `to`: the same reads and scales, over a matrix whose
    /// axes another reader names otherwise.
    pub fn renamed(self, from: Axis, to: Axis) -> Self {
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

    /// The operand `scheme` describes when its words run along `axis`.
    ///
    /// **The axis is the caller's**: a buffer's strides say which dim its words
    /// run along, and nothing here re-derives it. What a scheme states is the
    /// two lengths — how many values share a word, and how many share a scale.
    /// `axes` names the tensor's dims in the order the scheme's block shape
    /// counts them.
    ///
    /// `None` where the block spans both axes, or where a read would straddle
    /// two scales: neither is a form this kernel streams.
    pub fn of_scheme(scheme: &QuantScheme, axis: Axis, axes: [Axis; 2]) -> Option<Self> {
        let dims = scheme.block_size()?.to_dim_vec(2);
        let position = axes.iter().position(|&a| a == axis)?;
        if dims[1 - position] != 1 {
            return None;
        }
        let scale = dims[position] as usize;
        let read = scheme.num_quants();
        let scales_per_word = Field::of_scale(scheme.block_scale()?.dtype).per_word();
        scale
            .is_multiple_of(read)
            .then(|| Operand::packed(axis, read, axis, scale, scales_per_word))
    }

    /// The axis a scheme's own block shape implies.
    ///
    /// For the two callers holding no tensor — a tuning row, and the rule asked
    /// before a launch — where the strides that decide it everywhere else are
    /// not in hand. `PackedU32(0)` counts from the innermost dim, so it is the
    /// second of `axes`.
    pub fn axis_of(scheme: &QuantScheme, axes: [Axis; 2]) -> Option<Axis> {
        match scheme.store {
            QuantStore::PackedU32(0) => Some(axes[1]),
            QuantStore::PackedU32(1) => Some(axes[0]),
            _ => None,
        }
    }

    /// What one read brings along `axis`.
    pub fn read(&self, axis: Axis) -> usize {
        self.reads.read.extent(axis)
    }

    /// Scales one read of them brings: as many as share a word, one where this
    /// operand carries none. The blocks a read of scales covers are one worker's.
    pub fn scales_per_word(&self) -> usize {
        self.scaled_along()
            .map_or(1, |axis| self.reads.scale_read.extent(axis))
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

    /// What one read of the scales brings along `axis`, in scales.
    pub fn scale_read(&self, axis: Axis) -> usize {
        self.reads.scale_read.extent(axis)
    }

    /// Values one read brings, every axis it spans.
    pub fn read_values(&self) -> usize {
        self.reads.read.values()
    }

    /// Scales one read of them brings, every axis it spans.
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
    /// this operand's scheme reads, along its own axis or along `across`, so no read of it is
    /// whole words; or one whose load would step along an axis that is neither, or come back to
    /// one after the other, which is no longer one run along each.
    pub fn in_tile_order(self, order: TileOrder, across: Axis, words: usize) -> Option<Self> {
        let values = order.values_storage(across, words);
        let scales = order.scales_storage(across, order.rows_a_load(across, words));
        let (values_axes, scales_axes) = (values_axes(across), scales_axes(across));
        let stored = self.stored(
            across,
            words,
            Some((&values_axes, values.tiles())),
            scales
                .as_ref()
                .map(|scales| (scales_axes.as_slice(), scales.tiles())),
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
        across: Axis,
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
            Some(axis) => read_off(
                self.reads.scale_read,
                axis,
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

    /// What one scale covers along `axis`, where this operand carries scales.
    pub fn scale(&self, axis: Axis) -> Option<usize> {
        self.reads.scale.as_ref().map(|tile| tile.extent(axis))
    }

    /// The axis this operand's scales are grouped along, where they group one.
    ///
    /// A scales operand asks, because the groups it counts are that axis's.
    pub fn scaled_along(&self) -> Option<Axis> {
        self.reads.scaled_along()
    }

    /// The axis this operand's words run along, and `K` where its reads are one
    /// value wide.
    ///
    /// A binding asks, because a buffer's dims are named in the order it steps
    /// them and this is the one it steps first.
    pub fn lined(&self) -> Axis {
        self.reads.read.axis().unwrap_or(K)
    }

    /// The length a scale groups `axis` into, where it groups it at all.
    ///
    /// **A scale covering one value groups nothing**, and neither does one
    /// covering the whole axis: a grouping is what makes a walk read the same
    /// scale twice, and both of those make it read a new one every step or
    /// never. This is the one rule the cuts and the space share.
    pub fn grouping(&self, axis: Axis, extent: usize) -> Option<usize> {
        self.scale(axis)
            .filter(|&scale| scale > 1 && scale < extent)
    }

    /// Which of the axes `axis` is split into a `width`-wide cut goes on, and how wide it is on
    /// each: the groups where a scale groups the axis, the position inside one
    /// where it does not — and **both** where the width is narrower than a
    /// group, which is one group and part of it.
    ///
    /// A cut wider than a group but not a whole number of them is none of the
    /// three and states nothing; a caller reading an empty answer is holding a
    /// width this operand's scales do not admit.
    pub fn cut(&self, axis: Axis, extent: usize, width: usize) -> Vec<(Axis, usize)> {
        let (groups, inside) = blocked(axis);
        let Some(scale) = self.grouping(axis, extent) else {
            return vec![(inside, width)];
        };
        match (width.is_multiple_of(scale), scale.is_multiple_of(width)) {
            (true, _) => vec![(groups, width / scale)],
            (false, true) => vec![(groups, 1), (inside, width)],
            (false, false) => Vec::new(),
        }
    }

    /// The axes `axis` is split into that a `width`-wide box spans, each with its extent: whole groups and the
    /// whole of each where `width` covers whole groups a scale makes of the axis, the position
    /// inside one where it does not, and the position alone where no scale groups the axis.
    ///
    /// Where a [`cut`](Self::cut) states what a level counts, the groups alone, a box states
    /// every split axis it covers, so whole groups bring the position inside each with them.
    pub fn span(&self, axis: Axis, extent: usize, width: usize) -> Vec<(Axis, usize)> {
        let (groups, inside) = blocked(axis);
        match self.grouping(axis, extent) {
            Some(scale) if width.is_multiple_of(scale) => {
                vec![(groups, width / scale), (inside, scale)]
            }
            Some(_) => vec![(groups, 1), (inside, width)],
            None => vec![(inside, width)],
        }
    }

    /// How many `tile`-wide tiles of `axis` one `span`-wide tile holds, per
    /// split axis: the counts a level states, the level below it covering `tile`
    /// and this one `span`.
    ///
    /// Both widths are ones this operand's scales admit ([`cut`](Self::cut)).
    /// A split axis that multiplies by one is left out, a level stating only what
    /// it multiplies.
    pub fn tiles(&self, axis: Axis, extent: usize, tile: usize, span: usize) -> Vec<(Axis, usize)> {
        let (groups, inside) = blocked(axis);
        let Some(scale) = self.grouping(axis, extent) else {
            return vec![(inside, span / tile)];
        };
        // Whole blocks of the axis, or part of one block: the two split axes any
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

    /// The two axes `axis` becomes in a space at `extent`: the groups a scale
    /// makes of it, and the position inside one. An axis no scale groups is one
    /// group over the whole of it.
    pub fn axes(&self, axis: Axis, extent: usize) -> [(Axis, usize); 2] {
        let (groups, inside) = blocked(axis);
        let scale = self.grouping(axis, extent).unwrap_or(extent);
        // The one division that declares a tiling, and it is exact: a space a scale does not
        // tile would be a space short of the extent, and nothing downstream could tell.
        assert!(
            extent.is_multiple_of(scale),
            "{extent} along {axis:?} is not whole scale tiles of {scale}"
        );
        [(groups, extent / scale), (inside, scale)]
    }

    /// The widest group of at most `units` units that distributes `axis` evenly, and
    /// that tiles the `units` it came from.
    ///
    /// A grouped axis is distributed a word of scales a unit per turn
    /// ([`spread_across`](Self::spread_across)), so a group distributes it evenly
    /// where its width divides the axis's words of scales. `units` as they are
    /// where they already do, where no scale groups the axis, and where its
    /// blocks are not whole words, which no group distributes.
    pub fn units_distributing(&self, axis: Axis, extent: usize, units: usize) -> usize {
        if self.grouping(axis, extent).is_none() {
            return units;
        }
        let [(_, blocks), _] = self.axes(axis, extent);
        if !blocks.is_multiple_of(self.scales_per_word()) {
            return units;
        }
        let words = blocks / self.scales_per_word();
        (1..=units)
            .rev()
            .find(|&group| units.is_multiple_of(group) && words.is_multiple_of(group))
            .unwrap_or(1)
    }

    /// How `units` spread `axis` between them, stated from the leaf up.
    ///
    /// A grouped axis is distributed by the scales: each unit takes the blocks one
    /// read of them covers — as many as share a word — and the units take
    /// turns, so neighbouring units hold neighbouring words of scales. The
    /// position inside a block is never cut: whoever holds a block's scale
    /// holds the whole block, so the walk over what the units do not span sits
    /// *above* them and counts blocks alone. An ungrouped axis is a contiguous
    /// run each, walked *below* the units — and no unit cut at all where they
    /// do not divide it, every unit then walking the whole of it and computing
    /// the same sum.
    ///
    /// # Errors
    ///
    /// Units that do not distribute the scales' words evenly.
    pub fn spread_across(
        &self,
        axis: Axis,
        extent: usize,
        units: usize,
    ) -> Result<Spread, Refusal> {
        let (groups, inside) = blocked(axis);
        let read = self.read(axis);
        let Some(scale) = self.grouping(axis, extent) else {
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
        let [(_, blocks), _] = self.axes(axis, extent);
        let word = self.scales_per_word();
        // Each unit takes a word of scales a turn, so a turn is `units` words: the floor a
        // contraction has to reach for this many units, and the unit a longer one is distributed in.
        if !blocks.is_multiple_of(word * units) {
            return Err(Refusal::Declined {
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
        lined: Axis,
        read: usize,
        tiled: Axis,
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
/// fit one load and stay inside `inside` along each axis it names. The tiles past the load are the
/// buffer's to step, not the read's.
///
/// `None` where the buffer does not start with that word, or where a tile inside the load lies
/// along neither axis or comes back to one after the other.
fn read_off(
    tile: MicroTile,
    along: Axis,
    across: Axis,
    words: usize,
    inside: &MicroTile,
    stored: Option<Stored<'_>>,
) -> Option<MicroTile> {
    let Some((axes, tiles)) = stored else {
        return Some(tile);
    };
    let per_word = tile.extent(along);
    // The word starts at the finer of the axes an axis is split into; a tile above it continues
    // the run along an axis on that same split axis, or on the blocks once `read` holds a whole
    // one of `inside`: before then the next block's values are not the next ones along the axis.
    let axis_of = |tile_axis: Axis, read: &MicroTile| {
        [along, across].into_iter().find(|&axis| {
            let (groups, within) = blocked(axis);
            let block = inside.extent(axis);
            finest_axis(axes, axis) == Some(tile_axis)
                || (tile_axis == groups
                    && axes.contains(&within)
                    && block > 1
                    && read.extent(axis) == block)
        })
    };
    let (&(tile_axis, first), above) = tiles.split_first()?;
    let word_along = axis_of(tile_axis, &MicroTile::unit()).filter(|_| first == per_word)?;
    let mut read = MicroTile::along(word_along, per_word);
    let mut taken = 1;
    let mut last = word_along;
    for &(tile_axis, count) in above.iter().filter(|&&(_, count)| count > 1) {
        if taken * count > words {
            break;
        }
        let axis = axis_of(tile_axis, &read)?;
        let bound = inside.extent(axis);
        if bound > 1 && read.extent(axis) * count > bound {
            break;
        }
        if axis != last && read.extent(axis) > 1 {
            return None;
        }
        read = read.times(axis, count);
        (taken, last) = (taken * count, axis);
    }
    Some(read)
}

/// The finer of the two axes `axis` is split into ([`blocked`]) a buffer over `axes` carries:
/// the position inside a block where it has it, the blocks where it omits it (as scales do).
fn finest_axis(axes: &[Axis], axis: Axis) -> Option<Axis> {
    let (groups, within) = blocked(axis);
    [within, groups]
        .into_iter()
        .find(|split| axes.contains(split))
}

/// A buffer stored in tiles: the axes its dims carry, and its stored tiles, finest first.
type Stored<'a> = (&'a [Axis], &'a [(Axis, usize)]);

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

/// How a matrix is packed: what one read of its words brings and how far one block scale covers,
/// over its two axes, and the scheme that stores them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Packing {
    pub operand: Operand,
    pub scheme: QuantScheme,
}

impl Packing {
    /// Bytes the words of a matrix of `values` values hold, packed.
    pub fn values_bytes(&self, values: usize) -> usize {
        values * self.scheme.size_bits_stored() / self.scheme.num_quants() / 8
    }

    /// Bytes a matrix of `values` values holds: its words and its block scales. The per-tensor
    /// scale is one word and left out.
    pub fn bytes(&self, values: usize) -> usize {
        let scales = match (self.operand.scaled_along(), self.scheme.block_scale()) {
            (Some(axis), Some(block)) => {
                let covered = self.operand.scale(axis).unwrap_or(1);
                (values / covered) * 4 / Field::of_scale(block.dtype).per_word()
            }
            _ => 0,
        };
        self.values_bytes(values) + scales
    }
}

/// One axis of an operand, stated from the leaf up for the units that spread
/// it: what one unit holds at the bottom, how many of it each split axis distributes
/// to the units, which split axes they take in turns, and the walk over what they do
/// not cover between them.
///
/// The two split axes are the axis's own ([`blocked`]): the scale groups, and the
/// position inside one. An axis no scale groups names only the second.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Spread {
    /// The tile one unit holds at the bottom, per split axis.
    pub leaf: Vec<(Axis, usize)>,
    /// How many of the thing below each split axis distributes to the units. Empty where
    /// the units take none of this axis and every one of them walks it whole.
    pub units: Vec<(Axis, usize)>,
    /// Split axes the units take in turns rather than in runs.
    pub interleaved: Vec<Axis>,
    /// The walk over what the units do not hold between them.
    pub walk: Walked,
}

/// Where the walk over an axis sits, relative to the units that spread it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Walked {
    /// Above the units, taking every tile of these split axes the level over it
    /// hands down: a grouped axis, whose scales and reads the units take in
    /// turns.
    Above(Vec<Axis>),
    /// Below the units, one unit stepping this many of the leaf: a contiguous
    /// run each.
    Below(Axis, usize),
}

#[cfg(test)]
mod tests {
    use crate::layout::stored::Operand;
    use crate::layout::tile_order::{self, TileOrder, TileWords};
    use crate::matmul_axes::{K, KB, KI, N, NI};

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
                Some((&tile_order::values_axes(N), &one_level)),
                None,
            )
            .expect("tile order is read");
        assert_eq!((stored.read(K), stored.read(N)), (16, 1));

        for (columns, read) in [(1, 1), (2, 2), (4, 4), (8, 1)] {
            let tiles = [(KI, 8), (NI, columns)];
            let stored = nvfp4()
                .stored(N, WORDS, Some((&tile_order::values_axes(N), &tiles)), None)
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
                Some((&tile_order::values_axes(N), &half_a_block_then_the_next)),
                None
            ),
            None
        );
    }

    /// A buffer that does not start with the scheme's word, or whose load would come back to
    /// `k` after the columns, is no run along each axis: refused rather than misread.
    #[test]
    fn a_read_that_is_not_whole_words_in_one_run_each_is_refused() {
        let half_word = [(KI, 4), (KI, 4)];
        let interleaved = [(KI, 8), (NI, 2), (KI, 2)];
        for tiles in [&half_word[..], &interleaved[..]] {
            assert_eq!(
                nvfp4().stored(N, WORDS, Some((&tile_order::values_axes(N), tiles)), None),
                None,
                "{tiles:?}"
            );
        }
    }
}
