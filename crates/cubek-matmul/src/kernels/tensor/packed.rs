//! A packed tensor bound as one tile: its words, under its block scales, under its per-tensor scale
//! where its scheme has one. Every kernel that reads a packed operand reads it through
//! [`PackedTensor::bind`], whichever family it belongs to; the family names the operand's
//! dimensions and how its block scales are read ([`BlockScales`]), nothing else.

use super::PlainTensor;
use core::marker::PhantomData;
use cubecl::client::Client;
use cubecl::ir::{ElemType, IntKind, UIntKind};
use cubecl::prelude::{TensorBinding, TensorMapArg, TensorMapSwizzle};
use cubecl::quant::scheme::{BlockScale, QuantScheme, QuantStore, QuantValue};
use cubecl::tile::launch::{Arg, Decline, Delivery, Labelled, TmaCoordinates, TmaLanding, TmaTerm};
use cubecl::tile::{
    Dimension, Launcher, Projection, TileTensorLaunch,
    kind::Field,
    layout::{AxesBuilder, Offset, PhysicalAxisMap, split},
};
use cubecl::zspace::metadata::Metadata;
use cubecl::zspace::{Shape, Strides};
use cubek_microkernel::matmul::MatmulStoredTile;
use cubek_std::Hardware;
use cubek_std::launch::tma::{
    swizzle_span_bytes, tma_meta_tiled, tma_operand_tiled_typed, tma_scale_words_in_tiles,
};
use cubek_std::layout::stored::{BlockScales, Operand, OuterScale, Packing};
use cubek_std::layout::tile_order::{self, TileOrder, TileWords};
use cubek_std::matmul_dimensions::{EXPERT, K, blocked};
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Breakable};

/// The element a packed matrix's words and scales are bound in: whole `u32` words.
const WORD: ElemType = ElemType::UInt(UIntKind::U32);

/// What should hold of the words the TMA engine fills from.
const TMA_FILLS_FROM_ROWS: &str =
    "the TMA engine fills from words in rows, or in tiles laid row by row";

/// What should hold of a stage the TMA engine fills from words in tiles.
const TMA_STAGE_OF_WHOLE_TILES: &str = "a stage the TMA engine fills from words in tiles is whole \
     tiles along its rows, and whole tiles or a part dividing one along k";

/// What should hold of a stage the TMA engine fills from scales in tiles.
const TMA_STAGE_OF_WHOLE_STRIPS: &str =
    "a stage the TMA engine fills from scales in tiles is whole strips of a tile's rows";

/// What should hold of a packed matrix the TMA engine fills from.
const TMA_FILLS_FROM_ONE_SLAB: &str =
    "the TMA engine fills from no stack: a stack is read slab by slab";

/// What should hold of words in rows the TMA engine fills from.
const TMA_ROWS_DENSE: &str = "rows of words the TMA engine fills from are one dense run each";

/// What should hold of a matrix bound with its block scales as words.
const SCALE_WORDS_OF_WHOLE_ROWS: &str =
    "block scales bound as words lie in rows, and no window of rows reads them";

/// The extents of one scale, and the strides of a column of them.
const ONE_SCALE: [usize; 2] = [1, 1];

/// What [`PackedTensor::of_quantized`] declines: a scheme with no block scales along the dimension
/// its words run along, or with a coarser level of more than one scale.
pub const SCALES_NOT_ALONG_WORDS: Decline = Decline::Unserved {
    why: "the packed routines read block scales along the dimension the words run along, under \
          at most one scale for the whole tensor",
};

/// What [`PackedTensor::of_quantized`] declines: a batch of matrices, or values stored other than
/// in contiguous rows.
pub const PACKED_NOT_ONE_MATRIX_IN_ROWS: Decline = Decline::Unserved {
    why: "the packed routines read one quantized matrix in row-major rows",
};

/// What a read of a packed tensor declines: anything but one packed, block-scaled matrix.
pub const TENSOR_NOT_PACKED: Decline = Decline::Unserved {
    why: "a packed read takes one packed, block-scaled matrix",
};

/// What reading a packed tensor as a stack of slabs declines: slabs that split one read of its
/// words, one block of its scales or one word of block scales; and for words stored in tiles,
/// slabs along anything but its rows, or that split a strip of its tiles.
pub const PACKED_STACK_UNREAD: Decline = Decline::Unserved {
    why: "a packed stack is read slab by slab, each slab whole reads, whole scale blocks and \
          whole words of block scales, and in tiles whole strips of its rows",
};

/// What reading a packed tensor as a batch of matrices, or a window of its rows, declines: words
/// stored in tiles, a two-level scheme, whose per-tensor scale is the whole tensor's, a read
/// already stacked or windowed, rows a batch does not divide, or rows past the matrix's.
pub const PACKED_BATCH_UNREAD: Decline = Decline::Unserved {
    why: "a packed batch or window is read from words in rows under a one-level scheme, once, \
          over rows the matrix holds",
};

/// What a read of a packed tensor declines: values stored in tiles it reads no whole load of.
pub const STORED_TILES_UNREAD: Decline = Decline::Unserved {
    why: "a packed read takes values stored in tiles shaped [n, k / block, block] from a \
              whole word, a load one run along k and one along n",
};

/// A packed matrix as a kernel reads it: its words, in the order their buffer steps, its block
/// scales, and its per-tensor scale where its scheme has one, over two dimensions, each with its
/// extent. The words and scales are read where they lie, in rows or in the tiles they are stored
/// in.
///
/// Its scales split each dimension into blocks and the position inside one
/// ([`Operand::dimensions`]); it binds over that block-split space, and restates the dense rows of
/// the tensors read beside it over the same space.
#[derive(Debug, Clone)]
pub struct PackedTensor {
    words: PlainTensor,
    block_scales: PlainTensor,
    outer_scales: Option<PlainTensor>,
    scheme: QuantScheme,
    /// The dimension its words run along.
    lined: Dimension,
    /// Where its words lie in tile order, where they do.
    tiles: Option<StoredTiles>,
    dimensions: [(Dimension, usize); 2],
    stack: Option<PackedStack>,
    window: Option<RowWindow>,
}

/// Where a packed matrix's words lie in tile order, and what one load of the device reading them
/// takes: what, with its scheme and the dimension its words run along, says how it is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoredTiles {
    /// The order its words and scales are laid in.
    pub order: TileOrder,
    /// The dimension the matrix lies across, the one other than `K`.
    pub across: Dimension,
    /// The words one load of the device takes.
    pub words: usize,
}

/// A run of a packed matrix's rows read where they lie: the matrix's extent along its rows is the
/// run's, its words and scales bound over every row `stored` holds, and the dimension its rows
/// split into shifted by `at`, a dynamic offset, so one compiled launch reads every run of its
/// length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RowWindow {
    at: usize,
    stored: usize,
}

/// The matrices a packed tensor holds past its first, bound as a leading dim of its words and
/// scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackedStack {
    /// `slabs` matrices end to end along one of its dimensions, `along`: the words and scales of
    /// each slab one slab's extent further along it, or in tiles one slab's strips of them
    /// ([`TileOrder::stacked`]), bound as a leading [`EXPERT`] dim a table's entry picks one of.
    Slabs { slabs: usize, along: Dimension },
    /// `batches` matrices end to end along its rows, each `pitch` rows of its words and scales past
    /// the one before, bound as a leading dim over `dimension`, which the launch walks as it walks
    /// the matrix's own dimensions.
    Batch {
        batches: usize,
        pitch: usize,
        dimension: Dimension,
    },
}

/// No slab splits a read, a scale block or a word of block scales; in tile order each slab is whole
/// strips (rows of tiles).
fn slabs_are_whole(operand: &Operand, along: Dimension, extent: usize, slabs: usize) -> bool {
    if slabs == 0 || !extent.is_multiple_of(slabs) {
        return false;
    }
    let slab = extent / slabs;
    let blocks_whole = operand.grouping(along, extent).is_none_or(|block| {
        slab.is_multiple_of(block) && (slab / block).is_multiple_of(operand.scales_per_word())
    });
    let strips_whole = operand
        .tile_order()
        .is_none_or(|order| along != operand.lined() && order.holds_whole_strips(slab));
    slab.is_multiple_of(operand.read(along)) && blocks_whole && strips_whole
}

/// For a stack, the whole matrix's layout split into one layout per slab; for one matrix, the layout
/// unchanged.
fn stated_by_slab(
    slabs: Option<usize>,
    rows: usize,
    stated: impl Fn(usize) -> Option<Metadata>,
) -> Option<Metadata> {
    let whole = stated(slabs.unwrap_or(1) * rows)?;
    match slabs {
        Some(slabs) => Some(
            TileOrder::stacked(&whole, slabs)
                .expect("a stack in tile order is whole strips a slab, as `stacked` admits"),
        ),
        None => Some(whole),
    }
}

impl PackedTensor {
    /// A packed matrix over `dimensions`, each with its extent, its words running along `lined`
    /// under `scheme`: its words in the order their buffer steps, in the tile order `tiles` lays
    /// them or in rows, its block scales as whole words where they
    /// lie (rows of them, or for words laid block by block the tiles tile order stores them in),
    /// and its per-tensor scale, as whole words, where `scheme` has one. Its caller reads these
    /// parts off however it stores a packed tensor, and states `tiles` as
    /// [`read_in_tiles`](Self::read_in_tiles) returns them for a matrix in tiles.
    ///
    /// # Assumptions
    ///
    /// - `outer_scales` is given where `scheme` has two levels, and only there.
    /// - `scheme` reads words along `lined` over `dimensions` ([`Operand::of_scheme`]), and, in
    ///   `tiles`, in their order ([`Operand::in_tile_order`]).
    ///
    /// # Panics
    ///
    /// Outer scales given for a one-level scheme, or none for a two-level one, against its
    /// Assumption: it would bind none and drop them.
    pub fn new(
        words: PlainTensor,
        block_scales: PlainTensor,
        outer_scales: Option<PlainTensor>,
        scheme: QuantScheme,
        lined: Dimension,
        tiles: Option<StoredTiles>,
        dimensions: [(Dimension, usize); 2],
    ) -> Self {
        assert_eq!(
            outer_scales.is_some(),
            scheme.levels().count() > 1,
            "a two-level scheme carries a per-tensor scale and a one-level one does not"
        );
        Self {
            words,
            block_scales,
            outer_scales,
            scheme,
            lined,
            tiles,
            dimensions,
            stack: None,
            window: None,
        }
    }

    /// A quantized matrix over `dimensions`, in the order its shape states them, as a framework
    /// stores one: `values`, its buffer bound under its logical shape and strides, its values
    /// packed as `scheme` stores them, and its block scales and its per-tensor scale, where
    /// `scheme` has one, each bound where it lies. The words and the block scales are read as their
    /// buffers lie ([`PlainTensor::in_buffer_order`]): a transposed view of a packed matrix, a tied
    /// embedding read as a head, is the matrix it views, its rows along the dimension its words do
    /// not run along. The scales are laid in rows and bound on whole words, as [`new`](Self::new)
    /// takes them. Values laid in tile order are read in tile order
    /// (`of_tile_order`).
    ///
    /// # Errors
    ///
    /// [`SCALES_NOT_ALONG_WORDS`], and [`PACKED_NOT_ONE_MATRIX_IN_ROWS`]; for values in tile order,
    /// [`STORED_TILES_UNREAD`].
    ///
    /// # Panics
    ///
    /// A scheme storing sub-byte values natively, or natively packing other values than `E2M1`:
    /// no framework stores one.
    pub fn of_quantized(
        client: &Client,
        values: TensorBinding,
        scheme: QuantScheme,
        block_scales: PlainTensor,
        outer_scales: Option<PlainTensor>,
        dimensions: [Dimension; 2],
    ) -> Result<Self, Decline> {
        let lined = Operand::dimension_of(&scheme, dimensions).ok_or(SCALES_NOT_ALONG_WORDS)?;
        let operand =
            Operand::of_scheme(&scheme, lined, dimensions).ok_or(SCALES_NOT_ALONG_WORDS)?;
        if values.tiling.is_tiled() {
            let words = PlainTensor::new(values, WORD);
            return Self::of_tile_order(
                client,
                words,
                block_scales,
                outer_scales,
                scheme,
                operand,
                dimensions,
            );
        }
        let shape = values.shape.clone();
        if shape.rank() != 2 {
            return Err(PACKED_NOT_ONE_MATRIX_IN_ROWS);
        }
        let words = Self::stored_words(values, &scheme).in_buffer_order();
        if !words.is_contiguous() {
            return Err(PACKED_NOT_ONE_MATRIX_IN_ROWS);
        }
        let in_rows_of_words =
            |scale: PlainTensor| Self::bound_to_whole_words(scale.into_contiguous(client));
        Ok(Self::new(
            words,
            in_rows_of_words(block_scales.in_buffer_order()),
            outer_scales.map(in_rows_of_words),
            scheme,
            lined,
            None,
            [(dimensions[0], shape[0]), (dimensions[1], shape[1])],
        ))
    }

    /// This matrix read as `slabs` matrices laid end to end along `along`, each its extent there
    /// divided by `slabs`: a stacked input one slab of which a table's entry picks
    /// ([`MatmulIndex::Slab`](crate::kernels::MatmulIndex::Slab)).
    ///
    /// # Errors
    ///
    /// [`PACKED_STACK_UNREAD`]: slabs that split a read, a scale's block, or a word of block
    /// scales; and for words stored in tiles, slabs along `K`, or that split a strip of tiles.
    pub fn stacked(self, slabs: usize, along: Dimension) -> Result<Self, Decline> {
        let extent = self.extent(along);
        if !slabs_are_whole(&self.operand(), along, extent, slabs) {
            return Err(PACKED_STACK_UNREAD);
        }
        let slab = extent / slabs;
        let dimensions = self
            .dimensions
            .map(|(dimension, extent)| (dimension, if dimension == along { slab } else { extent }));
        Ok(Self {
            dimensions,
            stack: Some(PackedStack::Slabs { slabs, along }),
            ..self
        })
    }

    /// Where the words of a matrix `sized` so, read as `operand` in rows, lie in the tile order
    /// `matrix` states ([`TileOrder::of_operand`]), never one the reader works out again, and what
    /// one load of `hw` takes of them: read its values a load at a time, and where the order keeps
    /// them in tiles of their own ([`TileWords::BlockByBlock`]), its scales a read at a time
    /// beside them.
    ///
    /// The tile order stores a matrix over `K` and the dimension it lies across
    /// ([`values_dimensions`](tile_order::values_dimensions)): a weight states it `[k, n]`, an
    /// activation `[m, k]`, which is read swapped ([`swapped`](tile_order::swapped)).
    ///
    /// # Errors
    ///
    /// [`STORED_TILES_UNREAD`]: a matrix not over `K` and one other dimension, values in rows or at
    /// another block than this matrix's, or in tiles no read takes whole.
    pub fn read_in_tiles(
        hw: &Hardware,
        sized: [(Dimension, usize); 2],
        operand: Operand,
        matrix: &Metadata,
    ) -> Result<StoredTiles, Decline> {
        // Tile order reads a matrix stated `[k, rows]`; an `[m, k]` input states it swapped.
        let (k, across, matrix) = match sized {
            [(K, k), (across, _)] => (k, across, matrix.clone()),
            [(across, _), (K, k)] => (k, across, tile_order::swapped(matrix)),
            _ => return Err(STORED_TILES_UNREAD),
        };
        let block = operand.grouping(K, k).ok_or(STORED_TILES_UNREAD)?;
        let order = TileOrder::of_operand(&operand, &matrix).ok_or(STORED_TILES_UNREAD)?;
        if order.block() != block {
            return Err(STORED_TILES_UNREAD);
        }
        let words = hw.load_values(WORD);
        operand
            .in_tile_order(order, across, words)
            .ok_or(STORED_TILES_UNREAD)?;
        Ok(StoredTiles {
            order,
            across,
            words,
        })
    }

    /// This matrix read as `batches` matrices laid end to end along its rows, each its rows divided
    /// by `batches`, its rows those of the dimension its words do not run along: a leading dim over
    /// `dimension` the launch walks, as every head of a cache is one matrix of its positions.
    /// [`stacked`](Self::stacked)'s split, walked rather than picked by a table, so the words and
    /// scales stay bound over the whole matrix. Read before a [`window`](Self::window) of it.
    ///
    /// # Errors
    ///
    /// [`PACKED_BATCH_UNREAD`]: words stored in tiles, a two-level scheme, a stack or a window
    /// already, or rows `batches` does not divide.
    pub fn batched(self, batches: usize, dimension: Dimension) -> Result<Self, Decline> {
        let rows_dimension = self.rows_dimension();
        let rows = self.extent(rows_dimension);
        let unread = self.tiles.is_some()
            || self.outer_scales.is_some()
            || self.stack.is_some()
            || self.window.is_some()
            || batches == 0
            || !rows.is_multiple_of(batches);
        if unread {
            return Err(PACKED_BATCH_UNREAD);
        }
        let pitch = rows / batches;
        let dimensions = self
            .dimensions
            .map(|(own, extent)| (own, if own == rows_dimension { pitch } else { extent }));
        Ok(Self {
            dimensions,
            stack: Some(PackedStack::Batch {
                batches,
                pitch,
                dimension,
            }),
            ..self
        })
    }

    /// This matrix's rows `at..at + count` read where they lie, of the dimension its words do not
    /// run along: a matrix of `count` rows whose launch reads `at` as a dynamic offset, so one
    /// compiled launch reads every run of `count`, as a cache's write and read land at any
    /// position. Every matrix of a batch is read at the same rows. Its block scales are read as
    /// fields ([`BlockScales::Fields`]).
    ///
    /// # Errors
    ///
    /// [`PACKED_BATCH_UNREAD`]: words stored in tiles, a two-level scheme, slabs or a window
    /// already, or rows past the matrix's.
    pub fn window(self, at: usize, count: usize) -> Result<Self, Decline> {
        let rows_dimension = self.rows_dimension();
        let stored = self.extent(rows_dimension);
        let unread = self.tiles.is_some()
            || self.outer_scales.is_some()
            || matches!(self.stack, Some(PackedStack::Slabs { .. }))
            || self.window.is_some()
            || at + count > stored
            || i32::try_from(at).is_err();
        if unread {
            return Err(PACKED_BATCH_UNREAD);
        }
        let dimensions = self.dimensions.map(|(dimension, extent)| {
            (
                dimension,
                if dimension == rows_dimension {
                    count
                } else {
                    extent
                },
            )
        });
        Ok(Self {
            dimensions,
            window: Some(RowWindow { at, stored }),
            ..self
        })
    }

    /// The matrices it holds where it is read as a batch ([`batched`](Self::batched)).
    pub fn batches(&self) -> Option<usize> {
        match self.stack {
            Some(PackedStack::Batch { batches, .. }) => Some(batches),
            Some(PackedStack::Slabs { .. }) | None => None,
        }
    }

    /// The slabs it holds where it is read as a stack ([`stacked`](Self::stacked)).
    pub fn slabs(&self) -> Option<usize> {
        match self.stack {
            Some(PackedStack::Slabs { slabs, .. }) => Some(slabs),
            Some(PackedStack::Batch { .. }) | None => None,
        }
    }

    /// The tensor holding its words.
    pub fn words(&self) -> &PlainTensor {
        &self.words
    }

    /// The tensors holding its scales: the block scales, then the per-tensor scale where its
    /// scheme has one.
    pub fn scales(&self) -> impl Iterator<Item = &PlainTensor> {
        core::iter::once(&self.block_scales).chain(&self.outer_scales)
    }

    /// How its values are packed and scaled.
    pub fn scheme(&self) -> QuantScheme {
        self.scheme
    }

    /// What one read of it brings and what one of its scales covers, over its dimensions: what its
    /// scheme reads along the dimension its words run along, in the tile order its words lie in.
    pub fn operand(&self) -> Operand {
        let rows = Operand::of_scheme(&self.scheme, self.lined, self.dimensions())
            .expect("a packed tensor's scheme should read its words along `lined`, as it assumes");
        match self.tiles {
            Some(tiles) => rows
                .in_tile_order(tiles.order, tiles.across, tiles.words)
                .expect(
                    "a packed tensor's words should be read in their tile order, as it assumes",
                ),
            None => rows,
        }
    }

    /// How it is packed when read as `operand` states, a launch's planned reads of the same bytes
    /// ([`MatmulMicrokernel::planned_space`](cubek_microkernel::matmul::MatmulMicrokernel::planned_space)).
    pub fn read_as(&self, operand: Operand) -> Packing {
        Packing {
            operand,
            scheme: self.scheme,
        }
    }

    /// How it is packed: its reads and scales, and its scheme.
    pub fn packing(&self) -> Packing {
        Packing {
            operand: self.operand(),
            scheme: self.scheme,
        }
    }

    /// The matrix's two dimensions, in the order its caller stated them.
    pub fn dimensions(&self) -> [Dimension; 2] {
        self.dimensions.map(|(dimension, _)| dimension)
    }

    /// This matrix with its dimension `from` named `to`: the same bytes, read by a kernel that
    /// names the matrix's dimensions otherwise.
    pub fn renamed(self, from: Dimension, to: Dimension) -> Self {
        let rename = |dimension: Dimension| if dimension == from { to } else { dimension };
        Self {
            lined: rename(self.lined),
            tiles: self.tiles.map(|tiles| StoredTiles {
                across: rename(tiles.across),
                ..tiles
            }),
            dimensions: self
                .dimensions
                .map(|(dimension, extent)| (rename(dimension), extent)),
            stack: self.stack.map(|stack| match stack {
                PackedStack::Slabs { slabs, along } => PackedStack::Slabs {
                    slabs,
                    along: rename(along),
                },
                PackedStack::Batch {
                    batches,
                    pitch,
                    dimension,
                } => PackedStack::Batch {
                    batches,
                    pitch,
                    dimension: rename(dimension),
                },
            }),
            ..self
        }
    }

    /// The tile its words are stored in, which a descriptor lands a stage of whole tiles of row
    /// by row: its rows by its values of `K`, each a row's run, laid row by row. None in rows,
    /// and none for tiles laid block by block, whose tile rows are no row's run: a routine that
    /// lands tiles declines them rather than read them as rows.
    pub fn stored_tile(&self) -> Option<MatmulStoredTile> {
        let order = self.tiles?.order;
        let (columns, rows) = order.tile();
        (order.words() == TileWords::RowByRow).then_some(MatmulStoredTile { rows, columns })
    }

    /// What `dimension` of the matrix measures.
    ///
    /// # Assumptions
    ///
    /// `dimension` is one of the matrix's two.
    ///
    /// # Panics
    ///
    /// On a dimension the matrix does not have, against its Assumption.
    pub fn extent(&self, dimension: Dimension) -> usize {
        self.dimensions
            .iter()
            .find(|&&(own, _)| own == dimension)
            .map(|&(_, extent)| extent)
            .unwrap_or_else(|| panic!("{dimension:?} is not one of the packed matrix's dimensions"))
    }

    /// The dims [`restated`](Self::restated) states dense rows of `rows` by `columns`, each with
    /// its extent, `step` apart, in: one for each dimension of the block-split space, `(dimension,
    /// extent, stride)`, coarsest first. A dim's blocks are left out where there is one.
    pub fn restated_dims(
        &self,
        rows: (Dimension, usize),
        columns: (Dimension, usize),
        step: usize,
    ) -> Vec<(Dimension, usize, usize)> {
        let mut dims = Vec::new();
        for ((dimension, extent), dimension_step) in [(rows, step), (columns, 1)] {
            let [(blocks, block_count), (inside, inside_count)] =
                self.operand().dimensions(dimension, extent);
            if block_count > 1 {
                dims.push((blocks, block_count, inside_count * dimension_step));
            }
            dims.push((inside, inside_count, dimension_step));
        }
        dims
    }

    /// `tensor`, dense rows of `rows` by `columns`, each with its extent, restated over the
    /// dimensions each is in the block-split space, and those dimensions: the same bytes, a dim a
    /// dimension, which the tile engine stages and masks as it lies. A dim's blocks are left out
    /// where there is one, the tensor constant along them; its inside always stays, so the tensor
    /// is a matrix.
    pub fn restated(
        &self,
        tensor: &PlainTensor,
        rows: (Dimension, usize),
        columns: (Dimension, usize),
    ) -> (PlainTensor, Vec<Dimension>) {
        let dense = tensor.as_rows(rows.1, columns.1);
        let dims = self.restated_dims(rows, columns, dense.binding.strides[0]);
        let meta = Metadata::new(
            dims.iter()
                .map(|&(_, extent, _)| extent)
                .collect::<Vec<_>>(),
            dims.iter()
                .map(|&(_, _, stride)| stride)
                .collect::<Vec<_>>(),
        );
        (
            dense.viewed(meta),
            dims.iter().map(|&(dimension, _, _)| dimension).collect(),
        )
    }

    /// The matrix as one tile under `launcher`, its words `width` values wide and moved into a
    /// stage as `delivery` says: the words under their block scales, read as `scales` says, under
    /// the per-tensor scale where the scheme has one, multiplied in as `outer` says.
    ///
    /// Every level binds as the words it lies in and is served every scale a word holds: the unit
    /// that reads a word of scales holds the blocks of every field in it.
    ///
    /// # Assumptions
    ///
    /// - `launcher` was built for the problem of the product this matrix is an operand of.
    /// - `width` is a multiple of the values a word of its scheme packs.
    /// - Its scales are blocks along one of its dimensions: its scheme is block-scaled.
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] where the tile engine refuses what the routine prepared, and on block
    /// scales read as words that lie in tiles or under a window of rows.
    ///
    /// # Panics
    ///
    /// Where its scheme is not block-scaled, an Assumption above.
    pub fn bind(
        &self,
        launcher: &Launcher,
        operand: Operand,
        width: usize,
        scales: BlockScales,
        outer: OuterScale,
        delivery: Delivery,
    ) -> Result<TileTensorLaunch, Decline> {
        let lined = self.lined;
        // The buffer steps its rows before its words, so its dims are named that way.
        let rows_along = self.other_dimension(lined);
        // Bound over every row it stores; a window reads its run of them.
        let rows = self.stored_rows();
        let dimensions = self
            .stack_dim()
            .axis(self.rows_dim(rows_along))
            .axis(self.dim(lined))
            .build();
        let across = self.other_dimension(K);
        let values_dimensions = tile_order::values_dimensions(across);
        let values_bound: Vec<Dimension> = self.stacked_dimensions(&values_dimensions).collect();
        let scales_dimensions = tile_order::scales_dimensions(across);
        let scales_bound: Vec<Dimension> = self.stacked_dimensions(&scales_dimensions).collect();
        // The dimensions the bound words carry: every split dimension of both dimensions in rows,
        // the stored dims' in tiles.
        let (words_dimensions, words): (Vec<Dimension>, _) = match operand.tile_order() {
            None => (
                [rows_along, lined]
                    .into_iter()
                    .flat_map(|dimension| <[Dimension; 2]>::from(blocked(dimension)))
                    .collect(),
                // Strides in stored words, each the scheme's values: a read may take several
                // ([`Operand::by_block`]).
                launcher
                    .arg(
                        self.per_slab(self.words.rows_unpacked(
                            rows,
                            self.extent(lined),
                            self.scheme.num_quants(),
                        )),
                        WORD,
                    )
                    .gathered(dimensions),
            ),
            // Stated for the read the launch takes, in the tile the matrix is stored in: the
            // same bytes at every read.
            Some(order) => (
                values_dimensions.to_vec(),
                launcher
                    .arg(
                        self.words.bound_as(
                            self.values_in_tiles(order, width / self.scheme.num_quants()),
                        ),
                        WORD,
                    )
                    .dimensions(&values_bound),
            ),
        };
        let words = words.vectorize(width).delivery(delivery);
        // The scales count the blocks of the dimension they are constant inside.
        let tiled_along = operand
            .scaled_along()
            .expect("a block-scaled matrix groups one of its dimensions");
        let (blocks_dimension, inside) = blocked(tiled_along);
        let blocks = self.extent(tiled_along) / launcher.space().extent(inside);
        // A scale is looked up at its value's coordinates, so it names only the dimensions the
        // values carry: values in tiles carry the position inside the rows' dimension, not its
        // blocks, which no scale groups.
        let scale_rows = match operand.tile_order() {
            None => self.rows_dim(rows_along),
            Some(_) => PhysicalAxisMap::of(blocked(rows_along).1),
        };
        let outer_scales = self.outer_scales(launcher, &words_dimensions, tiled_along)?;
        match scales {
            // A read taking its one scale alone ([`Packing::by_block`]): each bound as the element
            // it is, a `ue4m3` the `e4m3` byte with its sign clear, so a unit loads the scale it
            // uses rather than the word around it.
            BlockScales::Fields if self.read_as(operand).reads_scales_alone() => {
                let block_scales = self.scales_alone(
                    launcher,
                    rows,
                    blocks,
                    scale_rows,
                    blocks_dimension,
                    &scales_bound,
                );
                self.shifted_words(words).and_then(|words| {
                    Ok(Self::under_scales(
                        words,
                        self.shifted(block_scales.build()?.arg()),
                        outer_scales,
                        outer,
                    ))
                })
            }
            BlockScales::Fields => {
                let block_scales = self.scale_fields(
                    &operand,
                    launcher,
                    rows,
                    blocks,
                    scale_rows,
                    blocks_dimension,
                    &scales_bound,
                );
                self.shifted_words(words).and_then(|words| {
                    let field = self.block_scale_field();
                    let block_scales = self.shifted(block_scales.packed(field).build()?.arg());
                    Ok(Self::under_scales(words, block_scales, outer_scales, outer))
                })
            }
            // A window of rows reads its block scales as fields, and a launch taking its scales as
            // words declines scales in tiles before it binds (`MATMUL_SCALE_WORDS_IN_TILES`):
            // their rows are no tile's.
            BlockScales::Words
                if self.window.is_some()
                    || operand.tile_order().is_some_and(TileOrder::scales_in_tiles) =>
            {
                Breakable::broken(SCALE_WORDS_OF_WHOLE_ROWS)
            }
            BlockScales::Words => self
                .scale_words(
                    &operand,
                    launcher,
                    scale_rows,
                    blocks_dimension,
                    blocks,
                    None,
                )
                .and_then(|block_scales| {
                    let words = words.packed(self.scheme.value).build()?.arg();
                    Ok(Self::under_scales(words, block_scales, outer_scales, outer))
                }),
        }
        .or_broken(ADMITTED_PLAN_RUNS)
    }

    /// The matrix as one tile under `launcher` as [`bind`](Self::bind) binds it under `scales`,
    /// served `width` values wide, its words moved by the TMA engine through a descriptor over
    /// the rows or the tiles ([`TileOrder::words_in_tiles`]) they are stored in: a box of
    /// `stage_rows` rows by `stage_k` values of `k`, swizzled as `swizzle` says. Its block scales
    /// are kept beside the words. As [`BlockScales::Words`], a word of a step's scales one load
    /// straight into the instruction's register, those in tiles of their own are landed by the
    /// engine too, a box of the stage's rows one tile deep
    /// ([`TileOrder::scale_words_in_tiles`]); those in rows are handed to the copy engine, a
    /// stage's row of them a few bytes, short of the 16 a descriptor's row lands, so the units that
    /// fill issue them beside the descriptors. As [`BlockScales::Fields`], each scale is read and
    /// narrowed on its own, by the copy engine.
    ///
    /// # Assumptions
    ///
    /// As [`bind`](Self::bind).
    ///
    /// # Errors
    ///
    /// As [`bind`](Self::bind), and [`Decline::Broken`] on words in tiles laid block by block,
    /// whose tile rows are no row's run; on a stage not whole tiles along its rows, nor whole
    /// tiles or a part dividing one along `k`; on a stack; on words in rows not one dense run a row;
    /// on scales in tiles and a stage not whole strips of a tile's rows.
    ///
    /// # Panics
    ///
    /// As [`bind`](Self::bind).
    pub fn bind_by_tma(
        &self,
        launcher: &Launcher,
        operand: Operand,
        width: usize,
        stage_rows: usize,
        stage_k: usize,
        swizzle: TensorMapSwizzle,
        scales: BlockScales,
    ) -> Result<TileTensorLaunch, Decline> {
        // The engine lands the words as they lie, so a stage keeps one scale beside them.
        let outer = OuterScale::Factored;
        let across = self.other_dimension(K);
        let values_dimensions = tile_order::values_dimensions(across);
        let words = match self.tiles.map(|tiles| tiles.order) {
            None => self.words_by_tma_in_rows(stage_rows, stage_k, swizzle, &values_dimensions)?,
            Some(order) if order.words() == TileWords::RowByRow => {
                self.words_by_tma_in_tiles(order, stage_rows, stage_k, swizzle, &values_dimensions)?
            }
            Some(_) => return Breakable::broken(TMA_FILLS_FROM_ROWS),
        }
        .served_wide(width / self.scheme.num_quants())
        .packed(Field::from(self.scheme.value).read_on(launcher.client()));
        let lined = self.lined;
        let rows_along = self.other_dimension(lined);
        let rows = self.extent(rows_along);
        let words_dimensions = values_dimensions.to_vec();
        let tiled_along = operand
            .scaled_along()
            .expect("a block-scaled matrix groups one of its dimensions");
        let (blocks_dimension, inside) = blocked(tiled_along);
        let blocks = self.extent(tiled_along) / launcher.space().extent(inside);
        let scale_rows = PhysicalAxisMap::of(blocked(rows_along).1);
        let scales_dimensions = tile_order::scales_dimensions(across);
        let outer_scales = self.outer_scales(launcher, &words_dimensions, tiled_along)?;
        match scales {
            // Each scale read alone, as `bind` binds it, beside the words the engine lands.
            BlockScales::Fields if self.read_as(operand).reads_scales_alone() => {
                let block_scales = self
                    .scales_alone(
                        launcher,
                        rows,
                        blocks,
                        scale_rows,
                        blocks_dimension,
                        &scales_dimensions,
                    )
                    .delivery(Delivery::AsyncPerUnit)
                    .build()
                    .or_broken(ADMITTED_PLAN_RUNS)?
                    .arg();
                Ok(Self::under_scales(words, block_scales, outer_scales, outer))
            }
            BlockScales::Fields => {
                let field = self.block_scale_field();
                let block_scales = self
                    .scale_fields(
                        &operand,
                        launcher,
                        rows,
                        blocks,
                        scale_rows,
                        blocks_dimension,
                        &scales_dimensions,
                    )
                    .delivery(Delivery::AsyncPerUnit)
                    .packed(field)
                    .build()
                    .or_broken(ADMITTED_PLAN_RUNS)?
                    .arg();
                Ok(Self::under_scales(words, block_scales, outer_scales, outer))
            }
            // Words of a step's scales, each one load straight into the instruction's register. In
            // tiles of their own, landed by the engine beside the words it lands, on their
            // barrier: a tile's scales are one run, a box's worth.
            BlockScales::Words if operand.tile_order().is_some_and(TileOrder::scales_in_tiles) => {
                let order = operand
                    .tile_order()
                    .expect("scales in tiles are a tile order's");
                let block_scales = self.scale_words_by_tma_in_tiles(
                    &operand,
                    order,
                    stage_rows,
                    [blocked(rows_along).1, blocks_dimension],
                    blocks,
                )?;
                Ok(Self::under_scales(words, block_scales, outer_scales, outer))
            }
            // In rows, handed to the copy engine as the words they are stored in: a stage's row of
            // them is a few bytes, short of the 16 a descriptor's row lands.
            BlockScales::Words => self
                .scale_words(
                    &operand,
                    launcher,
                    scale_rows,
                    blocks_dimension,
                    blocks,
                    Some(Delivery::AsyncPerUnit),
                )
                .map(|block_scales| Self::under_scales(words, block_scales, outer_scales, outer))
                .or_broken(ADMITTED_PLAN_RUNS),
        }
    }
}

impl PackedTensor {
    /// A quantized matrix whose `words` are laid in tile order. Values are read in the order
    /// stated by the bindings ([`read_in_tiles`](Self::read_in_tiles)): its words as they are
    /// stored, and its block scales where it keeps them. The per-tensor scale is bound as
    /// [`of_quantized`](Self::of_quantized) binds it.
    ///
    /// # Errors
    ///
    /// [`STORED_TILES_UNREAD`]: anything but one matrix, or tiles no read takes whole.
    fn of_tile_order(
        client: &Client,
        words: PlainTensor,
        block_scales: PlainTensor,
        outer_scales: Option<PlainTensor>,
        scheme: QuantScheme,
        operand: Operand,
        dimensions: [Dimension; 2],
    ) -> Result<Self, Decline> {
        let shape = words.shape();
        if shape.rank() != 2 {
            return Err(STORED_TILES_UNREAD);
        }
        let sized = [(dimensions[0], shape[0]), (dimensions[1], shape[1])];
        let hw = Hardware::from_client(client);
        let tiles = Self::read_in_tiles(&hw, sized, operand, &words.meta())?;
        let in_rows_of_words =
            |scale: PlainTensor| Self::bound_to_whole_words(scale.into_contiguous(client));
        // Where the scales lie is the order's own answer, whichever way its words lie.
        let block_scales = match tiles.order.scales_in_tiles() {
            true => Self::bound_to_whole_words(block_scales),
            false => in_rows_of_words(block_scales.in_buffer_order()),
        };
        Ok(Self::new(
            words,
            block_scales,
            outer_scales.map(in_rows_of_words),
            scheme,
            operand.lined(),
            Some(tiles),
            sized,
        ))
    }

    /// The words stored in `order`'s tiles laid row by row, through a descriptor over the tiles
    /// ([`TileOrder::words_in_tiles`]): a box of `stage_rows` rows by `stage_k` values of `k`,
    /// landing row by row, swizzled as `swizzle` says, over `values_dimensions`.
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] on a stack, or a stage not whole tiles along its rows, nor whole tiles or
    /// a part dividing one along `k`.
    fn words_by_tma_in_tiles(
        &self,
        order: TileOrder,
        stage_rows: usize,
        stage_k: usize,
        swizzle: TensorMapSwizzle,
        values_dimensions: &[Dimension; 3],
    ) -> Result<TileTensorLaunch, Decline> {
        if self.stack.is_some() {
            return Breakable::broken(TMA_FILLS_FROM_ONE_SLAB);
        }
        let (tile_k, tile_rows) = order.tile();
        let across = self.other_dimension(K);
        let (k, rows_across) = (self.extent(K), self.extent(across));
        if !(stage_k.is_multiple_of(tile_k) || tile_k.is_multiple_of(stage_k))
            || !stage_rows.is_multiple_of(tile_rows)
        {
            return Breakable::broken(TMA_STAGE_OF_WHOLE_TILES);
        }
        let words = self.words.bound_as(order.words_in_tiles(k, rows_across));
        let map = tma_operand_tiled_typed(
            words,
            (stage_rows, stage_k / order.values_per_word()),
            TmaLanding::Rows,
            WORD,
            swizzle,
        );
        // The descriptor after a row landing: the strips of rows, a row in the strip, the
        // tiles along `k`, a word in the tile; a box starts on whole tiles both ways.
        let coordinates = TmaCoordinates {
            axes: vec![
                vec![TmaTerm::dimension(0).over(tile_rows as u32)],
                vec![],
                vec![
                    TmaTerm::dimension(1)
                        .times(order.block() as u32)
                        .over(tile_k as u32),
                ],
                vec![
                    TmaTerm::dimension(1)
                        .times(order.block() as u32)
                        .modulo(tile_k as u32)
                        .over(order.values_per_word() as u32),
                ],
            ],
        };
        Ok(TileTensorLaunch::tensor_map_folded(
            map,
            values_dimensions,
            &[
                rows_across as u32,
                (k / order.block()) as u32,
                order.block() as u32,
            ],
            coordinates,
        ))
    }

    /// The block scales laid row by row in `order`'s tiles, as the words a block-scaled instruction
    /// takes, through a descriptor over the tiles ([`tma_scale_words_in_tiles`]): a box of
    /// `stage_rows` rows one tile deep, landing a row's words of the tile one run, the rows one
    /// after another, as a stage of scales holds them. Over `scale_dimensions`, the rows and `KB`,
    /// the latter `blocks` long.
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] on a stack, or a stage that is not whole strips of the tile's rows.
    fn scale_words_by_tma_in_tiles(
        &self,
        operand: &Operand,
        order: TileOrder,
        stage_rows: usize,
        scale_dimensions: [Dimension; 2],
        blocks: usize,
    ) -> Result<TileTensorLaunch, Decline> {
        if self.stack.is_some() {
            return Breakable::broken(TMA_FILLS_FROM_ONE_SLAB);
        }
        let (tile_k, tile_rows) = order.tile();
        if !stage_rows.is_multiple_of(tile_rows) {
            return Breakable::broken(TMA_STAGE_OF_WHOLE_STRIPS);
        }
        let across = self.other_dimension(K);
        let (k, rows_across) = (self.extent(K), self.extent(across));
        let scales = self
            .block_scales
            .bound_as(order.scale_words_in_tiles(k, rows_across));
        let map = tma_scale_words_in_tiles(scales, stage_rows / tile_rows);
        // The descriptor's dims: the strips of rows, the tiles along `k`, a tile's words; a box
        // starts on a whole strip and a whole tile, and takes the tile's words whole.
        let coordinates = TmaCoordinates {
            axes: vec![
                vec![TmaTerm::dimension(0).over(tile_rows as u32)],
                vec![TmaTerm::dimension(1).over((tile_k / order.block()) as u32)],
                vec![],
            ],
        };
        // Read as the units' gather reads them, a word every few blocks, so a stage of them is
        // the compacted window of words the box lands.
        let per_word = operand.scales_per_word();
        let projection = Projection::new(
            &scale_dimensions,
            &[
                PhysicalAxisMap::of(scale_dimensions[0]),
                PhysicalAxisMap::of(scale_dimensions[1]).over(per_word),
            ],
        );
        Ok(TileTensorLaunch::tensor_map_with_coordinates_through(
            map,
            projection,
            &[rows_across as u32, blocks as u32],
            coordinates,
        )
        .words(self.block_scale_field()))
    }

    /// The words stored in rows, through a descriptor over the rows: `k`'s words cut into rows of
    /// the swizzle's span, so a box of `stage_rows` rows by `stage_k` values of `k` lands a
    /// stage's blocks of `K` side by side, as a float operand's descriptor lands them, over
    /// `values_dimensions`.
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] on a stack, or on rows whose words are not one dense run each: a
    /// descriptor over the rows steps them by the run.
    fn words_by_tma_in_rows(
        &self,
        stage_rows: usize,
        stage_k: usize,
        swizzle: TensorMapSwizzle,
        values_dimensions: &[Dimension; 3],
    ) -> Result<TileTensorLaunch, Decline> {
        if self.stack.is_some() {
            return Breakable::broken(TMA_FILLS_FROM_ONE_SLAB);
        }
        let across = self.other_dimension(K);
        let (k, rows) = (self.extent(K), self.extent(across));
        let per_word = self.scheme.num_quants();
        let block = self
            .operand()
            .scale(K)
            .expect("a block-scaled matrix groups k");
        let words_k = k / per_word;
        if self.words.row_step().unwrap_or(words_k) != words_k {
            return Breakable::broken(TMA_ROWS_DENSE);
        }
        let stage_words = stage_k / per_word;
        // A box row is the swizzle's span, or the stage's depth where it swizzles nothing.
        let row = swizzle_span_bytes(swizzle)
            .map_or(stage_words, |span| span / size_of::<u32>())
            .min(stage_words);
        let metadata = Metadata::new(
            Shape::new([rows, words_k / row, row]),
            Strides::new(&[words_k, row, 1]),
        );
        let map = TensorMapArg {
            tensor: self.words.rows(rows, words_k).into_tensor_arg(),
            metadata: tma_meta_tiled(
                metadata,
                Shape::new([stage_rows, stage_words / row, row]),
                WORD,
                swizzle,
            ),
            _kind: PhantomData,
        };
        // A box starts on a stage's first block of `K`, a whole number of rows in.
        let coordinates = TmaCoordinates {
            axes: vec![
                vec![TmaTerm::dimension(0)],
                vec![
                    TmaTerm::dimension(1)
                        .times((block / per_word) as u32)
                        .over(row as u32),
                ],
                vec![],
            ],
        };
        Ok(TileTensorLaunch::tensor_map_folded(
            map,
            values_dimensions,
            &[rows as u32, (k / block) as u32, block as u32],
            coordinates,
        ))
    }

    /// `words` under `block_scales`, and under `outer_scales` where the scheme has a second level,
    /// grouped as `outer` says where those are multiplied in.
    fn under_scales(
        words: TileTensorLaunch,
        block_scales: TileTensorLaunch,
        outer_scales: Option<TileTensorLaunch>,
        outer: OuterScale,
    ) -> TileTensorLaunch {
        match (outer, outer_scales) {
            (OuterScale::Factored, Some(scales)) => words.scales(block_scales).scales(scales),
            (OuterScale::Kept, Some(scales)) => words
                .scales(block_scales.scales(scales))
                .scales(TileTensorLaunch::one()),
            (OuterScale::Factored | OuterScale::Kept, None) => {
                words.scales(block_scales).scales(TileTensorLaunch::one())
            }
        }
    }

    /// Its outer scales under `launcher`, where its scheme has some, over the dimensions the bound
    /// words carry: one for each of its rows, gathered over the rows, or one over all of it.
    fn outer_scales(
        &self,
        launcher: &Launcher,
        words_dimensions: &[Dimension],
        tiled_along: Dimension,
    ) -> Result<Option<TileTensorLaunch>, Decline> {
        let lined = self
            .dimensions()
            .iter()
            .position(|&dimension| dimension == self.lined);
        let (Some(level), Some(scales), Some(lined)) = (
            self.scheme.levels().nth(1),
            self.outer_scales.as_ref(),
            lined,
        ) else {
            return Ok(None);
        };
        let Some(along) = Operand::constant_along(&level, lined) else {
            return Ok(None);
        };
        let (binding, projection) = match along {
            None => (
                scales.bound_as(Metadata::new(ONE_SCALE, ONE_SCALE)),
                Self::over_whole_matrix(words_dimensions, tiled_along),
            ),
            Some(_) => self.outer_scales_of_rows(scales),
        };
        let outer = launcher
            .arg(binding, ElemType::from_scale_dtype(level.dtype))
            .gathered(projection)
            .build()
            .or_broken(ADMITTED_PLAN_RUNS)?;
        Ok(Some(outer.arg()))
    }

    /// `scales`, one for each row, bound as a dim of them, once per slab where
    /// the slabs lie along the rows; slabs along the other dimension share a row's.
    fn outer_scales_of_rows(&self, scales: &PlainTensor) -> (TensorBinding, Projection) {
        // The rows are one block, the position inside it the whole of them.
        let row = PhysicalAxisMap::of(blocked(self.rows_dimension()).1);
        let binding = scales.bound_as(Metadata::new([self.stored_rows(), 1], ONE_SCALE));
        let shared =
            matches!(self.stack, Some(PackedStack::Slabs { along, .. }) if along == self.lined);
        let (binding, axes) = match shared {
            true => (binding, Projection::axes()),
            false => (self.per_slab(binding), self.stack_dim()),
        };
        (
            binding,
            axes.axis(row).axis(PhysicalAxisMap::broadcast()).build(),
        )
    }

    /// `words` packed as the scheme's values, read from a window's start where it has one
    /// ([`window`](Self::window)).
    fn shifted_words(&self, words: Arg<'_, Labelled>) -> Result<TileTensorLaunch, Decline> {
        Ok(self.shifted(words.packed(self.scheme.value).build()?.arg()))
    }

    /// `arg` read from a window's start where it has one ([`window`](Self::window)): the dynamic
    /// offset its projection shifts its rows by. No window holds a per-tensor scale.
    fn shifted(&self, arg: TileTensorLaunch) -> TileTensorLaunch {
        match self.runtime_offsets() {
            offsets if offsets.is_empty() => arg,
            offsets => arg.runtime_map(Vec::new(), offsets),
        }
    }

    /// Its block scales bound one to a read ([`Packing::by_block`]), each as the element it is, a
    /// `ue4m3` the `e4m3` byte with its sign clear, so a unit loads the scale it uses rather than
    /// the word around it: in the tiles tile order stores them in, where it does, stated a row a
    /// read, counted in scales; gathered over `scale_rows` and the blocks otherwise. What every
    /// engine that moves the words binds them as.
    fn scales_alone<'l>(
        &self,
        launcher: &'l Launcher,
        rows: usize,
        blocks: usize,
        scale_rows: PhysicalAxisMap,
        blocks_dimension: Dimension,
        scales_dimensions: &'l [Dimension],
    ) -> Arg<'l, Labelled> {
        let scale = ElemType::from_scale_dtype(self.block_scales().dtype);
        match self.scales_in_tiles(1) {
            None => launcher
                .arg(self.per_slab(self.block_scale_bytes(rows, blocks)), scale)
                .gathered(self.scales_gathered(scale_rows, blocks_dimension)),
            Some(stated) => launcher
                .arg(self.block_scales.bound_as(stated), scale)
                .dimensions(scales_dimensions),
        }
        .vectorize(1)
    }

    /// Its block scales bound a whole read at a time as the words they are stored in, the scales of
    /// every row a read of the words spans: in the tiles tile order stores them in, where it does,
    /// since read as rows only a tile's first strip of rows would find its own; gathered over
    /// `scale_rows` and the blocks otherwise.
    fn scale_fields<'l>(
        &self,
        operand: &Operand,
        launcher: &'l Launcher,
        rows: usize,
        blocks: usize,
        scale_rows: PhysicalAxisMap,
        blocks_dimension: Dimension,
        scales_dimensions: &'l [Dimension],
    ) -> Arg<'l, Labelled> {
        let rows_a_read = operand.scale_read(self.other_dimension(K));
        match self.scales_in_tiles(rows_a_read) {
            None => launcher
                .arg(self.per_slab(self.block_scales.rows(rows, blocks)), WORD)
                .gathered(self.scales_gathered(scale_rows, blocks_dimension)),
            Some(stated) => launcher
                .arg(self.block_scales.bound_as(stated), WORD)
                .dimensions(scales_dimensions),
        }
        .vectorize(operand.scale_read_values())
    }

    /// Where its block scales in rows are read at a value's coordinates, over `scale_rows` and the
    /// blocks.
    fn scales_gathered(
        &self,
        scale_rows: PhysicalAxisMap,
        blocks_dimension: Dimension,
    ) -> Projection {
        self.stack_dim()
            .axis(scale_rows)
            .axis(self.dim(self.lined))
            .build()
            .scales_per(blocks_dimension)
    }

    /// The block scales a byte apiece, `blocks` a row, each row stepped at its stride counted in
    /// scales: a buffer holding them in wider elements steps its rows in those.
    fn block_scale_bytes(&self, rows: usize, blocks: usize) -> TensorBinding {
        // Counted in scales: a row's run of stored elements, each so many scales.
        let scale = ElemType::from_scale_dtype(self.block_scales().dtype).size();
        let per_element = self.block_scales.elem.size();
        let step = self
            .block_scales
            .row_step()
            .map_or(blocks, |step| step * per_element / scale);
        self.block_scales
            .bound_as(Metadata::new([rows, blocks], [step, 1]))
    }

    /// The block scales as the `u32` words a block-scaled instruction takes, each row a run of
    /// whole words read through a gather over `scale_rows` and `KB`, a word every
    /// [`scales_per_word`](Operand::scales_per_word) blocks: a stage of them is compacted to the
    /// words its window holds. Moved into a stage as `delivery` says where it states one.
    fn scale_words(
        &self,
        operand: &Operand,
        launcher: &Launcher,
        scale_rows: PhysicalAxisMap,
        blocks_dimension: Dimension,
        blocks: usize,
        delivery: Option<Delivery>,
    ) -> Result<TileTensorLaunch, Decline> {
        let per_word = operand.scales_per_word();
        let rows = self.stored_rows();
        let arg = launcher
            .arg(
                self.per_slab(self.block_scale_words(rows, blocks / per_word)),
                WORD,
            )
            .gathered(
                self.stack_dim()
                    .axis(scale_rows)
                    .axis(PhysicalAxisMap::of(blocks_dimension).over(per_word))
                    .build(),
            );
        let arg = match delivery {
            Some(delivery) => arg.delivery(delivery),
            None => arg,
        };
        Ok(arg.words(self.block_scale_field()).build()?.arg())
    }

    /// The block scales as the `u32` words they are stored in, `words` a row: each row a run of
    /// whole words, stepped from row to row at its stride, counted in words.
    fn block_scale_words(&self, rows: usize, words: usize) -> TensorBinding {
        let pitch = self.block_scales.binding.strides[0] * self.block_scales.elem.size();
        self.block_scales
            .bound_as(Metadata::new([rows, words], [pitch / 4, 1]))
    }

    /// The innermost level of its scheme: the block scales laid beside its words, which its
    /// operand was read off.
    fn block_scales(&self) -> BlockScale {
        self.scheme
            .levels()
            .next()
            .expect("a packed tensor's scheme should state its block scales")
    }

    /// The field a word of block scales holds, the scheme's block-scale type.
    fn block_scale_field(&self) -> Field {
        Field::of_scale(self.block_scales().dtype)
    }

    /// The dimension the matrices it holds past its first are bound along: the slabs' [`EXPERT`],
    /// or a batch's own. `None` for one matrix.
    fn stack_dimension(&self) -> Option<Dimension> {
        match self.stack {
            Some(PackedStack::Slabs { .. }) => Some(EXPERT),
            Some(PackedStack::Batch { dimension, .. }) => Some(dimension),
            None => None,
        }
    }

    /// The dims a binding of words or scales starts with: the slabs, where this tensor is a stack.
    fn stack_dim(&self) -> AxesBuilder {
        match self.stack_dimension() {
            Some(dimension) => Projection::axes().axis(PhysicalAxisMap::of(dimension)),
            None => Projection::axes(),
        }
    }

    /// `dimensions`, the dimensions one matrix's words or scales in tiles are bound over, led by
    /// the [`stack_dimension`](Self::stack_dimension) where it holds several.
    fn stacked_dimensions(&self, dimensions: &[Dimension]) -> impl Iterator<Item = Dimension> {
        self.stack_dimension()
            .into_iter()
            .chain(dimensions.iter().copied())
    }

    /// Its values in `order`'s tiles as the kernel states them, `words_per_load` words a load, by
    /// slab where it is a stack.
    fn values_in_tiles(&self, order: TileOrder, words_per_load: usize) -> Metadata {
        let (k, across) = (self.extent(K), self.other_dimension(K));
        stated_by_slab(self.slabs(), self.extent(across), |rows| {
            Some(order.kernel_values_read(k, (across, rows), words_per_load))
        })
        .expect("tile order states the values of any matrix it stores")
    }

    /// Its block scales as the kernel states them, `rows_a_read` rows a read, by slab where it is a
    /// stack. `None` where they are not stored in tiles of their own.
    fn scales_in_tiles(&self, rows_a_read: usize) -> Option<Metadata> {
        let (k, across) = (self.extent(K), self.other_dimension(K));
        let order = self.tiles?.order;
        stated_by_slab(self.slabs(), self.extent(across), |rows| {
            order.kernel_scales_read(k, (across, rows), rows_a_read)
        })
    }

    /// `binding`, words or scales of one matrix over its two dimensions in the order the buffer
    /// steps them, rows then line, once per matrix where this tensor holds several: a leading dim,
    /// each slab one slab's run of the dim the stack lies along further, each matrix of a batch its
    /// pitch of rows. As it is otherwise.
    fn per_slab(&self, binding: TensorBinding) -> TensorBinding {
        let (count, step) = match self.stack {
            None => return binding,
            Some(PackedStack::Slabs { slabs, along }) => {
                let dim = usize::from(along == self.lined);
                (slabs, binding.shape[dim] * binding.strides[dim])
            }
            Some(PackedStack::Batch { batches, pitch, .. }) => {
                (batches, pitch * binding.strides[0])
            }
        };
        let lead = |first: usize, dims: &[usize]| [&[first][..], dims].concat();
        TensorBinding {
            shape: lead(count, &binding.shape).into(),
            strides: lead(step, &binding.strides).into(),
            ..binding
        }
    }

    /// The dimension the matrix's rows run along: the one its words do not.
    fn rows_dimension(&self) -> Dimension {
        self.other_dimension(self.lined)
    }

    /// The rows its words and scales are bound over: every row stored, past a window's.
    fn stored_rows(&self) -> usize {
        self.window
            .map_or(self.extent(self.rows_dimension()), |window| window.stored)
    }

    /// The rows' dim, as [`dim`](Self::dim), shifted by a window's dynamic offset where it has one.
    fn rows_dim(&self, rows_along: Dimension) -> PhysicalAxisMap {
        match self.window {
            Some(_) => self.dim(rows_along).shifted(Offset::Dynamic),
            None => self.dim(rows_along),
        }
    }

    /// The value of each dynamic offset its parts' projections hold: a window's start.
    fn runtime_offsets(&self) -> Vec<i32> {
        self.window
            .map(|window| vec![i32::try_from(window.at).expect("a window's start fits an i32")])
            .unwrap_or_default()
    }

    /// The matrix's dimension other than `dimension`.
    fn other_dimension(&self, dimension: Dimension) -> Dimension {
        match self.dimensions {
            [(first, _), (second, _)] if first == dimension => second,
            [(first, _), _] => first,
        }
    }

    /// `dimension` of the matrix as one dim over the two dimensions it is split into, the blocks
    /// and the position inside one.
    fn dim(&self, dimension: Dimension) -> PhysicalAxisMap {
        split(&self.operand().dimensions(dimension, self.extent(dimension)))
    }

    /// One value over the whole matrix: every dimension of `words_dimensions`, the dimensions the
    /// bound words carry, that its block scales span, none addressed. A factor is looked up at the
    /// values' coordinates, so it spans no dimension they do not carry.
    fn over_whole_matrix(words_dimensions: &[Dimension], tiled_along: Dimension) -> Projection {
        let inside = blocked(tiled_along).1;
        let dimensions: Vec<Dimension> = words_dimensions
            .iter()
            .copied()
            .filter(|&dimension| dimension != inside)
            .collect();
        Projection::new(
            &dimensions,
            &[PhysicalAxisMap::broadcast(), PhysicalAxisMap::broadcast()],
        )
    }

    /// The values of a quantized matrix bound under its logical shape, `values`, as the elements
    /// `scheme` stores them in: one per value, or, packed, whole words of them, the packed dim's
    /// extent counting words.
    fn stored_words(values: TensorBinding, scheme: &QuantScheme) -> PlainTensor {
        let packed = |values: TensorBinding, packed_dim: usize, elem: ElemType| {
            let mut values = values;
            let dim = values.shape.rank() - packed_dim - 1;
            values.shape[dim] = values.shape[dim].div_ceil(scheme.num_quants());
            PlainTensor::new(values, elem)
        };
        match (scheme.store, scheme.value) {
            (QuantStore::Native, QuantValue::Q8F | QuantValue::Q8S) => {
                PlainTensor::new(values, ElemType::Int(IntKind::I8))
            }
            (QuantStore::Native, QuantValue::E4M3 | QuantValue::E5M2) => {
                PlainTensor::new(values, ElemType::UInt(UIntKind::U8))
            }
            (QuantStore::PackedU32(packed_dim), _) => packed(values, packed_dim, WORD),
            (QuantStore::PackedNative(packed_dim), QuantValue::E2M1) => {
                packed(values, packed_dim, ElemType::UInt(UIntKind::U8))
            }
            (store, value) => panic!("no framework stores {value:?} as {store:?}"),
        }
    }

    /// `tensor`, its binding ending on the `u32` word that holds its last element, the gaps between
    /// its rows or its tiles included: the routines read scales a word at a time. A quantized
    /// buffer starts each region on a word, so the word's tail is allocated.
    fn bound_to_whole_words(mut tensor: PlainTensor) -> PlainTensor {
        let (shape, strides) = (&tensor.binding.shape, &tensor.binding.strides);
        let elements = match shape.num_elements() {
            0 => 0,
            _ => {
                1 + (0..shape.rank())
                    .map(|dim| (shape[dim] - 1) * strides[dim])
                    .sum::<usize>()
            }
        };
        let bytes = (elements * tensor.elem.size()) as u64;
        let handle = &mut tensor.binding.handle;
        let end = handle.offset_start.unwrap_or(0) + bytes.next_multiple_of(4);
        handle.offset_end = Some(handle.size - end);
        tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubek_std::layout::tile_order::nvfp4_scheme;
    use cubek_std::matmul_dimensions::N;
    use cubek_test_utils::client;

    /// Values of `k` each matrix holds.
    const DEPTH: usize = 256;

    /// Rows each matrix holds: three strips of eight.
    const ROWS: usize = 24;

    /// NVFP4 in rows: eight values a word, sixteen a scale, four scales a word.
    fn in_rows() -> Operand {
        Operand::of_scheme(&nvfp4_scheme(), K, [K, N]).expect("NVFP4 blocks k")
    }

    /// NVFP4 in tiles of eight rows, row by row.
    fn in_tiles() -> (Operand, TileOrder) {
        let order = TileOrder::laid(&nvfp4_scheme(), TileWords::RowByRow, (128, 8))
            .expect("NVFP4 is tile order");
        let operand = in_rows()
            .in_tile_order(order, N, 4)
            .expect("tile order is read");
        (operand, order)
    }

    /// NVFP4 in [`in_tiles`]' tiles, as a matrix states where they lie.
    fn tiles() -> StoredTiles {
        StoredTiles {
            order: in_tiles().1,
            across: N,
            words: 4,
        }
    }

    /// A matrix in rows, or in `tiles`, over buffers of zeros.
    fn matrix(tiles: Option<StoredTiles>) -> PackedTensor {
        let operand = tiles.map_or(in_rows(), |_| in_tiles().0);
        let words = DEPTH / operand.read(K).min(8);
        let blocks = DEPTH / 16;
        let client = client();
        let u8s = ElemType::UInt(UIntKind::U8);
        PackedTensor::new(
            PlainTensor::zeros(&client, &[ROWS, words], vec![words, 1], WORD),
            PlainTensor::zeros(&client, &[ROWS, blocks], vec![blocks, 1], u8s),
            Some(PlainTensor::zeros(&client, &[1, 1], vec![1, 1], WORD)),
            nvfp4_scheme(),
            K,
            tiles,
            [(K, DEPTH), (N, ROWS)],
        )
    }

    #[test]
    fn no_slab_is_whole() {
        assert!(!slabs_are_whole(&in_rows(), N, ROWS, 0));
    }

    #[test]
    fn slabs_that_do_not_divide_the_extent_are_not_whole() {
        assert!(!slabs_are_whole(&in_rows(), N, ROWS, 5));
    }

    /// Unscaled reads of four values, two slabs of six.
    #[test]
    fn slabs_that_split_a_read_are_not_whole() {
        assert!(!slabs_are_whole(&Operand::packed(K, 4, K, 1, 1), K, 12, 2));
    }

    #[test]
    fn slabs_that_split_a_scale_s_block_are_not_whole() {
        assert!(!slabs_are_whole(&in_rows(), K, DEPTH, 32));
    }

    #[test]
    fn slabs_that_split_a_word_of_scales_are_not_whole() {
        assert!(!slabs_are_whole(&in_rows(), K, DEPTH, 8));
    }

    #[test]
    fn slabs_of_whole_words_of_scales_are_whole() {
        assert!(slabs_are_whole(&in_rows(), K, DEPTH, 4));
    }

    #[test]
    fn slabs_in_tiles_that_split_a_strip_are_not_whole() {
        assert!(!slabs_are_whole(&in_tiles().0, N, ROWS, 2));
    }

    /// In tiles, slabs along `k` lie across every strip.
    #[test]
    fn slabs_in_tiles_along_k_are_not_whole() {
        assert!(!slabs_are_whole(&in_tiles().0, K, DEPTH, 2));
    }

    #[test]
    fn slabs_in_tiles_of_whole_strips_are_whole() {
        assert!(slabs_are_whole(&in_tiles().0, N, ROWS, 3));
    }

    #[test]
    fn a_stack_whose_slabs_are_not_whole_is_refused() {
        let stacked = matrix(Some(tiles())).stacked(2, N);
        assert_eq!(stacked.err(), Some(PACKED_STACK_UNREAD));
    }

    #[test]
    fn a_stack_measures_one_slab_along_its_slabs() {
        let stacked = matrix(Some(tiles())).stacked(3, N).expect("whole strips");
        assert_eq!(stacked.extent(N), ROWS / 3);
    }

    /// A stack states the matrix its slabs are stored as, split into them.
    #[test]
    fn a_stack_states_its_values_by_slab() {
        let order = in_tiles().1;
        let stack = matrix(Some(tiles())).stacked(3, N).expect("whole strips");
        let whole = order.kernel_values_read(DEPTH, (N, ROWS), 4);
        assert_eq!(
            Some(stack.values_in_tiles(order, 4)),
            TileOrder::stacked(&whole, 3)
        );
    }

    #[test]
    fn one_matrix_states_its_values_whole() {
        let order = in_tiles().1;
        assert_eq!(
            matrix(Some(tiles())).values_in_tiles(order, 4),
            order.kernel_values_read(DEPTH, (N, ROWS), 4)
        );
    }

    #[test]
    fn a_stack_states_its_scales_by_slab() {
        let order = in_tiles().1;
        let stack = matrix(Some(tiles())).stacked(3, N).expect("whole strips");
        let whole = order
            .kernel_scales_read(DEPTH, (N, ROWS), 1)
            .expect("scales in tiles");
        assert_eq!(stack.scales_in_tiles(1), TileOrder::stacked(&whole, 3));
    }

    #[test]
    fn scales_in_rows_are_stated_in_no_tiles() {
        assert_eq!(matrix(None).scales_in_tiles(1), None);
    }

    /// A descriptor addresses one matrix: it picks no slab by a table.
    #[test]
    fn a_stack_in_tiles_is_landed_by_no_descriptor() {
        let order = in_tiles().1;
        let stack = matrix(Some(tiles())).stacked(3, N).expect("whole strips");
        let landed = stack.words_by_tma_in_tiles(
            order,
            8,
            128,
            TensorMapSwizzle::None,
            &tile_order::values_dimensions(N),
        );
        assert_eq!(
            landed.err(),
            Some(Decline::Broken {
                condition: TMA_FILLS_FROM_ONE_SLAB,
                cause: None,
            })
        );
    }
}
