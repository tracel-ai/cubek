//! A packed tensor bound as one tile: its words, under its block scales, under its per-tensor scale
//! where its scheme has one. Every kernel that reads a packed operand reads it through
//! [`PackedTensor::bind`], whichever family it belongs to; the family names the operand's axes and
//! how its block scales are read ([`BlockScales`]), nothing else.

use super::PlainTensor;
use core::marker::PhantomData;
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind, IntKind, UIntKind};
use cubecl::prelude::{TensorBinding, TensorMapArg, TensorMapSwizzle};
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue};
use cubecl::tile::launch::{Arg, Delivery, Labelled, Refusal, TmaCoordinates, TmaLanding, TmaTerm};
use cubecl::tile::{
    Axis, Launcher, Projection, TileTensorLaunch,
    kind::Field,
    layout::{DimsBuilder, Offset, PhysicalAxisMap, split},
};
use cubecl::zspace::metadata::Metadata;
use cubecl::zspace::{Shape, Strides};
use cubek_microkernel::matmul::MatmulStoredTile;
use cubek_quant::quantized_arg_shifted;
use cubek_std::Hardware;
use cubek_std::launch::tma::{swizzle_span_bytes, tma_meta_tiled, tma_operand_tiled_typed};
use cubek_std::layout::stored::{BlockScales, Operand, Packing};
use cubek_std::layout::tile_order::{self, TileOrder, TileWords};
use cubek_std::matmul_axes::{EXPERT, K, blocked};
use cubek_std::routine::ADMITTED_PLAN_RUNS;

/// The element a packed matrix's words and scales are bound in: whole `u32` words.
const WORD: ElemType = ElemType::UInt(UIntKind::U32);

/// What [`PackedTensor::of_quantized`] declines: a scheme with no block scales along the axis its
/// words run along.
pub const SCALES_NOT_ALONG_WORDS: Refusal = Refusal::Declined {
    why: "the packed routines read block scales along the axis the words run along",
};

/// What [`PackedTensor::of_quantized`] declines: a batch of matrices, or values stored other than
/// in contiguous rows.
pub const PACKED_NOT_ONE_MATRIX_IN_ROWS: Refusal = Refusal::Declined {
    why: "the packed routines read one quantized matrix in row-major rows",
};

/// What a read of a packed tensor declines: anything but one packed, block-scaled matrix.
pub const TENSOR_NOT_PACKED: Refusal = Refusal::Declined {
    why: "a packed read takes one packed, block-scaled matrix",
};

/// What reading a packed tensor as a stack of slabs declines: words stored in tiles, or slabs that
/// split one read of its words, one block of its scales or one word of block scales.
pub const PACKED_STACK_UNREAD: Refusal = Refusal::Declined {
    why: "a packed stack is read slab by slab from words in rows, each slab whole reads, whole \
          scale blocks and whole words of block scales",
};

/// What reading a packed tensor as a batch of matrices, or a window of its rows, declines: words
/// stored in tiles, a two-level scheme, whose per-tensor scale is the whole tensor's, a read
/// already stacked or windowed, rows a batch does not divide, or rows past the matrix's.
pub const PACKED_BATCH_UNREAD: Refusal = Refusal::Declined {
    why: "a packed batch or window is read from words in rows under a one-level scheme, once, \
          over rows the matrix holds",
};

/// What a read of a packed tensor declines: values stored in tiles it reads no whole load of.
pub const STORED_TILES_UNREAD: Refusal = Refusal::Declined {
    why: "a packed read takes values stored in tiles shaped [n, k / block, block] from a \
              whole word, a load one run along k and one along n",
};

/// A packed matrix as a kernel reads it: its words, in the order their buffer steps, its block
/// scales, and its per-tensor scale where its scheme has one, over two axes, each with its extent.
/// The words and scales are read where they lie, in rows or in the tiles they are stored in.
///
/// Its scales split each axis into blocks and the position inside one
/// ([`Operand::axes`]); it binds over that block-split space, and restates the dense rows of the
/// tensors read beside it over the same space.
#[derive(Debug, Clone)]
pub struct PackedTensor {
    words: PlainTensor,
    block_scales: PlainTensor,
    global_scale: Option<PlainTensor>,
    scheme: QuantScheme,
    operand: Operand,
    axes: [(Axis, usize); 2],
    stack: Option<PackedStack>,
    window: Option<RowWindow>,
}

/// A run of a packed matrix's rows read where they lie: the matrix's extent along its rows is the
/// run's, its words and scales bound over every row `stored` holds, and the axis its rows split
/// into shifted by `at`, a dynamic offset, so one compiled launch reads every run of its length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RowWindow {
    at: usize,
    stored: usize,
}

/// The matrices a packed tensor holds past its first, bound as a leading dim of its words and
/// scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackedStack {
    /// `slabs` matrices end to end along one of its axes, `along`: the words and scales of each
    /// slab one slab's extent further along it, bound as a leading [`EXPERT`] dim a table's entry
    /// picks one of.
    Slabs { slabs: usize, along: Axis },
    /// `batches` matrices end to end along its rows, each `pitch` rows of its words and scales
    /// past the one before, bound as a leading dim over `axis`, which the launch walks as it walks
    /// the matrix's own axes.
    Batch {
        batches: usize,
        pitch: usize,
        axis: Axis,
    },
}

impl PackedTensor {
    /// A packed matrix over `axes`, each with its extent, read as `operand` says under `scheme`:
    /// its words in the order their buffer steps, where the operand's
    /// [tile order](Operand::tile_order) lays them or in rows, its block scales as whole words
    /// where they lie (rows of them, or for words laid block by block the tiles tile order stores
    /// them in), and its per-tensor scale, as whole words, where `scheme` has one. Its caller reads
    /// these parts off however it stores a packed tensor, and states `operand` as
    /// [`read_in_tiles`](Self::read_in_tiles) returns it for a matrix in tiles.
    ///
    /// # Panics
    ///
    /// A per-tensor scale given for a one-level scheme, or none for a two-level one: it would
    /// bind none and drop it.
    pub fn new(
        words: PlainTensor,
        block_scales: PlainTensor,
        global_scale: Option<PlainTensor>,
        scheme: QuantScheme,
        operand: Operand,
        axes: [(Axis, usize); 2],
    ) -> Self {
        assert_eq!(
            global_scale.is_some(),
            scheme.num_levels() > 1,
            "a two-level scheme carries a per-tensor scale and a one-level one does not"
        );
        Self {
            words,
            block_scales,
            global_scale,
            scheme,
            operand,
            axes,
            stack: None,
            window: None,
        }
    }

    /// A quantized matrix over `axes`, in the order its shape states them, as a framework stores
    /// one: `values`, its buffer bound under its logical shape and strides, its values packed as
    /// `scheme` stores them, and its block scales and its per-tensor scale, where `scheme` has one,
    /// each bound where it lies. The words and the block scales are read as their buffers lie
    /// ([`PlainTensor::in_buffer_order`]): a transposed view of a packed matrix, a tied embedding
    /// read as a head, is the matrix it views, its rows along the axis its words do not run
    /// along. The scales are laid in rows and bound on whole words, as [`new`](Self::new) takes
    /// them.
    ///
    /// # Errors
    ///
    /// [`SCALES_NOT_ALONG_WORDS`], and [`PACKED_NOT_ONE_MATRIX_IN_ROWS`].
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
        global_scale: Option<PlainTensor>,
        axes: [Axis; 2],
    ) -> Result<Self, Refusal> {
        let operand = Operand::axis_of(&scheme, axes)
            .and_then(|axis| Operand::of_scheme(&scheme, axis, axes))
            .ok_or(SCALES_NOT_ALONG_WORDS)?;
        let shape = values.shape.clone();
        if shape.rank() != 2 || values.tiling.is_tiled() {
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
            global_scale.map(in_rows_of_words),
            scheme,
            operand,
            [(axes[0], shape[0]), (axes[1], shape[1])],
        ))
    }

    /// This matrix read as `slabs` matrices laid end to end along `along`, each its extent there
    /// divided by `slabs`: a stacked input one slab of which a table's entry picks
    /// ([`MatmulIndex::Slab`](crate::kernels::MatmulIndex::Slab)).
    ///
    /// # Errors
    ///
    /// [`PACKED_STACK_UNREAD`]: words stored in tiles, or slabs that split a read, a scale's block,
    /// or a word of block scales.
    pub fn stacked(self, slabs: usize, along: Axis) -> Result<Self, Refusal> {
        let extent = self.extent(along);
        if self.operand.tile_order().is_some() || slabs == 0 || !extent.is_multiple_of(slabs) {
            return Err(PACKED_STACK_UNREAD);
        }
        let slab = extent / slabs;
        let operand = &self.operand;
        let blocks_whole = match operand.grouping(along, extent) {
            Some(block) => {
                slab.is_multiple_of(block)
                    && (slab / block).is_multiple_of(operand.scales_per_word())
            }
            None => true,
        };
        if !slab.is_multiple_of(operand.read(along)) || !blocks_whole {
            return Err(PACKED_STACK_UNREAD);
        }
        let axes = self
            .axes
            .map(|(axis, extent)| (axis, if axis == along { slab } else { extent }));
        Ok(Self {
            axes,
            stack: Some(PackedStack::Slabs { slabs, along }),
            ..self
        })
    }

    /// `operand`, of a matrix `sized` so whose values `matrix` states in tile order, read what one
    /// load of `hw` takes of its tiles, in the tile order the matrix states
    /// ([`TileOrder::of_operand`]), never one the reader works out again: its values a load at a
    /// time, and where the order keeps them in tiles of their own ([`TileWords::BlockByBlock`]),
    /// its scales a read at a time beside them. The operand keeps the order.
    ///
    /// The tile order stores a matrix over `K` and the axis it lies across
    /// ([`values_axes`](tile_order::values_axes)), so only one over `K` then another axis is read
    /// from it.
    ///
    /// # Errors
    ///
    /// [`STORED_TILES_UNREAD`]: a matrix over other axes, values in rows or at another block than
    /// this matrix's, or in tiles no read takes whole.
    pub fn read_in_tiles(
        hw: &Hardware,
        sized: [(Axis, usize); 2],
        operand: Operand,
        matrix: &Metadata,
    ) -> Result<Operand, Refusal> {
        let [(K, k), (across, _)] = sized else {
            return Err(STORED_TILES_UNREAD);
        };
        let block = operand.grouping(K, k).ok_or(STORED_TILES_UNREAD)?;
        let order = TileOrder::of_operand(&operand, matrix).ok_or(STORED_TILES_UNREAD)?;
        if order.block() != block {
            return Err(STORED_TILES_UNREAD);
        }
        operand
            .in_tile_order(order, across, hw.load_values(WORD))
            .ok_or(STORED_TILES_UNREAD)
    }

    /// This matrix read as `batches` matrices laid end to end along its rows, each its rows
    /// divided by `batches`, its rows those of the axis its words do not run along: a leading dim
    /// over `axis` the launch walks, as every head of a cache is one matrix of its positions.
    /// [`stacked`](Self::stacked)'s split, walked rather than picked by a table, so the words and
    /// scales stay bound over the whole matrix. Read before a [`window`](Self::window) of it.
    ///
    /// # Errors
    ///
    /// [`PACKED_BATCH_UNREAD`]: words stored in tiles, a two-level scheme, a stack or a window
    /// already, or rows `batches` does not divide.
    pub fn batched(self, batches: usize, axis: Axis) -> Result<Self, Refusal> {
        let rows_axis = self.rows_axis();
        let rows = self.extent(rows_axis);
        let unread = self.operand.tile_order().is_some()
            || self.global_scale.is_some()
            || self.stack.is_some()
            || self.window.is_some()
            || batches == 0
            || !rows.is_multiple_of(batches);
        if unread {
            return Err(PACKED_BATCH_UNREAD);
        }
        let pitch = rows / batches;
        let axes = self
            .axes
            .map(|(own, extent)| (own, if own == rows_axis { pitch } else { extent }));
        Ok(Self {
            axes,
            stack: Some(PackedStack::Batch {
                batches,
                pitch,
                axis,
            }),
            ..self
        })
    }

    /// This matrix's rows `at..at + count` read where they lie, of the axis its words do not run
    /// along: a matrix of `count` rows whose launch reads `at` as a dynamic offset, so one compiled
    /// launch reads every run of `count`, as a cache's write and read land at any position. Every
    /// matrix of a batch is read at the same rows. Its block scales are read as fields
    /// ([`BlockScales::Fields`]).
    ///
    /// # Errors
    ///
    /// [`PACKED_BATCH_UNREAD`]: words stored in tiles, a two-level scheme, slabs or a window
    /// already, or rows past the matrix's.
    pub fn window(self, at: usize, count: usize) -> Result<Self, Refusal> {
        let rows_axis = self.rows_axis();
        let stored = self.extent(rows_axis);
        let unread = self.operand.tile_order().is_some()
            || self.global_scale.is_some()
            || matches!(self.stack, Some(PackedStack::Slabs { .. }))
            || self.window.is_some()
            || at + count > stored
            || i32::try_from(at).is_err();
        if unread {
            return Err(PACKED_BATCH_UNREAD);
        }
        let axes = self
            .axes
            .map(|(axis, extent)| (axis, if axis == rows_axis { count } else { extent }));
        Ok(Self {
            axes,
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
        core::iter::once(&self.block_scales).chain(&self.global_scale)
    }

    /// How its values are packed and scaled.
    pub fn scheme(&self) -> QuantScheme {
        self.scheme
    }

    /// What one read of it brings and what one of its scales covers, over its axes.
    pub fn operand(&self) -> Operand {
        self.operand
    }

    /// This matrix read a scale block at a time ([`Packing::by_block`]), its block scales bound
    /// one to a read. The same bytes; `None` where its packing reads no block so.
    pub fn by_block(&self) -> Option<Self> {
        Some(Self {
            operand: self.packing().by_block()?.operand,
            ..self.clone()
        })
    }

    /// How it is packed: its reads and scales, and its scheme.
    pub fn packing(&self) -> Packing {
        Packing {
            operand: self.operand,
            scheme: self.scheme,
        }
    }

    /// The matrix's two axes, in the order its caller stated them.
    pub fn axes(&self) -> [Axis; 2] {
        self.axes.map(|(axis, _)| axis)
    }

    /// This matrix with its axis `from` named `to`: the same bytes, read by a kernel that names
    /// the matrix's axes otherwise.
    pub fn renamed(self, from: Axis, to: Axis) -> Self {
        let rename = |axis: Axis| if axis == from { to } else { axis };
        Self {
            operand: self.operand.renamed(from, to),
            axes: self.axes.map(|(axis, extent)| (rename(axis), extent)),
            stack: self.stack.map(|stack| match stack {
                PackedStack::Slabs { slabs, along } => PackedStack::Slabs {
                    slabs,
                    along: rename(along),
                },
                PackedStack::Batch {
                    batches,
                    pitch,
                    axis,
                } => PackedStack::Batch {
                    batches,
                    pitch,
                    axis: rename(axis),
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
        let order = self.operand.tile_order()?;
        let (columns, rows) = order.tile();
        (order.words() == TileWords::RowByRow).then_some(MatmulStoredTile { rows, columns })
    }

    /// What `axis` of the matrix measures.
    ///
    /// # Panics
    ///
    /// On an axis the matrix does not have.
    pub fn extent(&self, axis: Axis) -> usize {
        self.axes
            .iter()
            .find(|&&(own, _)| own == axis)
            .map(|&(_, extent)| extent)
            .unwrap_or_else(|| panic!("{axis:?} is not one of the packed matrix's axes"))
    }

    /// The dims [`restated`](Self::restated) states dense rows of `rows` by `columns`, each with
    /// its extent, `step` apart, in: one for each axis of the block-split space, `(axis, extent,
    /// stride)`, coarsest first. A dim's blocks are left out where there is one.
    pub fn restated_dims(
        &self,
        rows: (Axis, usize),
        columns: (Axis, usize),
        step: usize,
    ) -> Vec<(Axis, usize, usize)> {
        let mut dims = Vec::new();
        for ((dim, extent), dim_step) in [(rows, step), (columns, 1)] {
            let [(blocks, block_count), (inside, inside_count)] = self.operand.axes(dim, extent);
            if block_count > 1 {
                dims.push((blocks, block_count, inside_count * dim_step));
            }
            dims.push((inside, inside_count, dim_step));
        }
        dims
    }

    /// `tensor`, dense rows of `rows` by `columns`, each with its extent, restated over the axes
    /// each is in the block-split space, and those axes: the same bytes, a dim an axis, which the
    /// tile engine stages and masks as it lies. A dim's blocks are left out where there is one, the
    /// tensor constant along them; its inside always stays, so the tensor is a matrix.
    pub fn restated(
        &self,
        tensor: &PlainTensor,
        rows: (Axis, usize),
        columns: (Axis, usize),
    ) -> (PlainTensor, Vec<Axis>) {
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
            dims.iter().map(|&(axis, _, _)| axis).collect(),
        )
    }

    /// The matrix as one tile under `launcher`, its words `width` values wide and moved into a
    /// stage as `delivery` says: the words under their block scales, read as `scales` says, under
    /// the per-tensor scale where the scheme has one.
    ///
    /// Every level binds as the words it lies in and is served every scale a word holds: the unit
    /// that reads a word of scales holds the blocks of every field in it.
    pub fn bind(
        &self,
        launcher: &Launcher,
        width: usize,
        scales: BlockScales,
        delivery: Delivery,
    ) -> TileTensorLaunch {
        let operand = &self.operand;
        let lined = operand.lined();
        // The buffer steps its rows before its words, so its dims are named that way.
        let rows_along = self.other_axis(lined);
        // Bound over every row it stores; a window reads its run of them.
        let rows = self.stored_rows();
        let axes = self
            .stack_dim()
            .dim(self.rows_dim(rows_along))
            .dim(self.dim(lined))
            .build();
        let across = self.other_axis(K);
        let values_axes = tile_order::values_axes(across);
        // The axes the bound words carry: every split axis of both axes in rows, the stored dims'
        // in tiles.
        let (words_axes, words): (Vec<Axis>, _) = match operand.tile_order() {
            None => (
                [rows_along, lined]
                    .into_iter()
                    .flat_map(|axis| {
                        let (blocks, within) = blocked(axis);
                        [blocks, within]
                    })
                    .collect(),
                launcher
                    .arg(
                        // Strides in stored words, each the scheme's values: a read may take
                        // several ([`Operand::by_block`]).
                        self.per_slab(self.words.rows_unpacked(
                            rows,
                            self.extent(lined),
                            self.scheme.num_quants(),
                        )),
                        WORD,
                    )
                    .gathered(axes.clone()),
            ),
            // Stated for the read the launch takes, in the tile the matrix is stored in: the
            // same bytes at every read.
            Some(order) => (
                values_axes.to_vec(),
                launcher
                    .arg(
                        self.words.bound_as(order.kernel_values_read(
                            self.extent(K),
                            (across, self.extent(across)),
                            width / self.scheme.num_quants(),
                        )),
                        WORD,
                    )
                    .axes(&values_axes),
            ),
        };
        let words = words.vectorize(width).delivery(delivery);
        // The scales count the blocks of the axis they are constant inside.
        let tiled_along = operand
            .scaled_along()
            .expect("a block-scaled matrix groups one of its axes");
        let (blocks_axis, inside) = blocked(tiled_along);
        let blocks = self.extent(tiled_along) / launcher.space().extent(inside);
        // A scale is looked up at its value's coordinates, so it names only the axes the values
        // carry: values in tiles carry the position inside the rows' axis, not its blocks, which
        // no scale groups.
        let scale_rows = match operand.tile_order() {
            None => self.rows_dim(rows_along),
            Some(_) => PhysicalAxisMap::of(blocked(rows_along).1),
        };
        let global = self.global_scale.as_ref().map(|scale| {
            launcher
                .arg(scale.rows(1, 1), WORD)
                .gathered(Self::over_whole_matrix(&words_axes, tiled_along))
        });
        match scales {
            // A read taking its one `ue4m3` scale alone ([`Packing::by_block`]): each a byte of
            // its own, bound as the `e4m3` byte it is with its sign clear, so a unit loads the byte
            // it uses rather than the word around it.
            BlockScales::Fields if self.packing().reads_scale_bytes() => {
                let e4m3 = ElemType::Float(FloatKind::E4M3);
                let scales_axes = tile_order::scales_axes(across);
                // In the tiles tile order stores them in, where it does, stated a row a read: a
                // statement counted in scales, which bound as bytes is counted in bytes.
                let in_tiles = operand.tile_order().and_then(|order| {
                    order.kernel_scales_read(self.extent(K), (across, self.extent(across)), 1)
                });
                let block_scales = match in_tiles {
                    None => launcher
                        .arg(self.per_slab(self.block_scale_bytes(rows, blocks)), e4m3)
                        .gathered(
                            self.stack_dim()
                                .dim(scale_rows)
                                .dim(self.dim(lined))
                                .build()
                                .scales_per(blocks_axis),
                        ),
                    Some(stated) => launcher
                        .arg(self.block_scales.bound_as(stated), e4m3)
                        .axes(&scales_axes),
                }
                .vectorize(1);
                self.under_scale_bytes(words, block_scales, global)
            }
            BlockScales::Fields => {
                let scales_axes = tile_order::scales_axes(across);
                // In the tiles tile order stores them in beside the words, where it does, stated
                // for the read the operand takes of them, which they are bound at below: the rows
                // a read of the words spans, side by side.
                let in_tiles = operand.tile_order().and_then(|order| {
                    order.kernel_scales_read(
                        self.extent(K),
                        (across, self.extent(across)),
                        operand.scale_read(across),
                    )
                });
                let block_scales = match in_tiles {
                    None => launcher
                        .arg(self.per_slab(self.block_scales.rows(rows, blocks)), WORD)
                        .gathered(
                            self.stack_dim()
                                .dim(scale_rows)
                                .dim(self.dim(lined))
                                .build()
                                .scales_per(blocks_axis),
                        ),
                    Some(stated) => launcher
                        .arg(self.block_scales.bound_as(stated), WORD)
                        .axes(&scales_axes),
                }
                // A whole read of the scales at once: the scales of every row a read spans.
                .vectorize(operand.scale_read_values());
                quantized_arg_shifted(
                    &self.scheme,
                    words,
                    [block_scales].into_iter().chain(global),
                    self.runtime_offsets(),
                )
            }
            BlockScales::Words => {
                assert!(
                    self.window.is_none(),
                    "bind: a window of rows reads its block scales as fields"
                );
                // A launch taking its scales as words declines scales in tiles before it binds
                // (`MATMUL_SCALE_WORDS_IN_TILES`): their rows are no tile's.
                assert_ne!(
                    operand.tile_order().map(TileOrder::words),
                    Some(TileWords::BlockByBlock),
                    "bind: block scales are bound as words of their rows, and these lie in tiles"
                );
                let per_word = operand.scales_per_word();
                let block_scales = launcher
                    .arg(
                        self.per_slab(self.block_scale_words(rows, blocks / per_word)),
                        WORD,
                    )
                    .gathered(
                        self.stack_dim()
                            .dim(scale_rows)
                            .dim(PhysicalAxisMap::of(blocks_axis).over(per_word))
                            .build(),
                    );
                self.under_scale_words(words, block_scales, global)
            }
        }
        .expect(ADMITTED_PLAN_RUNS)
    }

    /// The matrix as one tile under `launcher` as [`bind`](Self::bind) binds it under `scales`,
    /// served `width` values wide, its words moved by the TMA engine through a descriptor over
    /// the rows or the tiles ([`TileOrder::words_in_tiles`]) they are stored in: a box of
    /// `stage_rows` rows by `stage_k` values of `k`, swizzled as `swizzle` says. Its block scales,
    /// kept beside the words, are handed to the copy engine: a stage's row of them is a few bytes,
    /// short of the 16 a descriptor's row lands, so the units that fill issue them beside the
    /// descriptors rather than loading and storing them. Under [`BlockScales::Words`] a word of a
    /// step's scales is one load straight into the instruction's register; under
    /// [`BlockScales::Fields`] each scale is read and narrowed on its own.
    ///
    /// # Panics
    ///
    /// On words in tiles laid block by block, whose tile rows are no row's run; on a stage not
    /// whole tiles along its rows, nor whole tiles or a part dividing one along `k`; on words in
    /// rows of a stack, or not one dense run a row.
    pub fn bind_by_tma(
        &self,
        launcher: &Launcher,
        width: usize,
        stage_rows: usize,
        stage_k: usize,
        swizzle: TensorMapSwizzle,
        scales: BlockScales,
    ) -> TileTensorLaunch {
        let across = self.other_axis(K);
        let values_axes = tile_order::values_axes(across);
        let words = match self.operand.tile_order() {
            None => self.words_by_tma_in_rows(stage_rows, stage_k, swizzle, &values_axes),
            Some(order) if order.words() == TileWords::RowByRow => {
                self.words_by_tma_in_tiles(order, stage_rows, stage_k, swizzle, &values_axes)
            }
            Some(_) => panic!("bind_by_tma: words in rows, or in tiles laid row by row"),
        }
        .served_wide(width / self.scheme.num_quants())
        .packed(Field::from(self.scheme.value).read_on(launcher.client()));
        let operand = &self.operand;
        let lined = operand.lined();
        let rows_along = self.other_axis(lined);
        let rows = self.extent(rows_along);
        let words_axes = values_axes.to_vec();
        let tiled_along = operand
            .scaled_along()
            .expect("a block-scaled matrix groups one of its axes");
        let (blocks_axis, inside) = blocked(tiled_along);
        let blocks = self.extent(tiled_along) / launcher.space().extent(inside);
        let scale_rows = PhysicalAxisMap::of(blocked(rows_along).1);
        let global = self.global_scale.as_ref().map(|scale| {
            launcher
                .arg(scale.rows(1, 1), WORD)
                .gathered(Self::over_whole_matrix(&words_axes, tiled_along))
        });
        match scales {
            BlockScales::Fields => {
                let block_scales = launcher
                    .arg(self.per_slab(self.block_scales.rows(rows, blocks)), WORD)
                    .gathered(
                        self.stack_dim()
                            .dim(scale_rows)
                            .dim(self.dim(lined))
                            .build()
                            .scales_per(blocks_axis),
                    )
                    .vectorize(operand.scale_read_values())
                    .delivery(Delivery::AsyncPerUnit);
                let fields = self
                    .scheme
                    .block_scale()
                    .map(|block| block.dtype)
                    .into_iter()
                    .chain(self.scheme.tensor_scale())
                    .map(Field::of_scale);
                let mut tile = words;
                for (scale, field) in [block_scales].into_iter().chain(global).zip(fields) {
                    tile =
                        tile.scales(scale.packed(field).build().expect(ADMITTED_PLAN_RUNS).arg());
                }
                tile
            }
            // Words of a step's scales, each one load straight into the instruction's register,
            // handed to the copy engine as the words they are stored in.
            BlockScales::Words => {
                let per_word = operand.scales_per_word();
                let block_scales = launcher
                    .arg(
                        self.per_slab(self.block_scale_words(rows, blocks / per_word)),
                        WORD,
                    )
                    .gathered(
                        self.stack_dim()
                            .dim(scale_rows)
                            .dim(PhysicalAxisMap::of(blocks_axis).over(per_word))
                            .build(),
                    )
                    .delivery(Delivery::AsyncPerUnit);
                self.scaled_by_words(words, block_scales, global)
                    .expect(ADMITTED_PLAN_RUNS)
            }
        }
    }

    /// The words stored in `order`'s tiles laid row by row, through a descriptor over the tiles
    /// ([`TileOrder::words_in_tiles`]): a box of `stage_rows` rows by `stage_k` values of `k`,
    /// landing row by row, swizzled as `swizzle` says, over `values_axes`.
    fn words_by_tma_in_tiles(
        &self,
        order: TileOrder,
        stage_rows: usize,
        stage_k: usize,
        swizzle: TensorMapSwizzle,
        values_axes: &[Axis; 3],
    ) -> TileTensorLaunch {
        let tile = order.tile();
        let (tile_k, tile_rows) = tile;
        let across = self.other_axis(K);
        let (k, rows_across) = (self.extent(K), self.extent(across));
        assert!(
            (stage_k.is_multiple_of(tile_k) || tile_k.is_multiple_of(stage_k))
                && stage_rows.is_multiple_of(tile_rows),
            "bind_by_tma: a stage of {stage_rows} x {stage_k} is not whole {tile:?} tiles along \
             its rows, nor whole tiles or a part dividing one along k"
        );
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
            dims: vec![
                vec![TmaTerm::axis(0).over(tile_rows as u32)],
                vec![],
                vec![
                    TmaTerm::axis(1)
                        .times(order.block() as u32)
                        .over(tile_k as u32),
                ],
                vec![
                    TmaTerm::axis(1)
                        .times(order.block() as u32)
                        .modulo(tile_k as u32)
                        .over(order.values_per_word() as u32),
                ],
            ],
        };
        TileTensorLaunch::tensor_map_folded(
            map,
            values_axes,
            &[
                rows_across as u32,
                (k / order.block()) as u32,
                order.block() as u32,
            ],
            coordinates,
        )
    }

    /// The words stored in rows, through a descriptor over the rows: `k`'s words cut into rows of
    /// the swizzle's span, so a box of `stage_rows` rows by `stage_k` values of `k` lands a
    /// stage's blocks of `K` side by side, as a float operand's descriptor lands them, over
    /// `values_axes`.
    ///
    /// # Panics
    ///
    /// On a stack, or on rows whose words are not one dense run each: a descriptor over the rows
    /// steps them by the run.
    fn words_by_tma_in_rows(
        &self,
        stage_rows: usize,
        stage_k: usize,
        swizzle: TensorMapSwizzle,
        values_axes: &[Axis; 3],
    ) -> TileTensorLaunch {
        assert!(
            self.stack.is_none(),
            "bind_by_tma: a stack is read slab by slab"
        );
        let across = self.other_axis(K);
        let (k, rows) = (self.extent(K), self.extent(across));
        let per_word = self.scheme.num_quants();
        let block = self
            .operand
            .scale(K)
            .expect("a block-scaled matrix groups k");
        let words_k = k / per_word;
        assert_eq!(
            self.words.row_step().unwrap_or(words_k),
            words_k,
            "bind_by_tma: rows of words a descriptor over the rows steps by their run"
        );
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
            dims: vec![
                vec![TmaTerm::axis(0)],
                vec![
                    TmaTerm::axis(1)
                        .times((block / per_word) as u32)
                        .over(row as u32),
                ],
                vec![],
            ],
        };
        TileTensorLaunch::tensor_map_folded(
            map,
            values_axes,
            &[rows as u32, (k / block) as u32, block as u32],
            coordinates,
        )
    }

    /// `words` under `block_scales`, words of block scales read whole, each word the scheme's
    /// block-scale field four or more times, under `global`, the per-tensor scale packed as the
    /// scheme's field for it: what [`quantized_arg`] binds, but for the block scales' level. The
    /// words state their field: the block-scaled instruction reads an `e4m3` scale and a `ue8m0`
    /// one as different types, and a word that states none is read as one scale.
    fn under_scale_words(
        &self,
        words: Arg<'_, Labelled>,
        block_scales: Arg<'_, Labelled>,
        global: Option<Arg<'_, Labelled>>,
    ) -> Result<TileTensorLaunch, Refusal> {
        let words = words.packed(self.scheme.value).build()?.arg();
        self.scaled_by_words(words, block_scales, global)
    }

    /// `words`, bound packed, under `block_scales` read as words of the scheme's block-scale field
    /// and `global` packed as its per-tensor one: [`under_scale_words`](Self::under_scale_words)
    /// past the words, which a descriptor binds as well as the launcher.
    fn scaled_by_words(
        &self,
        words: TileTensorLaunch,
        block_scales: Arg<'_, Labelled>,
        global: Option<Arg<'_, Labelled>>,
    ) -> Result<TileTensorLaunch, Refusal> {
        let block = self
            .scheme
            .block_scale()
            .expect("block scales bound as words are a block-scaled scheme's");
        let tile = words.scales(
            block_scales
                .words(Field::of_scale(block.dtype))
                .build()?
                .arg(),
        );
        self.under_global(tile, global)
    }

    /// `words` under `block_scales`, bound a byte a scale as plain `e4m3` elements, under
    /// `global`, the per-tensor scale packed as the scheme's field for it. A window's words and
    /// block scales read its start as their dynamic offset, as [`quantized_arg_shifted`] reads
    /// it; no window holds a per-tensor scale ([`window`](Self::window)).
    fn under_scale_bytes(
        &self,
        words: Arg<'_, Labelled>,
        block_scales: Arg<'_, Labelled>,
        global: Option<Arg<'_, Labelled>>,
    ) -> Result<TileTensorLaunch, Refusal> {
        let offsets = self.runtime_offsets();
        let shifted = |arg: TileTensorLaunch| match offsets.is_empty() {
            true => arg,
            false => arg.runtime_map(Vec::new(), offsets.clone()),
        };
        let tile = shifted(words.packed(self.scheme.value).build()?.arg())
            .scales(shifted(block_scales.build()?.arg()));
        self.under_global(tile, global)
    }

    /// `tile` under `global`, the per-tensor scale packed as the scheme's field for it, where the
    /// scheme has one; as it is otherwise.
    fn under_global(
        &self,
        tile: TileTensorLaunch,
        global: Option<Arg<'_, Labelled>>,
    ) -> Result<TileTensorLaunch, Refusal> {
        match (global, self.scheme.tensor_scale()) {
            (Some(global), Some(field)) => {
                Ok(tile.scales(global.packed(Field::of_scale(field)).build()?.arg()))
            }
            _ => Ok(tile),
        }
    }

    /// The block scales a byte apiece, `blocks` a row, each row stepped at its stride counted in
    /// scales: a buffer holding them in wider elements steps its rows in those.
    fn block_scale_bytes(&self, rows: usize, blocks: usize) -> TensorBinding {
        let per_element = self.block_scales.elem.size();
        let step = self
            .block_scales
            .row_step()
            .map_or(blocks, |step| step * per_element);
        self.block_scales
            .bound_as(Metadata::new([rows, blocks], [step, 1]))
    }

    /// The block scales as the `u32` words they are stored in, `words` a row: each row a run of
    /// whole words, stepped from row to row at its stride, counted in words.
    fn block_scale_words(&self, rows: usize, words: usize) -> TensorBinding {
        let pitch = self.block_scales.binding.strides[0] * self.block_scales.elem.size();
        self.block_scales
            .bound_as(Metadata::new([rows, words], [pitch / 4, 1]))
    }

    /// The dims a binding of words or scales starts with: the slabs, where this tensor is a stack.
    fn stack_dim(&self) -> DimsBuilder {
        match self.stack {
            Some(PackedStack::Slabs { .. }) => Projection::dims().dim(PhysicalAxisMap::of(EXPERT)),
            Some(PackedStack::Batch { axis, .. }) => {
                Projection::dims().dim(PhysicalAxisMap::of(axis))
            }
            None => Projection::dims(),
        }
    }

    /// `binding`, words or scales of one matrix over its two axes in the order the buffer steps
    /// them, rows then line, once per matrix where this tensor holds several: a leading dim, each
    /// slab one slab's run of the dim the stack lies along further, each matrix of a batch its
    /// pitch of rows. As it is otherwise.
    fn per_slab(&self, binding: TensorBinding) -> TensorBinding {
        let (count, step) = match self.stack {
            None => return binding,
            Some(PackedStack::Slabs { slabs, along }) => {
                let dim = usize::from(along == self.operand.lined());
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

    /// The axis the matrix's rows run along: the one its words do not.
    fn rows_axis(&self) -> Axis {
        self.other_axis(self.operand.lined())
    }

    /// The rows its words and scales are bound over: every row stored, past a window's.
    fn stored_rows(&self) -> usize {
        self.window
            .map_or(self.extent(self.rows_axis()), |window| window.stored)
    }

    /// The rows' dim, as [`dim`](Self::dim), shifted by a window's dynamic offset where it has one.
    fn rows_dim(&self, rows_along: Axis) -> PhysicalAxisMap {
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

    /// The matrix's axis other than `axis`.
    fn other_axis(&self, axis: Axis) -> Axis {
        match self.axes {
            [(first, _), (second, _)] if first == axis => second,
            [(first, _), _] => first,
        }
    }

    /// `axis` of the matrix as one dim over the two axes it is split into, the blocks and the position inside
    /// one.
    fn dim(&self, axis: Axis) -> PhysicalAxisMap {
        split(&self.operand.axes(axis, self.extent(axis)))
    }

    /// One value over the whole matrix: every axis of `words_axes`, the axes the bound words
    /// carry, that its block scales span, none addressed. A factor is looked up at the values'
    /// coordinates, so it spans no axis they do not carry.
    fn over_whole_matrix(words_axes: &[Axis], tiled_along: Axis) -> Projection {
        let inside = blocked(tiled_along).1;
        let axes: Vec<Axis> = words_axes
            .iter()
            .copied()
            .filter(|&axis| axis != inside)
            .collect();
        Projection::new(
            &axes,
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

    /// `tensor`, contiguous, its binding ending on the `u32` word that holds its last element: the
    /// routines read scales a word at a time. A quantized buffer starts each region on a word, so the
    /// word's tail is allocated.
    fn bound_to_whole_words(mut tensor: PlainTensor) -> PlainTensor {
        let handle = &mut tensor.binding.handle;
        let bytes = (tensor.binding.shape.num_elements() * tensor.elem.size()) as u64;
        let end = handle.offset_start.unwrap_or(0) + bytes.next_multiple_of(4);
        handle.offset_end = Some(handle.size - end);
        tensor
    }
}
