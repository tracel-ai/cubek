//! A packed tensor bound as one tile: its words, under its block scales, under its per-tensor scale
//! where its scheme has one. Every kernel that reads a packed operand reads it through
//! [`PackedTensor::bind`], whichever family it belongs to; the family names the operand's axes and
//! how its block scales are read ([`BlockScales`]), nothing else.

use super::PlainTensor;
use cubecl::client::Client;
use cubecl::ir::{ElemType, IntKind, UIntKind};
use cubecl::prelude::{TensorBinding, TensorMapSwizzle};
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue};
use cubecl::tile::launch::{Arg, Delivery, Labelled, Refusal, TmaCoordinates, TmaLanding, TmaTerm};
use cubecl::tile::{
    Axis, Launcher, Projection, TileArgLaunch,
    kind::Field,
    layout::{DimsBuilder, PhysicalAxisMap, split},
};
use cubecl::zspace::metadata::Metadata;
use cubek_microkernel::matmul::MatmulStoredTile;
use cubek_quant::quantized_arg;
use cubek_std::Hardware;
use cubek_std::launch::tma::tma_operand_tiled_typed;
use cubek_std::layout::stored::{BlockScales, Operand, Packing};
use cubek_std::layout::tile_order::{self, TileOrder};
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
    laid: WordsLaid,
    block_scales: PlainTensor,
    global_scale: Option<PlainTensor>,
    scheme: QuantScheme,
    operand: Operand,
    axes: [(Axis, usize); 2],
    stack: Option<PackedStack>,
}

/// A packed tensor holding `slabs` matrices end to end along one of its axes, `along`: the words
/// and scales of each slab one slab's extent further along it, bound as a leading [`EXPERT`] dim
/// a table's entry picks one of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PackedStack {
    slabs: usize,
    along: Axis,
}

/// How a packed matrix's words lie in their buffer: read off the matrix once, and bound as it says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WordsLaid {
    /// In the rows they are packed in.
    InRows,
    /// In [tile order](tile_order), in tiles of `(k, n)` values.
    InTiles((usize, usize)),
}

impl PackedTensor {
    /// A packed matrix over `axes`, each with its extent, its words `laid` so, read as `operand`
    /// says under `scheme`: its words in the order their buffer steps, its block scales as rows of
    /// whole words, and its per-tensor scale, as whole words, where `scheme` has one. Its caller
    /// reads these parts off however it stores a packed tensor.
    ///
    /// # Panics
    ///
    /// A per-tensor scale given for a one-level scheme, or none for a two-level one: it would
    /// bind none and drop it.
    pub fn new(
        words: PlainTensor,
        laid: WordsLaid,
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
            laid,
            block_scales,
            global_scale,
            scheme,
            operand,
            axes,
            stack: None,
        }
    }

    /// A quantized matrix over `axes`, in the order its shape states them, as a framework stores
    /// one: `values`, its buffer bound under its logical shape and strides, its values packed as
    /// `scheme` stores them, and its block scales and its per-tensor scale, where `scheme` has one,
    /// each bound where it lies. The scales are laid in rows and bound on whole words, as
    /// [`new`](Self::new) takes them.
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
        let words = Self::stored_words(values, &scheme);
        if !words.is_contiguous() {
            return Err(PACKED_NOT_ONE_MATRIX_IN_ROWS);
        }
        let in_rows_of_words =
            |scale: PlainTensor| Self::bound_to_whole_words(scale.into_contiguous(client));
        Ok(Self::new(
            words,
            WordsLaid::InRows,
            in_rows_of_words(block_scales),
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
        if self.laid != WordsLaid::InRows || slabs == 0 || !extent.is_multiple_of(slabs) {
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
            stack: Some(PackedStack { slabs, along }),
            ..self
        })
    }

    /// `operand`, of a matrix `sized` so and stored in `tile`, read what one load of `hw` takes of
    /// the tiles.
    ///
    /// The tile order stores a matrix over `K` and the axis it lies across
    /// ([`values_axes`](tile_order::values_axes)), so only one over `K` then another axis is read
    /// from it.
    ///
    /// # Errors
    ///
    /// [`STORED_TILES_UNREAD`]: a matrix over other axes, values at another shape than this
    /// matrix's, or in tiles no read takes whole.
    pub fn read_in_tiles(
        hw: &Hardware,
        sized: [(Axis, usize); 2],
        operand: Operand,
        tile: (usize, usize),
    ) -> Result<Operand, Refusal> {
        let [(K, k), (across, _)] = sized else {
            return Err(STORED_TILES_UNREAD);
        };
        let block = operand.grouping(K, k).ok_or(STORED_TILES_UNREAD)?;
        let order = TileOrder::of_operand(&operand).ok_or(STORED_TILES_UNREAD)?;
        if order.block != block || !tile.0.is_multiple_of(block) {
            return Err(STORED_TILES_UNREAD);
        }
        let words = hw.load_values(WORD);
        let values = order.values_storage(across, tile, words);
        operand
            .stored(
                across,
                words,
                Some((&tile_order::values_axes(across), values.tiles())),
                None,
            )
            .ok_or(STORED_TILES_UNREAD)
    }

    /// How it lies in tile order, as its scheme packs it ([`TileOrder::of`]). Read from the scheme
    /// and not from its operand: a matrix read in tiles holds the operand
    /// [`read_in_tiles`](Self::read_in_tiles) restated at the device's load, whose read along `K`
    /// is a whole load of words, not one word's values.
    ///
    /// # Panics
    ///
    /// On a matrix tile order does not store, which [`read_in_tiles`](Self::read_in_tiles) refuses
    /// before any is laid in tiles.
    pub fn tile_order(&self) -> TileOrder {
        TileOrder::of(&self.scheme).expect("a matrix in tiles is in tile order")
    }

    /// The slabs it holds where it is read as a stack ([`stacked`](Self::stacked)).
    pub fn slabs(&self) -> Option<usize> {
        self.stack.map(|stack| stack.slabs)
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
            stack: self.stack.map(|stack| PackedStack {
                along: rename(stack.along),
                ..stack
            }),
            ..self
        }
    }

    /// The tile its words are stored in, which a descriptor lands a stage of whole tiles of row
    /// by row: its lines by its values of `K`. None in rows.
    pub fn stored_tile(&self) -> Option<MatmulStoredTile> {
        match self.laid {
            WordsLaid::InRows => None,
            WordsLaid::InTiles((columns, rows)) => Some(MatmulStoredTile { rows, columns }),
        }
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
    ) -> TileArgLaunch {
        let operand = &self.operand;
        let lined = operand.lined();
        // The buffer steps its rows before its words, so its dims are named that way.
        let rows_along = self.other_axis(lined);
        let rows = self.extent(rows_along);
        let axes = self
            .stack_dim()
            .dim(self.dim(rows_along))
            .dim(self.dim(lined))
            .build();
        let across = self.other_axis(K);
        let values_axes = tile_order::values_axes(across);
        // The axes the bound words carry: every split axis of both axes in rows, the stored dims'
        // in tiles.
        let (words_axes, words): (Vec<Axis>, _) = match self.laid {
            WordsLaid::InRows => (
                [rows_along, lined]
                    .into_iter()
                    .flat_map(|axis| {
                        let (blocks, within) = blocked(axis);
                        [blocks, within]
                    })
                    .collect(),
                launcher
                    .arg(
                        self.per_slab(self.words.rows_unpacked(
                            rows,
                            self.extent(lined),
                            operand.read(lined),
                        )),
                        WORD,
                    )
                    .gathered(axes.clone()),
            ),
            // Stated for the read the launch takes, in the tile the matrix is stored in: the
            // same bytes at every read.
            WordsLaid::InTiles(tile) => (
                values_axes.to_vec(),
                launcher
                    .arg(
                        self.words.bound_as(self.tile_order().kernel_values_read(
                            self.extent(K),
                            (across, self.extent(across)),
                            tile,
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
        let scale_rows = match self.laid {
            WordsLaid::InRows => self.dim(rows_along),
            WordsLaid::InTiles(_) => PhysicalAxisMap::of(blocked(rows_along).1),
        };
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
                    // A whole read of the scales at once: a word of them.
                    .vectorize(operand.scale_read_values());
                quantized_arg(
                    &self.scheme,
                    words,
                    [block_scales].into_iter().chain(global),
                )
            }
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
                    );
                self.under_scale_words(words, block_scales, global)
            }
        }
        .expect(ADMITTED_PLAN_RUNS)
    }

    /// The matrix as one tile under `launcher` as [`bind`](Self::bind) binds it under
    /// `BlockScales::Fields`, served `width` values wide, its words moved by the TMA engine
    /// through a descriptor over the tiles they are stored in
    /// ([`TileOrder::words_in_tiles`]): a box of `stage_lines` lines by `stage_k` values of `k`,
    /// landing row by row, swizzled as `swizzle` says. Its block scales, kept beside the words, are
    /// handed to the copy engine: a stage's row of them is a few bytes, short of the 16 a
    /// descriptor's row lands, so the units that fill issue them beside the descriptors rather
    /// than loading and storing them.
    ///
    /// # Panics
    ///
    /// On words in rows, or a stage not whole tiles along its lines, nor whole tiles or a part
    /// dividing one along `k`.
    pub fn bind_by_tma(
        &self,
        launcher: &Launcher,
        width: usize,
        stage_lines: usize,
        stage_k: usize,
        swizzle: TensorMapSwizzle,
    ) -> TileArgLaunch {
        let WordsLaid::InTiles(tile) = self.laid else {
            panic!("bind_by_tma: words stored in tiles only");
        };
        let (tile_k, tile_lines) = tile;
        let operand = &self.operand;
        let lined = operand.lined();
        let rows_along = self.other_axis(lined);
        let rows = self.extent(rows_along);
        let across = self.other_axis(K);
        let (k, lines) = (self.extent(K), self.extent(across));
        assert!(
            (stage_k.is_multiple_of(tile_k) || tile_k.is_multiple_of(stage_k))
                && stage_lines.is_multiple_of(tile_lines),
            "bind_by_tma: a stage of {stage_lines} x {stage_k} is not whole {tile:?} tiles along \
             its lines, nor whole tiles or a part dividing one along k"
        );
        let order = self.tile_order();
        let words = self.words.bound_as(order.words_in_tiles(k, lines, tile));
        let map = tma_operand_tiled_typed(
            words,
            (stage_lines, stage_k / order.values_per_word),
            TmaLanding::Rows,
            WORD,
            swizzle,
        );
        let values_axes = tile_order::values_axes(across);
        // The descriptor after a row landing: the strips of lines, a line in the strip, the
        // tiles along `k`, a word in the tile; a box starts on whole tiles both ways.
        let coordinates = TmaCoordinates {
            dims: vec![
                vec![TmaTerm::axis(0).over(tile_lines as u32)],
                vec![],
                vec![
                    TmaTerm::axis(1)
                        .times(order.block as u32)
                        .over(tile_k as u32),
                ],
                vec![
                    TmaTerm::axis(1)
                        .times(order.block as u32)
                        .modulo(tile_k as u32)
                        .over(order.values_per_word as u32),
                ],
            ],
        };
        let words = TileArgLaunch::tensor_map_folded(
            map,
            &values_axes,
            &[lines as u32, (k / order.block) as u32, order.block as u32],
            coordinates,
        )
        .served_wide(width / self.scheme.num_quants())
        .packed(Field::from(self.scheme.value).read_on(launcher.client()));
        let words_axes = values_axes.to_vec();
        let tiled_along = operand
            .scaled_along()
            .expect("a block-scaled matrix groups one of its axes");
        let (blocks_axis, inside) = blocked(tiled_along);
        let blocks = self.extent(tiled_along) / launcher.space().extent(inside);
        let scale_rows = PhysicalAxisMap::of(blocked(rows_along).1);
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
        let global = self.global_scale.as_ref().map(|scale| {
            launcher
                .arg(scale.rows(1, 1), WORD)
                .gathered(Self::over_whole_matrix(&words_axes, tiled_along))
        });
        let mut tile = words;
        for (scale, field) in [block_scales].into_iter().chain(global).zip(fields) {
            tile = tile.scales(scale.packed(field).build().expect(ADMITTED_PLAN_RUNS).arg());
        }
        tile
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
    ) -> Result<TileArgLaunch, Refusal> {
        let block = self
            .scheme
            .block_scale()
            .expect("block scales bound as words are a block-scaled scheme's");
        let mut tile = words.packed(self.scheme.value).build()?.arg().scales(
            block_scales
                .words(Field::of_scale(block.dtype))
                .build()?
                .arg(),
        );
        if let (Some(global), Some(field)) = (global, self.scheme.tensor_scale()) {
            tile = tile.scales(global.packed(Field::of_scale(field)).build()?.arg());
        }
        Ok(tile)
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
            Some(_) => Projection::dims().dim(PhysicalAxisMap::of(EXPERT)),
            None => Projection::dims(),
        }
    }

    /// `binding`, words or scales of one matrix over its two axes in the order the buffer steps
    /// them, rows then line, once per slab where this tensor is a stack: the slabs a leading dim,
    /// each one slab's run of the dim the stack lies along further. As it is otherwise.
    fn per_slab(&self, binding: TensorBinding) -> TensorBinding {
        let Some(PackedStack { slabs, along }) = self.stack else {
            return binding;
        };
        let dim = usize::from(along == self.operand.lined());
        let step = binding.shape[dim] * binding.strides[dim];
        let lead = |slab: usize, dims: &[usize]| [&[slab][..], dims].concat();
        TensorBinding {
            shape: lead(slabs, &binding.shape).into(),
            strides: lead(step, &binding.strides).into(),
            ..binding
        }
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
