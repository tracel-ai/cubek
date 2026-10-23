//! A packed tensor bound as one tile: its words, under its block scales, under its per-tensor scale
//! where its scheme has one. Every kernel that reads a packed operand reads it through
//! [`PackedTensor::bind`], whichever family it belongs to; the family names the operand's axes and
//! how its block scales are read ([`BlockScales`]), nothing else.

use super::PlainTensor;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::prelude::TensorBinding;
use cubecl::quant::scheme::QuantScheme;
use cubecl::server::BufferBinding;
use cubecl::tile::launch::{Arg, Labelled, Refusal};
use cubecl::tile::{
    Axis, Launcher, Projection, TileArgLaunch,
    kind::Field,
    layout::{DimsBuilder, PhysicalAxisMap, split},
};
use cubecl::zspace::metadata::Metadata;
use cubek_microkernel::matmul::stored::{BlockScales, Operand, Packing};
use cubek_microkernel::matmul::tile_order::{self, VALUES_AXES};
use cubek_quant::quantized_arg;
use cubek_std::Hardware;
use cubek_std::matmul_axes::{EXPERT, K, N, blocked};
use cubek_std::routine::ADMITTED_PLAN_RUNS;

/// The element a packed matrix's words and scales are bound in: whole `u32` words.
const WORD: ElemType = ElemType::UInt(UIntKind::U32);

/// A tensor as it is stored, which is how a kernel reads it: its values where they lie, or packed
/// under its scales. Each binds as one tile, so a kernel reads whichever it is handed.
#[derive(Debug, Clone)]
#[expect(
    clippy::large_enum_variant,
    reason = "one per product's tensors, moved once into them and read there"
)]
pub enum StoredTensor {
    /// Its values as they are.
    Plain(PlainTensor),
    /// Packed words under their scales.
    Packed(PackedTensor),
}

impl StoredTensor {
    /// The tensor holding its values: itself, or a packed one's words.
    pub fn values(&self) -> &PlainTensor {
        match self {
            Self::Plain(tensor) => tensor,
            Self::Packed(tensor) => &tensor.words,
        }
    }

    /// What `axis` measures, of the two `axes` it states last.
    ///
    /// # Panics
    ///
    /// On an axis not among them.
    pub fn extent(&self, axes: [Axis; 2], axis: Axis) -> usize {
        match self {
            Self::Plain(tensor) => {
                let shape = tensor.shape();
                let position = axes
                    .iter()
                    .position(|&own| own == axis)
                    .unwrap_or_else(|| panic!("{axis:?} is not one of {axes:?}"));
                shape[shape.rank() - 2 + position]
            }
            Self::Packed(tensor) => tensor.extent(axis),
        }
    }

    /// Every buffer it binds, in the order its binding holds them: its values, then a packed one's
    /// block scales and its per-tensor scale. What a kept binding is rebound to, in place of the
    /// buffers it was made over ([`TileArgLaunch::rebound`]).
    pub fn buffers(&self) -> Vec<BufferBinding> {
        self.tensors()
            .map(|tensor| tensor.binding.handle.clone())
            .collect()
    }

    /// How each of [`buffers`](Self::buffers) lies, in its order: what a binding derived over them
    /// reads of them, so a kept binding is reused only over buffers that lie the same.
    pub fn layouts(&self) -> impl Iterator<Item = Metadata> + '_ {
        self.tensors().map(PlainTensor::meta)
    }

    /// The tensors behind [`buffers`](Self::buffers), in its order.
    fn tensors(&self) -> impl Iterator<Item = &PlainTensor> {
        let (values, scales) = match self {
            Self::Plain(tensor) => (tensor, None),
            Self::Packed(tensor) => (
                &tensor.words,
                Some(core::iter::once(&tensor.block_scales).chain(&tensor.global_scale)),
            ),
        };
        core::iter::once(values).chain(scales.into_iter().flatten())
    }
}

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

    /// This matrix read as `slabs` matrices laid end to end along `along`, each its extent there
    /// divided by `slabs`: a stacked weight one slab of which a table's entry picks
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
    /// The tile order stores a `[k, n]` matrix ([`VALUES_AXES`]), so only one over `[K, N]` is
    /// read from it.
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
        let [(K, k), (N, _)] = sized else {
            return Err(STORED_TILES_UNREAD);
        };
        let block = operand.grouping(K, k).ok_or(STORED_TILES_UNREAD)?;
        if block != tile_order::BLOCK || !tile.0.is_multiple_of(block) {
            return Err(STORED_TILES_UNREAD);
        }
        let words = hw.load_values(WORD);
        let values = tile_order::values_storage(tile, words);
        operand
            .stored(N, words, Some((&VALUES_AXES[..], values.tiles())), None)
            .ok_or(STORED_TILES_UNREAD)
    }

    /// The slabs it holds where it is read as a stack ([`stacked`](Self::stacked)).
    pub fn slabs(&self) -> Option<usize> {
        self.stack.map(|stack| stack.slabs)
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

    /// The matrix as one tile under `launcher`, its words `width` values wide: the words under
    /// their block scales, read as `scales` says, under the per-tensor scale where the scheme has
    /// one.
    ///
    /// Every level binds as the words it lies in and is served every scale a word holds: the unit
    /// that reads a word of scales holds the blocks of every field in it.
    pub fn bind(&self, launcher: &Launcher, width: usize, scales: BlockScales) -> TileArgLaunch {
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
        // The axes the bound words carry: every digit of both axes in rows, the stored dims'
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
                VALUES_AXES.to_vec(),
                launcher
                    .arg(
                        self.words.bound_as(tile_order::kernel_values_read(
                            self.extent(K),
                            self.extent(N),
                            tile,
                            width / self.scheme.num_quants(),
                        )),
                        WORD,
                    )
                    .axes(&VALUES_AXES),
            ),
        };
        let words = words.vectorize(width);
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

    /// `words` under `block_scales`, words of block scales read whole and so bound plain, under
    /// `global`, the per-tensor scale packed as the scheme's field for it: what
    /// [`quantized_arg`] binds, but for the block scales' level.
    fn under_scale_words(
        &self,
        words: Arg<'_, Labelled>,
        block_scales: Arg<'_, Labelled>,
        global: Option<Arg<'_, Labelled>>,
    ) -> Result<TileArgLaunch, Refusal> {
        let mut tile = words
            .packed(self.scheme.value)
            .build()?
            .arg()
            .scales(block_scales.build()?.arg());
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

    /// `axis` of the matrix as one dim over its two digits, the blocks and the position inside
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
}

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
