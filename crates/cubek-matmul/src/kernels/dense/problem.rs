use crate::kernels::tensor::PlainTensor;
use core::num::NonZeroUsize;
use std::borrow::Cow;

use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::Geometry;
use cubecl::tile::launch::{Decline, VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Dimension, Indexing};
use cubecl::tune::Work;
use cubecl::zspace::metadata::Metadata;
use cubek_microkernel::matmul::{
    MatmulBlocks, MatmulBuffers, MatmulDestination, MatmulDim, MatmulDimensions, MatmulLayout,
    MatmulOperand, MatmulOperands, MatmulStoredTile,
};
use cubek_std::layout::BindingOrder;
use cubek_std::layout::stored::{BlockScales, Packing, StoredInput};
use cubek_std::layout::tile_order::TileOrder;
use cubek_std::matmul_dimensions::{EXPERT, K, batch_dimension};
use cubek_std::routine::EstimatedProblem;

/// `out[b] = lhs[b] · rhs[b]` for every batch `b`, as the host describes it, each input as it is
/// stored ([`inputs`](Self::inputs)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulRoutineProblem {
    /// The output's batch extents, outermost first. An operand of extent one there is broadcast.
    pub batches: Vec<usize>,
    /// The lhs's rows as the problem was stated, which mean one batch's rows for a problem read
    /// off a tuning key, and every row, its batches already merged, for one read off tensors that
    /// hold every row whole. A launch reads it through
    /// [`MatmulView::rows`](crate::kernels::MatmulView::rows); only what is right for both forms
    /// reads it as stated: a problem that states no batches (the grouped and routed cases), and the
    /// demand estimate, whose batches multiply it either way.
    pub stated_rows: usize,
    pub n: usize,
    pub k: usize,
    /// The element each operand is read at: a float operand's stored one, a packed one's the
    /// element its words unpack into ([`inputs`](Self::inputs) says how it is stored).
    pub elements: MatmulOperands<ElemType>,
    /// Batches each operand holds: the output's, or fewer where an operand is broadcast.
    pub batches_held: MatmulOperands<usize>,
    /// The alignment of each operand's buffer in elements, over the dimensions the product names:
    /// the line widths a launch may bind it at, before the device or any component narrows them.
    pub elem_alignments: MatmulOperands<ElemAlignment>,
    /// Bytes each operand's view starts past its buffer's own start: a slice of a larger tensor
    /// starts partway into it.
    pub start_bytes: MatmulOperands<u64>,
    /// How each operand's buffer lies, which a microkernel may bind it in the order of.
    pub layouts: MatmulOperands<MatmulLayout>,
    /// Each operand's alignment over the order its buffer steps, which a col operand bound in
    /// that order is read at; the stated one for any other.
    pub buffer_alignments: MatmulOperands<ElemAlignment>,
    /// The tile each operand is stored in, where it is one a descriptor lands whole rows of; none
    /// for an operand in rows, or stored in tiles no descriptor here reads.
    pub stored_tiles: MatmulOperands<Option<MatmulStoredTile>>,
    /// Where the output goes.
    pub destination: MatmulDestination,
    /// How each operand is stored: what its values are, the element they stand for, and how a read
    /// of them and of their scales lies. A float operand is the case with no scales.
    pub inputs: MatmulOperands<StoredInput>,
    /// The table an input reads one entry of per batch, where the product has one: which slab of
    /// a stacked rhs a batch contracts against, or the run of `K` it sums.
    pub index: Option<MatmulIndex>,
}

/// What each batch's entry of a product's table says, the table's dims being the batches: one
/// entry per batch, read where the launch cuts them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulIndex {
    /// The slab of the rhs the batch contracts against: the rhs is a stack of `slabs` `[k, n]`
    /// slabs along [`EXPERT`], which no level cuts, and an entry past it names no slab, so its
    /// batch sums nothing.
    Slab { slabs: usize },
    /// Where the batch's run of `K` ends, both inputs holding every run end to end: the entry
    /// before it along the batches is where it starts. A run that ends before it starts sums
    /// nothing.
    RunOfK,
    /// Row groups of a flat, unpadded lhs `[rows, k]`, each against the slab of a stack of `slabs` its
    /// entry names, written to the same rows of a flat output `[rows, n]`. A second table holds
    /// where each row group ends, the one before it along the batches being where it starts; rows past a
    /// group's end read zero and are never written, and no row group is longer than the table's `max_rows`.
    GroupedRows { slabs: usize },
}

impl MatmulIndex {
    /// The dimension the index adds to the product's space, with its extent: the stack's slabs,
    /// which an entry picks one of.
    pub fn dimension(self) -> Option<(Dimension, usize)> {
        match self {
            Self::Slab { slabs } | Self::GroupedRows { slabs } => Some((EXPERT, slabs)),
            Self::RunOfK => None,
        }
    }

    /// How an entry places each operand that reads the table, none for one that does not. `rows`
    /// is the dimension the product's rows lie along.
    pub fn placements(self, rows: Dimension) -> MatmulOperands<Option<Indexing>> {
        match self {
            Self::Slab { .. } => MatmulOperands {
                lhs: None,
                rhs: Some(Indexing::Coordinate(EXPERT)),
                out: None,
            },
            Self::RunOfK => MatmulOperands {
                lhs: Some(Indexing::Run(K)),
                rhs: Some(Indexing::Run(K)),
                out: None,
            },
            Self::GroupedRows { .. } => MatmulOperands {
                lhs: Some(Indexing::Run(rows)),
                rhs: Some(Indexing::Coordinate(EXPERT)),
                out: Some(Indexing::Run(rows)),
            },
        }
    }
}

impl MatmulRoutineProblem {
    /// Checks every operand is float: what a routine whose kernel reads no packed operand
    /// declines first.
    ///
    /// # Errors
    ///
    /// [`MATMUL_PACKED_UNREAD`], for a packed operand.
    pub fn check_plain(&self) -> Result<(), Decline> {
        match self.packings() == MatmulOperands::all(None) {
            true => Ok(()),
            false => Err(MATMUL_PACKED_UNREAD),
        }
    }

    /// This problem with its packed inputs read as `planned` states: the problem a microkernel
    /// that re-read them plans and is bound on ([`MatmulMicrokernel::planned_space`](cubek_microkernel::matmul::MatmulMicrokernel::planned_space)). Itself where
    /// `planned` is `None` or reads every input as it is.
    pub fn read_as(&self, planned: Option<MatmulBlocks>) -> Cow<'_, Self> {
        let Some(planned) = planned else {
            return Cow::Borrowed(self);
        };
        let inputs = self
            .inputs
            .map(|operand, input| match planned.packings.get(operand) {
                Some(packing) => StoredInput {
                    operand: packing.operand,
                    ..input
                },
                None => input,
            });
        match inputs == self.inputs {
            true => Cow::Borrowed(self),
            false => Cow::Owned(Self {
                inputs,
                ..self.clone()
            }),
        }
    }

    /// Checks no input reads a table: what a routine whose kernel walks no indexed input
    /// declines first.
    ///
    /// # Errors
    ///
    /// [`MATMUL_INDEX_UNREAD`], for a product with a table.
    pub fn check_unindexed(&self) -> Result<(), Decline> {
        match self.index {
            None => Ok(()),
            Some(_) => Err(MATMUL_INDEX_UNREAD),
        }
    }

    /// Checks every packed input's block scales lie where a launch binding them as `block_scales`,
    /// its stages filled by the TMA engine where `by_tma`, reads them: what a routine whose
    /// microkernel may take them as words declines. Words copied by the units are bound from rows,
    /// through a gather that cannot address scales kept in tiles of their own; the TMA engine lands
    /// those tiles whole ([`TileOrder::scale_words_in_tiles`]).
    ///
    /// # Errors
    ///
    /// [`MATMUL_SCALE_WORDS_IN_TILES`], for scales taken as words over an input keeping them in
    /// tiles of their own, its stages filled by anything but the TMA engine.
    pub fn check_block_scales(
        &self,
        block_scales: BlockScales,
        by_tma: bool,
    ) -> Result<(), Decline> {
        let scales_in_tiles = [self.inputs.lhs, self.inputs.rhs]
            .into_iter()
            .filter_map(|input| input.operand.tile_order())
            .any(TileOrder::scales_in_tiles);
        match block_scales == BlockScales::Words && scales_in_tiles && !by_tma {
            true => Err(MATMUL_SCALE_WORDS_IN_TILES),
            false => Ok(()),
        }
    }

    /// The product of `lhs` and `rhs` written into `out`, which share a rank of at least two
    /// and broadcast their batch dims.
    pub fn new(lhs: &PlainTensor, rhs: &PlainTensor, out: &PlainTensor) -> Self {
        Self::with_output(
            lhs,
            rhs,
            OutputLayout {
                shape: &out.binding.shape,
                strides: &out.binding.strides,
                element: out.elem,
                start_bytes: out.start_bytes(),
                destination: MatmulDestination::Buffer,
            },
        )
    }

    /// The product of `lhs` and `rhs` handed to a sink whose lines are counted over `layout`, each
    /// of the output's extents with its stride, in the operands' element.
    pub fn sink(lhs: &PlainTensor, rhs: &PlainTensor, layout: &Geometry) -> Self {
        Self::with_output(
            lhs,
            rhs,
            OutputLayout {
                shape: layout.shape(),
                strides: layout.strides(),
                element: lhs.elem,
                start_bytes: 0,
                destination: MatmulDestination::Sink,
            },
        )
    }

    fn with_output(lhs: &PlainTensor, rhs: &PlainTensor, out: OutputLayout<'_>) -> Self {
        let (lhs_shape, rhs_shape) = (lhs.shape(), rhs.shape());
        let rank = lhs_shape.num_dims();
        let held = |shape: &[usize]| shape[..rank - 2].iter().product();
        let start = |tensor: &PlainTensor| tensor.start_bytes();
        let layouts = MatmulOperands {
            lhs: MatmulLayout::new(&lhs.meta(), MatmulOperand::Lhs),
            rhs: MatmulLayout::new(&rhs.meta(), MatmulOperand::Rhs),
            out: MatmulLayout::new(&Metadata::new(out.shape, out.strides), MatmulOperand::Out),
        };
        // An operand stored in tiles is aligned to the tile it is stored in, which its strides,
        // being the tiles', do not state.
        let buffer_alignment = |tensor: &PlainTensor, layout: MatmulLayout| match tensor.is_tiled()
        {
            true => tensor.elem_alignment(),
            false => layout.buffer_alignment(&tensor.binding.shape, &tensor.binding.strides),
        };
        Self {
            batches: out.shape[..rank - 2].to_vec(),
            stated_rows: lhs_shape[rank - 2],
            n: rhs_shape[rank - 1],
            k: lhs_shape[rank - 1],
            elements: MatmulOperands {
                lhs: lhs.elem,
                rhs: rhs.elem,
                out: out.element,
            },
            batches_held: MatmulOperands {
                lhs: held(lhs_shape.as_slice()),
                rhs: held(rhs_shape.as_slice()),
                out: held(out.shape),
            },
            elem_alignments: MatmulOperands {
                lhs: lhs.elem_alignment(),
                rhs: rhs.elem_alignment(),
                out: ElemAlignment::new(out.shape, out.strides),
            },
            start_bytes: MatmulOperands {
                lhs: start(lhs),
                rhs: start(rhs),
                out: out.start_bytes,
            },
            layouts,
            buffer_alignments: MatmulOperands {
                lhs: buffer_alignment(lhs, layouts.lhs),
                rhs: buffer_alignment(rhs, layouts.rhs),
                out: layouts.out.buffer_alignment(out.shape, out.strides),
            },
            stored_tiles: MatmulOperands::all(None),
            destination: out.destination,
            inputs: MatmulOperands {
                lhs: StoredInput::plain(lhs.elem),
                rhs: StoredInput::plain(rhs.elem),
                out: StoredInput::plain(out.element),
            },
            index: None,
        }
    }

    /// One unbatched product of dense operands of `element` throughout, every one in rows: the
    /// problem a key of these extents describes. [`laid`](Self::laid) lays an input another way.
    pub fn dense(m: usize, n: usize, k: usize, element: ElemType) -> Self {
        let rows = MatmulOperands {
            lhs: ElemAlignment::dense_rows(k),
            rhs: ElemAlignment::dense_rows(n),
            out: ElemAlignment::dense_rows(n),
        };
        Self {
            batches: Vec::new(),
            stated_rows: m,
            n,
            k,
            elements: MatmulOperands::all(element),
            batches_held: MatmulOperands::all(1),
            elem_alignments: rows,
            // A key describes operands a launch allocates or views whole.
            start_bytes: MatmulOperands::all(0),
            layouts: MatmulOperands::all(MatmulLayout::Row),
            buffer_alignments: rows,
            stored_tiles: MatmulOperands::all(None),
            // A launch the table times allocates its output.
            destination: MatmulDestination::Buffer,
            inputs: MatmulOperands::all(StoredInput::plain(element)),
            index: None,
        }
    }

    /// This dense problem with `input` laid as `layout`, its alignments those of a dense buffer
    /// so laid.
    pub fn laid(mut self, input: MatmulOperand, layout: MatmulLayout) -> Self {
        let shape = input.dims().map(|dim| match dim {
            MatmulDim::M => self.stated_rows,
            MatmulDim::N => self.n,
            MatmulDim::K => self.k,
        });
        let (alignment, buffer_alignment) = match layout.dense_strides(shape) {
            Some(strides) => (
                ElemAlignment::new(&shape, &strides),
                layout.buffer_alignment(&shape, &strides),
            ),
            None => (ElemAlignment::ONE, ElemAlignment::ONE),
        };
        self.layouts = self.layouts.with(input, layout);
        self.elem_alignments = self.elem_alignments.with(input, alignment);
        self.buffer_alignments = self.buffer_alignments.with(input, buffer_alignment);
        self
    }

    /// `rows` rows of `element` against one slab of `stack`, its slabs `[slabs, k, n]` at
    /// `slab_strides`, the rhs laid as `rhs_layout`: what one slab of a grouped or routed product
    /// states. The rows and the output are dense rows a launch allocates; the slab is aligned as a
    /// slab of the stack is, so a line never reads across two slabs, and read where the stack's
    /// view starts.
    pub fn slab(
        rows: usize,
        element: ElemType,
        stack: &PlainTensor,
        (slab_shape, slab_strides): ([usize; 3], [usize; 3]),
        rhs_layout: MatmulLayout,
    ) -> Self {
        let [_, k, n] = slab_shape;
        let dense = Self::dense(rows, n, k, element).laid(MatmulOperand::Rhs, rhs_layout);
        Self {
            elements: MatmulOperands {
                rhs: stack.elem,
                ..dense.elements
            },
            elem_alignments: MatmulOperands {
                rhs: ElemAlignment::new(&slab_shape, &slab_strides),
                ..dense.elem_alignments
            },
            start_bytes: MatmulOperands {
                rhs: stack.start_bytes(),
                ..dense.start_bytes
            },
            buffer_alignments: MatmulOperands {
                rhs: rhs_layout.buffer_alignment(&slab_shape, &slab_strides),
                ..dense.buffer_alignments
            },
            ..dense
        }
    }

    /// How each input is packed, where its values are codes: `None` for an input of elements.
    pub fn packings(&self) -> MatmulOperands<Option<Packing>> {
        self.inputs.map(|_, input| input.packing())
    }

    /// The element a packed operand's words unpack into, so nothing rounds before the sum.
    pub const PACKED: ElemType = ElemType::Float(cubecl::ir::FloatKind::F32);

    /// `m` rows by `n` columns, `k` deep, each operand of `element` or packed as `packings` says,
    /// a packed one read at [`PACKED`](Self::PACKED): one unbatched product, its rows every row
    /// of the lhs, its leading dims merged, since each of them reads the same rhs. Its
    /// operands are whole and dense, as a key describes them; the packed routines read none of
    /// a packed operand's buffer facts.
    pub fn packed(
        m: usize,
        n: usize,
        k: usize,
        element: ElemType,
        packings: MatmulOperands<Option<Packing>>,
    ) -> Self {
        let dense = MatmulOperands {
            lhs: ElemAlignment::dense_rows(k),
            rhs: ElemAlignment::dense_rows(n),
            out: ElemAlignment::dense_rows(n),
        };
        let alignments = dense.map(|operand, alignment| match packings.get(operand) {
            Some(_) => ElemAlignment::ONE,
            None => alignment,
        });
        Self {
            batches: Vec::new(),
            stated_rows: m,
            n,
            k,
            elements: packings.map(|_, packing| match packing {
                Some(_) => Self::PACKED,
                None => element,
            }),
            batches_held: MatmulOperands::all(1),
            elem_alignments: alignments,
            start_bytes: MatmulOperands::all(0),
            layouts: packings.map(|operand, packing| match packing {
                Some(packing) => MatmulLayout::of_packing(&packing, operand),
                None => MatmulLayout::Row,
            }),
            buffer_alignments: alignments,
            stored_tiles: MatmulOperands::all(None),
            destination: MatmulDestination::Buffer,
            inputs: packings.map(|_, packing| match packing {
                Some(packing) => StoredInput::packed(packing, element),
                None => StoredInput::plain(element),
            }),
            index: None,
        }
    }

    /// `m` rows of an NVFP4 lhs against `n` of an NVFP4 rhs, `k` deep: the
    /// [`dense`](Self::dense) problem, its operands stored in `e2m1` and its output in `out`.
    pub fn nvfp4(m: usize, n: usize, k: usize, out: ElemType) -> Self {
        let e2m1 = ElemType::Float(cubecl::ir::FloatKind::E2M1);
        Self {
            elements: MatmulOperands {
                lhs: e2m1,
                rhs: e2m1,
                out,
            },
            ..Self::dense(m, n, k, out).laid(MatmulOperand::Rhs, MatmulLayout::Col)
        }
    }

    /// `operand`'s alignment bound in `order`: over its buffer's order, or as the product names
    /// it. Which order a launch binds it in is its microkernel's choice
    /// ([`MatmulMicrokernel::orders`](cubek_microkernel::matmul::MatmulMicrokernel::orders)).
    pub fn alignment(&self, operand: MatmulOperand, order: BindingOrder) -> ElemAlignment {
        match order {
            BindingOrder::InBufferOrder => self.buffer_alignments.get(operand),
            BindingOrder::Stated => self.elem_alignments.get(operand),
        }
    }

    /// What each operand asks of a read of it, bound in the order `orders` says: to lie inside a
    /// run its layout aligns; a view starting partway into its buffer, to line up with where it
    /// starts; and an input a table indexes along its innermost dimension — a run of `K` read
    /// along a row — to bring one value, which an entry places.
    pub fn require_reads(
        &self,
        orders: MatmulOperands<BindingOrder>,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        for operand in MatmulOperand::ALL {
            vectorization.require(VectorRequirement::Aligned {
                operand,
                alignment: self.alignment(operand, orders.get(operand)),
            });
            vectorization.require(VectorRequirement::Aligned {
                operand,
                alignment: ElemAlignment::at_start(
                    self.start_bytes.get(operand),
                    self.elements.get(operand).size(),
                ),
            });
            let innermost = self.dimensions().of(orders.innermost(operand));
            let indexed_along_a_line = self
                .placements()
                .get_ref(operand)
                .as_ref()
                .is_some_and(|indexing| indexing.dimension() == innermost);
            if indexed_along_a_line {
                vectorization.require(VectorRequirement::Elementwise { operand });
            }
        }
    }

    /// Where each operand's buffer starts and how it is aligned over the order `orders` binds it
    /// in ([`alignment`](Self::alignment)).
    pub fn buffers(&self, orders: MatmulOperands<BindingOrder>) -> MatmulBuffers {
        MatmulBuffers {
            alignments: orders.map(|operand, order| self.alignment(operand, order)),
            start_bytes: self.start_bytes,
            stored_tiles: self.stored_tiles,
        }
    }

    /// Whether `operand`, bound in `order`, may be rows one element wide, more than one of them:
    /// a `[k, 1]` rhs, or its `[m, 1]` output. Each row steps to the next by the one element it
    /// holds, a stride no tensor map takes, while its alignment, which a dim of one element leaves
    /// out, reads as the column it is.
    ///
    /// Read as stated, those are its columns. Read in its buffer's order, a col operand's two
    /// dims swap only where its strides say the second is the coarser, which a dense `[k, 1]`
    /// buffer, both strides one, does not: so either extent of one counts there.
    pub fn rows_one_element_wide(&self, operand: MatmulOperand, order: BindingOrder) -> bool {
        let [rows, columns] = match operand {
            MatmulOperand::Lhs => [self.stated_rows, self.k],
            MatmulOperand::Rhs => [self.k, self.n],
            MatmulOperand::Out => [self.stated_rows, self.n],
        };
        let one_wide = |rows: usize, columns: usize| columns == 1 && rows > 1;
        match (order, self.layouts.get(operand)) {
            (BindingOrder::InBufferOrder, MatmulLayout::Col) => {
                one_wide(rows, columns) || one_wide(columns, rows)
            }
            _ => one_wide(rows, columns),
        }
    }

    /// The element the product sums in, whatever it reads and writes: an integer product its own
    /// integer, which sums exactly, and a float one `f32`. An output its cubes add into is held in
    /// it too.
    pub fn sum(&self) -> ElemType {
        match self.elements.lhs {
            integer @ ElemType::Int(_) => integer,
            _ => ElemType::Float(FloatKind::F32),
        }
    }

    /// This problem of flat rows restated as `row_groups` batches of `max_rows` rows, each a row group of
    /// the lhs and of the output ([`MatmulIndex::GroupedRows`]).
    pub fn batched_by_row_groups(self, max_rows: NonZeroUsize, row_groups: Vec<usize>) -> Self {
        let held = row_groups.iter().product();
        Self {
            batches: row_groups,
            stated_rows: max_rows.get(),
            batches_held: MatmulOperands {
                lhs: held,
                rhs: 1,
                out: held,
            },
            ..self
        }
    }

    /// The dimensions the product is stated over: matmul's own, or those inside a packed input's
    /// blocks.
    pub fn dimensions(&self) -> MatmulDimensions {
        match self.packings() == MatmulOperands::all(None) {
            true => MatmulDimensions::MNK,
            false => MatmulDimensions::INSIDE_BLOCKS,
        }
    }

    /// Whether a table ends row groups ([`MatmulIndex::GroupedRows`]), which no level cuts.
    pub fn grouped_rows(&self) -> bool {
        matches!(self.index, Some(MatmulIndex::GroupedRows { .. }))
    }

    /// How the table places each operand, none for one that reads no table.
    pub fn placements(&self) -> MatmulOperands<Option<Indexing>> {
        match self.index {
            Some(index) => index.placements(self.dimensions().m),
            None => MatmulOperands {
                lhs: None,
                rhs: None,
                out: None,
            },
        }
    }

    /// Every batch dimension, one per leading dim: what an operand's leading dims bind to.
    pub fn batch_dimensions(&self) -> Vec<Dimension> {
        Self::batch_dimensions_of(self.batches.len())
    }

    /// The dimensions `dims` batch dims are labelled with, outermost first: past `M`, `N` and `K`.
    pub fn batch_dimensions_of(dims: usize) -> Vec<Dimension> {
        (0..dims).map(batch_dimension).collect()
    }
}

/// What a problem reads of its output: the layout over the product's dimensions, the element, where
/// its view starts, and where it goes.
struct OutputLayout<'a> {
    shape: &'a [usize],
    strides: &'a [usize],
    element: ElemType,
    start_bytes: u64,
    destination: MatmulDestination,
}

impl EstimatedProblem for MatmulRoutineProblem {
    /// Two operations per multiply-add of every batch, and every operand's elements in the batches
    /// it holds, a broadcast one counted once; a packed one as its words and block scales.
    fn demand(&self) -> Work {
        let batches = self.batches.iter().product::<usize>();
        let values = MatmulOperands {
            lhs: self.stated_rows * self.k,
            rhs: self.k * self.n,
            out: self.stated_rows * self.n,
        };
        let bytes = values.map(|operand, values| {
            let held = self.inputs.get(operand).bytes(values);
            self.batches_held.get(operand) * held
        });
        Work {
            compute_ops: 2 * batches * self.stated_rows * self.n * self.k,
            bytes: bytes.lhs + bytes.rhs + bytes.out,
        }
    }
}

/// What a routine whose kernel reads no packed operand declines: a packed one, which the packed
/// routines read.
pub const MATMUL_PACKED_UNREAD: Decline = Decline::Unserved {
    why: "this routine reads float operands; a packed one runs on the packed routines",
};

/// What a routine whose planes cut the rows declines: grouped rows, which no level cuts.
pub const MATMUL_GROUPED_ROWS_UNREAD: Decline = Decline::Unserved {
    why: "this routine's planes cut the rows, which a table ending row groups places whole",
};

/// What a launch declines whatever the device: a cube box that is not as tall as a row group, which
/// no level cuts.
pub const MATMUL_ROW_GROUP_NOT_A_BOX: Decline = Decline::Unserved {
    why: "a row group is held whole by a box as tall as `max_rows`, which no level cuts",
};

/// What a launch declines whatever the device: grouped rows under a swizzled cube order, which
/// has no cut rows to put back.
pub const MATMUL_GROUPED_ROWS_SWIZZLED: Decline = Decline::Unserved {
    why: "grouped rows are not cut among the cubes, so a swizzled cube order has nothing to permute",
};

/// What a launch declines whatever the device: grouped rows over an lhs stored in tiles, which a
/// row group is not placed over.
pub const MATMUL_GROUPED_ROWS_OVER_TILED_LHS: Decline = Decline::Unserved {
    why: "grouped rows are read in rows, which an lhs stored in tiles is not",
};

/// What a launch declines whatever the device: grouped rows written anywhere but a buffer.
pub const MATMUL_GROUPED_ROWS_BUFFER_ONLY: Decline = Decline::Unserved {
    why: "grouped rows are written into a buffer, which masks their end; a sink or an accumulating destination cannot",
};

/// What a routine whose kernel walks no indexed input declines: a product whose inputs read a
/// table, which the routines walking within their inputs serve.
pub const MATMUL_INDEX_UNREAD: Decline = Decline::Unserved {
    why: "this routine's kernel walks no indexed input; a routed or grouped product runs on the \
          routines that walk within it",
};

/// What a fill the TMA engine delivers declines: a product whose inputs read a table, which a
/// tensor map, addressing its tensor whole, does not.
pub const MATMUL_INDEX_UNREAD_BY_TMA: Decline = Decline::Unserved {
    why: "a tensor map reads no table; an indexed input is staged by the cube's units",
};

/// What a launch taking its block scales as words copied by the units declines: scales kept in
/// tiles of their own ([`TileOrder::scales_in_tiles`]). The units bind the words from the rows a
/// matrix in rows keeps them in; only the TMA engine lands the tiles whole.
pub const MATMUL_SCALE_WORDS_IN_TILES: Decline = Decline::Unserved {
    why: "this launch's units copy its block scales as words of their rows, and this matrix keeps \
          them in tiles of their own, which only the TMA engine lands",
};

/// What a fill or a store the TMA engine moves declines: an operand of rows one element wide
/// ([`rows_one_element_wide`](MatmulRoutineProblem::rows_one_element_wide)), whose rows a tensor
/// map cannot step between, so the cube's units move them.
pub const MATMUL_ONE_ELEMENT_ROWS_UNMOVED_BY_TMA: Decline = Decline::Unserved {
    why: "a tensor map steps between rows by whole 16-byte lines, and this operand's rows are one \
          element wide; the cube's units move them",
};

/// What a launch over a packed input declines where its levels reach past the end of `K`: the
/// input is bound a whole read wide along `K`, and no mask checks a vector line of its words.
pub const MATMUL_PACKED_WALK_PAST_K: Decline = Decline::Unserved {
    why: "a packed input is read a whole read wide along k, which no mask checks, and this \
          launch's levels reach past the end of k",
};

/// What the hypercube routines decline: a packed lhs. They read a packed rhs beside a float lhs;
/// two packed inputs contract on the block-scaled instruction, whose operands only the cube
/// staged routine binds, so a hypercube launch over them would fail to expand.
pub const MATMUL_HYPERCUBE_PACKED_LHS: Decline = Decline::Unserved {
    why: "the hypercube routines read a packed rhs beside a float lhs, and this lhs is packed",
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{MatmulBatching, MatmulShape, OperandKey};
    use cubecl::ir::FloatKind;
    use cubecl::zspace::metadata::Metadata;
    use cubek_std::matmul_dimensions::{K, N};

    /// The problem a key describes states the layout and alignments of the tensors that key was
    /// made from.
    #[test]
    fn a_key_states_the_layout_of_its_tensors() {
        let (m, k, n) = (128, 64, 48);
        let shape = |rhs| MatmulShape {
            m,
            n,
            k,
            batch: 1,
            elem: ElemType::Float(FloatKind::F16),
            lhs: OperandKey::float(Some(K), MatmulBatching::Batched),
            rhs: OperandKey::float(rhs, MatmulBatching::Shared),
        };
        let cases: [(Option<Dimension>, [usize; 2]); 3] =
            [(Some(K), [1, k]), (Some(N), [n, 1]), (None, [2, 2 * k])];
        for (dimension, strides) in cases {
            let problem = MatmulRoutineProblem::from(&shape(dimension));
            assert_eq!(
                problem.layouts.rhs,
                MatmulLayout::new(&Metadata::new([k, n], strides), MatmulOperand::Rhs),
                "{dimension:?}"
            );
            assert_eq!(
                problem.elem_alignments.rhs,
                ElemAlignment::new(&[k, n], &strides),
                "{dimension:?}"
            );
        }
    }

    /// A shared rhs is held once, however many batches read it; a batched one, by every batch.
    /// Only the rhs's count moves.
    #[test]
    fn a_shared_rhs_is_held_once() {
        let shape = |batching| MatmulShape {
            m: 16,
            n: 48,
            k: 64,
            batch: 4,
            elem: ElemType::Float(FloatKind::F16),
            lhs: OperandKey::float(Some(K), MatmulBatching::Batched),
            rhs: OperandKey::float(Some(N), batching),
        };
        let shared = MatmulRoutineProblem::from(&shape(MatmulBatching::Shared));
        let batched = MatmulRoutineProblem::from(&shape(MatmulBatching::Batched));
        assert_eq!(shared.batches_held.rhs, 1);
        assert_eq!(batched.batches_held.rhs, 4);
        assert_eq!(
            MatmulRoutineProblem {
                batches_held: batched.batches_held,
                ..shared
            },
            batched
        );
    }

    /// A col rhs's buffer steps along `K`, so its alignment in the buffer's order is not the one
    /// the product names; a row rhs's buffer steps as the product names it, so the two agree.
    #[test]
    fn the_rhs_aligns_in_the_order_it_is_bound() {
        let f16 = ElemType::Float(FloatKind::F16);
        let rhs = MatmulOperand::Rhs;
        let col =
            MatmulRoutineProblem::dense(4, 48, 64, f16).laid(MatmulOperand::Rhs, MatmulLayout::Col);
        assert_eq!(
            col.alignment(rhs, BindingOrder::InBufferOrder),
            col.buffer_alignments.rhs
        );
        assert_ne!(
            col.alignment(rhs, BindingOrder::InBufferOrder),
            col.alignment(rhs, BindingOrder::Stated)
        );
        let row = MatmulRoutineProblem::dense(4, 48, 64, f16);
        assert_eq!(
            row.alignment(rhs, BindingOrder::InBufferOrder),
            row.alignment(rhs, BindingOrder::Stated)
        );
    }

    /// Where the lines lie is read off the strides, so an rhs with no single
    /// lined dimension has no problem at all — it is not a col rhs and not a row
    /// rhs, and there is no third tiling to sort it into.
    ///
    /// Neither stride one: an rhs narrowed along both dims, which no line walks.
    /// Both stride one at full extents: the two dims alias, and the addressability
    /// check refuses what the contiguity read alone could not. And a stride below
    /// the row it steps over is a narrowed or aliasing view this binding cannot
    /// describe.
    #[test]
    fn an_rhs_with_no_single_lined_dimension_has_no_problem() {
        for strides in [[2, 8192], [1, 1], [1, 4092]] {
            assert!(
                MatmulLayout::lines([K, N], [4096, 4096], strides).is_none(),
                "{strides:?}"
            );
        }
        assert_eq!(
            MatmulLayout::lines([K, N], [4096, 4096], [1, 4096 + 4]),
            Some((K, 4096 + 4)),
            "padding is served: the launch binds the real stride"
        );
        assert_eq!(
            MatmulLayout::lines([K, N], [4096, 4096], [4096, 1]),
            Some((N, 4096))
        );
    }
}
