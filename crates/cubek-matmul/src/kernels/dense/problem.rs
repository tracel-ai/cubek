use crate::kernels::tensor::PlainTensor;
use cubecl::ir::ElemType;
use cubecl::tile::Geometry;
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Axis, Indexing};
use cubecl::tune::Work;
use cubecl::zspace::metadata::Metadata;
use cubek_microkernel::matmul::stored::Packing;
use cubek_microkernel::matmul::{
    MatmulBuffers, MatmulContraction, MatmulDestination, MatmulLayout, MatmulOperand,
    MatmulOperands, MatmulOrder, MatmulStoredTile,
};
use cubek_std::matmul_axes::{EXPERT, K, batch_axis};
use cubek_std::routine::EstimatedProblem;

/// `out[b] = lhs[b] · rhs[b]` for every batch `b`, as the host describes it, each input float or
/// packed ([`packings`](Self::packings)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulRoutineProblem {
    /// The output's batch extents, outermost first. An operand of extent one there is broadcast.
    pub batches: Vec<usize>,
    /// The lhs's rows as the problem was stated, which mean one batch's rows for a problem read
    /// off a tuning key, and every row, its batches already folded, for one read off tensors that
    /// hold every row whole. A launch reads it through
    /// [`MatmulView::rows`](crate::kernels::MatmulView::rows); only what is right for both forms
    /// reads it as stated: a problem that states no batches (the grouped and routed cases), and the
    /// demand estimate, whose batches multiply it either way.
    pub stated_rows: usize,
    pub n: usize,
    pub k: usize,
    /// The element each operand is read at: a float operand's stored one, a packed one's the
    /// element its words unpack into ([`packings`](Self::packings) says how it is stored).
    pub elements: MatmulOperands<ElemType>,
    /// The elements the instruction multiplies them at, where one is not its stored one.
    pub contraction: MatmulContraction,
    /// Batches each operand holds: the output's, or fewer where an operand is broadcast.
    pub batches_held: MatmulOperands<usize>,
    /// The alignment of each operand's buffer in elements, over the axes the product names: the
    /// line widths a launch may bind it at, before the device or any component narrows them.
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
    /// How each operand is stored where it is packed: its words, its scales and their scheme.
    /// `None` for a float operand, read as stored.
    pub packings: MatmulOperands<Option<Packing>>,
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
}

impl MatmulIndex {
    /// The inputs that read the table.
    pub fn readers(&self) -> MatmulOperands<bool> {
        match self {
            Self::Slab { .. } => MatmulOperands {
                lhs: false,
                rhs: true,
                out: false,
            },
            Self::RunOfK => MatmulOperands {
                lhs: true,
                rhs: true,
                out: false,
            },
        }
    }

    /// What an entry places in the input that reads it.
    pub fn indexing(&self) -> Indexing {
        match self {
            Self::Slab { .. } => Indexing::Coordinate(EXPERT),
            Self::RunOfK => Indexing::Run(K),
        }
    }

    /// The axis the index adds to the product's space, with its extent: the stack's slabs, which
    /// an entry picks one of.
    pub fn axis(&self) -> Option<(Axis, usize)> {
        match self {
            Self::Slab { slabs } => Some((EXPERT, *slabs)),
            Self::RunOfK => None,
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
    pub fn check_plain(&self) -> Result<(), Refusal> {
        match self.packings == MatmulOperands::all(None) {
            true => Ok(()),
            false => Err(MATMUL_PACKED_UNREAD),
        }
    }

    /// Checks no input reads a table: what a routine whose kernel walks no indexed input
    /// declines first.
    ///
    /// # Errors
    ///
    /// [`MATMUL_INDEX_UNREAD`], for a product with a table.
    pub fn check_unindexed(&self) -> Result<(), Refusal> {
        match self.index {
            None => Ok(()),
            Some(_) => Err(MATMUL_INDEX_UNREAD),
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
            contraction: MatmulContraction::AsStored,
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
            packings: MatmulOperands::all(None),
            index: None,
        }
    }

    /// One unbatched product of dense operands of `element` throughout, the weight laid as
    /// `rhs_layout`: the problem a key of these extents describes.
    pub fn dense(
        m: usize,
        n: usize,
        k: usize,
        element: ElemType,
        rhs_layout: MatmulLayout,
    ) -> Self {
        let rhs_shape = [k, n];
        let (rhs_alignment, rhs_buffer_alignment) = match rhs_layout.dense_strides(rhs_shape) {
            Some(strides) => (
                ElemAlignment::new(&rhs_shape, &strides),
                rhs_layout.buffer_alignment(&rhs_shape, &strides),
            ),
            None => (ElemAlignment::ONE, ElemAlignment::ONE),
        };
        Self {
            batches: Vec::new(),
            stated_rows: m,
            n,
            k,
            contraction: MatmulContraction::AsStored,
            elements: MatmulOperands {
                lhs: element,
                rhs: element,
                out: element,
            },
            batches_held: MatmulOperands {
                lhs: 1,
                rhs: 1,
                out: 1,
            },
            elem_alignments: MatmulOperands {
                lhs: ElemAlignment::dense_rows(k),
                rhs: rhs_alignment,
                out: ElemAlignment::dense_rows(n),
            },
            // A key describes operands a launch allocates or views whole.
            start_bytes: MatmulOperands {
                lhs: 0,
                rhs: 0,
                out: 0,
            },
            layouts: MatmulOperands {
                rhs: rhs_layout,
                ..MatmulOperands::all(MatmulLayout::Row)
            },
            buffer_alignments: MatmulOperands {
                lhs: ElemAlignment::dense_rows(k),
                rhs: rhs_buffer_alignment,
                out: ElemAlignment::dense_rows(n),
            },
            stored_tiles: MatmulOperands::all(None),
            // A launch the table times allocates its output.
            destination: MatmulDestination::Buffer,
            packings: MatmulOperands::all(None),
            index: None,
        }
    }

    /// `rows` rows of `element` against one slab of `stack`, its slabs `[slabs, k, n]` at
    /// `slab_strides`, the weight laid as `rhs_layout`: what one slab of a grouped or routed product
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
        let dense = Self::dense(rows, n, k, element, rhs_layout);
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

    /// The element a packed operand's words unpack into, so nothing rounds before the sum.
    pub const PACKED: ElemType = ElemType::Float(cubecl::ir::FloatKind::F32);

    /// `m` rows by `n` columns, `k` deep, each operand of `element` or packed as `packings` says,
    /// a packed one read at [`PACKED`](Self::PACKED): one unbatched product, its rows every row
    /// of the activation, its leading dims folded, since each of them reads the same weight. Its
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
            contraction: MatmulContraction::AsStored,
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
            packings,
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
            ..Self::dense(m, n, k, out, MatmulLayout::Col)
        }
    }

    /// `operand`'s alignment bound in `order`: over its buffer's order, or as the product names
    /// it. Which order a launch binds it in is its microkernel's choice
    /// ([`MatmulMicrokernel::orders`](cubek_microkernel::matmul::MatmulMicrokernel::orders)).
    pub fn alignment(&self, operand: MatmulOperand, order: MatmulOrder) -> ElemAlignment {
        match order {
            MatmulOrder::InBufferOrder => self.buffer_alignments.get(operand),
            MatmulOrder::Stated => self.elem_alignments.get(operand),
        }
    }

    /// Where each operand's buffer starts and how it is aligned over the order `orders` binds it
    /// in ([`alignment`](Self::alignment)).
    pub fn buffers(&self, orders: MatmulOperands<MatmulOrder>) -> MatmulBuffers {
        MatmulBuffers {
            alignments: orders.map(|operand, order| self.alignment(operand, order)),
            start_bytes: self.start_bytes,
            stored_tiles: self.stored_tiles,
        }
    }

    /// Every batch axis, one per leading dim: what an operand's leading dims bind to.
    pub fn batch_axes(&self) -> Vec<Axis> {
        Self::batch_axes_of(self.batches.len())
    }

    /// The axes `dims` batch dims are labelled with, outermost first: past `M`, `N` and `K`.
    pub fn batch_axes_of(dims: usize) -> Vec<Axis> {
        (0..dims).map(batch_axis).collect()
    }
}

/// What a problem reads of its output: the layout over the product's axes, the element, where its
/// view starts, and where it goes.
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
            let held = match self.packings.get(operand) {
                Some(packing) => packing.bytes(values),
                None => values * self.elements.get(operand).size(),
            };
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
pub const MATMUL_PACKED_UNREAD: Refusal = Refusal::Declined {
    why: "this routine reads float operands; a packed one runs on the packed routines",
};

/// What a routine whose kernel walks no indexed input declines: a product whose inputs read a
/// table, which the routines walking within their inputs serve.
pub const MATMUL_INDEX_UNREAD: Refusal = Refusal::Declined {
    why: "this routine's kernel walks no indexed input; a routed or grouped product runs on the \
          routines that walk within it",
};

/// What a fill the TMA engine delivers declines: a product whose inputs read a table, which a
/// tensor map, addressing its tensor whole, does not.
pub const MATMUL_INDEX_UNREAD_BY_TMA: Refusal = Refusal::Declined {
    why: "a tensor map reads no table; an indexed input is staged by the cube's units",
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::tune::key::{MatmulBatching, MatmulShape, OperandKey};
    use cubecl::ir::FloatKind;
    use cubecl::zspace::metadata::Metadata;
    use cubek_std::matmul_axes::{K, N};

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
        let cases: [(Option<Axis>, [usize; 2]); 3] =
            [(Some(K), [1, k]), (Some(N), [n, 1]), (None, [2, 2 * k])];
        for (axis, strides) in cases {
            let problem = MatmulRoutineProblem::from(&shape(axis));
            assert_eq!(
                problem.layouts.rhs,
                MatmulLayout::new(&Metadata::new([k, n], strides), MatmulOperand::Rhs),
                "{axis:?}"
            );
            assert_eq!(
                problem.elem_alignments.rhs,
                ElemAlignment::new(&[k, n], &strides),
                "{axis:?}"
            );
        }
    }

    /// A shared weight is held once, however many batches read it; a batched one, by every batch.
    /// Only the weight's count moves.
    #[test]
    fn a_shared_weight_is_held_once() {
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

    /// A col weight's buffer steps along `K`, so its alignment in the buffer's order is not the one
    /// the product names; a row weight's buffer steps as the product names it, so the two agree.
    #[test]
    fn the_rhs_aligns_in_the_order_it_is_bound() {
        let f16 = ElemType::Float(FloatKind::F16);
        let rhs = MatmulOperand::Rhs;
        let col = MatmulRoutineProblem::dense(4, 48, 64, f16, MatmulLayout::Col);
        assert_eq!(
            col.alignment(rhs, MatmulOrder::InBufferOrder),
            col.buffer_alignments.rhs
        );
        assert_ne!(
            col.alignment(rhs, MatmulOrder::InBufferOrder),
            col.alignment(rhs, MatmulOrder::Stated)
        );
        let row = MatmulRoutineProblem::dense(4, 48, 64, f16, MatmulLayout::Row);
        assert_eq!(
            row.alignment(rhs, MatmulOrder::InBufferOrder),
            row.alignment(rhs, MatmulOrder::Stated)
        );
    }

    /// Where the lines lie is read off the strides, so a weight with no single
    /// lined axis has no problem at all — it is not a col weight and not a row
    /// weight, and there is no third tiling to sort it into.
    ///
    /// Neither stride one: a weight narrowed along both dims, which no line walks.
    /// Both stride one at full extents: the two dims alias, and the addressability
    /// check refuses what the contiguity read alone could not. And a stride below
    /// the row it steps over is a narrowed or aliasing view this binding cannot
    /// describe.
    #[test]
    fn a_weight_with_no_single_lined_axis_has_no_problem() {
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
