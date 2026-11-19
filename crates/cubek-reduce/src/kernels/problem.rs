//! The problem a routine computes.

use cubecl::ir::ElemType;
use cubecl::prelude::TensorBinding;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Dimension, Partitioning, Space};
use cubecl::tune::Work;
use cubek_microkernel::reduce::{ReduceLayout, ReduceOperation, ReduceWorkItem};
use cubek_std::routine::EstimatedProblem;

/// The rows: every axis of the input but the reduced one, as one dimension, the first axis
/// coarsest.
pub const BATCH: Dimension = Dimension(0);
/// The axis a row runs along, which the reduction reduces away.
pub const REDUCED: Dimension = Dimension(1);
/// The cells a row writes, `k` for a top-k: the output's dimension in place of the reduced one.
pub const SLOTS: Dimension = Dimension(2);
/// A hypercube's partials, every part's carrier for a row side by side: the partials' dimension in
/// place of the reduced one.
pub const PARTIALS: Dimension = Dimension(3);

/// The elements of a reduction: what the input is stored in, what it accumulates in, and what the
/// output is written in (an index's, where it writes indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ReduceElements {
    pub input: ElemType,
    pub accumulator: ElemType,
    pub out: ElemType,
}

/// `out = operation(input)` along axis `axis` of an input of `shape`, as the host describes it:
/// the output has the input's shape, with `axis` one wide (`k` wide for a top-k).
///
/// Its space has the same dimensions whatever the rank: [`BATCH`], every axis but `axis` as one,
/// [`REDUCED`], `axis` itself, and [`SLOTS`], the cells a row writes, which the output holds in
/// place of `axis`. Each operand's layout maps its axes onto them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReduceRoutineProblem {
    pub shape: Vec<usize>,
    pub axis: usize,
    pub operation: ReduceOperation,
    pub elements: ReduceElements,
    /// The axis both operands step one element along, if they share one: what a line runs along.
    pub contiguous: Option<usize>,
    /// The alignment of the input's and the output's buffers in elements.
    pub elem_alignments: [ElemAlignment; 2],
    /// The element of the indices written beside the values, to a second output laid out as
    /// the first, where they are.
    pub indices: Option<ElemType>,
}

impl ReduceRoutineProblem {
    /// `binding`, an operand of the input's axes holding `along` in place of the reduced axis
    /// ([`REDUCED`] for the input, [`SLOTS`] for the output and its indices, [`PARTIALS`] for a
    /// hypercube's partials), with the dimension each of its axes carries: `along` for the reduced
    /// axis, [`BATCH`] for every other, so [`BATCH`] spans the row axes, the first coarsest, on
    /// either side of the reduced one. Neighbouring row axes that step as one axis (the outer's
    /// stride the inner's whole span) merge into one, so a contiguous operand's rows before and
    /// after the reduced axis are one axis each. An operand of one axis is given a row axis of one.
    pub fn labelled(
        &self,
        mut binding: TensorBinding,
        along: Dimension,
    ) -> (TensorBinding, Vec<Dimension>) {
        let mut axes: Vec<(Dimension, (usize, usize))> = Vec::new();
        for d in 0..binding.shape.len() {
            let dimension = match d == self.axis {
                true => along,
                false => BATCH,
            };
            let axis = (binding.shape[d], binding.strides[d]);
            let merged = match axes.last() {
                Some(&(BATCH, outer)) if dimension == BATCH => merged_rows(outer, axis),
                _ => None,
            };
            match merged {
                Some(merged) => axes.last_mut().expect("a merged axis has an outer one").1 = merged,
                None => axes.push((dimension, axis)),
            }
        }
        if !axes.iter().any(|&(dimension, _)| dimension == BATCH) {
            axes.insert(0, (BATCH, (1, 1)));
        }
        binding.shape = axes.iter().map(|&(_, (extent, _))| extent).collect();
        binding.strides = axes.iter().map(|&(_, (_, stride))| stride).collect();
        (
            binding,
            axes.into_iter().map(|(dimension, _)| dimension).collect(),
        )
    }

    /// Values a row reduces.
    pub fn reduced_extent(&self) -> usize {
        self.shape[self.axis]
    }

    /// Rows reduced.
    pub fn row_count(&self) -> usize {
        self.shape.iter().product::<usize>() / self.reduced_extent().max(1)
    }

    /// How the lines lie: along the reduced axis where it is the contiguous one, across it where
    /// another is, each lane of a line its own row; along it, a value a line, where none is.
    pub fn layout(&self) -> ReduceLayout {
        match self.contiguous {
            Some(axis) if axis != self.axis => ReduceLayout::Across,
            _ => ReduceLayout::Along,
        }
    }

    /// What every microkernel of this problem is handed.
    pub fn work_item(&self) -> ReduceWorkItem {
        ReduceWorkItem {
            operation: self.operation,
            input: self.elements.input,
            accumulator: self.elements.accumulator,
            layout: self.layout(),
            with_indices: self.indices.is_some(),
        }
    }

    /// Every dimension of the problem at its extent: what a routine partitions.
    pub fn space(&self) -> Space {
        Space::new(&[
            (BATCH, self.row_count()),
            (REDUCED, self.reduced_extent()),
            (SLOTS, self.operation.written_per_row()),
        ])
    }

    /// [`space`](Self::space) and a hypercube's partials, `partial_extent` values of each box of
    /// rows'.
    pub fn split_space(&self, partial_extent: usize) -> Space {
        Space::new(&[
            (BATCH, self.row_count()),
            (REDUCED, self.reduced_extent()),
            (SLOTS, self.operation.written_per_row()),
            (PARTIALS, partial_extent),
        ])
    }

    /// The box an owner holds: one row, or `width` rows side by side across the reduced axis, and
    /// the cells it writes. It leaves the reduced axis to the space, so every owner spans its rows
    /// whole.
    pub fn owner_box(&self, width: usize) -> Vec<(Dimension, usize)> {
        vec![
            (BATCH, self.box_rows(width)),
            (SLOTS, self.operation.written_per_row()),
        ]
    }

    /// How many boxes of rows the owners, `width` rows across the reduced axis, take.
    pub fn row_boxes(&self, width: usize) -> usize {
        self.row_count().div_ceil(self.box_rows(width))
    }

    /// Rows an owner holds: `width` side by side across the reduced axis, one along it.
    fn box_rows(&self, width: usize) -> usize {
        match self.layout() {
            ReduceLayout::Across => width,
            ReduceLayout::Along => 1,
        }
    }

    /// `partitioning` as a kernel compiles it: every dimension but the cells a row writes left to
    /// the launch, the row length too, as today's reduce kernels read it, so one kernel serves
    /// every shape of a rank, layout, elements and widths. An owner's window is one row high, which
    /// its flat view reads with no division by the runtime length.
    pub fn kernel(&self, partitioning: &Partitioning) -> Partitioning {
        let space = partitioning.space();
        let dynamic: Vec<Dimension> = space
            .dimensions()
            .filter(|dimension| *dimension != SLOTS)
            .collect();
        Partitioning::new(
            space.clone().with_dynamic(&dynamic),
            partitioning.levels().to_vec(),
        )
    }
}

/// Two neighbouring row axes as one, the outer `(extent, stride)` and the inner, where the outer
/// steps the inner's whole span or either is one wide.
fn merged_rows(outer: (usize, usize), inner: (usize, usize)) -> Option<(usize, usize)> {
    match (outer, inner) {
        ((1, _), _) => Some(inner),
        (_, (1, _)) => Some(outer),
        ((extent, stride), (inner_extent, inner_stride)) => {
            (stride == inner_extent * inner_stride).then_some((extent * inner_extent, inner_stride))
        }
    }
}

impl EstimatedProblem for ReduceRoutineProblem {
    /// Every value read once and accumulated once, every cell written once.
    fn demand(&self) -> Work {
        let values: usize = self.shape.iter().product();
        let cells = self.row_count() * self.operation.written_per_row();
        Work {
            compute_ops: values,
            bytes: values * self.elements.input.size() + cells * self.elements.out.size(),
        }
    }
}
