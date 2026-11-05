//! The problem a routine computes.

use cubecl::ir::ElemType;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Axis, Partitioning, Space};
use cubecl::tune::Work;
use cubek_microkernel::reduce::{ReduceLayout, ReduceOperation, ReduceWorkItem};
use cubek_std::routine::EstimatedProblem;

/// The elements of a reduction: what the input is stored in, what it accumulates in, and what the
/// output is written in (an index's, where it writes indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceElements {
    pub input: ElemType,
    pub accumulator: ElemType,
    pub out: ElemType,
}

/// `out = operation(input)` along dim `axis` of an input of `shape`, as the host describes it:
/// the output has the input's shape, with `axis` one wide (`k` wide for a top-k).
///
/// Every dim is an axis of the problem's space, `Axis(d)` for dim `d`; the reduced dim is the
/// reduced axis, and every other dim a row axis. The output's `axis` is its own,
/// [`slots`](Self::slots), since it holds the cells a row writes, not the values it read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReduceRoutineProblem {
    pub shape: Vec<usize>,
    pub axis: usize,
    pub operation: ReduceOperation,
    pub elements: ReduceElements,
    /// The dim both operands step one element along, if they share one: what a line runs along.
    pub contiguous: Option<usize>,
    /// The alignment of the input's and the output's buffers in elements.
    pub elem_alignments: [ElemAlignment; 2],
    /// The element of the indices written beside the values, to a second output laid out as
    /// the first, where they are.
    pub indices: Option<ElemType>,
}

impl ReduceRoutineProblem {
    /// The reduced axis.
    pub fn reduced(&self) -> Axis {
        dim_axis(self.axis)
    }

    /// The axis of the cells a row writes, in place of the reduced one in the output.
    pub fn slots(&self) -> Axis {
        dim_axis(self.shape.len())
    }

    /// The axis a hypercube's partials run along in place of the reduced one: every part's
    /// carrier for a row, side by side.
    pub fn partials(&self) -> Axis {
        dim_axis(self.shape.len() + 1)
    }

    /// Every row axis, outermost first.
    pub fn rows(&self) -> Vec<Axis> {
        (0..self.shape.len())
            .filter(|d| *d != self.axis)
            .map(dim_axis)
            .collect()
    }

    /// The axes the owners are placed along: the row axes, or the slots where there is none, a
    /// tensor of one dim reduced whole into its one row.
    pub fn owner_axes(&self) -> Vec<Axis> {
        match self.rows() {
            rows if rows.is_empty() => vec![self.slots()],
            rows => rows,
        }
    }

    /// The input's axes, one a dim.
    pub fn input_axes(&self) -> Vec<Axis> {
        (0..self.shape.len()).map(dim_axis).collect()
    }

    /// The output's axes, one a dim: the input's, the reduced dim's being the slots.
    pub fn out_axes(&self) -> Vec<Axis> {
        (0..self.shape.len())
            .map(|d| match d == self.axis {
                true => self.slots(),
                false => dim_axis(d),
            })
            .collect()
    }

    /// A hypercube's partials' axes, one a dim: the input's, the reduced dim's being the
    /// partials.
    pub fn partial_axes(&self) -> Vec<Axis> {
        (0..self.shape.len())
            .map(|d| match d == self.axis {
                true => self.partials(),
                false => dim_axis(d),
            })
            .collect()
    }

    /// Values a row reduces.
    pub fn reduced_extent(&self) -> usize {
        self.shape[self.axis]
    }

    /// Rows reduced.
    pub fn row_count(&self) -> usize {
        self.shape.iter().product::<usize>() / self.reduced_extent().max(1)
    }

    /// How the lines lie: along the reduced dim where it is the contiguous one, across it where
    /// another is, each lane of a line its own row; along it, a value a line, where none is.
    pub fn layout(&self) -> ReduceLayout {
        match self.contiguous {
            Some(dim) if dim != self.axis => ReduceLayout::Across,
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

    /// Every axis of the problem at its extent, the slots among them: what a routine partitions.
    pub fn space(&self) -> Space {
        let mut axes: Vec<(Axis, usize)> = self
            .shape
            .iter()
            .enumerate()
            .map(|(d, extent)| (dim_axis(d), *extent))
            .collect();
        axes.push((self.slots(), self.operation.written_per_row()));
        Space::new(&axes)
    }

    /// [`space`](Self::space) and a hypercube's partials, `partial_extent` values of each box of
    /// rows' along their axis.
    pub fn split_space(&self, partial_extent: usize) -> Space {
        let space = self.space();
        let mut axes: Vec<(Axis, usize)> = space
            .axes()
            .map(|axis| (axis, space.extent(axis)))
            .collect();
        axes.push((self.partials(), partial_extent));
        Space::new(&axes)
    }

    /// The box an owner holds: one row, or `width` rows side by side across the reduced dim, and
    /// the cells it writes. It leaves the reduced dim to the space, so every owner spans its rows
    /// whole.
    pub fn owner_box(&self, width: usize) -> Vec<(Axis, usize)> {
        let mut extents: Vec<(Axis, usize)> = (0..self.shape.len())
            .filter(|d| *d != self.axis)
            .map(|d| {
                let extent = match (self.layout(), self.contiguous) {
                    (ReduceLayout::Across, Some(inner)) if d == inner => width,
                    _ => 1,
                };
                (dim_axis(d), extent)
            })
            .collect();
        extents.push((self.slots(), self.operation.written_per_row()));
        extents
    }

    /// How many boxes of rows the owners, `width` rows across the reduced dim, take along each
    /// row axis, outermost first.
    pub fn row_boxes(&self, width: usize) -> Vec<(Axis, usize)> {
        let owner = self.owner_box(width);
        self.rows()
            .into_iter()
            .map(|axis| {
                let tile = owner
                    .iter()
                    .find_map(|&(a, extent)| (a == axis).then_some(extent))
                    .expect("the owner's box names every row axis");
                (axis, self.shape[usize::from(axis.0)].div_ceil(tile))
            })
            .collect()
    }

    /// `partitioning` as a kernel compiles it: every axis but the cells a row writes left to the
    /// launch, the row length too, as today's reduce kernels read it, so one kernel serves every
    /// shape of a rank, layout, elements and widths. An owner's window is one row high, which its
    /// flat view reads with no division by the runtime length.
    pub fn kernel(&self, partitioning: &Partitioning) -> Partitioning {
        let space = partitioning.space();
        let dynamic: Vec<Axis> = space.axes().filter(|axis| *axis != self.slots()).collect();
        Partitioning::new(
            space.clone().with_dynamic(&dynamic),
            partitioning.levels().to_vec(),
        )
    }
}

/// The axis of dim `d`.
pub fn dim_axis(d: usize) -> Axis {
    Axis(u8::try_from(d).expect("a tensor has fewer than 256 dims"))
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
