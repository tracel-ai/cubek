//! The problem a routine computes.

use cubecl::ir::{ElemType, FloatKind, IntKind, UIntKind};
use cubecl::prelude::TensorBinding;
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::BoundaryPolicy;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Dimension, Geometry, Partitioning, Space};
use cubecl::tune::Work;
use cubek_microkernel::reduce::{ReduceLayout, ReduceOperation, ReduceWorkItem};
use cubek_std::routine::EstimatedProblem;

/// The elements of a reduction: what the input is stored in, what it accumulates in, and what the
/// output is written in (an index's, where it writes indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ReduceElements {
    pub input: ElemType,
    pub accumulator: ElemType,
    pub out: ElemType,
}

impl ReduceElements {
    /// The elements a reduction of `input` by `operation` runs in. It writes `out` where it writes
    /// no value of the input (an index, or the flag of `Any` and `All`), and the input's element
    /// otherwise. A sum, a product or a mean accumulates in widened elements (`f32`, `i32` or `u32`
    /// for a narrower float or integer), every other operation in the input's own: an extremum
    /// gains nothing from a wider carrier, and an index or a flag is carried apart from the values.
    ///
    /// # Panics
    ///
    /// Where `operation` writes an index or a flag and `out` is `None`, or `input` is a boolean or
    /// a complex number.
    pub fn for_operation(
        operation: ReduceOperation,
        input: ElemType,
        out: Option<ElemType>,
    ) -> Self {
        let out = match operation {
            ReduceOperation::ArgMax
            | ReduceOperation::ArgMin
            | ReduceOperation::ArgTopK { .. }
            | ReduceOperation::Any
            | ReduceOperation::All => {
                out.expect("ArgMax, ArgMin, ArgTopK, Any and All write an element the caller names")
            }
            _ => input,
        };
        let accumulator = match operation {
            ReduceOperation::Sum | ReduceOperation::Prod | ReduceOperation::Mean => widened(input),
            _ => input,
        };
        Self {
            input,
            accumulator,
            out,
        }
    }
}

/// What a sum of `input`'s elements accumulates in: `f32` for a float narrower than 64 bits,
/// `i32` or `u32` for a narrower integer, the element itself otherwise.
///
/// # Panics
///
/// Where `input` is a boolean or a complex number, which no reduction sums.
fn widened(input: ElemType) -> ElemType {
    match input {
        ElemType::Float(FloatKind::F64)
        | ElemType::Int(IntKind::I64)
        | ElemType::UInt(UIntKind::U64)
        | ElemType::Index => input,
        ElemType::Float(_) => ElemType::Float(FloatKind::F32),
        ElemType::Int(_) => ElemType::Int(IntKind::I32),
        ElemType::UInt(_) => ElemType::UInt(UIntKind::U32),
        ElemType::Bool | ElemType::Complex(_) => panic!("{input:?} is not reduced"),
    }
}

/// `out = operation(input)` along axis `axis` of an input of `shape`, as the host describes it:
/// the output has the input's shape, with `axis` one wide (`k` wide for a top-k).
///
/// Every axis is a dimension of the problem's space, `Dimension(d)` for axis `d`; the reduced axis
/// is the reduced dimension, and every other axis a row dimension. The output's `axis` is its own,
/// `slots`, since it holds the cells a row writes, not the values it read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReduceRoutineProblem {
    /// The input's extent along each axis, read by axis (Position Usage).
    pub shape: Vec<usize>,
    pub axis: usize,
    pub operation: ReduceOperation,
    pub elements: ReduceElements,
    /// The axis both operands step one element along, if they share one: what a line runs along.
    pub contiguous: Option<usize>,
    /// The alignment in elements of the input's buffer, then the output's, read by that position
    /// (Position Usage).
    pub elem_alignments: [ElemAlignment; 2],
    /// The indices written beside the values, to a second output of their shape, where they are.
    pub indices: Option<ReduceIndices>,
}

/// The indices a reduction writes beside its values: their element, and the alignment of the
/// buffer they go to, which lies at its own strides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceIndices {
    pub element: ElemType,
    pub elem_alignment: ElemAlignment,
}

impl ReduceRoutineProblem {
    /// The problem `out = operation(input)` along axis `axis` states, `out` having `input`'s shape
    /// with `axis` as wide as the operation writes per row.
    ///
    /// # Panics
    ///
    /// Where `out`'s shape is not that.
    pub fn stated(
        input: &Geometry,
        out: &Geometry,
        axis: usize,
        operation: ReduceOperation,
        elements: ReduceElements,
    ) -> Self {
        let shape = input.shape().to_vec();
        let expected = with_axis_extent(&shape, axis, operation.written_per_row());
        assert_eq!(
            out.shape(),
            expected.as_slice(),
            "a reduction along axis {axis} writes the input's shape with that axis {} wide",
            operation.written_per_row()
        );
        // A line runs along the innermost axis where both operands step one element along it.
        let inner = shape.len() - 1;
        let contiguous = (steps_one(input, inner) && steps_one(out, inner)).then_some(inner);
        Self {
            elem_alignments: [input, out]
                .map(|geometry| ElemAlignment::new(geometry.shape(), geometry.strides())),
            shape,
            axis,
            operation,
            elements,
            contiguous,
            indices: None,
        }
    }

    /// The problem `out = operation(input)` along axis `axis` states, both read where they lie.
    ///
    /// # Panics
    ///
    /// As [`stated`](Self::stated).
    pub(crate) fn of_buffers(
        input: &TensorBinding,
        out: &TensorBinding,
        axis: usize,
        operation: ReduceOperation,
        elements: ReduceElements,
    ) -> Self {
        Self::stated(
            &Geometry::from(input),
            &Geometry::from(out),
            axis,
            operation,
            elements,
        )
    }

    /// The problem `out = operation(input)` along axis `axis` states, the input bound by the caller
    /// and read at exactly `width`.
    ///
    /// # Panics
    ///
    /// As [`stated`](Self::stated).
    pub(crate) fn of_unbound(
        input: &Geometry,
        width: usize,
        out: &TensorBinding,
        axis: usize,
        operation: ReduceOperation,
        elements: ReduceElements,
    ) -> Self {
        let stated = Self::stated(input, &Geometry::from(out), axis, operation, elements);
        let [_, out_alignment] = stated.elem_alignments;
        Self {
            elem_alignments: [ElemAlignment::stored_piece(width), out_alignment],
            ..stated
        }
    }

    /// This problem with the indices of its values written beside them in `element`, to
    /// `indices`: a line runs along the innermost axis only where that buffer steps one element
    /// along it too.
    pub(crate) fn writing_indices(mut self, element: ElemType, indices: &TensorBinding) -> Self {
        let indices = Geometry::from(indices);
        self.contiguous = self.contiguous.filter(|&inner| steps_one(&indices, inner));
        self.indices = Some(ReduceIndices {
            element,
            elem_alignment: ElemAlignment::new(indices.shape(), indices.strides()),
        });
        self
    }

    pub(crate) fn reduced(&self) -> Dimension {
        axis_dimension(self.axis)
    }

    /// The dimension of the cells a row writes, in place of the reduced one in the output.
    pub(crate) fn slots(&self) -> Dimension {
        axis_dimension(self.shape.len())
    }

    /// The dimension a hypercube's partials run along in place of the reduced one: every part's
    /// carrier for a row, side by side.
    pub(crate) fn partials(&self) -> Dimension {
        axis_dimension(self.shape.len() + 1)
    }

    /// Every row dimension, outermost first.
    pub(crate) fn rows(&self) -> Vec<Dimension> {
        (0..self.shape.len())
            .filter(|d| *d != self.axis)
            .map(axis_dimension)
            .collect()
    }

    /// The dimensions the owners are placed along, outermost first: the row
    /// dimensions, or the slots where there is none, a tensor of one axis reduced whole into its one row.
    pub(crate) fn owner_dimensions(&self) -> Vec<Dimension> {
        let rows = self.rows();
        if rows.is_empty() {
            return vec![self.slots()];
        }
        rows
    }

    /// [`owner_dimensions`](Self::owner_dimensions), innermost first. A launch deals its owners
    /// first dimension fastest, so neighbouring units and planes take neighbouring rows: their
    /// reads land a row apart instead of a whole outer axis apart. Rows across the reduced axis
    /// share their lines' memory, so there it is what keeps a unit or plane routine level with a
    /// kernel reading the rows in order (on Metal, a strided f16 sum: 1083 µs outermost first,
    /// 752 µs innermost first). A cube routine, whose cube spreads one owner's rows over all its
    /// units, measured slower this way (1045 µs to 1236 µs), so it keeps them outermost first.
    pub(crate) fn owner_dimensions_innermost_first(&self) -> Vec<Dimension> {
        let mut dimensions = self.owner_dimensions();
        dimensions.reverse();
        dimensions
    }

    /// The input's dimensions, one an axis.
    pub(crate) fn input_dimensions(&self) -> Vec<Dimension> {
        (0..self.shape.len()).map(axis_dimension).collect()
    }

    /// The output's dimensions, one an axis: the input's, the reduced axis's being the slots.
    pub(crate) fn out_dimensions(&self) -> Vec<Dimension> {
        self.dimensions_with_reduced(self.slots())
    }

    /// A hypercube's partials' dimensions, one an axis: the input's, the reduced axis's being the
    /// partials.
    pub(crate) fn partial_dimensions(&self) -> Vec<Dimension> {
        self.dimensions_with_reduced(self.partials())
    }

    /// The input's dimensions, one an axis, the reduced axis's being `reduced`.
    fn dimensions_with_reduced(&self, reduced: Dimension) -> Vec<Dimension> {
        let mut dimensions = self.input_dimensions();
        dimensions[self.axis] = reduced;
        dimensions
    }

    /// The input's shape with the reduced axis `extent` long.
    pub(crate) fn shape_with_reduced_extent(&self, extent: usize) -> Vec<usize> {
        with_axis_extent(&self.shape, self.axis, extent)
    }

    /// How the input is checked where a tile reaches past its edge: a read there returns the
    /// identity of the operation's edge monoid ([`ReduceOperation::edge_monoid`]), which leaves
    /// the total unchanged (`tracel:cubecl::edge-reads-identity`).
    pub(crate) fn input_boundary(&self) -> BoundaryPolicy {
        BoundaryPolicy::Overhanging(Boundary::Identity(self.operation.edge_monoid()))
    }

    /// Values a row reduces.
    pub(crate) fn reduced_extent(&self) -> usize {
        self.shape[self.axis]
    }

    pub(crate) fn row_count(&self) -> usize {
        self.shape.iter().product::<usize>() / self.reduced_extent().max(1)
    }

    /// How the lines lie: along the reduced axis where it is the contiguous one, across it where
    /// another is, each lane of a line its own row; along it, a value a line, where none is.
    pub(crate) fn layout(&self) -> ReduceLayout {
        match self.contiguous {
            Some(axis) if axis != self.axis => ReduceLayout::Across,
            _ => ReduceLayout::Along,
        }
    }

    /// What every microkernel of this problem is handed.
    pub(crate) fn work_item(&self) -> ReduceWorkItem {
        ReduceWorkItem {
            operation: self.operation,
            input: self.elements.input,
            accumulator: self.elements.accumulator,
            layout: self.layout(),
            with_indices: self.indices.is_some(),
        }
    }

    /// Every dimension of the problem at its extent, the slots among them: what a routine
    /// partitions.
    pub(crate) fn space(&self) -> Space {
        let mut dimensions: Vec<(Dimension, usize)> = self
            .shape
            .iter()
            .enumerate()
            .map(|(d, extent)| (axis_dimension(d), *extent))
            .collect();
        dimensions.push((self.slots(), self.operation.written_per_row()));
        Space::new(&dimensions)
    }

    /// [`space`](Self::space) and a hypercube's partials, `partial_extent` values of each box of
    /// rows' along their dimension.
    pub(crate) fn split_space_with_partial_extent(&self, partial_extent: usize) -> Space {
        let space = self.space();
        let mut dimensions: Vec<(Dimension, usize)> = space
            .dimensions()
            .map(|dimension| (dimension, space.extent(dimension)))
            .collect();
        dimensions.push((self.partials(), partial_extent));
        Space::new(&dimensions)
    }

    /// The values of a row one cube of `units` units reduces when the row is split across cubes,
    /// each unit `unit_lines` lines of `width` values: along, a line holds `width` values of the
    /// row; across, one.
    pub(crate) fn split_part(&self, units: usize, unit_lines: usize, width: usize) -> usize {
        let values_a_line = match self.layout() {
            ReduceLayout::Along => width,
            ReduceLayout::Across => 1,
        };
        units * unit_lines * values_a_line
    }

    /// The box a cube owns where a row is split across cubes: the owner's box at `width`, `part`
    /// values long along the reduced dim.
    pub(crate) fn split_owner_box(&self, width: usize, part: usize) -> Vec<(Dimension, usize)> {
        [
            self.owner_box_at_width(width).as_slice(),
            &[(self.reduced(), part)],
        ]
        .concat()
    }

    /// The dims a split row's cubes run along: the reduced dim first, on the grid's `X`, then the
    /// row dims on its `Y` and `Z`.
    pub(crate) fn split_cube_dimensions(&self) -> Vec<Dimension> {
        [&[self.reduced()], self.rows().as_slice()].concat()
    }

    /// The box an owner holds: one row, or `width` rows side by side across the reduced axis, and
    /// the cells it writes. It leaves the reduced axis to the space, so every owner spans its rows
    /// whole.
    pub(crate) fn owner_box_at_width(&self, width: usize) -> Vec<(Dimension, usize)> {
        let mut extents: Vec<(Dimension, usize)> = (0..self.shape.len())
            .filter(|d| *d != self.axis)
            .map(|d| {
                let across_lines =
                    self.layout() == ReduceLayout::Across && self.contiguous == Some(d);
                let extent = if across_lines { width } else { 1 };
                (axis_dimension(d), extent)
            })
            .collect();
        extents.push((self.slots(), self.operation.written_per_row()));
        extents
    }

    /// How many boxes of rows the owners, `width` rows across the reduced axis, take along each
    /// row dimension, outermost first.
    pub(crate) fn row_boxes_at_width(&self, width: usize) -> Vec<(Dimension, usize)> {
        let owner = self.owner_box_at_width(width);
        self.rows()
            .into_iter()
            .map(|dimension| {
                let tile = owner
                    .iter()
                    .find_map(|&(d, extent)| (d == dimension).then_some(extent))
                    .expect("the owner's box names every row dimension");
                (
                    dimension,
                    self.shape[usize::from(dimension.0)].div_ceil(tile),
                )
            })
            .collect()
    }

    /// `partitioning` as a kernel compiles it: every dimension but the cells a row writes left to
    /// the launch, the row length too, as today's reduce kernels read it, so one kernel serves
    /// every shape of a rank, layout, elements and widths. An owner's window is one row high, which
    /// its flat view reads with no division by the runtime length.
    pub(crate) fn kernel(&self, partitioning: &Partitioning) -> Partitioning {
        let space = partitioning.space();
        let dynamic: Vec<Dimension> = space
            .dimensions()
            .filter(|dimension| *dimension != self.slots())
            .collect();
        Partitioning::new(
            space.clone().with_dynamic(&dynamic),
            partitioning.levels().to_vec(),
        )
    }
}

/// The dimension of axis `d`.
pub(crate) fn axis_dimension(d: usize) -> Dimension {
    Dimension(u8::try_from(d).expect("a tensor has fewer than 256 axes"))
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

/// `shape` with `axis` `extent` long.
fn with_axis_extent(shape: &[usize], axis: usize, extent: usize) -> Vec<usize> {
    let mut shape = shape.to_vec();
    shape[axis] = extent;
    shape
}

/// Whether `geometry` steps one element along `axis`, or holds one element there.
fn steps_one(geometry: &Geometry, axis: usize) -> bool {
    geometry.strides()[axis] == 1 || geometry.shape()[axis] == 1
}

#[cfg(test)]
impl ReduceRoutineProblem {
    /// A dense row-major `shape` reduced along `axis` by `operation`, in f32, an index or a flag
    /// in i32: what a routine's Small Tests prepare.
    pub(crate) fn dense(shape: &[usize], axis: usize, operation: ReduceOperation) -> Self {
        let f32 = ElemType::Float(FloatKind::F32);
        let i32 = ElemType::Int(IntKind::I32);
        let innermost = shape.len() - 1;
        Self {
            shape: shape.to_vec(),
            axis,
            operation,
            elements: ReduceElements::for_operation(operation, f32, Some(i32)),
            contiguous: Some(innermost),
            elem_alignments: [ElemAlignment::dense_rows(shape[innermost]); 2],
            indices: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);
    const U32: ElemType = ElemType::UInt(UIntKind::U32);

    #[test]
    fn a_sum_of_halves_accumulates_in_f32() {
        let elements = ReduceElements::for_operation(ReduceOperation::Sum, F16, None);
        assert_eq!(
            elements,
            ReduceElements {
                input: F16,
                accumulator: F32,
                out: F16
            }
        );
    }

    #[test]
    fn an_extremum_accumulates_in_its_input() {
        let elements = ReduceElements::for_operation(ReduceOperation::Max, F16, Some(U32));
        assert_eq!(
            elements,
            ReduceElements {
                input: F16,
                accumulator: F16,
                out: F16
            }
        );
    }

    #[test]
    fn an_index_is_written_in_the_element_the_caller_names() {
        let elements =
            ReduceElements::for_operation(ReduceOperation::ArgTopK { k: 3 }, F16, Some(U32));
        assert_eq!(
            elements,
            ReduceElements {
                input: F16,
                accumulator: F16,
                out: U32
            }
        );
    }

    #[test]
    fn a_narrow_integer_sum_accumulates_in_i32() {
        let i8 = ElemType::Int(IntKind::I8);
        let elements = ReduceElements::for_operation(ReduceOperation::Mean, i8, None);
        assert_eq!(elements.accumulator, ElemType::Int(IntKind::I32));
    }

    /// A boolean is not reduced: `for_operation`'s documented panic.
    #[test]
    #[should_panic]
    fn a_boolean_input_is_not_reduced() {
        ReduceElements::for_operation(ReduceOperation::Sum, ElemType::Bool, None);
    }

    /// An output not the input's shape with the reduced axis as wide as a row writes is refused.
    #[test]
    #[should_panic]
    fn a_stated_output_of_another_shape_panics() {
        let input = Geometry::row_major(&[4, 8]);
        let out = Geometry::row_major(&[4, 2]);
        ReduceRoutineProblem::stated(
            &input,
            &out,
            1,
            ReduceOperation::Sum,
            ReduceElements::for_operation(ReduceOperation::Sum, F32, None),
        );
    }

    /// Rows lie along the reduced axis where it is the contiguous one, across it where another is.
    #[test]
    fn the_lines_lie_along_the_contiguous_axis() {
        let along = ReduceRoutineProblem::dense(&[4, 8], 1, ReduceOperation::Sum);
        let across = ReduceRoutineProblem::dense(&[4, 8], 0, ReduceOperation::Sum);
        assert_eq!(along.layout(), ReduceLayout::Along);
        assert_eq!(across.layout(), ReduceLayout::Across);
    }

    /// Across the reduced axis an owner holds a line of rows, `width` side by side; along it, one
    /// row. Each holds the cells a row writes.
    #[test]
    fn an_owner_holds_a_line_of_rows_across_and_one_along() {
        const WIDTH: usize = 4;
        const K: usize = 3;
        let across = ReduceRoutineProblem::dense(&[5, 8], 0, ReduceOperation::TopK { k: K });
        assert_eq!(
            across.owner_box_at_width(WIDTH),
            vec![(axis_dimension(1), WIDTH), (across.slots(), K)]
        );
        let along = ReduceRoutineProblem::dense(&[5, 8], 1, ReduceOperation::TopK { k: K });
        assert_eq!(
            along.owner_box_at_width(WIDTH),
            vec![(axis_dimension(0), 1), (along.slots(), K)]
        );
    }

    /// The boxes of rows along each row dim are its extent over the owner's, rounded up.
    #[test]
    fn row_boxes_round_up_the_rows_over_an_owner() {
        const WIDTH: usize = 4;
        let across = ReduceRoutineProblem::dense(&[3, 10], 0, ReduceOperation::Sum);
        assert_eq!(
            across.row_boxes_at_width(WIDTH),
            vec![(axis_dimension(1), 10_usize.div_ceil(WIDTH))]
        );
    }

    /// A tensor of one axis has one row, its owner along the slots.
    #[test]
    fn a_tensor_of_one_axis_owns_its_slots() {
        let problem = ReduceRoutineProblem::dense(&[9], 0, ReduceOperation::Sum);
        assert!(problem.rows().is_empty());
        assert_eq!(problem.owner_dimensions(), vec![problem.slots()]);
        assert_eq!(problem.row_count(), 1);
    }

    /// The output and the partials keep the input's dimensions, the reduced axis's swapped.
    #[test]
    fn the_output_and_partials_swap_the_reduced_dimension() {
        let problem = ReduceRoutineProblem::dense(&[2, 3, 4], 1, ReduceOperation::Sum);
        assert_eq!(
            problem.out_dimensions(),
            vec![axis_dimension(0), problem.slots(), axis_dimension(2)]
        );
        assert_eq!(
            problem.partial_dimensions(),
            vec![axis_dimension(0), problem.partials(), axis_dimension(2)]
        );
        assert_eq!(problem.shape_with_reduced_extent(7), vec![2, 7, 4]);
    }

    /// A split row's part is its cube's units, each `unit_lines` lines, of `width` values along
    /// the row and of one across it.
    #[test]
    fn a_split_part_counts_values_a_line_by_layout() {
        const UNITS: usize = 64;
        const UNIT_LINES: usize = 8;
        const WIDTH: usize = 4;
        let along = ReduceRoutineProblem::dense(&[2, 4096], 1, ReduceOperation::Sum);
        let across = ReduceRoutineProblem::dense(&[4096, 2], 0, ReduceOperation::Sum);
        assert_eq!(
            along.split_part(UNITS, UNIT_LINES, WIDTH),
            UNITS * UNIT_LINES * WIDTH
        );
        assert_eq!(
            across.split_part(UNITS, UNIT_LINES, WIDTH),
            UNITS * UNIT_LINES
        );
        assert_eq!(
            along.split_cube_dimensions(),
            vec![along.reduced(), axis_dimension(0)]
        );
    }

    /// A kernel leaves every dimension to the launch but the cells a row writes.
    #[test]
    fn a_kernel_compiles_only_the_cells_a_row_writes() {
        let problem = ReduceRoutineProblem::dense(&[2, 300], 1, ReduceOperation::Sum);
        let partitioning = Partitioning::new(
            problem.space(),
            cubecl::tile::Levels::leaf(&problem.owner_box_at_width(1))
                .cubes(&problem.owner_dimensions())
                .build(),
        );
        let space = problem.kernel(&partitioning).space().clone();
        assert!(space.is_dynamic(axis_dimension(1)));
        assert!(!space.is_dynamic(problem.slots()));
    }

    /// 256 axes do not fit a dimension's label.
    #[test]
    #[should_panic]
    fn an_axis_past_255_panics() {
        axis_dimension(256);
    }
}
