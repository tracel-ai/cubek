//! The problem a routine computes.

use crate::kernels::ReduceKey;
use cubecl::ir::{ElemType, FloatKind, IntKind, UIntKind};
use cubecl::prelude::TensorBinding;
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::BoundaryPolicy;
use cubecl::tile::layout::{ElemAlignment, VectorTile};
use cubecl::tile::{Dimension, Geometry, Partitioning, Space};
use cubecl::tune::Work;
use cubek_microkernel::reduce::{BATCH, PARTIALS, REDUCED, ReduceOperation, ReduceWorkItem, SLOTS};
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
    /// # Assumptions
    ///
    /// - `out` names an element where `operation` writes an index or a flag.
    /// - A sum, a product or a mean reduces neither booleans nor complex numbers.
    ///
    /// # Panics
    ///
    /// Where an Assumption is broken.
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
/// Its space has the same dimensions whatever the rank: `BATCH`, every axis but `axis` as one,
/// `REDUCED`, `axis` itself, and `SLOTS`, the cells a row writes, which the output holds in place
/// of `axis`. Each operand's layout maps its axes onto them.
///
/// # Assumptions
///
/// - `axis` is an axis of `shape`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReduceRoutineProblem {
    /// The input's extent along each axis, read by axis (Position Usage) and walked in their order
    /// (Arrival Usage).
    pub shape: Vec<usize>,
    pub axis: usize,
    pub operation: ReduceOperation,
    pub elements: ReduceElements,
    /// The innermost axis, where a line runs along it: the buffers a line is read from and written
    /// to whole step one element along it. `None` where none does, and a line holds one value.
    pub contiguous: Option<usize>,
    /// The alignment in elements of the input's buffer, then the output's, read by that position
    /// (Position Usage) and walked in that order (Arrival Usage).
    pub elem_alignments: [ElemAlignment; 2],
    /// The indices written beside the values, to a second output of their shape, where they are.
    pub indices: Option<ReduceIndices>,
    /// The partials a row reduced in parts is written to, where the caller gave buffers for them.
    pub partials: Option<ReducePartials>,
}

/// The indices a reduction writes beside its values: their element, and the alignment of the
/// buffer they go to, which lies at its own strides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceIndices {
    pub element: ElemType,
    pub elem_alignment: ElemAlignment,
}

/// The operands a row reduced in parts is written to before its parts merge: the cells each
/// holds, one after another ([`ReduceWorkspace`](crate::kernels::ReduceWorkspace)), which a
/// routine that splits rows checks are enough before it launches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReducePartials {
    /// The parts' values, in the accumulator's element.
    pub value_count: usize,
    /// The parts' indices, in `u32`, where the operation's carrier keeps any.
    pub index_count: usize,
    /// The boxes of rows' counts of their arrived parts, in `u32`, at zero.
    pub arrival_count: usize,
}

impl ReducePartials {
    /// As many cells as any problem's rows are written to.
    pub(crate) const UNBOUNDED: Self = Self {
        value_count: usize::MAX,
        index_count: usize::MAX,
        arrival_count: usize::MAX,
    };

    /// Whether each operand holds at least the cells `needs` asks of it.
    pub(crate) fn holds(&self, needs: &Self) -> bool {
        self.value_count >= needs.value_count
            && self.index_count >= needs.index_count
            && self.arrival_count >= needs.arrival_count
    }

    /// Each operand the larger of this and `other`.
    pub(crate) fn max(self, other: Self) -> Self {
        Self {
            value_count: self.value_count.max(other.value_count),
            index_count: self.index_count.max(other.index_count),
            arrival_count: self.arrival_count.max(other.arrival_count),
        }
    }
}

impl ReduceRoutineProblem {
    /// The problem `out = operation(input)` along axis `axis` states, `out` having `input`'s shape
    /// with `axis` as wide as the operation writes per row.
    ///
    /// # Assumptions
    ///
    /// - `axis` is an axis of `input`, which has one at least.
    /// - `out` has `input`'s shape, `axis` as wide as the operation writes per row.
    ///
    /// # Panics
    ///
    /// Where an Assumption is broken.
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
        // A line runs along the innermost axis where the operands it is read from and written to
        // step one element along it, past the row axes one long, which hold no step (as
        // `labelled` leaves them out). A line along the reduced axis is only read: its row goes out
        // a cell.
        let inner = (0..shape.len())
            .rev()
            .find(|&d| d == axis || shape[d] > 1)
            .expect("the reduced axis is one of the input's");
        let written_whole = inner == axis || steps_one(out, inner);
        let contiguous = (steps_one(input, inner) && written_whole).then_some(inner);
        Self {
            elem_alignments: [input, out]
                .map(|geometry| ElemAlignment::new(geometry.shape(), geometry.strides())),
            shape,
            axis,
            operation,
            elements,
            contiguous,
            indices: None,
            partials: None,
        }
    }
}

impl ReduceRoutineProblem {
    /// The problem `out = operation(input)` along axis `axis` states, both read where they lie.
    ///
    /// # Panics
    ///
    /// Where [`stated`](Self::stated) does, of their geometries.
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
    /// and read at exactly `lane_count`.
    ///
    /// # Panics
    ///
    /// Where the geometry and `out`'s state no problem ([`stated`](Self::stated)).
    pub(crate) fn of_unbound(
        input: &Geometry,
        lane_count: usize,
        out: &TensorBinding,
        axis: usize,
        operation: ReduceOperation,
        elements: ReduceElements,
    ) -> Self {
        let stated = Self::stated(input, &Geometry::from(out), axis, operation, elements);
        let [_, out_alignment] = stated.elem_alignments;
        Self {
            elem_alignments: [ElemAlignment::stored_piece(lane_count), out_alignment],
            ..stated
        }
    }

    /// This problem with its input's buffer aligned as one laid out as `input` is.
    pub(crate) fn with_input_aligned_as(&self, input: &Geometry) -> Self {
        let [_, out_alignment] = self.elem_alignments;
        Self {
            elem_alignments: [
                ElemAlignment::new(input.shape(), input.strides()),
                out_alignment,
            ],
            ..self.clone()
        }
    }

    /// This problem with the indices of its values written beside them in `element`, to
    /// `indices`: a line of rows runs along the innermost axis only where that buffer steps one
    /// element along it too.
    pub(crate) fn writing_indices(&self, element: ElemType, indices: &TensorBinding) -> Self {
        let indices = Geometry::from(indices);
        Self {
            contiguous: self
                .contiguous
                .filter(|&inner| inner == self.axis || steps_one(&indices, inner)),
            indices: Some(ReduceIndices {
                element,
                elem_alignment: ElemAlignment::new(indices.shape(), indices.strides()),
            }),
            ..self.clone()
        }
    }

    /// `binding`, an operand of the input's axes holding `along` in place of the reduced axis
    /// ([`REDUCED`] for the input, [`SLOTS`] for the output and its indices, [`PARTIALS`] for a
    /// hypercube's partials), with the dimension each of its axes carries
    /// ([`labelled_axes`](Self::labelled_axes)).
    pub(crate) fn labelled(
        &self,
        mut binding: TensorBinding,
        along: Dimension,
    ) -> (TensorBinding, Vec<Dimension>) {
        let (axes, dimensions) = self.labelled_axes(&binding.shape, &binding.strides, along);
        binding.shape = axes.iter().map(|&(extent, _)| extent).collect();
        binding.strides = axes.iter().map(|&(_, stride)| stride).collect();
        (binding, dimensions)
    }

    /// [`labelled`](Self::labelled) for an operand stated by its `geometry` alone, which the
    /// caller binds.
    pub(crate) fn labelled_geometry(
        &self,
        geometry: &Geometry,
        along: Dimension,
    ) -> (Geometry, Vec<Dimension>) {
        let (axes, dimensions) = self.labelled_axes(geometry.shape(), geometry.strides(), along);
        (Geometry::new(&axes), dimensions)
    }

    /// The `(extent, stride)` of each axis of an operand of `shape` at `strides`, in its order,
    /// and the dimension each carries: `along` for the reduced axis, [`BATCH`] for every other, so
    /// [`BATCH`] spans the row axes, the first coarsest, on either side of the reduced one.
    /// A row axis one long holds no step and is left out. Neighbouring row axes that step as one
    /// (the outer's stride the inner's whole span) merge into one, so a contiguous operand's rows
    /// before and after the reduced axis are one axis each. An operand with no row axis left is
    /// given a row axis of one.
    fn labelled_axes(
        &self,
        shape: &[usize],
        strides: &[usize],
        along: Dimension,
    ) -> (Vec<(usize, usize)>, Vec<Dimension>) {
        let stepping = (0..shape.len()).filter(|&d| d == self.axis || shape[d] != 1);
        let mut labelled: Vec<(Dimension, (usize, usize))> = Vec::new();
        for d in stepping {
            let dimension = match d == self.axis {
                true => along,
                false => BATCH,
            };
            let axis = (shape[d], strides[d]);
            let merged = match labelled.last() {
                Some(&(BATCH, outer)) if dimension == BATCH => merged_rows(outer, axis),
                _ => None,
            };
            match merged {
                Some(merged) => {
                    labelled
                        .last_mut()
                        .expect("a merged axis has an outer one")
                        .1 = merged
                }
                None => labelled.push((dimension, axis)),
            }
        }
        if !labelled.iter().any(|&(dimension, _)| dimension == BATCH) {
            labelled.insert(0, (BATCH, ONE_ROW));
        }
        labelled
            .into_iter()
            .map(|(dimension, axis)| (axis, dimension))
            .unzip()
    }

    /// The input's shape with the reduced axis `reduced_count` long.
    pub(crate) fn shape_with_reduced_count(&self, reduced_count: usize) -> Vec<usize> {
        with_axis_extent(&self.shape, self.axis, reduced_count)
    }

    /// How the input is checked where a tile reaches past its edge: a read there returns the
    /// identity of the operation's edge monoid ([`ReduceOperation::edge_monoid`]), which leaves
    /// the total unchanged.
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

    /// The dimension the input's lines run along: [`BATCH`] where another axis than the reduced
    /// one is the contiguous one, each lane of a line its own row's; [`REDUCED`] otherwise, a
    /// value a line where no axis is.
    pub(crate) fn lines_along(&self) -> Dimension {
        match self.contiguous {
            Some(axis) if axis != self.axis => BATCH,
            _ => REDUCED,
        }
    }

    /// How many values the axis a line runs along holds, which the line stays in: one where no axis
    /// is contiguous.
    pub(crate) fn line_stays_in(&self) -> usize {
        self.contiguous.map_or(1, |inner| self.shape[inner])
    }

    /// What one load of the input covers, its lines `lane_count` wide: a run along
    /// [`lines_along`](Self::lines_along), the load the input's tile carries on the device
    /// ([`Tile::vector_tile`](cubecl::tile::Tile::vector_tile)).
    pub(crate) fn input_load_at_lane_count(&self, lane_count: usize) -> VectorTile {
        VectorTile::run(self.lines_along(), lane_count)
    }

    /// What every microkernel of this problem is handed.
    pub(crate) fn work_item(&self) -> ReduceWorkItem {
        ReduceWorkItem {
            operation: self.operation,
            input: self.elements.input,
            accumulator: self.elements.accumulator,
            lines_along: self.lines_along(),
            with_indices: self.indices.is_some(),
        }
    }

    /// Every dimension of the problem at its extent: what a routine partitions.
    pub(crate) fn space(&self) -> Space {
        Space::new(&self.extents())
    }

    /// [`space`](Self::space) and a hypercube's partials, `partial_count` values of each box of
    /// rows'.
    pub(crate) fn split_space_with_partial_count(&self, partial_count: usize) -> Space {
        let [rows, reduced, slots] = self.extents();
        Space::new(&[rows, reduced, slots, (PARTIALS, partial_count)])
    }

    /// The values of a row one cube of `unit_count` units reduces when the row is split across
    /// cubes, each unit `unit_line_count` lines of `lane_count` values.
    pub(crate) fn split_part(
        &self,
        unit_count: usize,
        unit_line_count: usize,
        lane_count: usize,
    ) -> usize {
        unit_count * unit_line_count * self.row_value_count_at_lane_count(lane_count)
    }

    /// The values of one row a line `lane_count` wide holds: `lane_count` where the lines lie along
    /// the reduced dim, one where they lie across it.
    pub(crate) fn row_value_count_at_lane_count(&self, lane_count: usize) -> usize {
        self.input_load_at_lane_count(lane_count)
            .extent_along(REDUCED)
    }

    /// The box a cube owns where a row is split across cubes: the owner's box at `lane_count`,
    /// `part` values long along [`REDUCED`].
    pub(crate) fn split_owner_box(
        &self,
        lane_count: usize,
        part: usize,
    ) -> [(Dimension, usize); 3] {
        let [rows, slots] = self.owner_box_at_lane_count(lane_count);
        [rows, slots, (REDUCED, part)]
    }

    /// The box an owner holds: one row, or `lane_count` rows side by side across the reduced axis, and
    /// the cells it writes. It leaves [`REDUCED`] to the space, so every owner spans its rows
    /// whole.
    pub(crate) fn owner_box_at_lane_count(&self, lane_count: usize) -> [(Dimension, usize); 2] {
        [
            (BATCH, self.box_row_count_at_lane_count(lane_count)),
            (SLOTS, self.operation.written_per_row()),
        ]
    }

    /// How many boxes of rows the owners, `lane_count` rows across the reduced axis, take.
    pub(crate) fn row_box_count_at_lane_count(&self, lane_count: usize) -> usize {
        self.row_count()
            .div_ceil(self.box_row_count_at_lane_count(lane_count))
    }

    /// Rows an owner holds, and the lanes of a line it writes them in: those one load reaches along
    /// [`BATCH`], `lane_count` side by side across the reduced axis, one along it.
    pub(crate) fn box_row_count_at_lane_count(&self, lane_count: usize) -> usize {
        self.input_load_at_lane_count(lane_count)
            .extent_along(BATCH)
    }

    /// Each dimension the problem holds with its extent: the rows, the reduced dim and the cells a
    /// row writes.
    fn extents(&self) -> [(Dimension, usize); 3] {
        [
            (BATCH, self.row_count()),
            (REDUCED, self.reduced_extent()),
            (SLOTS, self.operation.written_per_row()),
        ]
    }

    /// The problem this one's Tune Key stands for ([`ReduceKey`]): what the rule picks a routine
    /// from.
    pub(crate) fn keyed(&self) -> Self {
        Self::from(&ReduceKey::from(self))
    }

    /// The elements a kernel compiles for: the input's, the accumulator's, the output's, and the
    /// indices', the output's where no launch of this problem writes them.
    pub(crate) fn compiled_elements(&self) -> [ElemType; 4] {
        let ReduceElements {
            input,
            accumulator,
            out,
        } = self.elements;
        let index = self.indices.map_or(out, |indices| indices.element);
        [input, accumulator, out, index]
    }

    /// A dense row-major input of `shape` reduced along `axis` by `operation` in `elements`, its
    /// output dense too, and `u32` indices beside it where `with_indices`: the problem a caller
    /// states by its shape alone, as [`ReduceWorkspace::bytes_for`](crate::kernels::ReduceWorkspace::bytes_for)
    /// sizes a workspace by.
    ///
    /// # Panics
    ///
    /// Where `axis` is past `shape`'s.
    pub(crate) fn of_shape(
        shape: &[usize],
        axis: usize,
        operation: ReduceOperation,
        elements: ReduceElements,
        with_indices: bool,
    ) -> Self {
        let input = Geometry::row_major(shape);
        let out = Geometry::row_major(&with_axis_extent(shape, axis, operation.written_per_row()));
        let stated = Self::stated(&input, &out, axis, operation, elements);
        let [_, out_alignment] = stated.elem_alignments;
        Self {
            indices: with_indices.then_some(ReduceIndices {
                element: ElemType::UInt(UIntKind::U32),
                elem_alignment: out_alignment,
            }),
            ..stated
        }
    }

    /// `partitioning` as a kernel compiles it: every dimension but the cells a row writes left to
    /// the launch, the row length too, as today's reduce kernels read it, so one kernel serves
    /// every shape of a rank, layout, elements and widths. An owner's window is one row high, which
    /// its flat view reads with no division by the runtime length.
    pub(crate) fn kernel(&self, partitioning: &Partitioning) -> Partitioning {
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

/// The `(extent, stride)` of a row axis that holds one row.
const ONE_ROW: (usize, usize) = (1, 1);

/// Two neighbouring row axes as one, the outer `(extent, stride)` and the inner, where the outer
/// steps the inner's whole span.
fn merged_rows(
    (extent, stride): (usize, usize),
    (inner_extent, inner_stride): (usize, usize),
) -> Option<(usize, usize)> {
    (stride == inner_extent * inner_stride).then_some((extent * inner_extent, inner_stride))
}

impl EstimatedProblem for ReduceRoutineProblem {
    /// Every value read once and accumulated once, every cell written once, and its index beside
    /// it where the problem writes any.
    ///
    /// # Panics
    ///
    /// Where an Assumption of [`ReduceRoutineProblem`] is broken.
    fn demand(&self) -> Work {
        let values: usize = self.shape.iter().product();
        let cells = self.row_count() * self.operation.written_per_row();
        let index_bytes = self.indices.map_or(0, |indices| indices.element.size());
        Work {
            compute_ops: values,
            bytes: values * self.elements.input.size()
                + cells * (self.elements.out.size() + index_bytes),
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
            partials: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::std::tensor::TensorHandle;
    use cubecl::tile::Monoid;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);
    const U32: ElemType = ElemType::UInt(UIntKind::U32);

    /// A sum of f32 values.
    fn sum_elements() -> ReduceElements {
        ReduceElements {
            input: F32,
            accumulator: F32,
            out: F32,
        }
    }

    /// A buffer of `shape` stated at `strides`, whose cells no test here reaches.
    fn buffer_at_strides(shape: &[usize], strides: &[usize]) -> TensorBinding {
        let client = cubecl::test_device().client();
        let mut buffer = TensorHandle::empty(&client, shape.to_vec(), F32).binding();
        buffer.strides = strides.iter().copied().collect();
        buffer
    }

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

    #[test]
    #[should_panic(expected = "write an element the caller names")]
    fn an_index_without_an_element_to_be_written_in_panics() {
        ReduceElements::for_operation(ReduceOperation::ArgMax, F32, None);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn an_axis_past_the_inputs_rank_panics() {
        let input = Geometry::row_major(&[4, 8]);
        let out = Geometry::row_major(&[4, 1]);
        ReduceRoutineProblem::stated(&input, &out, 2, ReduceOperation::Sum, sum_elements());
    }

    /// An output not the input's shape with the reduced axis as wide as a row writes is refused.
    #[test]
    #[should_panic]
    fn a_stated_output_of_another_shape_panics() {
        let input = Geometry::row_major(&[4, 8]);
        let out = Geometry::row_major(&[4, 2]);
        ReduceRoutineProblem::stated(&input, &out, 1, ReduceOperation::Sum, sum_elements());
    }

    /// Lines lie along the reduced axis where it is the contiguous one.
    #[test]
    fn the_lines_lie_along_a_contiguous_reduced_axis() {
        let along = ReduceRoutineProblem::dense(&[4, 8], 1, ReduceOperation::Sum);
        assert_eq!(along.lines_along(), REDUCED);
    }

    /// Lines lie across the reduced axis where another is the contiguous one.
    #[test]
    fn the_lines_lie_across_a_strided_reduced_axis() {
        let across = ReduceRoutineProblem::dense(&[4, 8], 0, ReduceOperation::Sum);
        assert_eq!(across.lines_along(), BATCH);
    }

    /// Rows side by side going out strided leave no axis for a line to run along: it holds a value.
    #[test]
    fn a_line_of_rows_going_out_strided_holds_one_value() {
        let input = Geometry::row_major(&[8, 4]);
        let out = Geometry::new(&[(1, 8), (4, 2)]);
        let problem =
            ReduceRoutineProblem::stated(&input, &out, 0, ReduceOperation::Sum, sum_elements());
        assert_eq!(problem.line_stays_in(), 1);
    }

    /// Along the reduced axis a line is only read, however its row's cells go out.
    #[test]
    fn a_line_along_the_reduced_axis_reads_whole_beside_a_strided_output() {
        const ROW_VALUE_COUNT: usize = 8;
        let input = Geometry::row_major(&[4, ROW_VALUE_COUNT]);
        let out = Geometry::new(&[(4, 1), (2, 4)]);
        let problem = ReduceRoutineProblem::stated(
            &input,
            &out,
            1,
            ReduceOperation::TopK { k: 2 },
            ReduceElements::for_operation(ReduceOperation::TopK { k: 2 }, F32, None),
        );
        assert_eq!(problem.line_stays_in(), ROW_VALUE_COUNT);
    }

    #[test]
    fn the_demand_counts_the_indices_written() {
        const ROW_COUNT: usize = 4;
        const ROW_VALUE_COUNT: usize = 300;
        let problem = ReduceRoutineProblem {
            indices: Some(ReduceIndices {
                element: U32,
                elem_alignment: ElemAlignment::ONE,
            }),
            ..ReduceRoutineProblem::dense(&[ROW_COUNT, ROW_VALUE_COUNT], 1, ReduceOperation::ArgMax)
        };
        let read = ROW_COUNT * ROW_VALUE_COUNT * problem.elements.input.size();
        let written = ROW_COUNT * (problem.elements.out.size() + U32.size());
        assert_eq!(problem.demand().bytes, read + written);
    }

    #[test]
    fn an_owner_across_the_axis_holds_a_line_of_rows() {
        const LANE_COUNT: usize = 4;
        const KEPT_COUNT: usize = 3;
        let across =
            ReduceRoutineProblem::dense(&[5, 8], 0, ReduceOperation::TopK { k: KEPT_COUNT });
        assert_eq!(
            across.owner_box_at_lane_count(LANE_COUNT),
            [(BATCH, LANE_COUNT), (SLOTS, KEPT_COUNT)]
        );
    }

    #[test]
    fn an_owner_along_the_axis_holds_one_row() {
        const LANE_COUNT: usize = 4;
        const KEPT_COUNT: usize = 3;
        let along =
            ReduceRoutineProblem::dense(&[5, 8], 1, ReduceOperation::TopK { k: KEPT_COUNT });
        assert_eq!(
            along.owner_box_at_lane_count(LANE_COUNT),
            [(BATCH, 1), (SLOTS, KEPT_COUNT)]
        );
    }

    /// The boxes of rows are the rows over an owner's, rounded up.
    #[test]
    fn row_boxes_round_up_the_rows_over_an_owner() {
        const LANE_COUNT: usize = 4;
        const ROW_COUNT: usize = 10;
        let across = ReduceRoutineProblem::dense(&[3, ROW_COUNT], 0, ReduceOperation::Sum);
        assert_eq!(
            across.row_box_count_at_lane_count(LANE_COUNT),
            ROW_COUNT.div_ceil(LANE_COUNT)
        );
    }

    #[test]
    fn a_tensor_of_one_axis_is_given_a_row_axis() {
        let problem = ReduceRoutineProblem::dense(&[9], 0, ReduceOperation::Sum);
        let (geometry, dimensions) = problem.labelled_geometry(&Geometry::row_major(&[9]), REDUCED);
        assert_eq!(
            (dimensions, geometry.shape()),
            (vec![BATCH, REDUCED], &[1, 9][..])
        );
    }

    /// A row axis one long after the reduced one holds no step: the reduced axis is innermost.
    #[test]
    fn a_row_axis_one_long_is_left_out() {
        let problem = ReduceRoutineProblem::dense(&[4, 8, 1], 1, ReduceOperation::Sum);
        let (geometry, dimensions) =
            problem.labelled_geometry(&Geometry::row_major(&[4, 8, 1]), REDUCED);
        assert_eq!(
            (dimensions, geometry.shape()),
            (vec![BATCH, REDUCED], &[4, 8][..])
        );
    }

    /// Lines lie along a reduced axis that only row axes one long follow.
    #[test]
    fn the_lines_lie_along_a_reduced_axis_before_axes_one_long() {
        let input = Geometry::row_major(&[4, 8, 1]);
        let out = Geometry::row_major(&[4, 1, 1]);
        let problem =
            ReduceRoutineProblem::stated(&input, &out, 1, ReduceOperation::Sum, sum_elements());
        assert_eq!(problem.lines_along(), REDUCED);
    }

    #[test]
    fn a_tensor_of_one_axis_is_one_row() {
        let problem = ReduceRoutineProblem::dense(&[9], 0, ReduceOperation::Sum);
        assert_eq!(problem.row_count(), 1);
    }

    /// Row axes that step as one merge into one: a dense tensor reduced along its last axis is one
    /// [`BATCH`] axis and the [`REDUCED`] one.
    #[test]
    fn dense_row_axes_merge_into_one() {
        let problem = ReduceRoutineProblem::dense(&[2, 3, 4], 2, ReduceOperation::Sum);
        let (geometry, dimensions) =
            problem.labelled_geometry(&Geometry::row_major(&[2, 3, 4]), REDUCED);
        assert_eq!(dimensions, vec![BATCH, REDUCED]);
        assert_eq!(geometry.shape(), &[6, 4]);
    }

    /// Rows on either side of the reduced axis do not step as one, so [`BATCH`] spans two axes.
    #[test]
    fn rows_on_either_side_of_the_reduced_axis_stay_two_axes() {
        let problem = ReduceRoutineProblem::dense(&[2, 3, 4], 1, ReduceOperation::Sum);
        let (_, dimensions) = problem.labelled_geometry(&Geometry::row_major(&[2, 3, 4]), SLOTS);
        assert_eq!(dimensions, vec![BATCH, SLOTS, BATCH]);
    }

    /// The shape with another reduced extent keeps the rows on either side.
    #[test]
    fn another_reduced_extent_keeps_the_rows() {
        const VALUE_COUNT: usize = 7;
        let problem = ReduceRoutineProblem::dense(&[2, 3, 4], 1, ReduceOperation::Sum);
        assert_eq!(
            problem.shape_with_reduced_count(VALUE_COUNT),
            vec![2, VALUE_COUNT, 4]
        );
    }

    /// Units of a split part, each [`UNIT_LINE_COUNT`] lines.
    const UNIT_COUNT: usize = 64;
    const UNIT_LINE_COUNT: usize = 8;

    /// Along the row, a split part's lines hold `LANE_COUNT` values each.
    #[test]
    fn a_split_part_along_the_row_holds_width_values_a_line() {
        const LANE_COUNT: usize = 4;
        let along = ReduceRoutineProblem::dense(&[2, 4096], 1, ReduceOperation::Sum);
        assert_eq!(
            along.split_part(UNIT_COUNT, UNIT_LINE_COUNT, LANE_COUNT),
            UNIT_COUNT * UNIT_LINE_COUNT * LANE_COUNT
        );
    }

    /// Across rows, a split part's lines hold one value of the row each.
    #[test]
    fn a_split_part_across_rows_holds_one_value_a_line() {
        const LANE_COUNT: usize = 4;
        let across = ReduceRoutineProblem::dense(&[4096, 2], 0, ReduceOperation::Sum);
        assert_eq!(
            across.split_part(UNIT_COUNT, UNIT_LINE_COUNT, LANE_COUNT),
            UNIT_COUNT * UNIT_LINE_COUNT
        );
    }

    #[test]
    fn an_unbound_input_is_read_at_exactly_the_callers_width() {
        const LANE_COUNT: usize = 2;
        let problem = ReduceRoutineProblem::of_unbound(
            &Geometry::row_major(&[4, 8]),
            LANE_COUNT,
            &buffer_at_strides(&[4, 1], &[1, 1]),
            1,
            ReduceOperation::Sum,
            sum_elements(),
        );
        assert_eq!(
            problem.elem_alignments[0],
            ElemAlignment::stored_piece(LANE_COUNT)
        );
    }

    #[test]
    fn an_input_aligned_as_a_geometry_takes_the_lines_that_geometry_takes() {
        let problem = ReduceRoutineProblem {
            elem_alignments: [ElemAlignment::stored_piece(2); 2],
            ..ReduceRoutineProblem::dense(&[4, 8], 1, ReduceOperation::Sum)
        };
        let aligned = problem.with_input_aligned_as(&Geometry::row_major(&[4, 8]));
        assert_eq!(
            aligned.elem_alignments,
            [ElemAlignment::dense_rows(8), ElemAlignment::stored_piece(2)]
        );
    }

    #[test]
    fn indices_are_written_in_their_element_at_their_buffers_alignment() {
        let problem = ReduceRoutineProblem::dense(&[8, 4], 0, ReduceOperation::ArgMax);
        let written = problem.writing_indices(U32, &buffer_at_strides(&[1, 4], &[4, 1]));
        assert_eq!(
            written.indices,
            Some(ReduceIndices {
                element: U32,
                elem_alignment: ElemAlignment::dense_rows(4),
            })
        );
    }

    #[test]
    fn a_line_of_rows_whose_indices_go_out_strided_holds_one_value() {
        let problem = ReduceRoutineProblem::dense(&[8, 4], 0, ReduceOperation::ArgMax);
        let written = problem.writing_indices(U32, &buffer_at_strides(&[1, 4], &[8, 2]));
        assert_eq!(written.line_stays_in(), 1);
    }

    /// Along the reduced axis no index goes out in a line.
    #[test]
    fn a_line_along_the_reduced_axis_reads_whole_beside_strided_indices() {
        const ROW_VALUE_COUNT: usize = 8;
        let problem = ReduceRoutineProblem::dense(
            &[4, ROW_VALUE_COUNT],
            1,
            ReduceOperation::ArgTopK { k: 2 },
        );
        let written = problem.writing_indices(U32, &buffer_at_strides(&[4, 2], &[4, 2]));
        assert_eq!(written.line_stays_in(), ROW_VALUE_COUNT);
    }

    #[test]
    fn a_problem_that_writes_no_index_compiles_it_in_the_outputs_element() {
        let problem = ReduceRoutineProblem::dense(&[4, 8], 1, ReduceOperation::Sum);
        assert_eq!(problem.compiled_elements(), [F32; 4]);
    }

    const GIVEN: ReducePartials = ReducePartials {
        value_count: 64,
        index_count: 32,
        arrival_count: 8,
    };

    #[test]
    fn partials_hold_as_many_cells() {
        assert!(GIVEN.holds(&GIVEN));
    }

    #[test]
    fn fewer_values_than_needed_hold_nothing() {
        let given = ReducePartials {
            value_count: GIVEN.value_count - 1,
            ..ReducePartials::UNBOUNDED
        };
        assert!(!given.holds(&GIVEN));
    }

    #[test]
    fn fewer_indices_than_needed_hold_nothing() {
        let given = ReducePartials {
            index_count: GIVEN.index_count - 1,
            ..ReducePartials::UNBOUNDED
        };
        assert!(!given.holds(&GIVEN));
    }

    #[test]
    fn fewer_arrivals_than_needed_hold_nothing() {
        let given = ReducePartials {
            arrival_count: GIVEN.arrival_count - 1,
            ..ReducePartials::UNBOUNDED
        };
        assert!(!given.holds(&GIVEN));
    }

    #[test]
    fn the_max_takes_the_larger_of_each_operand() {
        let other = ReducePartials {
            value_count: GIVEN.value_count * 2,
            index_count: GIVEN.index_count / 2,
            arrival_count: GIVEN.arrival_count * 2,
        };
        assert_eq!(
            GIVEN.max(other),
            ReducePartials {
                index_count: GIVEN.index_count,
                ..other
            }
        );
    }

    #[test]
    fn a_keyed_problem_is_as_large_as_its_keys_buckets() {
        let problem = ReduceRoutineProblem::dense(&[3, 7, 300], 2, ReduceOperation::Sum);
        assert_eq!(problem.keyed().shape, vec![64, 1024]);
    }

    /// A product's identity is one, where an unstated boundary reads zero.
    #[test]
    fn a_products_input_reads_the_products_identity_past_its_edge() {
        let problem = ReduceRoutineProblem::dense(&[4, 8], 1, ReduceOperation::Prod);
        assert_eq!(
            problem.input_boundary(),
            BoundaryPolicy::Overhanging(Boundary::Identity(Monoid::Prod))
        );
    }

    /// A kernel leaves every dimension to the launch but the cells a row writes.
    #[test]
    fn a_kernel_compiles_only_the_cells_a_row_writes() {
        let problem = ReduceRoutineProblem::dense(&[2, 300], 1, ReduceOperation::Sum);
        let partitioning = Partitioning::new(
            problem.space(),
            cubecl::tile::Levels::leaf(&problem.owner_box_at_lane_count(1))
                .cubes(&[BATCH])
                .build(),
        );
        let space = problem.kernel(&partitioning).space().clone();
        assert!(space.is_dynamic(REDUCED));
        assert!(space.is_dynamic(BATCH));
        assert!(!space.is_dynamic(SLOTS));
    }

    #[test]
    fn buffers_are_read_where_they_lie() {
        let problem = ReduceRoutineProblem::of_buffers(
            &buffer_at_strides(&[4, 8], &[16, 1]),
            &buffer_at_strides(&[4, 1], &[1, 1]),
            1,
            ReduceOperation::Sum,
            sum_elements(),
        );
        assert_eq!(
            problem.elem_alignments[0],
            ElemAlignment::new(&[4, 8], &[16, 1])
        );
    }

    #[test]
    fn a_labelled_buffer_steps_its_merged_row_axes() {
        let problem = ReduceRoutineProblem::dense(&[2, 3, 4], 2, ReduceOperation::Sum);
        let (labelled, _) = problem.labelled(buffer_at_strides(&[2, 3, 4], &[12, 4, 1]), REDUCED);
        let axes: Vec<(usize, usize)> = labelled
            .shape
            .iter()
            .copied()
            .zip(labelled.strides.iter().copied())
            .collect();
        assert_eq!(axes, vec![(6, 4), (4, 1)]);
    }

    #[test]
    fn a_work_item_says_whether_the_indices_are_written() {
        let problem = ReduceRoutineProblem {
            indices: Some(ReduceIndices {
                element: U32,
                elem_alignment: ElemAlignment::ONE,
            }),
            ..ReduceRoutineProblem::dense(&[4, 8], 1, ReduceOperation::ArgMax)
        };
        assert!(problem.work_item().with_indices);
    }

    #[test]
    fn a_split_space_holds_the_partials() {
        const PARTIAL_COUNT: usize = 12;
        let problem = ReduceRoutineProblem::dense(&[4, 8], 1, ReduceOperation::Sum);
        let space = problem.split_space_with_partial_count(PARTIAL_COUNT);
        assert_eq!(space.extent(PARTIALS), PARTIAL_COUNT);
    }

    #[test]
    fn a_split_owner_holds_a_part_of_its_rows() {
        const PART_VALUE_COUNT: usize = 5;
        let problem = ReduceRoutineProblem::dense(&[4, 8], 1, ReduceOperation::Sum);
        assert_eq!(
            problem.split_owner_box(1, PART_VALUE_COUNT),
            [(BATCH, 1), (SLOTS, 1), (REDUCED, PART_VALUE_COUNT)]
        );
    }

    #[test]
    fn a_shape_alone_states_a_dense_problem() {
        let problem =
            ReduceRoutineProblem::of_shape(&[4, 8], 1, ReduceOperation::Sum, sum_elements(), false);
        let dense = ReduceRoutineProblem {
            elem_alignments: [8, 4].map(ElemAlignment::dense_rows),
            ..ReduceRoutineProblem::dense(&[4, 8], 1, ReduceOperation::Sum)
        };
        assert_eq!(problem, dense);
    }
}
