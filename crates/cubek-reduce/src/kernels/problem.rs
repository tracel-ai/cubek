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

/// The rows: every axis of the input but the reduced one, folded into one dimension, the first
/// axis coarsest.
pub const BATCH: Dimension = Dimension(0);
/// The axis a row runs along, which the reduction folds away.
pub const REDUCED: Dimension = Dimension(1);
/// The cells a row writes, `k` for a top-k: the output's axis in place of the reduced one.
pub const TOP: Dimension = Dimension(2);
/// A hypercube's partials, every part's carrier for a row side by side: the partials' axis in place
/// of the reduced one.
pub const PARTIALS: Dimension = Dimension(3);

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
    /// otherwise. A sum, a product or a mean accumulates in [`widened`] elements, every other
    /// operation in the input's own: an extremum gains nothing from a wider carrier, and an index
    /// or a flag is carried apart from the values.
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
/// Its space has fixed dimensions, whatever the rank: [`BATCH`], every axis but `axis` folded into
/// one, [`REDUCED`], `axis` itself, and [`TOP`], the cells a row writes, which the output holds in
/// place of `axis`. Each operand's layout maps its axes onto them ([`fold`](Self::fold)).
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
        let expected: Vec<usize> = shape
            .iter()
            .enumerate()
            .map(|(d, extent)| match d == axis {
                true => operation.written_per_row(),
                false => *extent,
            })
            .collect();
        assert_eq!(
            out.shape(),
            expected.as_slice(),
            "a reduction along axis {axis} writes the input's shape with that axis {} wide",
            operation.written_per_row()
        );
        // A line runs along the innermost axis where both operands step one element along it.
        let inner = shape.len() - 1;
        let steps_one =
            |geometry: &Geometry| geometry.strides()[inner] == 1 || geometry.shape()[inner] == 1;
        let contiguous = (steps_one(input) && steps_one(out)).then_some(inner);
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

    /// `binding`, an operand of the input's axes holding `along` in place of the reduced axis
    /// (`REDUCED` for the input, `TOP` for the output and its indices, `PARTIALS` for a
    /// hypercube's partials), as its layout maps it onto the space, and the dimension each of its
    /// axes spans ([`folded`](Self::folded)).
    pub fn fold(
        &self,
        mut binding: TensorBinding,
        along: Dimension,
    ) -> (TensorBinding, Vec<Dimension>) {
        let (axes, labels) = self.folded(&binding.shape, &binding.strides, along);
        binding.shape = axes.iter().map(|&(extent, _)| extent).collect();
        binding.strides = axes.iter().map(|&(_, stride)| stride).collect();
        (binding, labels)
    }

    /// [`fold`](Self::fold) of an operand with no tensor, stated by its `geometry`.
    pub fn fold_geometry(
        &self,
        geometry: &Geometry,
        along: Dimension,
    ) -> (Geometry, Vec<Dimension>) {
        let (axes, labels) = self.folded(geometry.shape(), geometry.strides(), along);
        (Geometry::new(&axes), labels)
    }

    /// The `(extent, stride)` of each axis of an operand of `shape` at `strides`, in the order it
    /// holds them, and the dimension each spans: `along` for the reduced axis, [`BATCH`] for every
    /// other, so [`BATCH`] spans the row axes, the first coarsest, on either side of the reduced
    /// one. Neighbouring row axes that step as one axis (the outer's stride the inner's whole span)
    /// merge into one, so a contiguous operand's rows before and after the reduced axis are one
    /// axis each. An operand of one axis is given a row axis of one.
    fn folded(
        &self,
        shape: &[usize],
        strides: &[usize],
        along: Dimension,
    ) -> (Vec<(usize, usize)>, Vec<Dimension>) {
        let mut folded: Vec<(Dimension, (usize, usize))> = Vec::new();
        for d in 0..shape.len() {
            let label = match d == self.axis {
                true => along,
                false => BATCH,
            };
            let axis = (shape[d], strides[d]);
            let merged = match folded.last() {
                Some(&(BATCH, outer)) if label == BATCH => merged_rows(outer, axis),
                _ => None,
            };
            match merged {
                Some(merged) => {
                    folded.last_mut().expect("a merged axis has an outer one").1 = merged
                }
                None => folded.push((label, axis)),
            }
        }
        if !folded.iter().any(|&(label, _)| label == BATCH) {
            folded.insert(0, (BATCH, (1, 1)));
        }
        folded
            .into_iter()
            .map(|(label, axis)| (axis, label))
            .unzip()
    }

    /// How the input is checked where a tile reaches past its edge: a read there returns the
    /// identity of the operation's edge monoid ([`ReduceOperation::edge_monoid`]), which leaves
    /// the total unchanged (`tracel:cubecl::edge-reads-identity`).
    pub fn input_boundary(&self) -> BoundaryPolicy {
        BoundaryPolicy::Overhanging(Boundary::Identity(self.operation.edge_monoid()))
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
            (TOP, self.operation.written_per_row()),
        ])
    }

    /// [`space`](Self::space) and a hypercube's partials, `partial_extent` values of each box of
    /// rows'.
    pub fn split_space(&self, partial_extent: usize) -> Space {
        Space::new(&[
            (BATCH, self.row_count()),
            (REDUCED, self.reduced_extent()),
            (TOP, self.operation.written_per_row()),
            (PARTIALS, partial_extent),
        ])
    }

    /// The box an owner holds: one row, or `width` rows side by side across the reduced axis, and
    /// the cells it writes. It leaves the reduced axis to the space, so every owner spans its rows
    /// whole.
    pub fn owner_box(&self, width: usize) -> Vec<(Dimension, usize)> {
        let rows = match self.layout() {
            ReduceLayout::Across => width,
            ReduceLayout::Along => 1,
        };
        vec![(BATCH, rows), (TOP, self.operation.written_per_row())]
    }

    /// How many boxes of rows the owners, `width` rows across the reduced axis, take.
    pub fn row_boxes(&self, width: usize) -> usize {
        let rows = match self.layout() {
            ReduceLayout::Across => width,
            ReduceLayout::Along => 1,
        };
        self.row_count().div_ceil(rows)
    }

    /// `partitioning` as a kernel compiles it: every dimension but the cells a row writes left to
    /// the launch, the row length too, as today's reduce kernels read it, so one kernel serves
    /// every shape of a rank, layout, elements and widths. An owner's window is one row high, which
    /// its flat view reads with no division by the runtime length.
    pub fn kernel(&self, partitioning: &Partitioning) -> Partitioning {
        let space = partitioning.space();
        let dynamic: Vec<Dimension> = space
            .dimensions()
            .filter(|dimension| *dimension != TOP)
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
}
