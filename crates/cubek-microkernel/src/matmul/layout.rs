use crate::matmul::MatmulDim;
use crate::matmul::MatmulOperand;
use crate::matmul::stored::Packing;
use cubecl::tile::launch::{Arg, Labelled};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Axis, Geometry, Projection};
use cubecl::zspace::metadata::Metadata;
use cubek_std::layout::contiguous;

/// The order an operand is bound in: the axes the product names, or the order its buffer steps.
///
/// A col rhs (an `[n, k]` buffer behind a `[k, n]` view) bound over the axes the product names
/// steps its innermost bound axis by `k`, so it is read one element at a time; bound in its
/// buffer's order, the same bytes are read in lines. Only a fragment that loads its `B` col-major
/// reads an operand so, and the two orders compile two kernels, so the microkernel states which
/// ([`MatmulMicrokernel::orders`](crate::matmul::MatmulMicrokernel::orders)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulOrder {
    /// As the product names its axes: `[k, n]` for the rhs.
    Stated,
    /// As its buffer steps them: `[n, k]` for a col rhs.
    InBufferOrder,
}

impl core::fmt::Display for MatmulOrder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Stated => write!(f, "stated"),
            Self::InBufferOrder => write!(f, "in-buffer-order"),
        }
    }
}

impl MatmulOrder {
    /// `operand` bound in this order: as the product names it, or in the order its buffer steps.
    /// The one place a binding reads the order, which a blueprint's widths read through
    /// [`innermost`](Self::innermost) and a problem's alignment through
    /// `MatmulRoutineProblem::alignment`.
    pub fn applied(self, operand: Arg<'_, Labelled>) -> Arg<'_, Labelled> {
        match self {
            Self::Stated => operand,
            Self::InBufferOrder => operand.in_stride_order(),
        }
    }

    /// The dimension `operand` is bound innermost along under this order: the one its lines run
    /// along, the last the product names it by, or the first where it is bound as its buffer
    /// steps.
    pub fn innermost(self, operand: MatmulOperand) -> MatmulDim {
        let [outer, inner] = operand.dims();
        match self {
            Self::Stated => inner,
            Self::InBufferOrder => outer,
        }
    }
}

/// How an operand's buffer lies under the view the product names it by (`[k, n]` for the rhs), as
/// the tuning key reads its strides ([`contiguous`]), so a key and the tensors it was made from
/// state one layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulLayout {
    /// Contiguous along the second of its axes: its buffer steps in the order the product names
    /// them. A row rhs, an lhs in rows.
    Row,
    /// Contiguous along the first: its buffer is the view's transpose. A col rhs, an `[n, k]`
    /// buffer behind the view.
    Col,
    /// Contiguous along neither: read one element at a time in either order.
    Strided,
}

impl MatmulLayout {
    /// Which of `axes` a matrix of `extents` at `strides` is contiguous along, and what its buffer
    /// steps across its lines — or `None` for a matrix no cut here walks.
    ///
    /// The reading, apart from the tensor it is usually read off, because both of
    /// its refusals are about the strides alone. The cuts walk one axis in vectors
    /// and step rows of the other, so the matrix has to line along one of them; a
    /// dim of extent one addresses nothing and never competes for the unit stride.
    /// Padding is served, but a stride *below* the row it steps over is a
    /// narrowed, permuted or aliasing view this binding cannot describe.
    pub fn lines(
        axes: [Axis; 2],
        extents: [usize; 2],
        strides: [usize; 2],
    ) -> Option<(Axis, usize)> {
        let axis = contiguous(&Metadata::new(extents, strides), axes)?;
        if !axes.contains(&axis) {
            return None;
        }
        let matrix = Projection::direct(&axes);
        let dims = Geometry::new(&[(extents[0], strides[0]), (extents[1], strides[1])]);
        if !matrix.is_addressable(&dims) {
            return None;
        }
        let stepped = match axis == axes[0] {
            true => axes[1],
            false => axes[0],
        };
        // A direct projection names its dims in the order it spans them, so the
        // axis's position is its dim.
        Some((axis, strides[matrix.position(stepped)]))
    }

    /// The layout of `operand` stated by `meta`, its batch dims leading: the axis it is contiguous
    /// along, in rows or in tiles alike ([`contiguous`]).
    pub fn new(meta: &Metadata, operand: MatmulOperand) -> Self {
        Self::along(contiguous(meta, operand.axes()), operand)
    }

    /// The layout of `operand` contiguous along `axis`: col along the first of its axes, row
    /// along the second, strided along neither.
    pub fn along(axis: Option<Axis>, operand: MatmulOperand) -> Self {
        match axis {
            Some(axis) if axis == operand.axes()[0] => Self::Col,
            Some(_) => Self::Row,
            None => Self::Strided,
        }
    }

    /// How `operand` lies packed as `packing` says, as a float one of its shape would: a col one
    /// where its words run along the first of its axes, a row one otherwise.
    pub fn of_packing(packing: &Packing, operand: MatmulOperand) -> Self {
        match packing.operand.lined() == operand.axes()[0] {
            true => Self::Col,
            false => Self::Row,
        }
    }

    /// The axis `operand` laid this way is contiguous along; `None` for a strided one.
    pub fn contiguous_axis(self, operand: MatmulOperand) -> Option<Axis> {
        let [first, second] = operand.axes();
        match self {
            Self::Col => Some(first),
            Self::Row => Some(second),
            Self::Strided => None,
        }
    }

    /// The strides of a dense matrix of `extents` laid this way; `None` for a strided one, which
    /// no dense buffer is.
    pub fn dense_strides(self, [first, second]: [usize; 2]) -> Option<[usize; 2]> {
        match self {
            Self::Row => Some([second, 1]),
            Self::Col => Some([1, first]),
            Self::Strided => None,
        }
    }

    /// The alignment of an operand of `shape` at `strides` over the order its buffer steps: a col
    /// one's over its transposed buffer, any other's over the axes the product names.
    pub fn buffer_alignment(self, shape: &[usize], strides: &[usize]) -> ElemAlignment {
        match self {
            Self::Col => {
                let rank = shape.len();
                let (mut shape, mut strides) = (shape.to_vec(), strides.to_vec());
                shape.swap(rank - 2, rank - 1);
                strides.swap(rank - 2, rank - 1);
                ElemAlignment::new(&shape, &strides)
            }
            Self::Row | Self::Strided => ElemAlignment::new(shape, strides),
        }
    }
}

impl core::fmt::Display for MatmulLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Row => "row",
            Self::Col => "col",
            Self::Strided => "strided",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubek_std::matmul_axes::{K, N};

    /// A rhs's layout is the tuning key's reading of its strides: a pitched, batched or broadcast
    /// col rhs included, and a `[k, 1]` rhs whose one column strides by anything.
    #[test]
    fn the_layout_is_the_keys_reading() {
        let (k, n) = (64, 48);
        let cases: [(&[usize], &[usize]); 7] = [
            (&[k, n], &[1, k]),
            (&[k, n], &[1, k + 8]),
            (&[3, k, n], &[k * n, 1, k]),
            (&[3, k, n], &[0, 1, k]),
            (&[k, 1], &[1, 1]),
            (&[k, n], &[n, 1]),
            (&[k, n], &[2, 2 * k]),
        ];
        for (shape, strides) in cases {
            let meta = Metadata::new(shape.to_vec(), strides.to_vec());
            let layout = MatmulLayout::new(&meta, MatmulOperand::Rhs);
            let expected = match contiguous(&meta, [K, N]) {
                Some(axis) if axis == K => layout == MatmulLayout::Col,
                Some(_) => layout == MatmulLayout::Row,
                None => layout == MatmulLayout::Strided,
            };
            assert!(expected, "{shape:?} at {strides:?}: {layout:?}");
        }
    }

    /// Over its buffer's order a pitched col rhs is aligned to what its pitch divides, which
    /// its stated alignment, one element, does not show.
    #[test]
    fn a_col_rhs_buffer_alignment_reads_its_pitch() {
        let (k, n) = (64, 48);
        let (shape, strides) = ([k, n], [1, k + 8]);
        let pitched = MatmulLayout::Col.buffer_alignment(&shape, &strides);
        assert!(pitched.admits(8) && !pitched.admits(16));
        assert_eq!(
            MatmulLayout::Row.buffer_alignment(&shape, &strides),
            ElemAlignment::new(&shape, &strides)
        );
    }
}
