use cubecl::std::{
    FastDivmod,
    tensor::{
        View, ViewMut,
        layout::{Coords1d, Layout, LayoutExpand},
    },
};
use cubecl::{prelude::*, std::tensor::launch::ViewArg};

use crate::scheme::QuantScheme;

/// Layout for quantization scales, indexed by quant element index and returns the corresponding
/// scale based on the quantization type.
///
/// Per-tensor is the degenerate case where every axis's block covers its extent: each term of the
/// address folds away at comptime and a read is a constant-index broadcast, so it needs no layout
/// of its own.
#[derive(CubeType, CubeLaunch)]
pub struct ScalesLayout {
    tensor_shape: Sequence<FastDivmod<usize>>,
    tensor_len: usize,
    scales_strides: Sequence<usize>,
    /// Per-axis block edges in elements, resolved against the tensor so no dimension is
    /// [`BlockSize::FULL`], which divides as a zero.
    #[cube(comptime)]
    block_size: Vec<usize>,
    /// Per-axis extent in elements, so the last axis counts quants rather than packed stores.
    #[cube(comptime)]
    element_shape: Vec<usize>,
    #[cube(comptime)]
    scales_vector_size: usize,
}

#[cube]
impl ScalesLayout {
    pub fn new(
        tensor_shape: Sequence<FastDivmod<usize>>,
        tensor_len: usize,
        scales_strides: Sequence<usize>,
        #[comptime] block_size: Vec<usize>,
        #[comptime] element_shape: Vec<usize>,
        #[comptime] scales_vector_size: usize,
    ) -> Self {
        ScalesLayout {
            tensor_shape,
            tensor_len,
            scales_strides,
            block_size,
            element_shape,
            scales_vector_size,
        }
    }

    /// Whether axis `p` still distinguishes scales: an extent that fits inside its block leaves one
    /// scale for the whole axis, so the term is dropped at comptime rather than dividing a runtime
    /// coordinate that can only answer `0`.
    fn addresses(&self, #[comptime] p: usize) -> comptime_type!(bool) {
        comptime!(self.element_shape[p] > self.block_size[p])
    }

    /// The outermost axis that still addresses a scale, or the rank when none does. The divmod
    /// chain threads inward to outward, so every axis past this one is dead.
    fn outermost_addressed(&self) -> comptime_type!(usize) {
        comptime!(
            (0..self.block_size.len())
                .find(|&p| self.element_shape[p] > self.block_size[p])
                .unwrap_or(self.block_size.len())
        )
    }
}

#[cube]
impl Layout for ScalesLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let rank = comptime!(self.block_size.len());
        let outermost = self.outermost_addressed();

        if comptime!(outermost == rank) {
            // Every axis holds one scale, so the tensor holds exactly one.
            0usize.runtime()
        } else {
            let mut offs = pos;
            let mut scale_offs = 0;

            #[unroll]
            for i in 0..comptime!(rank - outermost) {
                let dim = comptime!(rank - i - 1);
                let (rem, offs_local) = self.tensor_shape[dim].div_mod(offs);
                offs = rem;

                // An unaddressed axis still has to divide, since the axes outside it read `rem`.
                if self.addresses(dim) {
                    let block_size_local = comptime!(self.block_size[dim]);
                    scale_offs += (offs_local / block_size_local) * self.scales_strides[dim];
                }
            }

            scale_offs / self.scales_vector_size
        }
    }

    fn shape(&self) -> Self::Coordinates {
        self.tensor_len
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos < self.tensor_len
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}

#[cube]
impl ScalesLayout {
    /// Whether the position is at the start of a new block. Used for electing a unit to write each
    /// scale.
    pub fn is_block_start(&self, pos: usize) -> bool {
        let rank = comptime!(self.block_size.len());
        let outermost = self.outermost_addressed();

        if comptime!(outermost == rank) {
            // Every axis covered leaves one scale for the whole tensor, and only its first element
            // starts a block. Dropping the terms instead would elect every unit to write it.
            pos == 0
        } else {
            let mut offs = pos;
            let mut is_start = true;

            #[unroll]
            for i in 0..rank {
                let dim = comptime!(rank - i - 1);
                let (rem, offs_local) = self.tensor_shape[dim].div_mod(offs);
                offs = rem;

                // No axis drops out here, unlike the address: a covered one still demands its
                // first element, and `offs_local` never reaching the block edge reduces that
                // modulo to a comparison.
                if self.addresses(dim) {
                    let block_size_local = comptime!(self.block_size[dim]);
                    is_start &= offs_local.is_multiple_of(block_size_local);
                } else {
                    is_start &= offs_local == 0;
                }
            }

            is_start
        }
    }
}

/// TensorView with a linear layout inferred from the shape/strides at launch.
/// Useful for elementwise kernels.
pub type ScalesView<'a, E> = View<'a, E, Coords1d>;
pub type ScalesViewMut<'a, E> = ViewMut<'a, E, Coords1d>;
/// Launch type for LinearTensorView.
pub type ScalesViewLaunch<R> = ViewArg<Coords1d, R>;

/// Create a scales view from the values and scales handle, vector size and quantization scheme.
/// `values` should be *the quantized tensor*, and will be adjusted by `num_quants`.
pub fn scales_view<R: Runtime>(
    values: TensorBinding<R>,
    scales: TensorBinding<R>,
    scales_vector_size: usize,
    quant_scheme: &QuantScheme,
) -> ScalesViewLaunch<R> {
    let layout = scales_layout(&values, &scales, scales_vector_size, quant_scheme);
    let len = scales.shape.iter().product::<usize>();
    let buffer = unsafe { BufferArg::from_raw_parts_binding(scales.handle, len) };
    ScalesViewLaunch::new_array::<ScalesLayout>(buffer, layout)
}

pub fn scales_layout<R: Runtime>(
    values: &TensorBinding<R>,
    scales: &TensorBinding<R>,
    scales_vector_size: usize,
    scheme: &QuantScheme,
) -> ScalesLayoutLaunch<R> {
    if scheme.num_levels() > 1 {
        unimplemented!("two-level quantization is not supported here, got {scheme:?}");
    }

    let element_shape = element_shape(&values.shape, scheme.num_quants());
    let values_len = element_shape.iter().product::<usize>();
    // Whole-tensor granularity is one block spanning every axis, which is what makes the layout
    // below branchless.
    let block_size = match scheme.block_size() {
        None => element_shape.clone(),
        Some(block) => block.resolved_dims(&element_shape),
    };

    ScalesLayoutLaunch::new(
        shape_divmod(&element_shape),
        values_len,
        strides_seq(&scales.strides),
        block_size,
        element_shape,
        scales_vector_size,
    )
}

/// The values' extents in elements. Its last axis is stored packed, `num_quants` to an entry, and
/// both the divmod chain and the block have to read the extents the coordinates actually span.
fn element_shape(shape: &[usize], num_quants: usize) -> Vec<usize> {
    let mut shape = shape.to_vec();
    *shape.last_mut().unwrap() *= num_quants;
    shape
}

fn shape_divmod<R: Runtime>(shape: &[usize]) -> SequenceArg<R, FastDivmod<usize>> {
    let mut out_seq = SequenceArg::new();
    for s in shape {
        out_seq.push(*s);
    }
    out_seq
}

fn strides_seq<R: Runtime>(strides: &[usize]) -> SequenceArg<R, usize> {
    let mut out_seq = SequenceArg::new();
    for s in strides {
        out_seq.push(*s);
    }
    out_seq
}
