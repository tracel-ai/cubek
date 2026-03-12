use cubecl::prelude::*;
use cubecl::std::{
    FastDivmod, FastDivmodArgs,
    tensor::{
        launch::{TypedView, TypedViewLaunch},
        layout::{Coords1d, Layout, LayoutExpand},
    },
};

use crate::scheme::{QuantLevel, QuantScheme};

/// Layout for quantization scales, indexed by quant element index and returns the corresponding
/// scale based on the quantization type.
#[derive(CubeType, CubeLaunch)]
pub enum ScalesLayout {
    PerTensor(PerTensorLayout),
    BlockScaled(BlockScaledLayout),
}

#[cube]
impl Layout for ScalesLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        match self {
            ScalesLayout::PerTensor(layout) => layout.to_source_pos(pos),
            ScalesLayout::BlockScaled(layout) => layout.to_source_pos(pos),
        }
    }

    fn shape(&self) -> Self::Coordinates {
        match self {
            ScalesLayout::PerTensor(layout) => layout.shape(),
            ScalesLayout::BlockScaled(layout) => layout.shape(),
        }
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        match self {
            ScalesLayout::PerTensor(layout) => layout.is_in_bounds(pos),
            ScalesLayout::BlockScaled(layout) => layout.is_in_bounds(pos),
        }
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        match self {
            ScalesLayout::PerTensor(layout) => layout.to_source_pos_checked(pos),
            ScalesLayout::BlockScaled(layout) => layout.to_source_pos_checked(pos),
        }
    }
}

#[cube]
impl ScalesLayout {
    /// Whether the position is at the start of a new block. Used for electing a unit to write each
    /// scale.
    pub fn is_block_start(&self, pos: usize) -> bool {
        match self {
            ScalesLayout::PerTensor(layout) => layout.is_block_start(pos),
            ScalesLayout::BlockScaled(layout) => layout.is_block_start(pos),
        }
    }
}

#[derive(CubeType, CubeLaunch)]
pub struct PerTensorLayout {
    tensor_len: usize,
}

#[cube]
impl PerTensorLayout {
    pub fn new(tensor_len: usize) -> Self {
        PerTensorLayout { tensor_len }
    }
}

#[cube]
impl Layout for PerTensorLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, _pos: Self::Coordinates) -> Self::SourceCoordinates {
        0usize.runtime()
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
impl PerTensorLayout {
    /// Whether the position is at the start of a new block. Used for electing a unit to write each
    /// scale.
    pub fn is_block_start(&self, pos: usize) -> bool {
        pos == 0
    }
}

#[derive(CubeType, CubeLaunch)]
pub struct BlockScaledLayout {
    tensor_shape: Sequence<FastDivmod<usize>>,
    tensor_len: usize,
    scales_strides: Sequence<usize>,
    #[cube(comptime)]
    block_size: Vec<u8>,
    #[cube(comptime)]
    scales_vector_size: usize,
}

#[cube]
impl BlockScaledLayout {
    pub fn new(
        tensor_shape: Sequence<FastDivmod<usize>>,
        tensor_len: usize,
        scales_strides: Sequence<usize>,
        #[comptime] block_size: Vec<u8>,
        #[comptime] scales_vector_size: usize,
    ) -> Self {
        BlockScaledLayout {
            tensor_shape,
            tensor_len,
            scales_strides,
            block_size,
            scales_vector_size,
        }
    }
}

#[cube]
impl Layout for BlockScaledLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let rank = self.scales_strides.len().comptime();
        let mut offs = pos;
        let mut scale_offs = 0;

        #[unroll]
        for i in 0..rank {
            let dim = rank - i - 1;
            let block_size_local = comptime![self.block_size[dim] as usize];
            let (rem, offs_local) = self.tensor_shape[dim].div_mod(offs);

            offs = rem;
            scale_offs += (offs_local / block_size_local) * self.scales_strides[dim];
        }

        scale_offs / self.scales_vector_size
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
impl BlockScaledLayout {
    /// Whether the position is at the start of a new block. Used for electing a unit to write each
    /// scale.
    pub fn is_block_start(&self, pos: usize) -> bool {
        let rank = self.scales_strides.len().comptime();
        let mut offs = pos;
        let mut is_start = true;

        #[unroll]
        for i in 0..rank {
            let dim = rank - i - 1;
            let block_size_local = comptime![self.block_size[dim] as usize];
            let (rem, offs_local) = self.tensor_shape[dim].div_mod(offs);
            offs = rem;
            is_start &= offs_local.is_multiple_of(block_size_local);
        }

        is_start
    }
}

/// TensorView with a linear layout inferred from the shape/strides at launch.
/// Useful for elementwise kernels.
pub type ScalesView<E, IO = ReadOnly> = TypedView<E, ScalesLayout, IO>;
/// Launch type for LinearTensorView.
pub type ScalesViewLaunch<R> = TypedViewLaunch<ScalesLayout, R>;

/// Create a scales view from the values and scales handle, vector size and quantization scheme.
/// `values` should be *the quantized tensor*, and will be adjusted by `num_quants`.
pub fn scales_view<R: Runtime>(
    client: &ComputeClient<R>,
    values: TensorBinding<R>,
    scales: TensorBinding<R>,
    scales_vector_size: usize,
    quant_scheme: &QuantScheme,
) -> ScalesViewLaunch<R> {
    let layout = scales_layout(client, &values, &scales, scales_vector_size, quant_scheme);
    let len = scales.shape.iter().product::<usize>();
    let buffer = unsafe { ArrayArg::from_raw_parts_binding(scales.handle, len) };
    ScalesViewLaunch::new(buffer, layout)
}

pub fn scales_layout<R: Runtime>(
    client: &ComputeClient<R>,
    values: &TensorBinding<R>,
    scales: &TensorBinding<R>,
    scales_vector_size: usize,
    scheme: &QuantScheme,
) -> ScalesLayoutArgs<R> {
    let values_len = values.shape.iter().product::<usize>() * scheme.num_quants();

    match &scheme.level {
        QuantLevel::Tensor => ScalesLayoutArgs::PerTensor(PerTensorLayoutLaunch::new(values_len)),
        QuantLevel::Block(block_size) => {
            let tensor_shape = shape_divmod_quant(client, &values.shape, scheme.num_quants());
            let scales_strides = strides_seq(client, &scales.strides);
            ScalesLayoutArgs::BlockScaled(BlockScaledLayoutLaunch::new(
                tensor_shape,
                values_len,
                scales_strides,
                block_size.to_dim_vec(values.shape.len()),
                scales_vector_size,
            ))
        }
    }
}

fn shape_divmod_quant<R: Runtime>(
    client: &ComputeClient<R>,
    shape: &[usize],
    num_quants: usize,
) -> SequenceArg<R, FastDivmod<usize>> {
    let mut out_seq = SequenceArg::new();
    for s in &shape[..shape.len() - 1] {
        out_seq.push(FastDivmodArgs::<usize>::new(client, *s));
    }
    let last = *shape.last().unwrap() * num_quants;
    out_seq.push(FastDivmodArgs::<usize>::new(client, last));
    out_seq
}

fn strides_seq<R: Runtime>(_client: &ComputeClient<R>, strides: &[usize]) -> SequenceArg<R, usize> {
    let mut out_seq = SequenceArg::new();
    for s in strides {
        out_seq.push(*s);
    }
    out_seq
}
