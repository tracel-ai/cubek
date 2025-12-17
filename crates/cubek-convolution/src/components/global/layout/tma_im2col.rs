use cubecl::{
    prelude::*,
    std::{
        FastDivmod,
        tensor::layout::{Coords3d, CoordsDyn, Layout, LayoutExpand},
    },
};

use crate::components::{ConvolutionOperation, ConvolutionParams, global::layout::NhwcCoords};

/// Im2col layout, producing both the position and offset
#[derive(CubeType, CubeLaunch)]
pub struct TmaIm2colLayout {
    shape_out: Sequence<FastDivmod>,
    padded_channels: FastDivmod,
    #[cube(comptime)]
    params: ConvolutionParams,
    #[cube(comptime)]
    check_kernel: bool,
}

#[cube]
impl TmaIm2colLayout {
    pub fn new(
        shape_out: Sequence<FastDivmod>,
        padded_channels: FastDivmod,
        #[comptime] params: ConvolutionParams,
        #[comptime] check_kernel: bool,
    ) -> Self {
        TmaIm2colLayout {
            shape_out,
            padded_channels,
            params,
            check_kernel,
        }
    }
}

#[cube]
impl Layout for TmaIm2colLayout {
    type Coordinates = Coords3d;
    type SourceCoordinates = (NhwcCoords, CoordsDyn);

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (_, m, k) = pos;
        let params = comptime![self.params];

        let (n_offs, spatial_offsets) = div_mod_seq(m, &self.shape_out);
        let spatial_dims = spatial_offsets.len();

        let mut in_offs = Sequence::<i32>::new();

        #[unroll]
        for dim in 0..spatial_dims {
            let dim = comptime![dim as usize];
            let stride = comptime!(params.stride[dim] as i32);
            let pad = comptime!(params.padding[dim]);
            let out_pos = *spatial_offsets.index(comptime![dim as u32]) as i32;
            let offs = match params.operation {
                ConvolutionOperation::Forward | ConvolutionOperation::BackwardWeight => {
                    out_pos * stride - pad
                }
                ConvolutionOperation::ForwardTransposed | ConvolutionOperation::BackwardData => {
                    let ksize = comptime!(params.kernel_size[dim] as i32);
                    (out_pos + pad - comptime!((ksize - 1) * params.dilation[dim] as i32)) / stride
                }
            };
            in_offs.push(offs);
        }

        let (mut k_idx, channel_start) = self.padded_channels.div_mod(k);

        let mut pos = NhwcCoords {
            batch: n_offs,
            spatial: in_offs,
            channel: channel_start,
        };

        let mut k_offs = Sequence::new();
        let k_rank = params.dimensionality.num_dims();

        #[unroll]
        for i in 0..k_rank {
            let dim = comptime![(k_rank - i - 1) as usize];
            let k_size = comptime!(params.kernel_size[dim]);
            let k_pos = k_idx % k_size;

            let k_pos = match params.operation {
                ConvolutionOperation::Forward | ConvolutionOperation::BackwardWeight => k_pos,
                ConvolutionOperation::ForwardTransposed | ConvolutionOperation::BackwardData => {
                    // Since kernels are always positive, we need to subtract the bottom right
                    // corner (see position above), then add the inverted index to it.
                    k_size - k_pos - 1
                }
            };
            k_offs.push(k_pos * comptime!(params.dilation[dim]));
            k_idx /= k_size;
        }

        if comptime![self.check_kernel] {
            // This is the largest index that's aligned to the channel count in all cases.
            // Alignment is 256, and that's the largest tile size possible with TMA.
            // Could alternatively solve this by only loading if in bounds, and adjusting the awaited
            // bytes by the in-bounds tiles but that's more complicated than just trying to load a very
            // large channel index and letting bounds checks handle it.
            let kernel_mask = (k_idx > 0) as u32 * 0x7FFFFF00u32;
            pos.channel = Max::max(pos.channel, kernel_mask);
        }

        (pos, k_offs.rev())
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }

    fn shape(&self) -> Self::Coordinates {
        (u32::MAX, u32::MAX, u32::MAX).runtime()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}

/// Decompose a linear index into local positions along each dimension in `shape`. Also returns the
/// left over remainder.
#[cube]
pub(crate) fn div_mod_seq(pos: u32, shape: &Sequence<FastDivmod>) -> (u32, Sequence<u32>) {
    let rank = comptime![shape.len()];
    let mut offs = pos;
    let mut out = Sequence::new();

    #[unroll]
    for i in 0..rank {
        let dim = comptime![rank - i - 1];
        let (rem, offs_local) = shape.index(dim).div_mod(offs);
        out.push(offs_local);
        offs = rem;
    }

    (offs, out.rev())
}
