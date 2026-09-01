use cubecl::{
    client::ComputeClient,
    ir::ElemType,
    prelude::Runtime,
    throughput::{ThroughputKey, ThroughputMode},
    tune::Work,
};
use cubek_matmul::definition::{MatmulCost, MatmulGlobalElems};

/// Minimal representation of a 2D convolution's cost: the shapes it moves and
/// the element types it moves them in.
#[derive(Debug, Clone)]
pub struct Conv2dCost {
    /// Batch, input channels, and the input's spatial extent.
    pub batch: usize,
    pub channels_in: usize,
    pub spatial_in: [usize; 2],
    /// Output channels and the filter's spatial extent.
    pub channels_out: usize,
    pub kernel: [usize; 2],
    /// The output's spatial extent, which stride, padding and dilation decide.
    pub spatial_out: [usize; 2],
    /// Elements of the bias, zero when the convolution has none.
    pub bias_elems: usize,
    /// Global element types of the operands.
    pub elems: MatmulGlobalElems,
}

impl Conv2dCost {
    /// The implicit gemm this convolution is: one output pixel per row, one
    /// output channel per column, contracting over the filter's input window.
    ///
    /// Its traffic is not the gemm's, since the operands are the maps and the
    /// filter rather than the unfolded matrices, so only the compute comes
    /// from here.
    fn gemm(&self) -> MatmulCost {
        let [h_out, w_out] = self.spatial_out;
        let [k_h, k_w] = self.kernel;

        MatmulCost {
            batches: 1,
            m: self.batch * h_out * w_out,
            n: self.channels_out,
            k: self.channels_in * k_h * k_w,
            elems: self.elems.clone(),
        }
    }

    /// Calculates the compute operations and compulsory memory traffic for the
    /// convolution.
    pub fn work(&self) -> Work {
        let (read, written) = self.traffic();

        Work {
            compute_ops: self.compute_ops(),
            bytes: read + written,
        }
    }

    /// Operations the contraction performs, `2 * k - 1` per output element.
    pub fn compute_ops(&self) -> usize {
        self.gemm().compute_ops()
    }

    /// Compulsory global traffic in bytes, split by direction, which
    /// [`work`](Self::work) sums.
    ///
    /// The maps and the filter are each moved once. An unfolded operand would
    /// re-read every overlapping window, which is the implementation's choice
    /// rather than the convolution's cost.
    pub fn traffic(&self) -> (usize, usize) {
        let [h_in, w_in] = self.spatial_in;
        let [h_out, w_out] = self.spatial_out;
        let [k_h, k_w] = self.kernel;

        let input = self.batch * self.channels_in * h_in * w_in;
        let filter = self.channels_out * self.channels_in * k_h * k_w;
        let output = self.batch * self.channels_out * h_out * w_out;

        (
            input * self.elems.lhs.size()
                + filter * self.elems.rhs.size()
                // A bias is added to the accumulator, so it is stored in the
                // output's type rather than the filter's.
                + self.bias_elems * self.elems.out.size(),
            output * self.elems.out.size(),
        )
    }

    /// The probe the contraction's arithmetic runs on, which is the matmul's:
    /// an accelerated convolution issues the same MMA a gemm does.
    pub fn compute_key<R: Runtime>(&self, client: &ComputeClient<R>) -> ThroughputKey {
        self.gemm().compute_key(client)
    }
}

/// Minimal representation of a depthwise convolution's cost. Each channel has
/// its own filter and contracts over nothing else, so this is not the implicit
/// gemm [`Conv2dCost`] describes.
#[derive(Debug, Clone, Copy)]
pub struct DepthwiseCost {
    pub batch: usize,
    pub channels: usize,
    /// The input's spatial extent, square.
    pub size: usize,
    /// The output's spatial extent, square.
    pub out_size: usize,
    /// The filter's spatial extent, square.
    pub kernel: usize,
    /// Element type of the maps and the filter.
    pub dtype: ElemType,
}

impl DepthwiseCost {
    /// Calculates the compute operations and compulsory memory traffic for the
    /// convolution.
    pub fn work(&self) -> Work {
        let (read, written) = self.traffic();

        Work {
            compute_ops: self.compute_ops(),
            bytes: read + written,
        }
    }

    /// `2 * taps - 1` per output element: one multiply per tap and the adds
    /// that join them.
    pub fn compute_ops(&self) -> usize {
        let taps = self.kernel * self.kernel;

        self.batch * self.out_size * self.out_size * self.channels * (2 * taps).saturating_sub(1)
    }

    /// Compulsory global traffic in bytes, split by direction, which
    /// [`work`](Self::work) sums.
    pub fn traffic(&self) -> (usize, usize) {
        let maps = self.batch * self.channels;
        let filter = self.channels * self.kernel * self.kernel;
        let size = self.dtype.size();

        (
            (maps * self.size * self.size + filter) * size,
            maps * self.out_size * self.out_size * size,
        )
    }

    /// A depthwise pass contracts over one filter window and issues no MMA, so
    /// its arithmetic is the scalar peak's.
    pub fn compute_key(&self) -> ThroughputKey {
        ThroughputKey {
            mode: ThroughputMode::ComputeDirect { dtype: self.dtype },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};

    fn cost() -> Conv2dCost {
        let f16 = ElemType::Float(FloatKind::F16);

        Conv2dCost {
            batch: 2,
            channels_in: 3,
            spatial_in: [8, 8],
            channels_out: 4,
            kernel: [3, 3],
            spatial_out: [6, 6],
            bias_elems: 0,
            elems: MatmulGlobalElems {
                lhs: f16,
                rhs: f16,
                out: f16,
            },
        }
    }

    #[test]
    fn contracts_over_the_filter_window_for_every_output_pixel() {
        let outputs = 2 * 6 * 6 * 4;
        let taps = 3 * 3 * 3;

        assert_eq!(cost().compute_ops(), outputs * (2 * taps - 1));
    }

    #[test]
    fn moves_the_maps_and_the_filter_once_each() {
        let (read, written) = cost().traffic();

        assert_eq!(read, (2 * 3 * 8 * 8 + 4 * 3 * 3 * 3) * 2);
        assert_eq!(written, 2 * 4 * 6 * 6 * 2);
    }

    #[test]
    fn a_bias_is_read_alongside_the_filter() {
        let mut with_bias = cost();
        with_bias.bias_elems = 4;

        assert_eq!(with_bias.traffic().0, cost().traffic().0 + 4 * 2);
    }
}
