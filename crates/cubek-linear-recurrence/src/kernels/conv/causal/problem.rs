//! The problem a causal conv routine computes.

use crate::kernels::conv::{CausalConvPast, ConvExtent, CubeCausalConvRoutine};
use crate::kernels::refusals::admit_element;
use crate::kernels::refusals::{
    CAUSAL_CONV_BINDING_BUFFERS, CAUSAL_CONV_CHANNELS_EMPTY, CAUSAL_CONV_OUTPUT_UNSUPPORTED,
    CONV_BINDINGS_PAST_DEVICE, CONV_NOTHING_TO_CONVOLVE,
};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Axis, Space};
use cubecl::tune::Work;
use cubecl::zspace::striding::row_major_contiguous_strides;
use cubek_microkernel::linear_recurrence::conv::CausalConvOperandMap;
use cubek_microkernel::linear_recurrence::conv::cube::CausalConvWorkItem;
use cubek_microkernel::linear_recurrence::{BATCH, CHANNEL, CHANNEL_BOX, LINE_IN_BOX, TAP};
use cubek_std::Hardware;
use cubek_std::routine::{EstimatedProblem, Routine};

const F32: ElemType = ElemType::Float(FloatKind::F32);

/// The depthwise causal conv, with bias and silu, over `batch` rows of one position each, or
/// prompts of positions: `channels` channels and a filter of `taps`, in activations of `element`,
/// written at `output_element`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalConvProblem {
    pub batch: usize,
    pub channels: usize,
    /// Taps of the filter, the newest the row's own value.
    pub taps: usize,
    /// The element of the input, the filters, the bias and the ring.
    pub element: ElemType,
    /// The element the convolved channels are written at: `f32`, or `element`.
    pub output_element: ElemType,
    /// Whether the conv adds a bias a channel.
    pub bias: bool,
    /// Where the past the taps read lives.
    pub past: CausalConvPast,
    /// The alignment of each operand's buffer in elements.
    pub alignments: CausalConvOperandMap<ElemAlignment>,
    /// Bytes each operand's view starts past its buffer's start. Both narrow the conv's lines.
    pub start_bytes: CausalConvOperandMap<u64>,
}

impl CausalConvProblem {
    /// Whether the conv writes `output` for activations of `element`.
    ///
    /// # Errors
    ///
    /// [`CAUSAL_CONV_OUTPUT_UNSUPPORTED`] for any but `f32` and `element`, and
    /// [`CONV_ELEMENT_UNSUPPORTED`](crate::kernels::refusals::CONV_ELEMENT_UNSUPPORTED) for an
    /// element the convs are not verified at.
    pub fn admit_elements(element: ElemType, output: ElemType) -> Result<(), Refusal> {
        admit_element(element)?;
        if output == element || output == F32 {
            Ok(())
        } else {
            Err(CAUSAL_CONV_OUTPUT_UNSUPPORTED)
        }
    }

    /// Whether a device launches the conv at all: the routine preparing.
    ///
    /// # Errors
    ///
    /// What admission declines: no channel, row or position, an unsupported element, a filter or
    /// ring the taps cannot be read back from, more buffers than the device binds.
    pub fn admit(&self, hw: &Hardware) -> Result<(), Refusal> {
        CubeCausalConvRoutine
            .prepare_first_served(hw, self)
            .map(|_| ())
    }

    /// What every launch declines a problem with before it places anything.
    pub(crate) fn admit_launch(&self, hw: &Hardware) -> Result<(), Refusal> {
        if self.channels == 0 {
            return Err(CAUSAL_CONV_CHANNELS_EMPTY);
        }
        if self.batch == 0 {
            return Err(CONV_NOTHING_TO_CONVOLVE);
        }
        Self::admit_elements(self.element, self.output_element)?;
        self.past.admit(self.taps)?;
        if CAUSAL_CONV_BINDING_BUFFERS > hw.properties.hardware.max_bindings {
            return Err(CONV_BINDINGS_PAST_DEVICE);
        }
        Ok(())
    }

    pub(crate) fn work_item(&self) -> CausalConvWorkItem {
        CausalConvWorkItem {
            channels: self.channels,
            taps: self.taps,
            past: self.past.kind(),
        }
    }

    /// Every axis of the problem at its extent, the channels cut into boxes of `units` lines of
    /// `width` channels.
    pub(crate) fn space(&self, units: usize, width: usize) -> Space {
        let boxes = (self.channels / width).div_ceil(units);
        let mut axes = vec![
            (BATCH, self.batch),
            (CHANNEL_BOX, boxes),
            (LINE_IN_BOX, units * width),
            (CHANNEL, self.channels),
            (TAP, self.taps),
        ];
        axes.extend(self.past.axes());
        Space::new(&axes)
    }

    /// The axes of a cube's leaf: one box, every channel and tap, `width` of the box's channels.
    pub(crate) fn leaf(&self, width: usize) -> Vec<(Axis, usize)> {
        let mut axes = vec![
            (CHANNEL_BOX, 1),
            (CHANNEL, self.channels),
            (LINE_IN_BOX, width),
            (TAP, self.taps),
        ];
        axes.splice(0..0, self.past.leaf_axes());
        axes
    }

    /// A problem of dense operands, each starting where its buffer does.
    pub fn dense(
        batch: usize,
        channels: usize,
        taps: usize,
        past: CausalConvPast,
        element: ElemType,
        output_element: ElemType,
    ) -> Self {
        let rows =
            |shape: &[usize]| ElemAlignment::new(shape, &row_major_contiguous_strides(shape));
        let dense = rows(&[past.lead_extents(batch).as_slice(), &[channels]].concat());
        let alignments = CausalConvOperandMap {
            input: dense,
            filters: rows(&[channels, taps]),
            bias: rows(&[channels]),
            ring: rows(&[past.ring().slots.get(), past.ring().columns, channels]),
            lengths: rows(&[batch]),
            output: dense,
        };
        Self {
            batch,
            channels,
            taps,
            element,
            output_element,
            bias: false,
            past,
            alignments,
            start_bytes: CausalConvOperandMap::splat(0),
        }
    }

    /// The problem with a bias a channel.
    pub fn with_bias(mut self) -> Self {
        self.bias = true;
        self
    }

    /// Filter values a channel reads: its taps, and its bias where there is one.
    pub(crate) fn filter_values(&self) -> usize {
        self.taps + usize::from(self.bias)
    }
}

impl EstimatedProblem for CausalConvProblem {
    /// Every operand once, and a handful of operations a channel a tap.
    fn demand(&self) -> Work {
        let size = self.element.size();
        let extent = ConvExtent {
            batch: self.batch,
            channels: self.channels,
            taps: self.taps,
        };
        let (read, written) = self.past.bytes(extent, size);
        let channels = self.batch * self.past.positions() * self.channels;
        Work {
            compute_ops: self.work_item().operations(channels),
            bytes: read
                + self.channels * self.filter_values() * size
                + written
                + channels * self.output_element.size(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::conv::RingStore;
    use crate::kernels::refusals::{CONV_ELEMENT_UNSUPPORTED, CONV_TAPS_PAST_RING};
    use core::num::NonZeroUsize;
    use cubecl::ir::UIntKind;
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    fn ring(columns: usize) -> RingStore {
        RingStore {
            slots: NonZeroUsize::new(4).unwrap(),
            columns,
        }
    }

    fn window(columns: usize) -> CausalConvPast {
        CausalConvPast::Window {
            ring: ring(columns),
        }
    }

    fn prompt(positions: usize) -> CausalConvPast {
        CausalConvPast::Prompt {
            ring: ring(3),
            positions,
        }
    }

    fn conv(past: CausalConvPast) -> CausalConvProblem {
        CausalConvProblem::dense(2, 1280, 4, past, F16, F16)
    }

    /// A launch binds fewer buffers than the q,k,v conv: a device binding seven declines it by
    /// name, eight admits.
    #[test]
    fn a_device_binding_fewer_buffers_than_the_launch_is_declined() {
        let problem = conv(window(3));
        let mut hw = devices::h100();
        devices::topology(&mut hw).max_bindings = CAUSAL_CONV_BINDING_BUFFERS - 1;
        assert_eq!(problem.admit(&hw), Err(CONV_BINDINGS_PAST_DEVICE));
        devices::topology(&mut hw).max_bindings = CAUSAL_CONV_BINDING_BUFFERS;
        assert_eq!(problem.admit(&hw), Ok(()));
    }

    /// An element the convs are not verified at, and an output that is neither `f32` nor the
    /// activations', are declined by name.
    #[test]
    fn elements_the_conv_is_not_verified_at_are_declined() {
        let hw = devices::h100();
        let dense = |element, output| {
            CausalConvProblem::dense(1, 64, 4, window(3), element, output).admit(&hw)
        };
        let bf16 = ElemType::Float(FloatKind::BF16);
        for (element, output) in [
            (F16, F16),
            (F16, F32),
            (bf16, bf16),
            (bf16, F32),
            (F32, F32),
        ] {
            assert_eq!(dense(element, output), Ok(()));
        }
        assert_eq!(dense(F32, F16), Err(CAUSAL_CONV_OUTPUT_UNSUPPORTED));
        assert_eq!(
            dense(ElemType::UInt(UIntKind::U32), F32),
            Err(CONV_ELEMENT_UNSUPPORTED)
        );
    }

    /// No channel, no position and a ring shorter than the taps read back are declined by name.
    #[test]
    fn what_has_nothing_to_convolve_or_no_taps_to_read_is_declined() {
        let hw = devices::h100();
        assert_eq!(
            CausalConvProblem::dense(1, 0, 4, window(3), F16, F16).admit(&hw),
            Err(CAUSAL_CONV_CHANNELS_EMPTY)
        );
        assert_eq!(conv(prompt(0)).admit(&hw), Err(CONV_NOTHING_TO_CONVOLVE));
        assert_eq!(conv(window(2)).admit(&hw), Err(CONV_TAPS_PAST_RING));
        assert_eq!(conv(prompt(8)).admit(&hw), Ok(()));
    }

    /// The bias is counted in the filters' bytes only where the conv has one.
    #[test]
    fn the_bias_is_counted_only_where_the_conv_has_one() {
        let without = conv(window(3));
        let with = without.clone().with_bias();
        let channel_bytes = without.channels * without.element.size();
        assert_eq!(with.demand().bytes - without.demand().bytes, channel_bytes);
    }
}
