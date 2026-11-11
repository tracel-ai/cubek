//! The problem a conv routine computes.

use crate::kernels::conv::admit_element;
use crate::kernels::conv::{CausalConvPast, ConvExtent, CubeQkNormedConvRoutine};
use crate::kernels::refusals::{
    CONV_BINDINGS_PAST_DEVICE, CONV_NOTHING_TO_CONVOLVE, QK_NORMED_CONV_BINDING_BUFFERS,
    QK_NORMED_CONV_HEADS_EMPTY,
};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Axis, Space};
use cubecl::tune::Work;
use cubecl::zspace::striding::row_major_contiguous_strides;
use cubek_microkernel::linear_recurrence::conv::QkNormedConvOperandMap;
use cubek_microkernel::linear_recurrence::conv::cube::QkNormedConvWorkItem;
use cubek_microkernel::linear_recurrence::{
    BATCH, CHANNEL, HEAD_CHANNEL, HEAD_IN_GROUP, KEY_FEATURE, KEY_HEAD, RecurrenceHeads, TAP,
    VALUE_FEATURE,
};
use cubek_std::Hardware;
use cubek_std::routine::{EstimatedProblem, Routine};

/// The element the queries, keys and values are written at: `f32`, as the walk reads them.
pub const QK_NORMED_CONV_OUTPUT_ELEMENT: ElemType = ElemType::Float(FloatKind::F32);

/// The conv over `batch` rows of one position each, or prompts of positions: the `q|k|v` stream
/// of `heads` and a filter of `taps`, in activations of `element`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QkNormedConvProblem {
    pub batch: usize,
    pub heads: RecurrenceHeads,
    /// Taps of the filter, the newest the row's own value.
    pub taps: usize,
    /// The element of the input, the filters and the ring.
    pub element: ElemType,
    /// Where the past the taps read lives.
    pub past: CausalConvPast,
    /// The alignment of each operand's buffer in elements.
    pub alignments: QkNormedConvOperandMap<ElemAlignment>,
    /// Bytes each operand's view starts past its buffer's start. Both narrow the conv's lines.
    pub start_bytes: QkNormedConvOperandMap<u64>,
}

impl QkNormedConvProblem {
    /// Whether a device launches the conv at all: the routine preparing.
    ///
    /// # Errors
    ///
    /// What admission declines: no head, no row or position, an unsupported element, a filter or
    /// window the taps cannot be read back from, more buffers than the device binds.
    pub fn admit(&self, hw: &Hardware) -> Result<(), Refusal> {
        CubeQkNormedConvRoutine
            .prepare_first_served(hw, self)
            .map(|_| ())
    }

    /// What every launch declines a problem with before it places anything.
    pub(crate) fn admit_launch(&self, hw: &Hardware) -> Result<(), Refusal> {
        if self.work_item().is_empty() {
            return Err(QK_NORMED_CONV_HEADS_EMPTY);
        }
        if self.batch == 0 {
            return Err(CONV_NOTHING_TO_CONVOLVE);
        }
        admit_element(self.element)?;
        self.past.admit(self.taps)?;
        if QK_NORMED_CONV_BINDING_BUFFERS > hw.properties.hardware.max_bindings {
            return Err(CONV_BINDINGS_PAST_DEVICE);
        }
        Ok(())
    }

    pub(crate) fn work_item(&self) -> QkNormedConvWorkItem {
        QkNormedConvWorkItem {
            heads: self.heads,
            taps: self.taps,
            past: self.past.kind(),
        }
    }

    /// Every axis of the problem at its extent.
    pub(crate) fn space(&self) -> Space {
        let heads = &self.heads;
        let mut axes = vec![
            (BATCH, self.batch),
            (KEY_HEAD, heads.key_heads),
            (HEAD_IN_GROUP, heads.heads_per_group),
            (KEY_FEATURE, heads.key_features),
            (VALUE_FEATURE, heads.value_features),
            (CHANNEL, heads.conv_dim()),
            (HEAD_CHANNEL, heads.channels_per_key_head()),
            (TAP, self.taps),
        ];
        axes.extend(self.past.axes());
        Space::new(&axes)
    }

    /// The axes of a cube's leaf: its key head, every channel and tap, `width` of the head's
    /// channels.
    pub(crate) fn leaf(&self, width: usize) -> Vec<(Axis, usize)> {
        let heads = &self.heads;
        let mut axes = vec![
            (KEY_HEAD, 1),
            (HEAD_IN_GROUP, heads.heads_per_group),
            (KEY_FEATURE, heads.key_features),
            (VALUE_FEATURE, heads.value_features),
            (CHANNEL, heads.conv_dim()),
            (HEAD_CHANNEL, width),
            (TAP, self.taps),
        ];
        axes.splice(0..0, self.past.leaf_axes());
        axes
    }

    /// A problem of dense operands, each starting where its buffer does.
    pub fn dense(
        batch: usize,
        heads: RecurrenceHeads,
        taps: usize,
        past: CausalConvPast,
        element: ElemType,
    ) -> Self {
        let rows =
            |shape: &[usize]| ElemAlignment::new(shape, &row_major_contiguous_strides(shape));
        let RecurrenceHeads {
            key_heads,
            heads_per_group,
            key_features,
            value_features,
        } = heads;
        let conv_dim = heads.conv_dim();
        let lead = past.lead_extents(batch);
        let with_lead = |tail: &[usize]| rows(&[lead.as_slice(), tail].concat());
        let input = with_lead(&[conv_dim]);
        let keyed = with_lead(&[key_heads, key_features]);
        let valued = with_lead(&[key_heads, heads_per_group, value_features]);
        let alignments = QkNormedConvOperandMap {
            qkv: input,
            filters: rows(&[conv_dim, taps]),
            ring: rows(&[past.ring().slots.get(), past.ring().columns, conv_dim]),
            lengths: rows(&[batch]),
            query: keyed,
            key: keyed,
            value: valued,
        };
        Self {
            batch,
            heads,
            taps,
            element,
            past,
            alignments,
            start_bytes: QkNormedConvOperandMap::splat(0),
        }
    }
}

impl EstimatedProblem for QkNormedConvProblem {
    /// Every operand once, and a handful of operations a channel a tap.
    fn demand(&self) -> Work {
        let conv_dim = self.heads.conv_dim();
        let size = self.element.size();
        let extent = ConvExtent {
            batch: self.batch,
            channels: conv_dim,
            taps: self.taps,
        };
        let (read, written) = self.past.bytes(extent, size);
        let channels = self.batch * self.past.positions() * conv_dim;
        Work {
            compute_ops: self.work_item().operations(channels),
            bytes: read
                + conv_dim * self.taps * size
                + written
                + channels * QK_NORMED_CONV_OUTPUT_ELEMENT.size(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::conv::RingStore;
    use crate::kernels::refusals::{CONV_ELEMENT_UNSUPPORTED, CONV_TAPS_PAST_RING};
    use core::num::NonZeroUsize;
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    fn heads() -> RecurrenceHeads {
        RecurrenceHeads {
            key_heads: 16,
            heads_per_group: 3,
            key_features: 128,
            value_features: 128,
        }
    }

    fn ring(slots: usize, columns: usize) -> RingStore {
        RingStore {
            slots: NonZeroUsize::new(slots).unwrap(),
            columns,
        }
    }

    fn window(slots: usize, columns: usize) -> CausalConvPast {
        CausalConvPast::Window {
            ring: ring(slots, columns),
        }
    }

    fn conv(heads: RecurrenceHeads, taps: usize, columns: usize) -> QkNormedConvProblem {
        QkNormedConvProblem::dense(8, heads, taps, window(9, columns), F16)
    }

    /// A window launch needs seven operands, the slot table and the metadata buffer: a device
    /// binding eight declines it by name, nine admits.
    #[test]
    fn a_device_binding_fewer_buffers_than_the_launch_is_declined() {
        let problem = conv(heads(), 4, 3);
        assert_eq!(
            problem.admit(&devices::wgpu_no_subgroups()),
            Err(CONV_BINDINGS_PAST_DEVICE)
        );
        let mut hw = devices::h100();
        devices::topology(&mut hw).max_bindings = QK_NORMED_CONV_BINDING_BUFFERS;
        assert_eq!(problem.admit(&hw), Ok(()));
    }

    /// The floats the conv is verified at are admitted, and a packed or integer element is
    /// declined.
    #[test]
    fn elements_the_conv_is_not_verified_at_are_declined() {
        use cubecl::ir::UIntKind;
        let hw = devices::h100();
        let dense = |element| QkNormedConvProblem::dense(1, heads(), 4, window(2, 3), element);
        for element in [F16, ElemType::Float(FloatKind::F32)] {
            assert_eq!(dense(element).admit(&hw), Ok(()));
        }
        for element in [
            ElemType::Float(FloatKind::BF16),
            ElemType::UInt(UIntKind::U32),
        ] {
            assert_eq!(dense(element).admit(&hw), Err(CONV_ELEMENT_UNSUPPORTED));
        }
    }

    /// A window shorter than the columns read back is declined by name.
    #[test]
    fn a_window_shorter_than_the_taps_read_back_is_declined() {
        let hw = devices::h100();
        assert_eq!(conv(heads(), 4, 2).admit(&hw), Err(CONV_TAPS_PAST_RING));
        assert_eq!(conv(heads(), 4, 3).admit(&hw), Ok(()));
    }

    /// A prompt with no position is declined by name; one with positions binds as many buffers as
    /// a window, so the devices that decline a window decline it.
    #[test]
    fn a_prompt_is_declined_without_positions_and_binds_as_a_window_does() {
        let prompt = |positions| {
            let past = CausalConvPast::Prompt {
                ring: ring(2, 3),
                positions,
            };
            QkNormedConvProblem::dense(1, heads(), 4, past, F16)
        };
        assert_eq!(
            prompt(0).admit(&devices::h100()),
            Err(CONV_NOTHING_TO_CONVOLVE)
        );
        assert_eq!(
            prompt(8).admit(&devices::wgpu_no_subgroups()),
            Err(CONV_BINDINGS_PAST_DEVICE)
        );
        assert_eq!(prompt(8).admit(&devices::h100()), Ok(()));
    }
}
