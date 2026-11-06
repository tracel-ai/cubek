//! The problem a conv routine computes.

use crate::kernels::conv::CubeGatedDeltaConvRoutine;
use crate::kernels::refusals::{
    GATED_DELTA_CONV_BINDINGS_PAST_DEVICE, GATED_DELTA_CONV_ELEMENT_UNSUPPORTED,
    GATED_DELTA_CONV_HEADS_EMPTY, GATED_DELTA_CONV_NOTHING_TO_CONVOLVE,
    GATED_DELTA_CONV_STREAM_BINDING_BUFFERS, GATED_DELTA_CONV_TAPS_PAST_WINDOW,
    GATED_DELTA_CONV_WINDOW_BINDING_BUFFERS,
};
use core::num::NonZeroUsize;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Axis, Space};
use cubecl::tune::Work;
use cubecl::zspace::striding::row_major_contiguous_strides;
use cubek_microkernel::gated_delta::conv::cube::GatedDeltaConvWorkItem;
use cubek_microkernel::gated_delta::conv::{GatedDeltaConvOperand, GatedDeltaConvOperandMap};
use cubek_microkernel::gated_delta::{
    BATCH, CHANNEL, GROUP, GatedDeltaHeads, HEAD_CHANNEL, KEY_FEATURE, KEY_HEAD, POSITION,
    RING_COLUMN, STORE_SLOT, STREAM_POSITION, TAP, VALUE_FEATURE,
};
use cubek_std::Hardware;
use cubek_std::routine::{EstimatedProblem, Routine};

/// The element the queries, keys and values are written at: `f32`, as the walk reads them.
pub const GATED_DELTA_CONV_OUTPUT_ELEMENT: ElemType = ElemType::Float(FloatKind::F32);

/// The activation elements the conv is verified at.
const ELEMENTS: [ElemType; 2] = [
    ElemType::Float(FloatKind::F16),
    ElemType::Float(FloatKind::F32),
];

/// Where a conv reads the past values its taps run over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GatedDeltaConvPast {
    /// A store of `slots` windows of `columns` ring columns. A row reads the columns its
    /// convolved positions fill and writes its newest values as the ring's next column.
    Window { slots: NonZeroUsize, columns: usize },
    /// The stream of `positions` positions of each sequence, `taps - 1` context columns before
    /// them: tap `j` of position `t` is the stream's value `t + j`, and nothing is written back.
    /// A caller that goes on with a window seeds it from the stream's tail: a row that has
    /// convolved `P` positions reads tap `j` at ring column `(P + j - (taps - 1)) % columns`.
    Stream { positions: usize },
}

impl GatedDeltaConvPast {
    /// The buffers a conv over this past binds, the metadata buffer included.
    pub fn binding_buffers(&self) -> u32 {
        match self {
            Self::Window { .. } => GATED_DELTA_CONV_WINDOW_BINDING_BUFFERS,
            Self::Stream { .. } => GATED_DELTA_CONV_STREAM_BINDING_BUFFERS,
        }
    }

    /// The ring length the kernel is told; over a stream it is zero and unread.
    pub fn columns(&self) -> usize {
        match self {
            Self::Window { columns, .. } => *columns,
            Self::Stream { .. } => 0,
        }
    }

    /// The operands of the conv this past binds, which a line of channels runs along.
    pub fn channels(&self) -> &'static [GatedDeltaConvOperand] {
        match self {
            Self::Window { .. } => &GatedDeltaConvOperand::WINDOW_CHANNELS,
            Self::Stream { .. } => &GatedDeltaConvOperand::STREAM_CHANNELS,
        }
    }

    /// The columns of a stream of `taps` taps: its context columns and its positions. Zero over a
    /// window, which has no stream.
    pub fn stream_len(&self, taps: usize) -> usize {
        match self {
            Self::Window { .. } => 0,
            Self::Stream { positions } => (positions + taps).saturating_sub(1),
        }
    }
}

/// The conv over `batch` rows of one position each, or sequences of positions: the `q|k|v` stream
/// of `heads` and a filter of `taps`, in activations of `element`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatedDeltaConvProblem {
    pub batch: usize,
    pub heads: GatedDeltaHeads,
    /// Taps of the filter, the newest the row's own value.
    pub taps: usize,
    /// The element of the stream, the filters and the window.
    pub element: ElemType,
    /// Where the past the taps read lives.
    pub past: GatedDeltaConvPast,
    /// The alignment of each operand's buffer in elements.
    pub alignments: GatedDeltaConvOperandMap<ElemAlignment>,
    /// Bytes each operand's view starts past its buffer's start. Both narrow the conv's lines.
    pub start_bytes: GatedDeltaConvOperandMap<u64>,
}

impl GatedDeltaConvProblem {
    /// Whether the conv reads and writes activations of `element`.
    ///
    /// # Errors
    ///
    /// [`GATED_DELTA_CONV_ELEMENT_UNSUPPORTED`] for any but `f16` and `f32`.
    pub fn admit_element(element: ElemType) -> Result<(), Refusal> {
        if ELEMENTS.contains(&element) {
            Ok(())
        } else {
            Err(GATED_DELTA_CONV_ELEMENT_UNSUPPORTED)
        }
    }

    /// Whether a device launches the conv at all: the routine preparing.
    ///
    /// # Errors
    ///
    /// What admission declines: no head, no row or position, an unsupported element, a filter or
    /// window the taps cannot be read back from, more buffers than the device binds.
    pub fn admit(&self, hw: &Hardware) -> Result<(), Refusal> {
        CubeGatedDeltaConvRoutine
            .prepare_first_served(hw, self)
            .map(|_| ())
    }

    /// What every launch declines a problem with before it places anything.
    pub(crate) fn admit_launch(&self, hw: &Hardware) -> Result<(), Refusal> {
        if self.work_item().is_empty() {
            return Err(GATED_DELTA_CONV_HEADS_EMPTY);
        }
        if self.batch == 0 || self.past == (GatedDeltaConvPast::Stream { positions: 0 }) {
            return Err(GATED_DELTA_CONV_NOTHING_TO_CONVOLVE);
        }
        Self::admit_element(self.element)?;
        let window_short = match self.past {
            GatedDeltaConvPast::Window { columns, .. } => columns + 1 < self.taps,
            GatedDeltaConvPast::Stream { .. } => false,
        };
        if self.taps < 2 || window_short {
            return Err(GATED_DELTA_CONV_TAPS_PAST_WINDOW);
        }
        if self.past.binding_buffers() > hw.properties.hardware.max_bindings {
            return Err(GATED_DELTA_CONV_BINDINGS_PAST_DEVICE);
        }
        Ok(())
    }

    pub(crate) fn work_item(&self) -> GatedDeltaConvWorkItem {
        GatedDeltaConvWorkItem {
            heads: self.heads,
            taps: self.taps,
            channels: self.past.channels(),
        }
    }

    /// Every axis of the problem at its extent.
    pub(crate) fn space(&self) -> Space {
        let heads = &self.heads;
        let mut axes = vec![
            (BATCH, self.batch),
            (KEY_HEAD, heads.key_heads),
            (GROUP, heads.group),
            (KEY_FEATURE, heads.key_features),
            (VALUE_FEATURE, heads.value_features),
            (CHANNEL, heads.conv_dim()),
            (HEAD_CHANNEL, heads.channels_per_key_head()),
            (TAP, self.taps),
        ];
        match self.past {
            GatedDeltaConvPast::Window { slots, columns } => {
                axes.extend([(STORE_SLOT, slots.get()), (RING_COLUMN, columns)]);
            }
            GatedDeltaConvPast::Stream { positions } => axes.extend([
                (POSITION, positions),
                (STREAM_POSITION, self.past.stream_len(self.taps)),
            ]),
        }
        Space::new(&axes)
    }

    /// The axes of a cube's leaf: its key head, every channel and tap, `width` of the head's
    /// channels.
    pub(crate) fn leaf(&self, width: usize) -> Vec<(Axis, usize)> {
        let heads = &self.heads;
        let mut axes = vec![
            (KEY_HEAD, 1),
            (GROUP, heads.group),
            (KEY_FEATURE, heads.key_features),
            (VALUE_FEATURE, heads.value_features),
            (CHANNEL, heads.conv_dim()),
            (HEAD_CHANNEL, width),
            (TAP, self.taps),
        ];
        if let GatedDeltaConvPast::Stream { .. } = self.past {
            axes.insert(0, (POSITION, 1));
        }
        axes
    }

    /// A problem of dense operands, each starting where its buffer does.
    pub fn dense(
        batch: usize,
        heads: GatedDeltaHeads,
        taps: usize,
        past: GatedDeltaConvPast,
        element: ElemType,
    ) -> Self {
        let rows =
            |shape: &[usize]| ElemAlignment::new(shape, &row_major_contiguous_strides(shape));
        let GatedDeltaHeads {
            key_heads,
            group,
            key_features,
            value_features,
        } = heads;
        let conv_dim = heads.conv_dim();
        let alignments = match past {
            GatedDeltaConvPast::Window { slots, columns } => {
                let keyed = rows(&[batch, key_heads, key_features]);
                GatedDeltaConvOperandMap {
                    qkv: rows(&[batch, conv_dim]),
                    filters: rows(&[conv_dim, taps]),
                    window: rows(&[slots.get(), columns, conv_dim]),
                    lengths: rows(&[batch]),
                    query: keyed,
                    key: keyed,
                    value: rows(&[batch, key_heads, group, value_features]),
                }
            }
            GatedDeltaConvPast::Stream { positions } => {
                let keyed = rows(&[batch, positions, key_heads, key_features]);
                GatedDeltaConvOperandMap {
                    qkv: rows(&[batch, past.stream_len(taps), conv_dim]),
                    filters: rows(&[conv_dim, taps]),
                    query: keyed,
                    key: keyed,
                    value: rows(&[batch, positions, key_heads, group, value_features]),
                    ..GatedDeltaConvOperandMap::splat(ElemAlignment::ONE)
                }
            }
        };
        Self {
            batch,
            heads,
            taps,
            element,
            past,
            alignments,
            start_bytes: GatedDeltaConvOperandMap::splat(0),
        }
    }
}

impl EstimatedProblem for GatedDeltaConvProblem {
    /// Every operand once, and a handful of operations a channel a tap.
    fn demand(&self) -> Work {
        let conv_dim = self.heads.conv_dim();
        let size = self.element.size();
        let filters = conv_dim * self.taps * size;
        let (positions, read, written) = match self.past {
            GatedDeltaConvPast::Window { .. } => {
                let channels = self.batch * conv_dim;
                (1, channels * self.taps * size, channels * size)
            }
            GatedDeltaConvPast::Stream { positions } => {
                let stream = self.batch * self.past.stream_len(self.taps) * conv_dim * size;
                (positions, stream, 0)
            }
        };
        let channels = self.batch * positions * conv_dim;
        Work {
            compute_ops: self.work_item().operations(channels),
            bytes: read + filters + written + channels * GATED_DELTA_CONV_OUTPUT_ELEMENT.size(),
        }
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    use cubek_std::Hardware;
    use cubek_std::devices;

    /// `hw` as a device that binds as many buffers as the conv's launch does.
    pub(crate) fn binding_enough(mut hw: Hardware) -> Hardware {
        devices::topology(&mut hw).max_bindings = 32;
        hw
    }

    pub(crate) fn fleet() -> [Hardware; 6] {
        [
            devices::h100(),
            devices::cdna_wave64(),
            devices::cpu_12(),
            devices::strix_halo(),
            devices::rdna_range(),
            devices::wgpu_no_subgroups(),
        ]
        .map(binding_enough)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    fn heads() -> GatedDeltaHeads {
        GatedDeltaHeads {
            key_heads: 16,
            group: 3,
            key_features: 128,
            value_features: 128,
        }
    }

    fn window(slots: usize, columns: usize) -> GatedDeltaConvPast {
        GatedDeltaConvPast::Window {
            slots: NonZeroUsize::new(slots).unwrap(),
            columns,
        }
    }

    fn conv(heads: GatedDeltaHeads, taps: usize, columns: usize) -> GatedDeltaConvProblem {
        GatedDeltaConvProblem::dense(8, heads, taps, window(9, columns), F16)
    }

    /// A window launch needs seven operands, the slot table and the metadata buffer: a device
    /// binding eight declines it by name, nine admits.
    #[test]
    fn a_device_binding_fewer_buffers_than_the_launch_is_declined() {
        let problem = conv(heads(), 4, 3);
        assert_eq!(
            problem.admit(&devices::wgpu_no_subgroups()),
            Err(GATED_DELTA_CONV_BINDINGS_PAST_DEVICE)
        );
        let mut hw = devices::h100();
        devices::topology(&mut hw).max_bindings = problem.past.binding_buffers();
        assert_eq!(problem.admit(&hw), Ok(()));
    }

    /// A packed or integer element, and a float the conv is not verified at, are declined.
    #[test]
    fn elements_the_conv_is_not_verified_at_are_declined() {
        use cubecl::ir::UIntKind;
        let hw = devices::h100();
        let dense = |element| GatedDeltaConvProblem::dense(1, heads(), 4, window(2, 3), element);
        assert_eq!(dense(F16).admit(&hw), Ok(()));
        for element in [
            ElemType::UInt(UIntKind::U32),
            ElemType::Float(FloatKind::BF16),
        ] {
            assert_eq!(
                dense(element).admit(&hw),
                Err(GATED_DELTA_CONV_ELEMENT_UNSUPPORTED)
            );
        }
    }

    /// A window shorter than the columns read back is declined by name.
    #[test]
    fn a_window_shorter_than_the_taps_read_back_is_declined() {
        let hw = devices::h100();
        assert_eq!(
            conv(heads(), 4, 2).admit(&hw),
            Err(GATED_DELTA_CONV_TAPS_PAST_WINDOW)
        );
        assert_eq!(conv(heads(), 4, 3).admit(&hw), Ok(()));
    }

    /// A stream with no position is declined by name; one with positions binds fewer buffers than
    /// a window, so the devices that decline a window admit it.
    #[test]
    fn a_stream_is_declined_without_positions_and_binds_less_than_a_window() {
        let stream = |positions| {
            let past = GatedDeltaConvPast::Stream { positions };
            GatedDeltaConvProblem::dense(1, heads(), 4, past, F16)
        };
        assert_eq!(
            stream(0).admit(&devices::h100()),
            Err(GATED_DELTA_CONV_NOTHING_TO_CONVOLVE)
        );
        assert_eq!(stream(8).admit(&devices::wgpu_no_subgroups()), Ok(()));
    }
}
