//! The past a causal conv's taps run over, which every conv of the family reads the same way.

use super::reads::bound;
use crate::kernels::refusals::{
    CONV_NOTHING_TO_CONVOLVE, QK_NORMED_CONV_STREAM_BINDING_BUFFERS,
    QK_NORMED_CONV_WINDOW_BINDING_BUFFERS,
};
use core::num::NonZeroUsize;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Indexing, Launcher, TileSpec, TileTensorLaunch};
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::linear_recurrence::conv::QkNormedConvOperand;
use cubek_microkernel::linear_recurrence::{BATCH, CHANNEL, RING_COLUMN, STORE_SLOT};
use cubek_std::routine::ADMITTED_PLAN_RUNS;

/// Where a conv reads the past values its taps run over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausalConvPast {
    /// A store of `slots` windows of `columns` ring columns. A row reads the columns its
    /// convolved positions fill and writes its newest values as the ring's next column.
    Window { slots: NonZeroUsize, columns: usize },
    /// The stream of `positions` positions of each sequence, `taps - 1` context columns before
    /// them: tap `j` of position `t` is the stream's value `t + j`, and nothing is written back.
    /// A caller that goes on with a window seeds it from the stream's tail: a row that has
    /// convolved `P` positions reads tap `j` at ring column `(P + j - (taps - 1)) % columns`.
    Stream { positions: usize },
}

impl CausalConvPast {
    /// The buffers a conv over this past binds, the metadata buffer included.
    pub fn binding_buffers(&self) -> u32 {
        match self {
            Self::Window { .. } => QK_NORMED_CONV_WINDOW_BINDING_BUFFERS,
            Self::Stream { .. } => QK_NORMED_CONV_STREAM_BINDING_BUFFERS,
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
    pub fn channels(&self) -> &'static [QkNormedConvOperand] {
        match self {
            Self::Window { .. } => &QkNormedConvOperand::WINDOW_CHANNELS,
            Self::Stream { .. } => &QkNormedConvOperand::STREAM_CHANNELS,
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

/// The past a conv's taps run over, as tensors.
// Built once a call and consumed by it, so the larger variant costs one copy of a few handles.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum CausalConvPastInputs {
    /// A ring store `[slots, columns, channels]`, its slots and columns at any stride. Each row
    /// has the slot it lies in (`slots`, `[rows]`, `u32`) and the positions it has convolved
    /// (`lengths`, `[rows]`, `u32`); a row with none uses nothing of its slot. Two rows may share
    /// a slot only if it is never read again, and a slot past the store's last reads and writes
    /// the last.
    Window {
        ring: PlainTensor,
        slots: PlainTensor,
        lengths: PlainTensor,
    },
    /// The stream's own context columns.
    Stream,
}

impl CausalConvPastInputs {
    /// The store, which only a window has.
    pub(crate) fn ring(&self) -> Option<&PlainTensor> {
        match self {
            Self::Window { ring, .. } => Some(ring),
            Self::Stream => None,
        }
    }

    /// The positions each row has convolved, which only a window has.
    pub(crate) fn lengths(&self) -> Option<&PlainTensor> {
        match self {
            Self::Window { lengths, .. } => Some(lengths),
            Self::Stream => None,
        }
    }

    /// The store and the lengths bound under `launcher`, `width` values a line, the ring placed at
    /// the slot its table names.
    pub(crate) fn bind(
        &self,
        launcher: &Launcher,
        width: usize,
    ) -> (Option<TileTensorLaunch>, Option<TileTensorLaunch>) {
        match self {
            Self::Window {
                ring,
                slots,
                lengths,
            } => {
                let entries = TileTensorLaunch::new(
                    slots.binding.clone().into_tensor_arg(),
                    slots.elem,
                    1,
                    TileSpec::direct(&[BATCH]),
                );
                let ring = bound(launcher, ring, &[STORE_SLOT, RING_COLUMN, CHANNEL], width)
                    .indexed(entries, Indexing::Coordinate(STORE_SLOT))
                    .expect(ADMITTED_PLAN_RUNS);
                let lengths = bound(launcher, lengths, &[BATCH], 1);
                (Some(ring), Some(lengths))
            }
            Self::Stream => (None, None),
        }
    }

    /// Views `input` of `channels` channels as the kernel reads it, and the store likewise, with
    /// the past they state: a window's `[rows, channels]` or a stream's
    /// `[batch, positions + taps - 1, channels]`.
    ///
    /// # Errors
    ///
    /// [`CONV_NOTHING_TO_CONVOLVE`] for no row or position.
    ///
    /// # Panics
    ///
    /// On shapes that disagree with `channels`, and on tables that are not `u32` or not
    /// `[rows]`.
    pub(crate) fn shaped(
        &mut self,
        input: PlainTensor,
        channels: usize,
        taps: usize,
    ) -> Result<(PlainTensor, CausalConvPast), Refusal> {
        match self {
            Self::Window {
                ring,
                slots,
                lengths,
            } => {
                let shape = input.shape().to_vec();
                let rows = shape[0];
                if rows == 0 {
                    return Err(CONV_NOTHING_TO_CONVOLVE);
                }
                assert!(
                    shape.last() == Some(&channels)
                        && shape[1..shape.len() - 1].iter().all(|&e| e == 1),
                    "the stream is [rows, channels]"
                );
                let input = input.viewed(Metadata::new(
                    vec![rows, channels],
                    vec![input.binding.strides[0], 1],
                ));
                let [slots_in_store, columns, ring_channels] = ring.shape().dims::<3>();
                assert_eq!(
                    ring_channels, channels,
                    "the ring holds every channel a column"
                );
                *ring = ring.clone().viewed(Metadata::new(
                    vec![slots_in_store, columns, channels],
                    vec![ring.binding.strides[0], ring.binding.strides[1], 1],
                ));
                for table in [&*slots, &*lengths] {
                    assert_eq!(table.elem, ElemType::UInt(UIntKind::U32), "a table is u32");
                    assert_eq!(table.shape().dims::<1>(), [rows]);
                }
                let past = CausalConvPast::Window {
                    slots: NonZeroUsize::new(slots_in_store).expect("a store holds a slot"),
                    columns,
                };
                Ok((input, past))
            }
            Self::Stream => {
                let [batch, stream_positions, stream_channels] = input.shape().dims::<3>();
                assert_eq!(
                    stream_channels, channels,
                    "the stream holds every channel a position"
                );
                let positions = stream_positions.saturating_sub(taps.saturating_sub(1));
                if batch == 0 || positions == 0 {
                    return Err(CONV_NOTHING_TO_CONVOLVE);
                }
                let input = input.viewed(Metadata::new(
                    vec![batch, stream_positions, channels],
                    vec![input.binding.strides[0], input.binding.strides[1], 1],
                ));
                Ok((input, CausalConvPast::Stream { positions }))
            }
        }
    }
}
