//! The past a causal conv's taps run over: where the ring lies, and who writes it.

use crate::kernels::SlotStore;
use crate::kernels::conv::CausalConvProblem;
use crate::kernels::refusals::{CAUSAL_CONV_NOTHING_TO_CONVOLVE, CAUSAL_CONV_TAPS_PAST_RING};
use core::num::NonZeroUsize;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::tile::Axis;
use cubecl::tile::launch::Refusal;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::linear_recurrence::conv::cube::RingWriter;
use cubek_microkernel::linear_recurrence::{POSITION, PROMPT_POSITION, RING_COLUMN, STORE_SLOT};
use cubek_std::microkernel::GlobalStream;

/// What one cube of the conv moves through global memory, apart from the past's own traffic: its
/// `channels` channels of a position, `run` values of each read contiguously.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CausalConvTraffic {
    pub channels: usize,
    pub taps: usize,
    pub run: usize,
    pub element_size: usize,
    pub filters: GlobalStream,
    pub outputs: GlobalStream,
}

/// A store of `slots` rings of `columns` columns, one ring a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RingLayout {
    pub slots: NonZeroUsize,
    pub columns: usize,
}

/// Where a conv reads the past values its taps run over, and who writes the ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausalConvPastLayout {
    /// One position of each row, which the conv's own launch writes to the ring as its next
    /// column, once the taps have read the column it replaces.
    DecodeStep { ring: RingLayout },
    /// `positions` positions of each row: tap `j` of position `t` is the prompt's value
    /// `t + j - (taps - 1)` where that is not before the prompt, and the row's ring through its
    /// length otherwise. The last `taps - 1` valid positions of each row are then written to its
    /// ring as its next columns, by a second launch. The caller advances the row's length by its
    /// valid positions.
    Prompt { ring: RingLayout, positions: usize },
}

impl CausalConvPastLayout {
    /// The store of rings.
    pub fn ring(&self) -> RingLayout {
        match self {
            Self::DecodeStep { ring } | Self::Prompt { ring, .. } => *ring,
        }
    }

    /// The positions a row convolves: one over a decode step.
    pub fn positions(&self) -> usize {
        match self {
            Self::DecodeStep { .. } => 1,
            Self::Prompt { positions, .. } => *positions,
        }
    }

    /// Who writes the convolved positions to the ring.
    pub fn writer(&self) -> RingWriter {
        match self {
            Self::DecodeStep { .. } => RingWriter::ConvLaunch,
            Self::Prompt { .. } => RingWriter::WriteBackLaunch,
        }
    }

    /// Whether the past holds a position to convolve.
    ///
    /// # Errors
    ///
    /// [`CAUSAL_CONV_NOTHING_TO_CONVOLVE`] for a prompt of no position, and
    /// [`CAUSAL_CONV_TAPS_PAST_RING`] for fewer than two `taps` or a ring of fewer columns than the
    /// taps read back.
    pub(crate) fn admit(&self, taps: usize) -> Result<(), Refusal> {
        if self.positions() == 0 {
            return Err(CAUSAL_CONV_NOTHING_TO_CONVOLVE);
        }
        if taps < 2 || self.ring().columns + 1 < taps {
            return Err(CAUSAL_CONV_TAPS_PAST_RING);
        }
        Ok(())
    }

    /// The streams one cube reads and writes: the input and the ring it reads, the ring a decode step
    /// writes, and a prompt's share of the write back, which reads each row's last `taps - 1`
    /// positions and writes them to its ring.
    pub(crate) fn streams(
        &self,
        conv: &CausalConvTraffic,
    ) -> (Vec<GlobalStream>, Vec<GlobalStream>) {
        let stream = |elements: usize| GlobalStream {
            bytes: (elements * conv.element_size) as u64,
            run_bytes: conv.run * conv.element_size,
        };
        let ring_columns = conv.channels * (conv.taps - 1);
        match self {
            Self::DecodeStep { .. } => (
                vec![stream(conv.channels), conv.filters, stream(ring_columns)],
                vec![stream(conv.channels), conv.outputs],
            ),
            Self::Prompt { positions, .. } => {
                let write_back = stream(ring_columns.div_ceil(*positions));
                (
                    vec![stream(conv.channels * conv.taps), conv.filters, write_back],
                    vec![conv.outputs, write_back],
                )
            }
        }
    }

    /// The bytes of activations the conv of `problem` reads and writes, apart from the filters:
    /// every tap of every position over a decode step, and a prompt's positions once and its write
    /// back.
    pub(crate) fn bytes(&self, problem: &CausalConvProblem) -> (usize, usize) {
        let size = problem.element.size();
        let channels = problem.batch * self.positions() * problem.channels;
        match self {
            Self::DecodeStep { .. } => (channels * problem.taps * size, channels * size),
            Self::Prompt { .. } => {
                let ring = problem.batch * (problem.taps - 1) * problem.channels * size;
                (channels * size + ring, ring)
            }
        }
    }

    /// The axes of what the write back places, with their extents: the store, and the input's
    /// positions. `POSITION` is no axis of its levels or operands.
    pub(crate) fn past_axes(&self) -> Vec<(Axis, usize)> {
        let ring = self.ring();
        vec![
            (STORE_SLOT, ring.slots.get()),
            (RING_COLUMN, ring.columns),
            (PROMPT_POSITION, self.positions()),
        ]
    }

    /// The axes of the past, and of the positions a cube convolves.
    pub(crate) fn axes(&self) -> Vec<(Axis, usize)> {
        let mut axes = self.past_axes();
        axes.push((POSITION, self.positions()));
        axes
    }
}

/// The past a conv's taps run over, as tensors: the ring store with the tables that place each row
/// in it.
#[derive(Debug, Clone)]
pub enum CausalConvPastInputs {
    /// A ring store `[slots, columns, channels]`, its slots and columns at any stride. Each row
    /// has the slot it lies in and the positions it has convolved: see [`SlotStore`].
    DecodeStep { ring: SlotStore },
    /// The same store and tables, and the positions of each row that are real (`valid`, `[rows]`,
    /// `u32`; a `valid` past the prompt's positions counts as all of them): the rest is right
    /// padding, which the ring takes nothing from and whose outputs are unspecified. The store's
    /// lengths are the fill before the prompt.
    Prompt { ring: SlotStore, valid: PlainTensor },
}

impl CausalConvPastInputs {
    /// The store and its tables.
    pub fn ring(&self) -> &SlotStore {
        match self {
            Self::DecodeStep { ring } | Self::Prompt { ring, .. } => ring,
        }
    }

    /// The past `input` of `channels` channels states, once its shapes are checked against the
    /// store and its tables.
    ///
    /// # Errors
    ///
    /// [`CAUSAL_CONV_NOTHING_TO_CONVOLVE`] for no row or position.
    ///
    /// # Panics
    ///
    /// On shapes that disagree with `channels`, on a decode step of more than one position, and on
    /// tables that are not `[rows]`.
    pub(crate) fn check(
        &self,
        input: &PlainTensor,
        channels: usize,
    ) -> Result<CausalConvPastLayout, Refusal> {
        let [rows, positions, input_channels] = input.shape().dims::<3>();
        assert_eq!(
            input_channels, channels,
            "the input holds every channel a position"
        );
        if rows == 0 || positions == 0 {
            return Err(CAUSAL_CONV_NOTHING_TO_CONVOLVE);
        }
        if matches!(self, Self::DecodeStep { .. }) {
            assert_eq!(positions, 1, "a decode step convolves one position a row");
        }
        let [_, columns, ring_channels] = self.ring().store().shape().dims::<3>();
        assert_eq!(
            ring_channels, channels,
            "the ring holds every channel a column"
        );
        assert_eq!(self.ring().rows(), rows, "a table is [rows]");
        let ring = RingLayout {
            slots: self.ring().slot_count(),
            columns,
        };
        Ok(match self {
            Self::DecodeStep { .. } => CausalConvPastLayout::DecodeStep { ring },
            Self::Prompt { valid, .. } => {
                assert_eq!(valid.elem, ElemType::UInt(UIntKind::U32), "a table is u32");
                assert_eq!(valid.shape().dims::<1>(), [rows], "a table is [rows]");
                CausalConvPastLayout::Prompt { ring, positions }
            }
        })
    }
}
