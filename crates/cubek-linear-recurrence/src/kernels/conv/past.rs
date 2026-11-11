//! The past a causal conv's taps run over, which every conv of the family reads the same way.

use crate::kernels::refusals::{
    CONV_ELEMENT_UNSUPPORTED, CONV_NOTHING_TO_CONVOLVE, CONV_TAPS_PAST_RING,
};
use core::num::NonZeroUsize;
use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Axis, Indexing, Launcher, TileSpec, TileTensorLaunch};
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::linear_recurrence::conv::cube::PastKind;
use cubek_microkernel::linear_recurrence::{
    BATCH, CHANNEL, POSITION, PROMPT_POSITION, RING_COLUMN, STORE_SLOT,
};
use cubek_std::microkernel::GlobalStream;
use cubek_std::routine::ADMITTED_PLAN_RUNS;

use super::reads::bound;

/// The activation elements the convs are verified at.
const ELEMENTS: [ElemType; 2] = [
    ElemType::Float(FloatKind::F16),
    ElemType::Float(FloatKind::F32),
];

/// Whether the convs read and write activations of `element`.
///
/// # Errors
///
/// [`CONV_ELEMENT_UNSUPPORTED`] for any but `f16` and `f32`.
pub(crate) fn admit_element(element: ElemType) -> Result<(), Refusal> {
    if ELEMENTS.contains(&element) {
        Ok(())
    } else {
        Err(CONV_ELEMENT_UNSUPPORTED)
    }
}

/// The extents a conv's taps run over: `batch` rows of `channels` channels, `taps` to a filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvExtent {
    pub batch: usize,
    pub channels: usize,
    pub taps: usize,
}

/// What one cube of a conv moves through global memory, apart from the past's own traffic: its
/// `channels` channels of a position, `run` values of each read contiguously.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConvStreams {
    pub channels: usize,
    pub taps: usize,
    pub run: usize,
    pub element_size: usize,
    pub filters: GlobalStream,
    pub outputs: GlobalStream,
}

/// A store of `slots` rings of `columns` columns, one ring a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RingStore {
    pub slots: NonZeroUsize,
    pub columns: usize,
}

/// Where a conv reads the past values its taps run over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausalConvPast {
    /// One position of each row. A row reads the columns its convolved positions fill and writes
    /// its newest values as the ring's next column.
    Window { ring: RingStore },
    /// `positions` positions of each row: tap `j` of position `t` is the prompt's value
    /// `t + j - (taps - 1)` where that is not before the prompt, and the row's ring through its
    /// length otherwise. The last `taps - 1` valid positions of each row are then written to its
    /// ring as its next columns. The caller advances the row's length by its valid positions.
    Prompt { ring: RingStore, positions: usize },
}

impl CausalConvPast {
    /// Where the microkernel reads the past from.
    pub fn kind(&self) -> PastKind {
        match self {
            Self::Window { .. } => PastKind::Window,
            Self::Prompt { .. } => PastKind::Prompt,
        }
    }

    /// The store of rings.
    pub fn ring(&self) -> RingStore {
        match self {
            Self::Window { ring } | Self::Prompt { ring, .. } => *ring,
        }
    }

    /// Whether a launch after the conv fills the rings: a window writes its own.
    pub(crate) fn writes_ring_after(&self) -> bool {
        matches!(self, Self::Prompt { .. })
    }

    /// The positions a row convolves: one over a window.
    pub fn positions(&self) -> usize {
        match self {
            Self::Window { .. } => 1,
            Self::Prompt { positions, .. } => *positions,
        }
    }

    /// Whether the past holds a position to convolve.
    ///
    /// # Errors
    ///
    /// [`CONV_NOTHING_TO_CONVOLVE`] for a prompt of no position, and [`CONV_TAPS_PAST_RING`] for
    /// fewer than two `taps` or a ring of fewer columns than the taps read back.
    pub(crate) fn admit(&self, taps: usize) -> Result<(), Refusal> {
        if self.positions() == 0 {
            return Err(CONV_NOTHING_TO_CONVOLVE);
        }
        if taps < 2 || self.ring().columns + 1 < taps {
            return Err(CONV_TAPS_PAST_RING);
        }
        Ok(())
    }

    /// The streams one cube reads and writes: the input and the ring it reads, the ring a window
    /// writes, and a prompt's share of the write back, which reads each row's last `taps - 1`
    /// positions and writes them to its ring.
    pub(crate) fn streams(&self, conv: &ConvStreams) -> (Vec<GlobalStream>, Vec<GlobalStream>) {
        let stream = |elements: usize| GlobalStream {
            bytes: (elements * conv.element_size) as u64,
            run_bytes: conv.run * conv.element_size,
        };
        let ring_columns = conv.channels * (conv.taps - 1);
        match self {
            Self::Window { .. } => (
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

    /// The bytes of activations a conv over `extent` reads and writes, apart from the filters:
    /// every tap of every position over a window, and a prompt's positions once and its write
    /// back.
    pub(crate) fn bytes(&self, extent: ConvExtent, element_size: usize) -> (usize, usize) {
        let channels = extent.batch * self.positions() * extent.channels;
        match self {
            Self::Window { .. } => (
                channels * extent.taps * element_size,
                channels * element_size,
            ),
            Self::Prompt { .. } => {
                let ring = extent.batch * (extent.taps - 1) * extent.channels * element_size;
                (channels * element_size + ring, ring)
            }
        }
    }

    /// The axes an input leads with, ahead of its channels: a row, and a prompt's position.
    pub(crate) fn input_lead(&self) -> &'static [Axis] {
        match self {
            Self::Window { .. } => &[BATCH],
            Self::Prompt { .. } => &[BATCH, PROMPT_POSITION],
        }
    }

    /// The axes an output leads with, ahead of its features: a row, and a prompt's position.
    pub(crate) fn output_lead(&self) -> &'static [Axis] {
        match self {
            Self::Window { .. } => &[BATCH],
            Self::Prompt { .. } => &[BATCH, POSITION],
        }
    }

    /// The extents of [`input_lead`](Self::input_lead) and [`output_lead`](Self::output_lead) over
    /// `batch` rows.
    pub(crate) fn lead_extents(&self, batch: usize) -> Vec<usize> {
        match self {
            Self::Window { .. } => vec![batch],
            Self::Prompt { positions, .. } => vec![batch, *positions],
        }
    }

    /// The axes of a cube's leaf the past adds: a cube holds one position of a prompt.
    pub(crate) fn leaf_axes(&self) -> Vec<(Axis, usize)> {
        match self {
            Self::Window { .. } => vec![],
            Self::Prompt { .. } => vec![(POSITION, 1)],
        }
    }

    /// The grid's axes, a prompt's positions ahead of the `boxes` of channels.
    pub(crate) fn cube_axes(&self, boxes: Axis) -> Vec<Axis> {
        match self {
            Self::Window { .. } => vec![boxes],
            Self::Prompt { .. } => vec![POSITION, boxes],
        }
    }

    /// The axes the past's operands carry: the store, and the prompt's positions.
    pub(crate) fn past_axes(&self) -> Vec<(Axis, usize)> {
        let ring = self.ring();
        let mut axes = vec![(STORE_SLOT, ring.slots.get()), (RING_COLUMN, ring.columns)];
        if let Self::Prompt { positions, .. } = self {
            axes.push((PROMPT_POSITION, *positions));
        }
        axes
    }

    /// The axes of the past, and of the positions a cube convolves.
    pub(crate) fn axes(&self) -> Vec<(Axis, usize)> {
        let mut axes = self.past_axes();
        if let Self::Prompt { positions, .. } = self {
            axes.push((POSITION, *positions));
        }
        axes
    }
}

/// The past a conv's taps run over, as tensors: the ring store, and the tables that place each row
/// in it.
// Built once a call and consumed by it, so the larger variant costs one copy of a few handles.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum CausalConvPastInputs {
    /// A ring store `[slots, columns, channels]`, its slots and columns at any stride. Each row
    /// has the slot it lies in (`slots`, `[rows]`, `u32`) and the positions it has convolved
    /// (`lengths`, `[rows]`, `u32`). Two rows may share a slot only if it is never read again, and a slot past the
    /// store's last reads and writes the last.
    Window {
        ring: PlainTensor,
        slots: PlainTensor,
        lengths: PlainTensor,
    },
    /// The same store and tables of a prompt `[rows, positions, channels]`, and the positions of
    /// each row that are real (`valid`, `[rows]`, `u32`, at most the prompt's): the rest is right
    /// padding, which the ring takes nothing from and whose outputs are unspecified. `lengths` is
    /// the fill before the prompt.
    Prompt {
        ring: PlainTensor,
        slots: PlainTensor,
        lengths: PlainTensor,
        valid: PlainTensor,
    },
}

/// The past's operands bound under one launcher.
pub(crate) struct BoundPast {
    pub ring: TileTensorLaunch,
    pub lengths: TileTensorLaunch,
}

impl CausalConvPastInputs {
    /// The store.
    pub(crate) fn ring(&self) -> &PlainTensor {
        match self {
            Self::Window { ring, .. } | Self::Prompt { ring, .. } => ring,
        }
    }

    /// The positions each row has convolved.
    pub(crate) fn lengths(&self) -> &PlainTensor {
        match self {
            Self::Window { lengths, .. } | Self::Prompt { lengths, .. } => lengths,
        }
    }

    /// The slot table.
    pub(crate) fn slots(&self) -> &PlainTensor {
        match self {
            Self::Window { slots, .. } | Self::Prompt { slots, .. } => slots,
        }
    }

    /// The store and the lengths bound under `launcher`, `width` values a line, the ring placed at
    /// the slot its table names.
    pub(crate) fn bind(&self, launcher: &Launcher, width: usize) -> BoundPast {
        let entries = TileTensorLaunch::new(
            self.slots().binding.clone().into_tensor_arg(),
            self.slots().elem,
            1,
            TileSpec::direct(&[BATCH]),
        );
        let ring = bound(
            launcher,
            self.ring(),
            &[STORE_SLOT, RING_COLUMN, CHANNEL],
            width,
        )
        .indexed(entries, Indexing::Coordinate(STORE_SLOT))
        .expect(ADMITTED_PLAN_RUNS);
        BoundPast {
            ring,
            lengths: bound(launcher, self.lengths(), &[BATCH], 1),
        }
    }

    /// Views the store as the kernel reads it and returns `input` of `channels` channels viewed
    /// likewise, with the past they state: a window's `[rows, channels]` or a prompt's
    /// `[rows, positions, channels]`.
    ///
    /// # Errors
    ///
    /// [`CONV_NOTHING_TO_CONVOLVE`] for no row or position.
    ///
    /// # Panics
    ///
    /// On shapes that disagree with `channels`, and on tables that are not `u32` or not
    /// `[rows]`.
    pub(crate) fn view(
        self,
        input: PlainTensor,
        channels: usize,
    ) -> Result<(Self, PlainTensor, CausalConvPast), Refusal> {
        let shape = input.shape().to_vec();
        assert!(
            shape.len() >= 2,
            "the input has a row axis and a channel axis"
        );
        let rows = shape[0];
        let strides = &input.binding.strides;
        let (positions, extents, input_strides) = match &self {
            Self::Window { .. } => {
                assert!(
                    shape.last() == Some(&channels)
                        && shape[1..shape.len() - 1].iter().all(|&e| e == 1),
                    "the input over a window is [rows, channels]"
                );
                (1, vec![rows, channels], vec![strides[0], 1])
            }
            Self::Prompt { .. } => {
                let [_, positions, prompt_channels] = input.shape().dims::<3>();
                assert_eq!(
                    prompt_channels, channels,
                    "the prompt holds every channel a position"
                );
                (
                    positions,
                    vec![rows, positions, channels],
                    vec![strides[0], strides[1], 1],
                )
            }
        };
        if rows == 0 || positions == 0 {
            return Err(CONV_NOTHING_TO_CONVOLVE);
        }
        let input = input.viewed(Metadata::new(extents, input_strides));
        let ring = self.ring();
        let [slots_in_store, columns, ring_channels] = ring.shape().dims::<3>();
        assert_eq!(
            ring_channels, channels,
            "the ring holds every channel a column"
        );
        let strides = ring.binding.strides.clone();
        let viewed = ring.clone().viewed(Metadata::new(
            vec![slots_in_store, columns, channels],
            vec![strides[0], strides[1], 1],
        ));
        for table in [self.slots(), self.lengths()] {
            assert_eq!(table.elem, ElemType::UInt(UIntKind::U32), "a table is u32");
            assert_eq!(table.shape().dims::<1>(), [rows]);
        }
        let ring = RingStore {
            slots: NonZeroUsize::new(slots_in_store).expect("a store holds a slot"),
            columns,
        };
        Ok(match self {
            Self::Window { slots, lengths, .. } => (
                Self::Window {
                    ring: viewed,
                    slots,
                    lengths,
                },
                input,
                CausalConvPast::Window { ring },
            ),
            Self::Prompt {
                slots,
                lengths,
                valid,
                ..
            } => {
                assert_eq!(valid.elem, ElemType::UInt(UIntKind::U32), "a table is u32");
                assert_eq!(valid.shape().dims::<1>(), [rows]);
                (
                    Self::Prompt {
                        ring: viewed,
                        slots,
                        lengths,
                        valid,
                    },
                    input,
                    CausalConvPast::Prompt { ring, positions },
                )
            }
        })
    }
}
