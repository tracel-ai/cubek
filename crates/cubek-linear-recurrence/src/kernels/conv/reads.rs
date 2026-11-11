//! The operands every conv reads, and how they bind them.

use crate::kernels::conv::past::BoundPast;
use crate::kernels::conv::{CausalConvPast, CausalConvPastInputs};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Axis, Launcher, TileTensorLaunch};
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, PlainTensor};
use cubek_microkernel::linear_recurrence::{CHANNEL, TAP};
use cubek_std::routine::ADMITTED_PLAN_RUNS;

/// The operands every conv reads: the input channels, the filters `[channels, taps]` (a unit dim
/// between allowed) and the past, in one element.
#[derive(Clone)]
pub(crate) struct CausalConvReads {
    client: Client,
    input: PlainTensor,
    filters: PlainTensor,
    past_inputs: CausalConvPastInputs,
    past: CausalConvPast,
    batch: usize,
    channels: usize,
    taps: usize,
    element: ElemType,
}

/// The reads bound under one launcher, `width` values a line.
pub(crate) struct BoundReads {
    pub input: TileTensorLaunch,
    pub filters: TileTensorLaunch,
    pub past: BoundPast,
}

impl CausalConvReads {
    /// The reads of `input` of `channels` channels, viewed as the kernels read them.
    ///
    /// # Errors
    ///
    /// [`CONV_NOTHING_TO_CONVOLVE`](crate::kernels::refusals::CONV_NOTHING_TO_CONVOLVE) for no row
    /// or position, and [`ELEMENT_NOT_ON_DEVICE`].
    ///
    /// # Panics
    ///
    /// On operands whose shapes disagree with `channels`, whose innermost axis is strided, on
    /// operands of different elements, on tables that are not `u32`, and on a store of no slot.
    pub(crate) fn new(
        client: Client,
        input: PlainTensor,
        filters: PlainTensor,
        past: CausalConvPastInputs,
        channels: usize,
    ) -> Result<Self, Refusal> {
        let element = input.elem;
        for tensor in [&input, &filters, past.ring()] {
            assert_eq!(tensor.elem, element, "the conv's operands share an element");
            assert!(
                tensor.binding.strides.last() == Some(&1),
                "the innermost axis of an operand of the conv is contiguous"
            );
        }
        if !input.runs_on(&client) {
            return Err(ELEMENT_NOT_ON_DEVICE);
        }
        let filter_shape = filters.shape().to_vec();
        assert!(
            filter_shape.len() >= 2,
            "the filters have a channel axis and a tap axis"
        );
        assert!(
            filter_shape[0] == channels
                && filter_shape[1..filter_shape.len() - 1]
                    .iter()
                    .all(|&e| e == 1),
            "the filters are [channels, taps]"
        );
        let taps = *filter_shape.last().expect("filters have a tap axis");
        let filters = filters.viewed(Metadata::new(
            vec![channels, taps],
            vec![filters.binding.strides[0], 1],
        ));
        let (past_inputs, input, past) = past.view(input, channels)?;
        Ok(Self {
            client,
            batch: input.shape()[0],
            input,
            filters,
            past_inputs,
            past,
            channels,
            taps,
            element,
        })
    }

    pub(crate) fn client(&self) -> &Client {
        &self.client
    }

    pub(crate) fn input(&self) -> &PlainTensor {
        &self.input
    }

    pub(crate) fn filters(&self) -> &PlainTensor {
        &self.filters
    }

    pub(crate) fn past_inputs(&self) -> &CausalConvPastInputs {
        &self.past_inputs
    }

    pub(crate) fn past(&self) -> CausalConvPast {
        self.past
    }

    /// The rows.
    pub(crate) fn batch(&self) -> usize {
        self.batch
    }

    pub(crate) fn channels(&self) -> usize {
        self.channels
    }

    pub(crate) fn taps(&self) -> usize {
        self.taps
    }

    pub(crate) fn element(&self) -> ElemType {
        self.element
    }

    /// The input bound under `launcher`, `width` values a line.
    pub(crate) fn bind_input(&self, launcher: &Launcher, width: usize) -> TileTensorLaunch {
        let axes = [self.past.input_lead(), &[CHANNEL]].concat();
        bound(launcher, &self.input, &axes, width)
    }

    /// The reads bound under `launcher`, `width` values a line, the ring placed at the slot its
    /// table names.
    pub(crate) fn bind(&self, launcher: &Launcher, width: usize) -> BoundReads {
        BoundReads {
            input: self.bind_input(launcher, width),
            filters: bound(launcher, &self.filters, &[CHANNEL, TAP], 1),
            past: self.past_inputs.bind(launcher, width),
        }
    }
}

/// `tensor` bound under `launcher` along `axes`, `width` values a line.
pub(crate) fn bound(
    launcher: &Launcher,
    tensor: &PlainTensor,
    axes: &[Axis],
    width: usize,
) -> TileTensorLaunch {
    launcher
        .arg(tensor.binding.clone(), tensor.elem)
        .axes(axes)
        .vectorize(width)
        .build()
        .expect(ADMITTED_PLAN_RUNS)
        .arg()
}
