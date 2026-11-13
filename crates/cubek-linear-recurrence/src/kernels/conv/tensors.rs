//! How a causal conv reads its operands and binds them.

use crate::kernels::bind::bind;
use crate::kernels::conv::{CausalConvBlueprint, CausalConvPastInputs, CausalConvProblem};
use crate::kernels::refusals::CAUSAL_CONV_CHANNELS_EMPTY;
use crate::kernels::{BoundSlots, SlotStore};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Launcher, TileTensorLaunch};
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, PlainTensor};
use cubek_microkernel::linear_recurrence::conv::CausalConvOperandMap;
use cubek_microkernel::linear_recurrence::{
    BATCH, CHANNEL, POSITION, PROMPT_POSITION, RING_COLUMN, STORE_SLOT, TAP,
};
use cubek_std::routine::Grid;

/// The inputs of a causal conv: the input `[rows, positions, channels]` (a decode step's positions are
/// one), the filters `[channels, taps]` (a unit dim between allowed), the bias `[channels]` where
/// there is one, and the past, all in one element. They are read where they lie, the innermost
/// axis contiguous; a padded row or an off-line start narrows the conv's lines.
#[derive(Clone)]
pub struct CausalConvInputs {
    /// The client every operand lives on.
    pub client: Client,
    pub input: PlainTensor,
    pub filters: PlainTensor,
    pub bias: Option<PlainTensor>,
    pub past: CausalConvPastInputs,
    /// The element the convolved channels are written at: `f32`, or the input's.
    pub output_element: ElemType,
}

impl core::fmt::Debug for CausalConvInputs {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The client says nothing about the conv.
        f.debug_struct("CausalConvInputs")
            .field("input", &self.input)
            .field("filters", &self.filters)
            .field("bias", &self.bias)
            .field("past", &self.past)
            .field("output_element", &self.output_element)
            .finish()
    }
}

/// What a causal conv leaves, `[rows, positions, channels]` at the output element.
#[derive(Clone)]
pub struct CausalConvOutputs {
    pub output: TensorHandle,
}

/// The tensors of one causal conv viewed as the kernel reads them, its output among them, and the
/// problem they state.
#[derive(Clone)]
pub struct CausalConvTensors {
    client: Client,
    input: PlainTensor,
    filters: PlainTensor,
    bias: Option<PlainTensor>,
    past: CausalConvPastInputs,
    output: PlainTensor,
    outputs: CausalConvOutputs,
    problem: CausalConvProblem,
}

/// The operands of a launch bound under one launcher. Not `Debug`: cubek's `Launcher` is not.
pub struct CausalConvBinding {
    pub launcher: Launcher,
    pub input: TileTensorLaunch,
    pub filters: TileTensorLaunch,
    pub bias: Option<TileTensorLaunch>,
    /// The ring placed at the slot its table names, and the lengths.
    pub past: BoundSlots,
    pub output: TileTensorLaunch,
}

impl CausalConvTensors {
    /// The causal conv of `inputs`, its output allocated.
    ///
    /// # Errors
    ///
    /// [`CAUSAL_CONV_CHANNELS_EMPTY`] for no channel, `CAUSAL_CONV_NOTHING_TO_CONVOLVE` for no row
    /// or position, and [`ELEMENT_NOT_ON_DEVICE`].
    ///
    /// # Panics
    ///
    /// On operands whose shapes disagree with the input's channels, whose innermost axis is
    /// strided, on operands of different elements, on a decode step of more than one position, and on
    /// tables that are not `[rows]`.
    pub fn new(inputs: CausalConvInputs) -> Result<Self, Refusal> {
        let CausalConvInputs {
            client,
            input,
            filters,
            bias,
            past,
            output_element,
        } = inputs;
        let channels = *input.shape().last().expect("an input has a channel axis");
        if channels == 0 {
            return Err(CAUSAL_CONV_CHANNELS_EMPTY);
        }
        let element = input.elem;
        for tensor in [&input, &filters, past.ring().store()]
            .into_iter()
            .chain(&bias)
        {
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
        if let Some(bias) = &bias {
            assert_eq!(
                bias.shape().dims::<1>(),
                [channels],
                "the bias is [channels]"
            );
        }
        let state = past.check(&input, channels)?;
        let batch = input.shape()[0];
        let (handle, output) = PlainTensor::empty_contiguous(
            &client,
            Shape::from(vec![batch, state.positions(), channels]),
            output_element,
        );
        if !output.runs_on(&client) {
            return Err(ELEMENT_NOT_ON_DEVICE);
        }
        let ring = past.ring().store();
        let alignments = CausalConvOperandMap {
            input: input.elem_alignment(),
            filters: filters.elem_alignment(),
            bias: bias
                .as_ref()
                .map_or(ElemAlignment::ONE, PlainTensor::elem_alignment),
            ring: ring.elem_alignment(),
            lengths: past.ring().lengths().elem_alignment(),
            output: output.elem_alignment(),
        };
        let start_bytes = CausalConvOperandMap {
            input: input.start_bytes(),
            filters: filters.start_bytes(),
            bias: bias.as_ref().map_or(0, PlainTensor::start_bytes),
            ring: ring.start_bytes(),
            lengths: past.ring().lengths().start_bytes(),
            output: output.start_bytes(),
        };
        let problem = CausalConvProblem {
            batch,
            channels,
            taps,
            element,
            output_element,
            bias: bias.is_some(),
            past: state,
            alignments,
            start_bytes,
        };
        Ok(Self {
            client,
            input,
            filters,
            bias,
            past,
            output,
            outputs: CausalConvOutputs { output: handle },
            problem,
        })
    }

    /// The client the operands live on, which a launch submits to.
    pub(crate) fn client(&self) -> &Client {
        &self.client
    }

    /// The problem the tensors state.
    pub fn problem(&self) -> &CausalConvProblem {
        &self.problem
    }

    /// The past the taps run over, as the kernels read it.
    pub(crate) fn past_inputs(&self) -> &CausalConvPastInputs {
        &self.past
    }

    /// The outputs, written.
    pub fn into_outputs(self) -> CausalConvOutputs {
        self.outputs
    }

    /// The input bound under `launcher`, `width` values a line.
    pub(crate) fn bind_input(&self, launcher: &Launcher, width: usize) -> TileTensorLaunch {
        bind(
            launcher,
            &self.input,
            &[BATCH, PROMPT_POSITION, CHANNEL],
            width,
        )
    }

    /// The ring bound under `launcher`, `width` values a line, placed at the slot its table names.
    pub(crate) fn bind_ring(&self, launcher: &Launcher, width: usize) -> BoundSlots {
        let ring: &SlotStore = self.past.ring();
        ring.bind(launcher, &[STORE_SLOT, RING_COLUMN, CHANNEL], width)
    }

    /// Every operand bound as `blueprint` states it, on `grid`.
    pub fn bind<M>(&self, blueprint: &CausalConvBlueprint<M>, grid: &Grid) -> CausalConvBinding {
        let launcher = blueprint.partitioning.launcher(&self.client, grid);
        let width = blueprint.width;
        CausalConvBinding {
            input: self.bind_input(&launcher, width),
            filters: bind(&launcher, &self.filters, &[CHANNEL, TAP], 1),
            bias: (self.bias.as_ref()).map(|bias| bind(&launcher, bias, &[CHANNEL], width)),
            past: self.bind_ring(&launcher, width),
            output: bind(&launcher, &self.output, &[BATCH, POSITION, CHANNEL], width),
            launcher,
        }
    }
}
