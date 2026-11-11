//! How a causal conv reads its operands and binds them.

use crate::kernels::conv::reads::{BoundReads, CausalConvReads, bound};
use crate::kernels::conv::{
    CausalConvBlueprint, CausalConvPast, CausalConvPastInputs, CausalConvProblem,
};
use crate::kernels::refusals::CAUSAL_CONV_CHANNELS_EMPTY;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Axis, Launcher, TileTensorLaunch};
use cubecl::zspace::Shape;
use cubek_matmul::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, PlainTensor};
use cubek_microkernel::linear_recurrence::conv::CausalConvOperandMap;
use cubek_microkernel::linear_recurrence::{BATCH, CHANNEL, POSITION};
use cubek_std::routine::Grid;

/// The inputs of a causal conv: the channels, the filters `[channels, taps]` (a unit dim between
/// allowed), the bias `[channels]` where there is one, and the past, all in one element. Over a
/// window the input holds the newest values `[rows, channels]`; over a prompt, every row's
/// `[rows, positions, channels]`. They are read where they lie, the innermost axis contiguous; a
/// padded row or an off-line start narrows the conv's lines.
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
        f.debug_struct("CausalConvInputs")
            .field("input", &self.input)
            .field("filters", &self.filters)
            .field("bias", &self.bias)
            .field("past", &self.past)
            .field("output_element", &self.output_element)
            .finish()
    }
}

/// What a causal conv leaves, `[rows, channels]` over a window and `[rows, positions, channels]`
/// over a prompt, at the output element.
#[derive(Clone)]
pub struct CausalConvOutputs {
    pub output: TensorHandle,
}

/// The tensors of one causal conv viewed as the kernel reads them, its output among them, and the
/// problem they state.
#[derive(Clone)]
pub struct CausalConvTensors {
    reads: CausalConvReads,
    bias: Option<PlainTensor>,
    output: PlainTensor,
    outputs: CausalConvOutputs,
    problem: CausalConvProblem,
}

/// The operands of a launch bound under one launcher. Not `Debug`: cubek's `Launcher` is not.
pub(crate) struct CausalConvBinding {
    pub launcher: Launcher,
    pub reads: BoundReads,
    pub bias: Option<TileTensorLaunch>,
    pub output: TileTensorLaunch,
}

impl CausalConvTensors {
    /// The causal conv of `inputs`, its output allocated.
    ///
    /// # Errors
    ///
    /// [`CAUSAL_CONV_CHANNELS_EMPTY`] for no channel,
    /// [`CONV_NOTHING_TO_CONVOLVE`](crate::kernels::refusals::CONV_NOTHING_TO_CONVOLVE) for no row
    /// or position, and [`ELEMENT_NOT_ON_DEVICE`].
    ///
    /// # Panics
    ///
    /// On operands whose shapes disagree with the input's channels, whose innermost axis is
    /// strided, on operands of different elements, on tables that are not `u32`, and on a store of
    /// no slot.
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
        let reads = CausalConvReads::new(client, input, filters, past, channels)?;
        if let Some(bias) = &bias {
            assert_eq!(
                bias.elem,
                reads.element(),
                "the conv's operands share an element"
            );
            assert_eq!(
                bias.shape().dims::<1>(),
                [channels],
                "the bias is [channels]"
            );
            assert!(
                bias.binding.strides.last() == Some(&1),
                "the innermost axis of an operand of the conv is contiguous"
            );
        }
        let shape = match reads.past() {
            CausalConvPast::Window { .. } => vec![reads.batch(), channels],
            CausalConvPast::Prompt { positions, .. } => vec![reads.batch(), positions, channels],
        };
        let (handle, output) =
            PlainTensor::empty_contiguous(reads.client(), Shape::from(shape), output_element);
        if !output.runs_on(reads.client()) {
            return Err(ELEMENT_NOT_ON_DEVICE);
        }
        let alignments = CausalConvOperandMap {
            input: reads.input().elem_alignment(),
            filters: reads.filters().elem_alignment(),
            bias: bias
                .as_ref()
                .map_or(ElemAlignment::ONE, PlainTensor::elem_alignment),
            ring: reads.past_inputs().ring().elem_alignment(),
            lengths: reads.past_inputs().lengths().elem_alignment(),
            output: output.elem_alignment(),
        };
        let start_bytes = CausalConvOperandMap {
            input: reads.input().start_bytes(),
            filters: reads.filters().start_bytes(),
            bias: bias.as_ref().map_or(0, PlainTensor::start_bytes),
            ring: reads.past_inputs().ring().start_bytes(),
            lengths: reads.past_inputs().lengths().start_bytes(),
            output: output.start_bytes(),
        };
        let problem = CausalConvProblem {
            batch: reads.batch(),
            channels,
            taps: reads.taps(),
            element: reads.element(),
            output_element,
            past: reads.past(),
            alignments,
            start_bytes,
        };
        Ok(Self {
            reads,
            bias,
            output,
            outputs: CausalConvOutputs { output: handle },
            problem,
        })
    }

    /// The client the operands live on, which a launch submits to.
    pub(crate) fn client(&self) -> &Client {
        self.reads.client()
    }

    /// The operands every conv reads.
    pub(crate) fn reads(&self) -> &CausalConvReads {
        &self.reads
    }

    /// The problem the tensors state.
    pub fn problem(&self) -> &CausalConvProblem {
        &self.problem
    }

    /// The output, written.
    pub(crate) fn into_outputs(self) -> CausalConvOutputs {
        self.outputs
    }

    /// Every operand bound as `blueprint` states it, on `grid`, a ring placed at the slot its
    /// table names.
    pub(crate) fn bind<M>(
        &self,
        blueprint: &CausalConvBlueprint<M>,
        grid: &Grid,
    ) -> CausalConvBinding {
        let launcher = blueprint.launcher(self.reads.client(), grid);
        let width = blueprint.width;
        let axes: &'static [Axis] = match self.problem.past {
            CausalConvPast::Window { .. } => &[BATCH, CHANNEL],
            CausalConvPast::Prompt { .. } => &[BATCH, POSITION, CHANNEL],
        };
        CausalConvBinding {
            reads: self.reads.bind(&launcher, width),
            bias: (self.bias.as_ref()).map(|bias| bound(&launcher, bias, &[CHANNEL], 1)),
            output: bound(&launcher, &self.output, axes, width),
            launcher,
        }
    }
}
