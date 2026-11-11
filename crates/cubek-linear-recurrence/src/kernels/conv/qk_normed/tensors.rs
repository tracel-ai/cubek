//! How a conv reads its operands and binds them.

use crate::kernels::conv::reads::{BoundReads, CausalConvReads, bound};
use crate::kernels::conv::{
    CausalConvPast, CausalConvPastInputs, QK_NORMED_CONV_OUTPUT_ELEMENT, QkNormedConvBlueprint,
    QkNormedConvProblem,
};
use cubecl::client::Client;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Axis, Launcher, TileTensorLaunch};
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubecl::zspace::striding::row_major_contiguous_strides;
use cubek_matmul::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, PlainTensor};
use cubek_microkernel::linear_recurrence::conv::QkNormedConvOperandMap;
use cubek_microkernel::linear_recurrence::{
    BATCH, HEAD_IN_GROUP, KEY_FEATURE, KEY_HEAD, POSITION, RecurrenceHeads, VALUE_FEATURE,
};
use cubek_std::routine::Grid;

/// The inputs of a conv: the `q|k|v` stream, the filters `[conv_dim, taps]` (a unit dim between
/// allowed) and the past, all in one element. Over a window the stream holds the newest values
/// `[rows, conv_dim]`; over a prompt, every row's `[rows, positions, conv_dim]`. They are read
/// where they lie, the innermost axis contiguous; a padded row or an off-line start narrows the
/// conv's lines.
#[derive(Clone)]
pub struct QkNormedConvInputs {
    /// The client every operand lives on.
    pub client: Client,
    pub qkv: PlainTensor,
    pub filters: PlainTensor,
    pub past: CausalConvPastInputs,
    /// The heads whose channels the stream holds.
    pub heads: RecurrenceHeads,
    /// Added to the sum of squares of a head's queries and of its keys before the square root.
    pub epsilon: f32,
}

impl core::fmt::Debug for QkNormedConvInputs {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QkNormedConvInputs")
            .field("qkv", &self.qkv)
            .field("filters", &self.filters)
            .field("past", &self.past)
            .field("heads", &self.heads)
            .field("epsilon", &self.epsilon)
            .finish()
    }
}

/// What a conv leaves in `f32`, in the walk's shapes: the queries and keys `[rows, 1, key_heads,
/// key_features]` and the values `[rows, 1, value_heads, value_features]` of a window conv, or
/// `[batch, positions, ..]` of a prompt conv.
#[derive(Clone)]
pub struct QkNormedConvOutputs {
    pub query: TensorHandle,
    pub key: TensorHandle,
    pub value: TensorHandle,
}

impl QkNormedConvOutputs {
    /// The outputs over `batch` rows, and the same buffers as the kernel binds them, without the
    /// unit position dim of a window conv.
    ///
    /// # Errors
    ///
    /// [`ELEMENT_NOT_ON_DEVICE`] where the device has no buffer of `f32`.
    fn allocate(
        client: &Client,
        heads: &RecurrenceHeads,
        batch: usize,
        past: &CausalConvPast,
    ) -> Result<(Self, [PlainTensor; 3]), Refusal> {
        let (lead, bound) = match *past {
            CausalConvPast::Window { .. } => (vec![batch, 1], vec![batch]),
            CausalConvPast::Prompt { positions, .. } => {
                (vec![batch, positions], vec![batch, positions])
            }
        };
        let make = |tail: &[usize], bound_tail: &[usize]| {
            let (handle, plain) = PlainTensor::empty_contiguous(
                client,
                Shape::from([lead.as_slice(), tail].concat()),
                QK_NORMED_CONV_OUTPUT_ELEMENT,
            );
            let shape = [bound.as_slice(), bound_tail].concat();
            let strides = row_major_contiguous_strides(&shape).to_vec();
            (handle, plain.viewed(Metadata::new(shape, strides)))
        };
        let keyed = [heads.key_heads, heads.key_features];
        let (query, bound_query) = make(&keyed, &keyed);
        let (key, bound_key) = make(&keyed, &keyed);
        let (value, bound_value) = make(
            &[heads.value_heads(), heads.value_features],
            &[heads.key_heads, heads.heads_per_group, heads.value_features],
        );
        if !bound_query.runs_on(client) {
            return Err(ELEMENT_NOT_ON_DEVICE);
        }
        Ok((
            Self { query, key, value },
            [bound_query, bound_key, bound_value],
        ))
    }
}

/// The tensors of one conv viewed as the kernel reads them, its outputs among them, and the
/// problem they state.
#[derive(Clone)]
pub struct QkNormedConvTensors {
    reads: CausalConvReads,
    query: PlainTensor,
    key: PlainTensor,
    value: PlainTensor,
    outputs: QkNormedConvOutputs,
    epsilon: f32,
    problem: QkNormedConvProblem,
}

/// The operands of a launch bound under one launcher. Not `Debug`: cubek's `Launcher` is not.
pub(crate) struct QkNormedConvBinding {
    pub launcher: Launcher,
    pub reads: BoundReads,
    pub query: TileTensorLaunch,
    pub key: TileTensorLaunch,
    pub value: TileTensorLaunch,
}

impl QkNormedConvTensors {
    /// The conv of `inputs`, its outputs allocated.
    ///
    /// # Errors
    ///
    /// [`CONV_NOTHING_TO_CONVOLVE`](crate::kernels::refusals::CONV_NOTHING_TO_CONVOLVE) for no row
    /// or position, and [`ELEMENT_NOT_ON_DEVICE`].
    ///
    /// # Panics
    ///
    /// On operands whose shapes disagree with the heads, whose innermost axis is strided, on
    /// operands of different elements, on tables that are not `u32`, and on a store of no slot.
    pub fn new(inputs: QkNormedConvInputs) -> Result<Self, Refusal> {
        let QkNormedConvInputs {
            client,
            qkv,
            filters,
            past,
            heads,
            epsilon,
        } = inputs;
        let reads = CausalConvReads::new(client, qkv, filters, past, heads.conv_dim())?;
        let (outputs, [query, key, value]) =
            QkNormedConvOutputs::allocate(&reads.client, &heads, reads.batch, &reads.past)?;
        let alignments = QkNormedConvOperandMap {
            qkv: reads.input.elem_alignment(),
            filters: reads.filters.elem_alignment(),
            ring: reads.past_inputs.ring().elem_alignment(),
            lengths: reads.past_inputs.lengths().elem_alignment(),
            query: query.elem_alignment(),
            key: key.elem_alignment(),
            value: value.elem_alignment(),
        };
        let start_bytes = QkNormedConvOperandMap {
            qkv: reads.input.start_bytes(),
            filters: reads.filters.start_bytes(),
            ring: reads.past_inputs.ring().start_bytes(),
            lengths: reads.past_inputs.lengths().start_bytes(),
            query: query.start_bytes(),
            key: key.start_bytes(),
            value: value.start_bytes(),
        };
        let problem = QkNormedConvProblem {
            batch: reads.batch,
            heads,
            taps: reads.taps,
            element: reads.element,
            past: reads.past,
            alignments,
            start_bytes,
        };
        Ok(Self {
            reads,
            query,
            key,
            value,
            outputs,
            epsilon,
            problem,
        })
    }

    /// The client the operands live on, which a launch submits to.
    pub(crate) fn client(&self) -> &Client {
        &self.reads.client
    }

    /// The operands every conv reads.
    pub(crate) fn reads(&self) -> &CausalConvReads {
        &self.reads
    }

    /// The problem the tensors state.
    pub fn problem(&self) -> &QkNormedConvProblem {
        &self.problem
    }

    /// The epsilon of the queries' and keys' norms.
    pub(crate) fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// The outputs, written.
    pub(crate) fn into_outputs(self) -> QkNormedConvOutputs {
        self.outputs
    }

    /// Every operand bound as `blueprint` states it, on `grid`, a ring placed at the slot its
    /// table names.
    pub(crate) fn bind(
        &self,
        blueprint: &QkNormedConvBlueprint,
        grid: &Grid,
    ) -> QkNormedConvBinding {
        let launcher = blueprint.launcher(&self.reads.client, grid);
        let width = blueprint.width;
        let (keyed, valued): (&[Axis], &[Axis]) = match self.problem.past {
            CausalConvPast::Window { .. } => (
                &[BATCH, KEY_HEAD, KEY_FEATURE],
                &[BATCH, KEY_HEAD, HEAD_IN_GROUP, VALUE_FEATURE],
            ),
            CausalConvPast::Prompt { .. } => (
                &[BATCH, POSITION, KEY_HEAD, KEY_FEATURE],
                &[BATCH, POSITION, KEY_HEAD, HEAD_IN_GROUP, VALUE_FEATURE],
            ),
        };
        QkNormedConvBinding {
            reads: self.reads.bind(&launcher, width),
            query: bound(&launcher, &self.query, keyed, width),
            key: bound(&launcher, &self.key, keyed, width),
            value: bound(&launcher, &self.value, valued, width),
            launcher,
        }
    }
}
