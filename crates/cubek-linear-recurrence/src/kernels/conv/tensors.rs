//! How a conv reads its operands and binds them.

use crate::kernels::conv::{
    GatedDeltaConvPast, QK_NORMED_CONV_OUTPUT_ELEMENT, QkNormedConvBlueprint, QkNormedConvProblem,
};
use crate::kernels::refusals::QK_NORMED_CONV_NOTHING_TO_CONVOLVE;
use core::num::NonZeroUsize;
use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Axis, Indexing, Launcher, TileSpec, TileTensorLaunch};
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubecl::zspace::striding::row_major_contiguous_strides;
use cubek_matmul::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, PlainTensor};
use cubek_microkernel::linear_recurrence::conv::QkNormedConvOperandMap;
use cubek_microkernel::linear_recurrence::{
    BATCH, CHANNEL, HEAD_IN_GROUP, KEY_FEATURE, KEY_HEAD, POSITION, RING_COLUMN, RecurrenceHeads,
    STORE_SLOT, STREAM_POSITION, TAP, VALUE_FEATURE,
};
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Grid};

/// The inputs of a conv: the `q|k|v` stream, the filters `[conv_dim, taps]` (a unit dim between
/// allowed) and the past, all in one element. Over a window the stream holds the newest values
/// `[rows, conv_dim]`; over a stream, every sequence's `[batch, positions + taps - 1, conv_dim]`,
/// context columns first. They are read where they lie, the innermost axis contiguous; a padded row
/// or an off-line start narrows the conv's lines.
#[derive(Clone)]
pub struct QkNormedConvInputs {
    /// The client every operand lives on.
    pub client: Client,
    pub qkv: PlainTensor,
    pub filters: PlainTensor,
    pub past: GatedDeltaConvPastInputs,
    /// The heads whose channels the stream holds.
    pub heads: RecurrenceHeads,
    /// Added to the sum of squares of a head's queries and of its keys before the square root.
    pub epsilon: f32,
}

/// The past a conv's taps run over.
// Built once a call and consumed by it, so the larger variant costs one copy of a few handles.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum GatedDeltaConvPastInputs {
    /// A window store `[slots, columns, conv_dim]`, its slots and columns at any stride. Each row
    /// has the slot it lies in (`slots`, `[rows]`, `u32`) and the positions it has convolved
    /// (`lengths`, `[rows]`, `u32`); a row with none uses nothing of its slot. Two rows may share
    /// a slot only if it is never read again, and a slot past the store's last reads and writes
    /// the last.
    Window {
        window: PlainTensor,
        slots: PlainTensor,
        lengths: PlainTensor,
    },
    /// The stream's own context columns.
    Stream,
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
/// `[batch, positions, ..]` of a stream conv.
#[derive(Clone)]
pub struct QkNormedConvOutputs {
    pub query: TensorHandle,
    pub key: TensorHandle,
    pub value: TensorHandle,
}

impl QkNormedConvOutputs {
    /// The outputs over `batch` rows or sequences, and the same buffers as the kernel binds them,
    /// without the unit position dim of a window conv.
    ///
    /// # Errors
    ///
    /// [`ELEMENT_NOT_ON_DEVICE`] where the device has no buffer of `f32`.
    fn allocate(
        client: &Client,
        heads: &RecurrenceHeads,
        batch: usize,
        past: &GatedDeltaConvPast,
    ) -> Result<(Self, [PlainTensor; 3]), Refusal> {
        let (lead, bound) = match *past {
            GatedDeltaConvPast::Window { .. } => (vec![batch, 1], vec![batch]),
            GatedDeltaConvPast::Stream { positions } => {
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
    client: Client,
    qkv: PlainTensor,
    filters: PlainTensor,
    query: PlainTensor,
    key: PlainTensor,
    value: PlainTensor,
    past: GatedDeltaConvPastInputs,
    outputs: QkNormedConvOutputs,
    epsilon: f32,
    problem: QkNormedConvProblem,
}

/// The operands of a launch bound under one launcher. Not `Debug`: cubek's `Launcher` is not.
pub(crate) struct QkNormedConvBinding {
    pub launcher: Launcher,
    pub qkv: TileTensorLaunch,
    pub filters: TileTensorLaunch,
    pub query: TileTensorLaunch,
    pub key: TileTensorLaunch,
    pub value: TileTensorLaunch,
    /// Over a window, the row's store placed by the slot table.
    pub window: Option<TileTensorLaunch>,
    /// Over a window, the positions each row has convolved.
    pub lengths: Option<TileTensorLaunch>,
}

impl GatedDeltaConvPastInputs {
    /// The store, which only a window has.
    fn window(&self) -> Option<&PlainTensor> {
        match self {
            Self::Window { window, .. } => Some(window),
            Self::Stream => None,
        }
    }

    /// The positions each row has convolved, which only a window has.
    fn lengths(&self) -> Option<&PlainTensor> {
        match self {
            Self::Window { lengths, .. } => Some(lengths),
            Self::Stream => None,
        }
    }

    /// The window and the lengths bound under `launcher`, `width` values a line.
    fn bind(
        &self,
        launcher: &Launcher,
        width: usize,
    ) -> (Option<TileTensorLaunch>, Option<TileTensorLaunch>) {
        match self {
            Self::Window {
                window,
                slots,
                lengths,
            } => {
                let entries = TileTensorLaunch::new(
                    slots.binding.clone().into_tensor_arg(),
                    slots.elem,
                    1,
                    TileSpec::direct(&[BATCH]),
                );
                let window = QkNormedConvTensors::bound(
                    launcher,
                    window,
                    &[STORE_SLOT, RING_COLUMN, CHANNEL],
                    width,
                )
                .indexed(entries, Indexing::Coordinate(STORE_SLOT))
                .expect(ADMITTED_PLAN_RUNS);
                let lengths = QkNormedConvTensors::bound(launcher, lengths, &[BATCH], 1);
                (Some(window), Some(lengths))
            }
            Self::Stream => (None, None),
        }
    }
}

impl QkNormedConvTensors {
    /// The conv of `inputs`, its outputs allocated.
    ///
    /// # Errors
    ///
    /// [`QkNormedConvProblem::admit_element`]'s refusal, [`QK_NORMED_CONV_NOTHING_TO_CONVOLVE`]
    /// for no row or position, and [`ELEMENT_NOT_ON_DEVICE`].
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
        let element = qkv.elem;
        QkNormedConvProblem::admit_element(element)?;
        for tensor in [Some(&qkv), Some(&filters), past.window()]
            .into_iter()
            .flatten()
        {
            assert_eq!(tensor.elem, element, "the conv's operands share an element");
            assert!(
                tensor.binding.strides.last() == Some(&1),
                "the innermost axis of an operand of the conv is contiguous"
            );
        }
        if !qkv.runs_on(&client) {
            return Err(ELEMENT_NOT_ON_DEVICE);
        }
        let conv_dim = heads.conv_dim();
        let filter_shape = filters.shape().to_vec();
        assert!(
            filter_shape[0] == conv_dim
                && filter_shape[1..filter_shape.len() - 1]
                    .iter()
                    .all(|&e| e == 1),
            "the filters are [conv_dim, taps]"
        );
        let taps = *filter_shape.last().expect("filters have a tap axis");
        let filters = filters.viewed(Metadata::new(
            vec![conv_dim, taps],
            vec![filters.binding.strides[0], 1],
        ));
        let (qkv, viewed, past) = match past {
            GatedDeltaConvPastInputs::Window {
                window,
                slots,
                lengths,
            } => Self::shape_window(qkv, conv_dim, window, slots, lengths)?,
            GatedDeltaConvPastInputs::Stream => Self::shape_stream(qkv, conv_dim, taps)?,
        };
        let batch = qkv.shape()[0];
        let (outputs, [query, key, value]) =
            QkNormedConvOutputs::allocate(&client, &heads, batch, &past)?;
        let alignments = QkNormedConvOperandMap {
            qkv: qkv.elem_alignment(),
            filters: filters.elem_alignment(),
            window: (viewed.window()).map_or(ElemAlignment::ONE, PlainTensor::elem_alignment),
            lengths: (viewed.lengths()).map_or(ElemAlignment::ONE, PlainTensor::elem_alignment),
            query: query.elem_alignment(),
            key: key.elem_alignment(),
            value: value.elem_alignment(),
        };
        let start_bytes = QkNormedConvOperandMap {
            qkv: qkv.start_bytes(),
            filters: filters.start_bytes(),
            window: viewed.window().map_or(0, PlainTensor::start_bytes),
            lengths: viewed.lengths().map_or(0, PlainTensor::start_bytes),
            query: query.start_bytes(),
            key: key.start_bytes(),
            value: value.start_bytes(),
        };
        let problem = QkNormedConvProblem {
            batch,
            heads,
            taps,
            element,
            past,
            alignments,
            start_bytes,
        };
        Ok(Self {
            client,
            qkv,
            filters,
            query,
            key,
            value,
            past: viewed,
            outputs,
            epsilon,
            problem,
        })
    }

    /// The inputs over a window: the stream `[rows, conv_dim]`, the store and its tables, viewed,
    /// and the past they state.
    fn shape_window(
        qkv: PlainTensor,
        conv_dim: usize,
        window: PlainTensor,
        slots: PlainTensor,
        lengths: PlainTensor,
    ) -> Result<(PlainTensor, GatedDeltaConvPastInputs, GatedDeltaConvPast), Refusal> {
        let qkv_shape = qkv.shape().to_vec();
        let rows = qkv_shape[0];
        if rows == 0 {
            return Err(QK_NORMED_CONV_NOTHING_TO_CONVOLVE);
        }
        assert!(
            qkv_shape.last() == Some(&conv_dim)
                && qkv_shape[1..qkv_shape.len() - 1].iter().all(|&e| e == 1),
            "the stream is [rows, conv_dim]"
        );
        let qkv = qkv.viewed(Metadata::new(
            vec![rows, conv_dim],
            vec![qkv.binding.strides[0], 1],
        ));
        let [slots_in_store, columns, window_channels] = window.shape().dims::<3>();
        assert_eq!(
            window_channels, conv_dim,
            "the window holds every channel a column"
        );
        let window = window.viewed(Metadata::new(
            vec![slots_in_store, columns, conv_dim],
            vec![window.binding.strides[0], window.binding.strides[1], 1],
        ));
        for table in [&slots, &lengths] {
            assert_eq!(table.elem, ElemType::UInt(UIntKind::U32), "a table is u32");
            assert_eq!(table.shape().dims::<1>(), [rows]);
        }
        let past = GatedDeltaConvPast::Window {
            slots: NonZeroUsize::new(slots_in_store).expect("a store holds a slot"),
            columns,
        };
        Ok((
            qkv,
            GatedDeltaConvPastInputs::Window {
                window,
                slots,
                lengths,
            },
            past,
        ))
    }

    /// The inputs over a stream `[batch, positions + taps - 1, conv_dim]`, viewed, and the past
    /// they state.
    fn shape_stream(
        qkv: PlainTensor,
        conv_dim: usize,
        taps: usize,
    ) -> Result<(PlainTensor, GatedDeltaConvPastInputs, GatedDeltaConvPast), Refusal> {
        let [batch, stream_positions, channels] = qkv.shape().dims::<3>();
        assert_eq!(
            channels, conv_dim,
            "the stream holds every channel a position"
        );
        let positions = stream_positions.saturating_sub(taps.saturating_sub(1));
        if batch == 0 || positions == 0 {
            return Err(QK_NORMED_CONV_NOTHING_TO_CONVOLVE);
        }
        let qkv = qkv.viewed(Metadata::new(
            vec![batch, stream_positions, conv_dim],
            vec![qkv.binding.strides[0], qkv.binding.strides[1], 1],
        ));
        Ok((
            qkv,
            GatedDeltaConvPastInputs::Stream,
            GatedDeltaConvPast::Stream { positions },
        ))
    }

    /// The client the operands live on, which a launch submits to.
    pub(crate) fn client(&self) -> &Client {
        &self.client
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

    /// Every operand bound as `blueprint` states it, on `grid`, a window placed at the slot its
    /// table names.
    pub(crate) fn bind(
        &self,
        blueprint: &QkNormedConvBlueprint,
        grid: &Grid,
    ) -> QkNormedConvBinding {
        let launcher = blueprint.launcher(&self.client, grid);
        let width = blueprint.width;
        let bind = |tensor: &PlainTensor, axes: &'static [Axis], width: usize| {
            Self::bound(&launcher, tensor, axes, width)
        };
        let (qkv, keyed, valued): (&[Axis], &[Axis], &[Axis]) = match self.past {
            GatedDeltaConvPastInputs::Window { .. } => (
                &[BATCH, CHANNEL],
                &[BATCH, KEY_HEAD, KEY_FEATURE],
                &[BATCH, KEY_HEAD, HEAD_IN_GROUP, VALUE_FEATURE],
            ),
            GatedDeltaConvPastInputs::Stream => (
                &[BATCH, STREAM_POSITION, CHANNEL],
                &[BATCH, POSITION, KEY_HEAD, KEY_FEATURE],
                &[BATCH, POSITION, KEY_HEAD, HEAD_IN_GROUP, VALUE_FEATURE],
            ),
        };
        let (window, lengths) = self.past.bind(&launcher, width);
        QkNormedConvBinding {
            qkv: bind(&self.qkv, qkv, width),
            filters: bind(&self.filters, &[CHANNEL, TAP], 1),
            query: bind(&self.query, keyed, width),
            key: bind(&self.key, keyed, width),
            value: bind(&self.value, valued, width),
            window,
            lengths,
            launcher,
        }
    }

    /// `tensor` bound under `launcher` along `axes`, `width` values a line.
    fn bound(
        launcher: &Launcher,
        tensor: &PlainTensor,
        axes: &'static [Axis],
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
}
