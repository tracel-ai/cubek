//! How a conv reads its operands and binds them.

use crate::kernels::conv::{
    GATED_DELTA_CONV_OUTPUT_ELEMENT, GatedDeltaConvBlueprint, GatedDeltaConvPast,
    GatedDeltaConvProblem,
};
use crate::kernels::refusals::GATED_DELTA_CONV_NOTHING_TO_CONVOLVE;
use core::num::NonZeroUsize;
use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Axis, Indexing, Launcher, TileSpec, TileTensorLaunch};
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubecl::zspace::striding::row_major_contiguous_strides;
use cubek_matmul::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, PlainTensor};
use cubek_microkernel::gated_delta::conv::GatedDeltaConvOperandMap;
use cubek_microkernel::gated_delta::{
    BATCH, CHANNEL, GROUP, GatedDeltaHeads, KEY_FEATURE, KEY_HEAD, RING_COLUMN, STORE_SLOT, TAP,
    VALUE_FEATURE,
};
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Grid};

/// The inputs of a conv: the newest values of the `q|k|v` stream `[rows, conv_dim]`, the filters
/// `[conv_dim, taps]` (a unit dim between allowed) and the past, all in one element. They are read
/// where they lie, the innermost axis contiguous; a padded row or an off-line start narrows the
/// conv's lines.
#[derive(Clone)]
pub struct GatedDeltaConvInputs {
    /// The client every operand lives on.
    pub client: Client,
    pub qkv: PlainTensor,
    pub filters: PlainTensor,
    pub past: GatedDeltaConvPastInputs,
    /// The heads whose channels the stream holds.
    pub heads: GatedDeltaHeads,
    /// Added to the sum of squares of a head's queries and of its keys before the square root.
    pub epsilon: f32,
}

/// The past a conv's taps run over.
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
}

impl core::fmt::Debug for GatedDeltaConvInputs {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GatedDeltaConvInputs")
            .field("qkv", &self.qkv)
            .field("filters", &self.filters)
            .field("past", &self.past)
            .field("heads", &self.heads)
            .field("epsilon", &self.epsilon)
            .finish()
    }
}

/// What a conv leaves in `f32`, in the walk's shapes: the queries and keys `[rows, 1, key_heads,
/// key_features]` and the values `[rows, 1, value_heads, value_features]`.
#[derive(Clone)]
pub struct GatedDeltaConvOutputs {
    pub query: TensorHandle,
    pub key: TensorHandle,
    pub value: TensorHandle,
}

impl GatedDeltaConvOutputs {
    /// The outputs over `rows` rows, and the same buffers as the kernel binds them, without the
    /// unit position dim.
    ///
    /// # Errors
    ///
    /// [`ELEMENT_NOT_ON_DEVICE`] where the device has no buffer of `f32`.
    fn allocate(
        client: &Client,
        heads: &GatedDeltaHeads,
        rows: usize,
    ) -> Result<(Self, [PlainTensor; 3]), Refusal> {
        let make = |tail: &[usize], bound_tail: &[usize]| {
            let (handle, plain) = PlainTensor::empty_contiguous(
                client,
                Shape::from([&[rows, 1], tail].concat()),
                GATED_DELTA_CONV_OUTPUT_ELEMENT,
            );
            let shape = [&[rows], bound_tail].concat();
            let strides = row_major_contiguous_strides(&shape).to_vec();
            (handle, plain.viewed(Metadata::new(shape, strides)))
        };
        let keyed = [heads.key_heads, heads.key_features];
        let (query, bound_query) = make(&keyed, &keyed);
        let (key, bound_key) = make(&keyed, &keyed);
        let (value, bound_value) = make(
            &[heads.value_heads(), heads.value_features],
            &[heads.key_heads, heads.group, heads.value_features],
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
pub struct GatedDeltaConvTensors {
    client: Client,
    qkv: PlainTensor,
    filters: PlainTensor,
    query: PlainTensor,
    key: PlainTensor,
    value: PlainTensor,
    past: GatedDeltaConvPastInputs,
    outputs: GatedDeltaConvOutputs,
    epsilon: f32,
    problem: GatedDeltaConvProblem,
}

/// The operands of a launch bound under one launcher. Not `Debug`: cubek's `Launcher` is not.
pub(crate) struct GatedDeltaConvBinding {
    pub launcher: Launcher,
    pub qkv: TileTensorLaunch,
    pub filters: TileTensorLaunch,
    pub window: TileTensorLaunch,
    pub lengths: TileTensorLaunch,
    pub query: TileTensorLaunch,
    pub key: TileTensorLaunch,
    pub value: TileTensorLaunch,
}

impl GatedDeltaConvTensors {
    /// The conv of `inputs`, its outputs allocated.
    ///
    /// # Errors
    ///
    /// [`GatedDeltaConvProblem::admit_element`]'s refusal, [`GATED_DELTA_CONV_NOTHING_TO_CONVOLVE`]
    /// for no row, and [`ELEMENT_NOT_ON_DEVICE`].
    ///
    /// # Panics
    ///
    /// On operands whose shapes disagree with the heads, whose innermost axis is strided, on
    /// operands of different elements, on tables that are not `u32`, and on a store of no slot.
    pub fn new(inputs: GatedDeltaConvInputs) -> Result<Self, Refusal> {
        let GatedDeltaConvInputs {
            client,
            qkv,
            filters,
            past,
            heads,
            epsilon,
        } = inputs;
        let GatedDeltaConvPastInputs::Window {
            window,
            slots,
            lengths,
        } = past;
        let element = qkv.elem;
        GatedDeltaConvProblem::admit_element(element)?;
        for tensor in [&qkv, &filters, &window] {
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
        let qkv_shape = qkv.shape().to_vec();
        let rows = qkv_shape[0];
        if rows == 0 {
            return Err(GATED_DELTA_CONV_NOTHING_TO_CONVOLVE);
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
        let (outputs, [query, key, value]) =
            GatedDeltaConvOutputs::allocate(&client, &heads, rows)?;
        let problem = GatedDeltaConvProblem {
            batch: rows,
            heads,
            taps,
            element,
            past: GatedDeltaConvPast::Window {
                slots: NonZeroUsize::new(slots_in_store).expect("a store holds a slot"),
                columns,
            },
            alignments: GatedDeltaConvOperandMap {
                qkv: qkv.elem_alignment(),
                filters: filters.elem_alignment(),
                window: window.elem_alignment(),
                lengths: lengths.elem_alignment(),
                query: query.elem_alignment(),
                key: key.elem_alignment(),
                value: value.elem_alignment(),
            },
            start_bytes: GatedDeltaConvOperandMap {
                qkv: qkv.start_bytes(),
                filters: filters.start_bytes(),
                window: window.start_bytes(),
                lengths: lengths.start_bytes(),
                query: query.start_bytes(),
                key: key.start_bytes(),
                value: value.start_bytes(),
            },
        };
        Ok(Self {
            client,
            qkv,
            filters,
            query,
            key,
            value,
            past: GatedDeltaConvPastInputs::Window {
                window,
                slots,
                lengths,
            },
            outputs,
            epsilon,
            problem,
        })
    }

    /// The client the operands live on, which a launch submits to.
    pub(crate) fn client(&self) -> &Client {
        &self.client
    }

    /// The problem the tensors state.
    pub fn problem(&self) -> &GatedDeltaConvProblem {
        &self.problem
    }

    /// The epsilon of the queries' and keys' norms.
    pub(crate) fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// The outputs, written.
    pub(crate) fn into_outputs(self) -> GatedDeltaConvOutputs {
        self.outputs
    }

    /// Every operand bound as `blueprint` states it, on `grid`, the window placed at the slot its
    /// table names.
    pub(crate) fn bind(
        &self,
        blueprint: &GatedDeltaConvBlueprint,
        grid: &Grid,
    ) -> GatedDeltaConvBinding {
        let launcher = blueprint.launcher(&self.client, grid);
        let width = blueprint.width;
        let bind = |tensor: &PlainTensor, axes: &'static [Axis], width: usize| {
            Self::bound(&launcher, tensor, axes, width)
        };
        let GatedDeltaConvPastInputs::Window {
            window,
            slots,
            lengths,
        } = &self.past;
        let entries = TileTensorLaunch::new(
            slots.binding.clone().into_tensor_arg(),
            slots.elem,
            1,
            TileSpec::direct(&[BATCH]),
        );
        let window = bind(window, &[STORE_SLOT, RING_COLUMN, CHANNEL], width)
            .indexed(entries, Indexing::Coordinate(STORE_SLOT))
            .expect(ADMITTED_PLAN_RUNS);
        GatedDeltaConvBinding {
            qkv: bind(&self.qkv, &[BATCH, CHANNEL], width),
            filters: bind(&self.filters, &[CHANNEL, TAP], 1),
            window,
            lengths: bind(lengths, &[BATCH], 1),
            query: bind(&self.query, &[BATCH, KEY_HEAD, KEY_FEATURE], width),
            key: bind(&self.key, &[BATCH, KEY_HEAD, KEY_FEATURE], width),
            value: bind(&self.value, &[BATCH, KEY_HEAD, GROUP, VALUE_FEATURE], width),
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
