//! How a walk reads its operands and binds them.

use crate::kernels::walk::{
    RECURRENCE_WALK_ELEMENT, RecurrenceWalkBlueprint, RecurrenceWalkProblem,
    RecurrenceWalkStateLayout,
};
use core::num::NonZeroUsize;
use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Axis, Indexing, Launcher, TileSpec, TileTensorLaunch};
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, PlainTensor};
use cubek_microkernel::linear_recurrence::walk::RecurrenceWalkOperandMap;
use cubek_microkernel::linear_recurrence::{
    BATCH, HEAD_IN_GROUP, KEPT_STATE, KEY_FEATURE, KEY_HEAD, POSITION, RATE, RecurrenceHeads,
    STORE_SLOT, VALUE_FEATURE,
};
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Grid};

/// The inputs of a walk, as a caller hands them: the queries and keys `[batch, positions,
/// key_heads, key_features]`, the values `[batch, positions, value_heads, value_features]`, the
/// rates `[batch, positions, value_heads, 2]` (the log decay, then the strength), and the states
/// ([`RecurrenceWalkStateInputs`]), `value_heads` a multiple of `key_heads`. All are `f32`. Each is
/// read where it lies, its rows padded or its view starting partway into a buffer included, the
/// innermost axis of the wide ones contiguous; a padded row or a start off a whole read narrows the
/// lines the walk reads.
#[derive(Clone)]
pub struct RecurrenceWalkInputs {
    /// The client every operand lives on.
    pub client: Client,
    pub query: PlainTensor,
    pub key: PlainTensor,
    pub value: PlainTensor,
    pub rates: PlainTensor,
    pub state: RecurrenceWalkStateInputs,
    /// The element the output is written at: the activations'.
    pub output_element: ElemType,
}

/// The states a walk resumes from, and where it leaves them.
// Built once a call and consumed by it, so the larger variant costs one copy of a few handles.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum RecurrenceWalkStateInputs {
    /// `[batch, value_heads, key_features, value_features]`, one state a sequence, and the number
    /// of states kept for a rewind: the state after each of the last `kept` positions, written to
    /// a tensor of their own.
    Owned {
        state: PlainTensor,
        kept: NonZeroUsize,
    },
    /// A store `[slots, value_heads, key_features, value_features]`, and for each sequence the slot
    /// it lies in (`slots`, `[batch]`, `u32`) and the positions it has walked already (`lengths`,
    /// `[batch]`, `u32`). A sequence that has walked none starts from zeros and reads nothing of
    /// its slot. The walk writes the state after its last position back to the sequence's slot.
    /// Two rows may share a slot only if it is never read. A slot past the store's last reads and
    /// writes the last.
    Slots {
        store: PlainTensor,
        slots: PlainTensor,
        lengths: PlainTensor,
    },
}

impl RecurrenceWalkInputs {
    /// Where the states live, as the problem states it.
    pub fn state_layout(&self) -> RecurrenceWalkStateLayout {
        match &self.state {
            RecurrenceWalkStateInputs::Owned { kept, .. } => {
                RecurrenceWalkStateLayout::Owned { kept: *kept }
            }
            RecurrenceWalkStateInputs::Slots { store, .. } => RecurrenceWalkStateLayout::Slots {
                slots: NonZeroUsize::new(store.shape().dims::<4>()[0])
                    .expect("a store holds a slot"),
            },
        }
    }
}

impl core::fmt::Debug for RecurrenceWalkInputs {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The client says nothing about the walk.
        f.debug_struct("RecurrenceWalkInputs")
            .field("query", &self.query)
            .field("key", &self.key)
            .field("value", &self.value)
            .field("rates", &self.rates)
            .field("state", &self.state)
            .field("output_element", &self.output_element)
            .finish()
    }
}

/// What a walk leaves: what each position read, at the activations' element, `[batch, positions,
/// value_heads, value_features]`, and over states it owns the kept states, oldest first, `[kept,
/// batch, value_heads, key_features, value_features]`. In a store's slots the states are left in
/// the store.
#[derive(Clone)]
pub struct RecurrenceWalkOutputs {
    /// The client the outputs live on, which the autotune checks read two candidates through.
    pub client: Client,
    pub out: TensorHandle,
    pub kept: Option<TensorHandle>,
}

/// What lets a walk be an autotune candidate's output: two candidates must have read and kept the
/// same values.
impl cubecl::tune::AutotuneOutput for RecurrenceWalkOutputs {
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        cubek_matmul::kernels::tune::assert_equivalent(&self.client, self.out.clone(), other.out);
        // A walk in a store's slots keeps its states in a scratch store the check cannot reach.
        if let (Some(kept), Some(other)) = (self.kept.clone(), other.kept) {
            cubek_matmul::kernels::tune::assert_equivalent(&self.client, kept, other);
        }
    }
}

/// The state operands of a walk, viewed as the kernel reads them: what differs between the walks.
#[derive(Clone)]
enum StateTensors {
    /// The state tensor and the tensor the kept states go to.
    Owned {
        state: PlainTensor,
        kept: PlainTensor,
    },
    /// The store, each sequence's window placed at the slot its table names, and the positions
    /// each sequence has walked.
    Slots {
        store: PlainTensor,
        slots: PlainTensor,
        lengths: PlainTensor,
    },
}

/// The tensors of one walk, viewed as the kernel reads them, its outputs among them, and the
/// problem they state.
#[derive(Clone)]
pub struct RecurrenceWalkTensors {
    client: Client,
    query: PlainTensor,
    key: PlainTensor,
    value: PlainTensor,
    rates: PlainTensor,
    out_view: PlainTensor,
    states: StateTensors,
    out: TensorHandle,
    kept_out: Option<TensorHandle>,
    problem: RecurrenceWalkProblem,
}

/// The operands of a launch bound under one launcher. Not `Debug`: cubek's `Launcher` is not.
pub struct RecurrenceWalkBinding {
    pub launcher: Launcher,
    pub query: TileTensorLaunch,
    pub key: TileTensorLaunch,
    pub value: TileTensorLaunch,
    pub rates: TileTensorLaunch,
    pub out: TileTensorLaunch,
    /// The states a walk reads: the state tensor, or the store placed by the slot table.
    pub state: TileTensorLaunch,
    /// The tensor the kept states go to, over states the walk owns.
    pub kept: Option<TileTensorLaunch>,
    /// The positions each sequence has walked, in a store's slots.
    pub lengths: Option<TileTensorLaunch>,
}

impl StateTensors {
    /// The tensor the states are read from.
    fn state(&self) -> &PlainTensor {
        match self {
            Self::Owned { state, .. } => state,
            Self::Slots { store, .. } => store,
        }
    }

    /// The tensor the kept states go to, which only the owned walk binds.
    fn kept(&self) -> Option<&PlainTensor> {
        match self {
            Self::Owned { kept, .. } => Some(kept),
            Self::Slots { .. } => None,
        }
    }

    /// The states, the kept states and the lengths bound under `launcher`, `width` values a line.
    fn bind(
        &self,
        launcher: &Launcher,
        width: usize,
    ) -> (
        TileTensorLaunch,
        Option<TileTensorLaunch>,
        Option<TileTensorLaunch>,
    ) {
        let bind = |tensor, axes: &'static [Axis], width| {
            RecurrenceWalkTensors::bound(launcher, tensor, axes, width)
        };
        match self {
            Self::Owned { state, kept } => (
                bind(
                    state,
                    &[BATCH, KEY_HEAD, HEAD_IN_GROUP, KEY_FEATURE, VALUE_FEATURE],
                    width,
                ),
                Some(bind(
                    kept,
                    &[
                        KEPT_STATE,
                        BATCH,
                        KEY_HEAD,
                        HEAD_IN_GROUP,
                        KEY_FEATURE,
                        VALUE_FEATURE,
                    ],
                    width,
                )),
                None,
            ),
            Self::Slots {
                store,
                slots,
                lengths,
            } => {
                let entries = TileTensorLaunch::new(
                    slots.binding.clone().into_tensor_arg(),
                    slots.elem,
                    1,
                    TileSpec::direct(&[BATCH]),
                );
                (
                    bind(
                        store,
                        &[
                            STORE_SLOT,
                            KEY_HEAD,
                            HEAD_IN_GROUP,
                            KEY_FEATURE,
                            VALUE_FEATURE,
                        ],
                        width,
                    )
                    .indexed(entries, Indexing::Coordinate(STORE_SLOT))
                    .expect(ADMITTED_PLAN_RUNS),
                    None,
                    Some(bind(lengths, &[BATCH], 1)),
                )
            }
        }
    }
}

impl RecurrenceWalkTensors {
    /// The walk of `inputs`, its outputs allocated.
    ///
    /// # Errors
    ///
    /// [`RecurrenceWalkProblem::admit_output`]'s refusals for the output element,
    /// [`RecurrenceWalkProblem::admit_operand`]'s for an input not `f32`, and
    /// [`ELEMENT_NOT_ON_DEVICE`]: what the routines decline whatever the device.
    ///
    /// # Panics
    ///
    /// On operands whose shapes disagree, whose innermost axis is strided, on value heads that
    /// are not a multiple of the key heads, on tables that are not `u32`, on more kept states
    /// than positions or none, on a store of no slot, and on a walk in slots of no position.
    pub fn new(inputs: RecurrenceWalkInputs) -> Result<Self, Refusal> {
        let layout = inputs.state_layout();
        let RecurrenceWalkInputs {
            client,
            query,
            key,
            value,
            rates,
            state,
            output_element,
        } = inputs;
        RecurrenceWalkProblem::admit_output(output_element)?;
        let states = match &state {
            RecurrenceWalkStateInputs::Owned { state, .. } => state,
            RecurrenceWalkStateInputs::Slots { store, .. } => store,
        };
        for tensor in [&query, &key, &value, &rates, states] {
            RecurrenceWalkProblem::admit_operand(tensor.elem)?;
        }
        if !query.runs_on(&client) {
            return Err(ELEMENT_NOT_ON_DEVICE);
        }
        for tensor in [&query, &key, &value, states] {
            let extent = tensor.shape().to_vec().last().copied();
            assert!(
                extent == Some(1) || tensor.binding.strides.last() == Some(&1),
                "the innermost axis of an operand of the walk is contiguous"
            );
        }
        let [batch, positions, key_heads, key_features] = query.shape().dims::<4>();
        let [_, _, value_heads, value_features] = value.shape().dims::<4>();
        assert_eq!(
            key.shape().dims::<4>(),
            [batch, positions, key_heads, key_features]
        );
        assert_eq!(value.shape().dims::<4>()[..2], [batch, positions]);
        assert_eq!(
            rates.shape().dims::<4>(),
            [batch, positions, value_heads, 2]
        );
        assert!(
            key_heads > 0 && value_heads.is_multiple_of(key_heads),
            "{value_heads} value heads are not grouped under {key_heads} key heads"
        );
        let heads_per_group = value_heads / key_heads;
        let heads = RecurrenceHeads {
            key_heads,
            heads_per_group,
            key_features,
            value_features,
        };

        // The value heads a key head groups are two axes, `[key_head, head_in_group]`, over the caller's
        // stride along the value heads.
        let grouped = |tensor: &PlainTensor, head_axis: usize| {
            let mut shape = tensor.shape().to_vec();
            let mut strides = tensor.binding.strides.to_vec();
            let stride = strides[head_axis];
            shape.splice(head_axis..=head_axis, [key_heads, heads_per_group]);
            strides.splice(head_axis..=head_axis, [stride * heads_per_group, stride]);
            tensor.viewed(Metadata::new(shape, strides))
        };
        let (out, out_view) = PlainTensor::empty_contiguous(
            &client,
            Shape::from(vec![batch, positions, value_heads, value_features]),
            output_element,
        );
        if !out_view.runs_on(&client) {
            return Err(ELEMENT_NOT_ON_DEVICE);
        }
        let (states, kept_out) = match &state {
            RecurrenceWalkStateInputs::Owned { state, kept } => {
                assert_eq!(
                    state.shape().dims::<4>(),
                    [batch, value_heads, key_features, value_features]
                );
                assert!(
                    kept.get() <= positions,
                    "a walk of {positions} positions keeps at most all of them, not {kept}"
                );
                let mut shape = state.shape().to_vec();
                shape.insert(0, kept.get());
                let (handle, plain) = PlainTensor::empty_contiguous(
                    &client,
                    Shape::from(shape),
                    RECURRENCE_WALK_ELEMENT,
                );
                let (state, kept) = (grouped(state, 1), grouped(&plain, 2));
                (StateTensors::Owned { state, kept }, Some(handle))
            }
            RecurrenceWalkStateInputs::Slots {
                store,
                slots,
                lengths,
            } => {
                assert!(positions > 0, "a walk in a store's slots walks a position");
                assert_eq!(
                    store.shape().dims::<4>()[1..],
                    [value_heads, key_features, value_features],
                    "a store is [slots, value_heads, key_features, value_features]"
                );
                for table in [slots, lengths] {
                    assert_eq!(table.elem, ElemType::UInt(UIntKind::U32), "a table is u32");
                    assert_eq!(table.shape().dims::<1>(), [batch]);
                }
                (
                    StateTensors::Slots {
                        store: grouped(store, 1),
                        slots: slots.clone(),
                        lengths: lengths.clone(),
                    },
                    None,
                )
            }
        };
        let (value, rates, out_view) = (
            grouped(&value, 2),
            grouped(&rates, 2),
            grouped(&out_view, 2),
        );
        let problem = RecurrenceWalkProblem {
            batch,
            positions,
            heads,
            output_element,
            state: layout,
            alignments: RecurrenceWalkOperandMap {
                query: query.elem_alignment(),
                key: key.elem_alignment(),
                value: value.elem_alignment(),
                rates: rates.elem_alignment(),
                state: states.state().elem_alignment(),
                out: out_view.elem_alignment(),
                kept: states
                    .kept()
                    .map_or(ElemAlignment::ONE, PlainTensor::elem_alignment),
            },
            start_bytes: RecurrenceWalkOperandMap {
                query: query.start_bytes(),
                key: key.start_bytes(),
                value: value.start_bytes(),
                rates: rates.start_bytes(),
                state: states.state().start_bytes(),
                out: out_view.start_bytes(),
                kept: states.kept().map_or(0, PlainTensor::start_bytes),
            },
        };
        Ok(Self {
            client,
            query,
            key,
            value,
            rates,
            out_view,
            states,
            out,
            kept_out,
            problem,
        })
    }

    /// The client the operands live on, which a launch submits to.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The problem the tensors state.
    pub fn problem(&self) -> &RecurrenceWalkProblem {
        &self.problem
    }

    /// The outputs, written.
    pub fn into_outputs(self) -> RecurrenceWalkOutputs {
        RecurrenceWalkOutputs {
            client: self.client,
            out: self.out,
            kept: self.kept_out,
        }
    }

    /// Every operand bound as `blueprint` states it, on `grid`: under its partitioning, at its
    /// width, the rates a scalar at a time.
    pub fn bind<M>(
        &self,
        blueprint: &RecurrenceWalkBlueprint<M>,
        grid: &Grid,
    ) -> RecurrenceWalkBinding {
        let partitioning = &blueprint.partitioning;
        // The grid's launcher, settled over the extents on its first launch and cloned after.
        let launcher = grid.launcher(self.client(), || {
            (
                partitioning.kernel(),
                partitioning.as_cubek().space().clone(),
            )
        });
        let width = blueprint.width;
        let bind = |tensor: &PlainTensor, axes: &'static [Axis], width: usize| {
            Self::bound(&launcher, tensor, axes, width)
        };
        let (state, kept, lengths) = self.states.bind(&launcher, width);
        RecurrenceWalkBinding {
            query: bind(
                &self.query,
                &[BATCH, POSITION, KEY_HEAD, KEY_FEATURE],
                width,
            ),
            key: bind(&self.key, &[BATCH, POSITION, KEY_HEAD, KEY_FEATURE], width),
            value: bind(
                &self.value,
                &[BATCH, POSITION, KEY_HEAD, HEAD_IN_GROUP, VALUE_FEATURE],
                width,
            ),
            rates: bind(
                &self.rates,
                &[BATCH, POSITION, KEY_HEAD, HEAD_IN_GROUP, RATE],
                1,
            ),
            out: bind(
                &self.out_view,
                &[BATCH, POSITION, KEY_HEAD, HEAD_IN_GROUP, VALUE_FEATURE],
                width,
            ),
            state,
            kept,
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
