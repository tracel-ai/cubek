//! How a walk reads its operands and binds them.

use crate::kernels::walk::{
    GATED_DELTA_WALK_ELEMENT, GatedDeltaWalkBlueprint, GatedDeltaWalkProblem,
    GatedDeltaWalkStateLayout,
};
use core::num::NonZeroUsize;
use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Axis, Indexing, Launcher, TileArgLaunch};
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, PlainTensor};
use cubek_microkernel::gated_delta::walk::GatedDeltaWalkOperandMap;
use cubek_microkernel::gated_delta::{
    BATCH, GROUP, GatedDeltaHeads, KEPT_STATE, KEY_FEATURE, KEY_HEAD, POSITION, STORE_SLOT,
    VALUE_FEATURE,
};
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Grid};

/// The inputs of a walk, as a caller hands them: the queries and keys `[batch, positions,
/// key_heads, key_features]`, the values `[batch, positions, value_heads, value_features]`, the
/// decay and the strength `[batch, positions, value_heads]`, and the states
/// ([`GatedDeltaWalkStateInputs`]), `value_heads` a multiple of `key_heads`. All are `f32`. Each is
/// read where it lies, its rows padded or its view starting partway into a buffer included, the
/// innermost axis of the wide ones contiguous; a padded row or a start off a whole read narrows the
/// lines the walk reads.
#[derive(Clone)]
pub struct GatedDeltaWalkInputs {
    /// The client every operand lives on.
    pub client: Client,
    pub query: PlainTensor,
    pub key: PlainTensor,
    pub value: PlainTensor,
    pub decay: PlainTensor,
    pub strength: PlainTensor,
    pub state: GatedDeltaWalkStateInputs,
    /// The element the output is written at: the activations'.
    pub output_element: ElemType,
    /// States kept for a rewind: the state after each of the last this many positions.
    pub kept: NonZeroUsize,
}

/// The states a walk resumes from, and where it leaves them.
// Built once a call and consumed by it, so the larger variant costs one copy of a few handles.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum GatedDeltaWalkStateInputs {
    /// `[batch, value_heads, key_features, value_features]`, one state a sequence. The kept states
    /// are written to a tensor of their own.
    Owned(PlainTensor),
    /// A store `[slots, value_heads, 1, key_features * value_features]`, its states row-major and
    /// its slots at any stride, and for each sequence the slot it lies in (`slots`, `[batch]`,
    /// `u32`) and the positions it has walked already (`positions`, `[batch]`, `u32`). A sequence
    /// that has walked none starts from zeros and reads nothing of its slot. The walk writes the
    /// state after its last position back to the sequence's slot.
    Slots {
        store: PlainTensor,
        slots: PlainTensor,
        positions: PlainTensor,
    },
}

impl GatedDeltaWalkInputs {
    /// Where the states live, as the problem states it.
    pub fn state_layout(&self) -> GatedDeltaWalkStateLayout {
        match &self.state {
            GatedDeltaWalkStateInputs::Owned(_) => {
                GatedDeltaWalkStateLayout::Owned { kept: self.kept }
            }
            GatedDeltaWalkStateInputs::Slots { store, .. } => GatedDeltaWalkStateLayout::Slots {
                slots: NonZeroUsize::new(store.shape().dims::<4>()[0])
                    .expect("a store holds a slot"),
                kept: self.kept,
            },
        }
    }
}

impl core::fmt::Debug for GatedDeltaWalkInputs {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The client says nothing about the walk.
        f.debug_struct("GatedDeltaWalkInputs")
            .field("query", &self.query)
            .field("key", &self.key)
            .field("value", &self.value)
            .field("decay", &self.decay)
            .field("strength", &self.strength)
            .field("state", &self.state)
            .field("output_element", &self.output_element)
            .field("kept", &self.kept)
            .finish()
    }
}

/// What a walk leaves: what each position read, at the activations' element, `[batch, positions,
/// value_heads, value_features]`, and over states it owns the kept states, oldest first, `[kept,
/// batch, value_heads, key_features, value_features]`. In a store's slots the states are left in
/// the store.
#[derive(Clone)]
pub struct GatedDeltaWalkOutputs {
    /// The client the outputs live on, which the autotune checks read two candidates through.
    pub client: Client,
    pub out: TensorHandle,
    pub kept: Option<TensorHandle>,
}

/// What lets a walk be an autotune candidate's output: two candidates must have read and kept the
/// same values.
impl cubecl::tune::AutotuneOutput for GatedDeltaWalkOutputs {
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        cubek_matmul::kernels::tune::assert_equivalent(&self.client, self.out.clone(), other.out);
        // A walk in a store's slots keeps its states in a scratch store the check cannot reach.
        if let (Some(kept), Some(other)) = (self.kept.clone(), other.kept) {
            cubek_matmul::kernels::tune::assert_equivalent(&self.client, kept, other);
        }
    }
}

/// The tables a walk in a store's slots places its sequences by.
#[derive(Clone)]
struct SlotTables {
    slots: PlainTensor,
    positions: PlainTensor,
}

/// The tensors of one walk, viewed as the kernel reads them, its outputs among them, and the
/// problem they state.
#[derive(Clone)]
pub struct GatedDeltaWalkTensors {
    client: Client,
    operands: GatedDeltaWalkOperandMap<PlainTensor>,
    tables: Option<SlotTables>,
    out: TensorHandle,
    kept: Option<TensorHandle>,
    problem: GatedDeltaWalkProblem,
}

/// The operands of a launch bound under one launcher. Not `Debug`: cubek's `Launcher` is not.
pub struct GatedDeltaWalkBinding {
    pub launcher: Launcher,
    pub query: TileArgLaunch,
    pub key: TileArgLaunch,
    pub value: TileArgLaunch,
    pub decay: TileArgLaunch,
    pub strength: TileArgLaunch,
    pub out: TileArgLaunch,
    pub state: GatedDeltaWalkStateBinding,
}

/// The state operands of a launch: what differs between the walks.
pub enum GatedDeltaWalkStateBinding {
    Owned {
        state: TileArgLaunch,
        kept: TileArgLaunch,
    },
    /// The store, placed by the slot table, and the positions table.
    Slots {
        store: TileArgLaunch,
        positions: TileArgLaunch,
    },
}

impl GatedDeltaWalkTensors {
    /// The walk of `inputs`, its outputs allocated.
    ///
    /// # Errors
    ///
    /// [`GatedDeltaWalkProblem::admit_output`]'s refusals for the output element,
    /// [`GatedDeltaWalkProblem::admit_operand`]'s for an input not `f32`, and
    /// [`ELEMENT_NOT_ON_DEVICE`]: what the routines decline whatever the device.
    ///
    /// # Panics
    ///
    /// On operands whose shapes disagree, whose innermost axis is strided, on value heads that
    /// are not a multiple of the key heads, on tables that are not `u32`, and on more kept states
    /// than positions or none.
    pub fn new(inputs: GatedDeltaWalkInputs) -> Result<Self, Refusal> {
        let layout = inputs.state_layout();
        let GatedDeltaWalkInputs {
            client,
            query,
            key,
            value,
            decay,
            strength,
            state,
            output_element,
            kept,
        } = inputs;
        let (state, tables) = match state {
            GatedDeltaWalkStateInputs::Owned(state) => (state, None),
            GatedDeltaWalkStateInputs::Slots {
                store,
                slots,
                positions,
            } => (store, Some(SlotTables { slots, positions })),
        };
        let all = [&query, &key, &value, &decay, &strength, &state];
        GatedDeltaWalkProblem::admit_output(output_element)?;
        for tensor in all {
            GatedDeltaWalkProblem::admit_operand(tensor.elem)?;
        }
        if !query.runs_on(&client) {
            return Err(ELEMENT_NOT_ON_DEVICE);
        }
        for tensor in [&query, &key, &value, &state] {
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
        assert_eq!(decay.shape().dims::<3>(), [batch, positions, value_heads]);
        assert_eq!(
            strength.shape().dims::<3>(),
            [batch, positions, value_heads]
        );
        match &tables {
            None => assert_eq!(
                state.shape().dims::<4>(),
                [batch, value_heads, key_features, value_features]
            ),
            Some(tables) => {
                let [_, heads, one, cells] = state.shape().dims::<4>();
                assert_eq!(
                    [heads, one, cells],
                    [value_heads, 1, key_features * value_features],
                    "a store is [slots, value_heads, 1, key_features * value_features]"
                );
                for table in [&tables.slots, &tables.positions] {
                    assert_eq!(table.elem, ElemType::UInt(UIntKind::U32), "a table is u32");
                    assert_eq!(table.shape().dims::<1>(), [batch]);
                }
            }
        }
        assert!(
            key_heads > 0 && value_heads.is_multiple_of(key_heads),
            "{value_heads} value heads are not grouped under {key_heads} key heads"
        );
        let kept = kept.get();
        assert!(
            kept <= positions,
            "a walk of {positions} positions keeps at most all of them, not {kept}"
        );
        let group = value_heads / key_heads;
        let heads = GatedDeltaHeads {
            key_heads,
            group,
            key_features,
            value_features,
        };

        // The value heads a key head groups are two axes, `[key_head, group]`, over the caller's
        // stride along the value heads.
        let grouped = |tensor: &PlainTensor, head_axis: usize| {
            let mut shape = tensor.shape().to_vec();
            let mut strides = tensor.binding.strides.to_vec();
            let stride = strides[head_axis];
            shape.splice(head_axis..=head_axis, [key_heads, group]);
            strides.splice(head_axis..=head_axis, [stride * group, stride]);
            tensor.viewed(Metadata::new(shape, strides))
        };
        let (out, out_plain) = PlainTensor::empty_contiguous(
            &client,
            Shape::from(vec![batch, positions, value_heads, value_features]),
            output_element,
        );
        if !out_plain.runs_on(&client) {
            return Err(ELEMENT_NOT_ON_DEVICE);
        }
        // A store's states are one `[key_features, value_features]` row-major block a slot and
        // head, over the caller's strides along the slots and the heads.
        let state_view = match &tables {
            None => grouped(&state, 1),
            Some(_) => {
                let [slot, head] = [state.binding.strides[0], state.binding.strides[1]];
                state.viewed(Metadata::new(
                    vec![
                        state.shape().dims::<4>()[0],
                        key_heads,
                        group,
                        key_features,
                        value_features,
                    ],
                    vec![slot, head * group, head, value_features, 1],
                ))
            }
        };
        let (kept_handle, kept_view) = match &tables {
            None => {
                let (handle, plain) = PlainTensor::empty_contiguous(
                    &client,
                    Shape::from(vec![kept, batch, value_heads, key_features, value_features]),
                    GATED_DELTA_WALK_ELEMENT,
                );
                let view = grouped(&plain, 2);
                (Some(handle), view)
            }
            // The kept states are the store: the owned walk's operand is read by no slot walk.
            Some(_) => (None, state_view.clone()),
        };
        let operands = GatedDeltaWalkOperandMap {
            query: query.clone(),
            key: key.clone(),
            value: grouped(&value, 2),
            decay: grouped(&decay, 2),
            strength: grouped(&strength, 2),
            state: state_view,
            out: grouped(&out_plain, 2),
            kept: kept_view,
        };
        let problem = GatedDeltaWalkProblem {
            batch,
            positions,
            heads,
            output_element,
            state: layout,
            alignments: GatedDeltaWalkOperandMap {
                query: operands.query.elem_alignment(),
                key: operands.key.elem_alignment(),
                value: operands.value.elem_alignment(),
                decay: operands.decay.elem_alignment(),
                strength: operands.strength.elem_alignment(),
                state: operands.state.elem_alignment(),
                out: operands.out.elem_alignment(),
                kept: operands.kept.elem_alignment(),
            },
            start_bytes: GatedDeltaWalkOperandMap {
                query: operands.query.start_bytes(),
                key: operands.key.start_bytes(),
                value: operands.value.start_bytes(),
                decay: operands.decay.start_bytes(),
                strength: operands.strength.start_bytes(),
                state: operands.state.start_bytes(),
                out: operands.out.start_bytes(),
                kept: operands.kept.start_bytes(),
            },
        };
        Ok(Self {
            client,
            operands,
            tables,
            out,
            kept: kept_handle,
            problem,
        })
    }

    /// The client the operands live on, which a launch submits to.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The problem the tensors state.
    pub fn problem(&self) -> &GatedDeltaWalkProblem {
        &self.problem
    }

    /// The outputs, written.
    pub fn into_outputs(self) -> GatedDeltaWalkOutputs {
        GatedDeltaWalkOutputs {
            client: self.client,
            out: self.out,
            kept: self.kept,
        }
    }

    /// Every operand bound as `blueprint` states it, on `grid`: under its partitioning, at its
    /// width, the decay and the strength a scalar at a time.
    pub fn bind<M>(
        &self,
        blueprint: &GatedDeltaWalkBlueprint<M>,
        grid: &Grid,
    ) -> GatedDeltaWalkBinding {
        let partitioning = &blueprint.partitioning;
        // The grid's launcher, settled over the extents on its first launch and cloned after.
        let launcher = grid.launcher(self.client(), || {
            (
                partitioning.kernel(),
                partitioning.as_cubek().space().clone(),
            )
        });
        let width = blueprint.width;
        let operands = &self.operands;
        let bind = |tensor: &PlainTensor, axes: &'static [Axis], width: usize| {
            Self::bound(&launcher, tensor, axes, width)
        };
        let state = match &self.tables {
            None => self.bind_owned(&launcher, width),
            Some(tables) => self.bind_slots(&launcher, tables, width),
        };
        GatedDeltaWalkBinding {
            query: bind(
                &operands.query,
                &[BATCH, POSITION, KEY_HEAD, KEY_FEATURE],
                width,
            ),
            key: bind(
                &operands.key,
                &[BATCH, POSITION, KEY_HEAD, KEY_FEATURE],
                width,
            ),
            value: bind(
                &operands.value,
                &[BATCH, POSITION, KEY_HEAD, GROUP, VALUE_FEATURE],
                width,
            ),
            decay: bind(&operands.decay, &[BATCH, POSITION, KEY_HEAD, GROUP], 1),
            strength: bind(&operands.strength, &[BATCH, POSITION, KEY_HEAD, GROUP], 1),
            out: bind(
                &operands.out,
                &[BATCH, POSITION, KEY_HEAD, GROUP, VALUE_FEATURE],
                width,
            ),
            state,
            launcher,
        }
    }

    /// The state tensor and the kept states, bound by sequence.
    fn bind_owned(&self, launcher: &Launcher, width: usize) -> GatedDeltaWalkStateBinding {
        GatedDeltaWalkStateBinding::Owned {
            state: Self::bound(
                launcher,
                &self.operands.state,
                &[BATCH, KEY_HEAD, GROUP, KEY_FEATURE, VALUE_FEATURE],
                width,
            ),
            kept: Self::bound(
                launcher,
                &self.operands.kept,
                &[
                    KEPT_STATE,
                    BATCH,
                    KEY_HEAD,
                    GROUP,
                    KEY_FEATURE,
                    VALUE_FEATURE,
                ],
                width,
            ),
        }
    }

    /// The store, each sequence's window placed at the slot its table names, and the positions.
    fn bind_slots(
        &self,
        launcher: &Launcher,
        tables: &SlotTables,
        width: usize,
    ) -> GatedDeltaWalkStateBinding {
        let entries = Self::bound(launcher, &tables.slots, &[BATCH], 1);
        let store = Self::bound(
            launcher,
            &self.operands.state,
            &[STORE_SLOT, KEY_HEAD, GROUP, KEY_FEATURE, VALUE_FEATURE],
            width,
        )
        .indexed(entries, Indexing::Coordinate(STORE_SLOT))
        .expect(ADMITTED_PLAN_RUNS);
        GatedDeltaWalkStateBinding::Slots {
            store,
            positions: Self::bound(launcher, &tables.positions, &[BATCH], 1),
        }
    }

    /// `tensor` bound under `launcher` along `axes`, `width` values a line.
    fn bound(
        launcher: &Launcher,
        tensor: &PlainTensor,
        axes: &'static [Axis],
        width: usize,
    ) -> TileArgLaunch {
        launcher
            .arg(tensor.binding.clone(), tensor.elem)
            .axes(axes)
            .vectorize(width)
            .build()
            .expect(ADMITTED_PLAN_RUNS)
            .arg()
    }
}
