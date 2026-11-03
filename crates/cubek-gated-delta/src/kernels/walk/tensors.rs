//! How a walk reads its operands and binds them.

use crate::kernels::walk::{
    GATED_DELTA_WALK_ELEMENT, GatedDeltaWalkBlueprint, GatedDeltaWalkProblem,
};
use core::num::NonZeroUsize;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Axis, Launcher, TileTensorLaunch};
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, PlainTensor};
use cubek_microkernel::gated_delta::walk::GatedDeltaWalkOperandMap;
use cubek_microkernel::gated_delta::{
    BATCH, GROUP, GatedDeltaHeads, KEPT_STATE, KEY_FEATURE, KEY_HEAD, POSITION, VALUE_FEATURE,
};
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Grid};

/// The inputs of a walk, as a caller hands them: the queries and keys `[batch, positions,
/// key_heads, key_features]`, the values `[batch, positions, value_heads, value_features]`, the
/// decay and the strength `[batch, positions, value_heads]`, and the state `[batch, value_heads,
/// key_features, value_features]`, `value_heads` a multiple of `key_heads`. All six are `f32`. Each
/// is read where it lies, its rows padded or its view starting partway into a buffer included, the
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
    pub state: PlainTensor,
    /// The element the output is written at: the activations'.
    pub output_element: ElemType,
    /// States kept for a rewind: the state after each of the last this many positions.
    pub kept: NonZeroUsize,
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
/// value_heads, value_features]`, and the kept states, oldest first, `[kept, batch, value_heads,
/// key_features, value_features]`.
#[derive(Clone)]
pub struct GatedDeltaWalkOutputs {
    /// The client the outputs live on, which the autotune checks read two candidates through.
    pub client: Client,
    pub out: TensorHandle,
    pub kept: TensorHandle,
}

/// What lets a walk be an autotune candidate's output: two candidates must have read and kept the
/// same values.
impl cubecl::tune::AutotuneOutput for GatedDeltaWalkOutputs {
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        cubek_matmul::kernels::tune::assert_equivalent(&self.client, self.out.clone(), other.out);
        cubek_matmul::kernels::tune::assert_equivalent(&self.client, self.kept.clone(), other.kept);
    }
}

/// The tensors of one walk, viewed as the kernel reads them, its outputs among them, and the
/// problem they state.
#[derive(Clone)]
pub struct GatedDeltaWalkTensors {
    client: Client,
    operands: GatedDeltaWalkOperandMap<PlainTensor>,
    out: TensorHandle,
    kept: TensorHandle,
    problem: GatedDeltaWalkProblem,
}

/// The operands of a launch bound under one launcher. Not `Debug`: cubek's `Launcher` is not.
pub struct GatedDeltaWalkBinding {
    pub launcher: Launcher,
    pub operands: GatedDeltaWalkOperandMap<TileTensorLaunch>,
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
    /// are not a multiple of the key heads, and on more kept states than positions or none.
    pub fn new(inputs: GatedDeltaWalkInputs) -> Result<Self, Refusal> {
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
        assert_eq!(
            state.shape().dims::<4>(),
            [batch, value_heads, key_features, value_features]
        );
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
        let (kept_handle, kept_plain) = PlainTensor::empty_contiguous(
            &client,
            Shape::from(vec![kept, batch, value_heads, key_features, value_features]),
            GATED_DELTA_WALK_ELEMENT,
        );
        let operands = GatedDeltaWalkOperandMap {
            query: query.clone(),
            key: key.clone(),
            value: grouped(&value, 2),
            decay: grouped(&decay, 2),
            strength: grouped(&strength, 2),
            state: grouped(&state, 1),
            out: grouped(&out_plain, 2),
            kept: grouped(&kept_plain, 2),
        };
        let problem = GatedDeltaWalkProblem {
            batch,
            positions,
            heads,
            output_element,
            kept: NonZeroUsize::new(kept).expect("at least one state is kept"),
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
        let bind = |tensor: &PlainTensor, axes: &'static [Axis], width: usize| {
            launcher
                .arg(tensor.binding.clone(), tensor.elem)
                .axes(axes)
                .vectorize(width)
                .build()
                .expect(ADMITTED_PLAN_RUNS)
                .arg()
        };
        let width = blueprint.width;
        let operands = &self.operands;
        let bound = GatedDeltaWalkOperandMap {
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
            state: bind(
                &operands.state,
                &[BATCH, KEY_HEAD, GROUP, KEY_FEATURE, VALUE_FEATURE],
                width,
            ),
            out: bind(
                &operands.out,
                &[BATCH, POSITION, KEY_HEAD, GROUP, VALUE_FEATURE],
                width,
            ),
            kept: bind(
                &operands.kept,
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
        };
        GatedDeltaWalkBinding {
            launcher,
            operands: bound,
        }
    }
}
