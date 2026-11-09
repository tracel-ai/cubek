//! The walk's autotune: the cube routine's settings, raced per key, over its floor.
//!
//! # A walk in a store's slots is measured on a scratch store
//!
//! It writes the states in place, so a round measures on a store of its own (`measured`), one slot
//! for each row it places: a round on the call's store would advance every live row's state once
//! per candidate. A walk over states it owns is measured on the call's inputs.

use crate::kernels::refusals::admit_on_device;
use crate::kernels::walk::tune::RecurrenceWalkCandidate;
use crate::kernels::walk::tune::key::RecurrenceWalkKey;
use crate::kernels::walk::{
    CubeDeltaWalkRoutine, CubeDeltaWalkRoutineSettings, RECURRENCE_WALK_ELEMENT,
    RecurrenceWalkInputs, RecurrenceWalkOutputs, RecurrenceWalkProblem, RecurrenceWalkStateInputs,
    RecurrenceWalkStateKind,
};
use cubecl::ir::ElemType;
use cubecl::tile::launch::Refusal;
use cubecl::tune::{LocalTuner, Tunable, local_tuner};
use cubecl::{CubeTuneId, Device};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::linear_recurrence::RecurrenceHeads;
use cubek_std::Hardware;
use cubek_std::autotune::{Autotune, GroupPriority};

/// The walk of `inputs` through the autotune on `device`. `microkernel` is the priority of the
/// cube routine's group; under it sits the floor, which serves every admitted problem.
///
/// # Panics
///
/// On a problem [`serves`] declines: every candidate declines it, and the autotune has nothing to
/// run.
pub fn launch_tuned(
    device: &Device,
    inputs: RecurrenceWalkInputs,
    microkernel: GroupPriority,
) -> RecurrenceWalkOutputs {
    let client = inputs.client.clone();
    static TUNER: LocalTuner<RecurrenceWalkKey, CubeTuneId> = local_tuner!("delta-walk");

    let id = CubeTuneId::new(&client, device);
    let hw = Hardware::from_client(&client);
    let tunables = TUNER.init(&id, move || {
        let mut table =
            Autotune::<RecurrenceWalkKey, RecurrenceWalkInputs, RecurrenceWalkOutputs>::measured_on(
                |inputs: &RecurrenceWalkInputs| RecurrenceWalkKey::from(inputs),
                "scratch",
                measured,
            );
        let group = table.group("microkernel", microkernel);
        let floor = RecurrenceWalkCandidate::new(
            CubeDeltaWalkRoutine,
            CubeDeltaWalkRoutineSettings::FLOOR,
            &hw,
        );
        table
            .prioritized(
                &group,
                RecurrenceWalkCandidate::list(CubeDeltaWalkRoutine, &hw),
                RecurrenceWalkCandidate::run,
            )
            // The floor, which every admitted problem prepares: taking `microkernel` out of the
            // table stops the family being measured, not being served.
            .floor(Tunable::new(
                "cube-delta-walk-floor",
                move |inputs: RecurrenceWalkInputs| floor.run(inputs),
            ))
            .finish()
    });

    TUNER.execute(&id, &client, tunables, inputs)
}

/// Whether the table serves a layer of `heads` with its states laid out as `state` and the output
/// written at `output_element` on `hw`: the inputs are `f32`, the output is a plain float the walk
/// is verified at (a packed one is declined, for the caller to unpack) that the device runs, and
/// the problem is admitted, which includes the floor preparing it at one sequence and one
/// position. An `Ok` is the guarantee that [`launch_tuned`] runs the layer, whichever candidate
/// wins, for as many sequences and positions as the device's grid holds.
///
/// # Errors
///
/// What the element or the layer's problem is declined with
/// ([`RecurrenceWalkProblem::admit_output`],
/// [`ELEMENT_NOT_ON_DEVICE`](cubek_matmul::kernels::tensor::ELEMENT_NOT_ON_DEVICE),
/// [`RecurrenceWalkProblem::admit`]).
pub fn serves(
    hw: &Hardware,
    heads: RecurrenceHeads,
    state: RecurrenceWalkStateKind,
    output_element: ElemType,
) -> Result<(), Refusal> {
    RecurrenceWalkProblem::admit_output(output_element)?;
    admit_on_device(hw, [RECURRENCE_WALK_ELEMENT, output_element])?;
    let state = state.layout();
    RecurrenceWalkProblem::dense(1, state.kept().get(), heads, output_element, state).admit(hw)
}

/// What a round runs on: the call's inputs, but for a walk in a store's slots, whose store is
/// zeros at the call's strides, with a slot for each row, each row at its own and past its first
/// position.
fn measured(_key: &RecurrenceWalkKey, inputs: &RecurrenceWalkInputs) -> RecurrenceWalkInputs {
    let RecurrenceWalkStateInputs::Slots { store, .. } = &inputs.state else {
        return inputs.clone();
    };
    let client = &inputs.client;
    let [_, heads, key_features, value_features] = store.shape().dims::<4>();
    let rows = inputs.rates.shape().dims::<4>()[0];
    let table = |entry: &dyn Fn(usize) -> u32| {
        let entries: Vec<u32> = (0..rows).map(entry).collect();
        PlainTensor::of_u32s(client, &[rows], &entries)
    };
    RecurrenceWalkInputs {
        state: RecurrenceWalkStateInputs::Slots {
            store: PlainTensor::zeros(
                client,
                &[rows, heads, key_features, value_features],
                store.binding.strides.clone(),
                store.elem,
            ),
            slots: table(&|row| row as u32),
            lengths: table(&|_| 1),
        },
        ..inputs.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::refusals::{
        RECURRENCE_WALK_BINDINGS_PAST_DEVICE, RECURRENCE_WALK_ELEMENT_NOT_FLOAT,
        RECURRENCE_WALK_HEADS_EMPTY,
    };
    use core::num::NonZeroUsize;
    use cubecl::features::TypeUsage;
    use cubecl::ir::{FloatKind, UIntKind};
    use cubek_matmul::kernels::tensor::ELEMENT_NOT_ON_DEVICE;
    use cubek_std::devices;

    const F32: ElemType = ElemType::Float(FloatKind::F32);
    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// `hw` as a device that computes in and stores each of `elements`: the fixtures grant none.
    fn running(mut hw: Hardware, elements: &[ElemType]) -> Hardware {
        for element in elements {
            devices::features(&mut hw)
                .types
                .elem
                .insert(*element, TypeUsage::Arithmetic | TypeUsage::Buffer);
        }
        hw
    }

    fn owned() -> RecurrenceWalkStateKind {
        RecurrenceWalkStateKind::Owned {
            kept: NonZeroUsize::MIN,
        }
    }

    fn heads() -> RecurrenceHeads {
        RecurrenceHeads {
            key_heads: 16,
            heads_per_group: 3,
            key_features: 128,
            value_features: 128,
        }
    }

    /// The floor is served wherever the problem is admitted, and a device that binds too few
    /// buffers declines the problem it cannot bind by name: of the devices that bind eight, the
    /// walk over owned states is served, the walk in slots is not.
    #[test]
    fn the_table_serves_what_admission_admits() {
        let h100 = running(devices::h100(), &[F32, F16]);
        for state in [owned(), RecurrenceWalkStateKind::Slots] {
            assert_eq!(serves(&h100, heads(), state, F32), Ok(()));
            assert_eq!(serves(&h100, heads(), state, F16), Ok(()));
            let binds = RecurrenceWalkProblem::dense(1, 1, heads(), F32, state.layout())
                .state
                .binding_buffers();
            for hw in [
                devices::wgpu_no_subgroups(),
                devices::rdna_range(),
                devices::strix_halo(),
            ] {
                let served = serves(&running(hw.clone(), &[F32, F16]), heads(), state, F32);
                if binds > hw.properties.hardware.max_bindings {
                    assert_eq!(served, Err(RECURRENCE_WALK_BINDINGS_PAST_DEVICE), "{hw:?}");
                } else {
                    assert_ne!(served, Err(RECURRENCE_WALK_BINDINGS_PAST_DEVICE), "{hw:?}");
                }
            }
        }
    }

    /// An output element the device does not run is declined by name, so [`launch_tuned`] is never
    /// reached with a layer none of its candidates can launch.
    #[test]
    fn an_element_the_device_does_not_run_is_declined() {
        let no_f16 = running(devices::h100(), &[F32]);
        assert_eq!(serves(&no_f16, heads(), owned(), F32), Ok(()));
        assert_eq!(
            serves(&no_f16, heads(), owned(), F16),
            Err(ELEMENT_NOT_ON_DEVICE)
        );
        let no_f32 = running(devices::h100(), &[F16]);
        assert_eq!(
            serves(&no_f32, heads(), owned(), F16),
            Err(ELEMENT_NOT_ON_DEVICE)
        );
    }

    /// A layer with no key head is declined, not divided by.
    #[test]
    fn a_layer_with_no_key_head_is_declined() {
        let heads = RecurrenceHeads {
            key_heads: 0,
            ..heads()
        };
        assert_eq!(
            serves(&running(devices::h100(), &[F32]), heads, owned(), F32),
            Err(RECURRENCE_WALK_HEADS_EMPTY)
        );
    }

    /// A packed activation is declined by name through `serves`, for the caller to fall back on.
    #[test]
    fn a_packed_operand_is_declined() {
        assert_eq!(
            serves(
                &running(devices::h100(), &[F32]),
                heads(),
                owned(),
                ElemType::UInt(UIntKind::U32)
            ),
            Err(RECURRENCE_WALK_ELEMENT_NOT_FLOAT)
        );
    }
}
