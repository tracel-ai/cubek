//! The walk's autotune: the cube routine's settings, raced per key, over its floor.
//!
//! # A walk in a store's slots is measured on a scratch store
//!
//! It writes the states in place, so a round measures on a store of its own (`measured`), one slot
//! for each row it places: a round on the call's store would advance every live row's state once
//! per candidate. A walk over states it owns is measured on the call's inputs.

use crate::kernels::walk::tune::GatedDeltaWalkCandidate;
use crate::kernels::walk::tune::key::GatedDeltaWalkKey;
use crate::kernels::walk::{
    CubeGatedDeltaWalkRoutine, CubeGatedDeltaWalkRoutineSettings, GATED_DELTA_WALK_ELEMENT,
    GatedDeltaWalkInputs, GatedDeltaWalkOutputs, GatedDeltaWalkProblem, GatedDeltaWalkStateInputs,
    GatedDeltaWalkStateLayout,
};
use cubecl::features::TypeUsage;
use cubecl::ir::ElemType;
use cubecl::tile::launch::Refusal;
use cubecl::tune::{LocalTuner, Tunable, local_tuner};
use cubecl::zspace::striding::row_major_contiguous_strides;
use cubecl::{CubeTuneId, Device};
use cubek_matmul::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, PlainTensor};
use cubek_microkernel::gated_delta::GatedDeltaHeads;
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
    inputs: GatedDeltaWalkInputs,
    microkernel: GroupPriority,
) -> GatedDeltaWalkOutputs {
    let client = inputs.client.clone();
    static TUNER: LocalTuner<GatedDeltaWalkKey, CubeTuneId> = local_tuner!("gated-delta-walk");

    let id = CubeTuneId::new(&client, device);
    let hw = Hardware::from_client(&client);
    let tunables = TUNER.init(&id, move || {
        let mut table =
            Autotune::<GatedDeltaWalkKey, GatedDeltaWalkInputs, GatedDeltaWalkOutputs>::measured_on(
                |inputs: &GatedDeltaWalkInputs| GatedDeltaWalkKey::from(inputs),
                "scratch",
                measured,
            );
        let group = table.group("microkernel", microkernel);
        let floor = GatedDeltaWalkCandidate::new(
            CubeGatedDeltaWalkRoutine,
            CubeGatedDeltaWalkRoutineSettings::FLOOR,
            &hw,
        );
        table
            .prioritized(
                &group,
                GatedDeltaWalkCandidate::list(CubeGatedDeltaWalkRoutine, &hw),
                GatedDeltaWalkCandidate::run,
            )
            // The floor, which every admitted problem prepares: taking `microkernel` out of the
            // table stops the family being measured, not being served.
            .floor(Tunable::new(
                "cube-gated-delta-walk-floor",
                move |inputs: GatedDeltaWalkInputs| floor.run(inputs),
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
/// ([`GatedDeltaWalkProblem::admit_output`], [`ELEMENT_NOT_ON_DEVICE`],
/// [`GatedDeltaWalkProblem::admit`]).
pub fn serves(
    hw: &Hardware,
    heads: GatedDeltaHeads,
    state: GatedDeltaWalkStateLayout,
    output_element: ElemType,
) -> Result<(), Refusal> {
    GatedDeltaWalkProblem::admit_output(output_element)?;
    for element in [GATED_DELTA_WALK_ELEMENT, output_element] {
        let usage = hw.properties.features.type_usage(element);
        if !usage.is_superset(TypeUsage::Arithmetic | TypeUsage::Buffer) {
            return Err(ELEMENT_NOT_ON_DEVICE);
        }
    }
    GatedDeltaWalkProblem::dense(1, state.kept().get(), heads, output_element, state).admit(hw)
}

/// What a round runs on: the call's inputs, but for a walk in a store's slots, whose store is
/// zeros with a slot for each row (as many as the call's store holds, if fewer), each row at a slot
/// of its own and past its first position.
fn measured(_key: &GatedDeltaWalkKey, inputs: &GatedDeltaWalkInputs) -> GatedDeltaWalkInputs {
    let GatedDeltaWalkStateInputs::Slots { store, .. } = &inputs.state else {
        return inputs.clone();
    };
    let client = &inputs.client;
    let [slots, heads, one, cells] = store.shape().dims::<4>();
    let rows = inputs.decay.shape().dims::<3>()[0];
    let scratch = [rows.min(slots), heads, one, cells];
    let table = |entry: &dyn Fn(usize) -> u32| {
        let entries: Vec<u32> = (0..rows).map(entry).collect();
        PlainTensor::of_u32s(client, &[rows], &entries)
    };
    GatedDeltaWalkInputs {
        state: GatedDeltaWalkStateInputs::Slots {
            store: PlainTensor::zeros(
                client,
                &scratch,
                row_major_contiguous_strides(scratch),
                store.elem,
            ),
            slots: table(&|row| (row % scratch[0]) as u32),
            positions: table(&|_| 1),
        },
        ..inputs.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::refusals::{
        GATED_DELTA_WALK_BINDINGS_PAST_DEVICE, GATED_DELTA_WALK_ELEMENT_NOT_FLOAT,
        GATED_DELTA_WALK_HEADS_EMPTY, GATED_DELTA_WALK_SLOTS_BINDING_BUFFERS,
    };
    use core::num::NonZeroUsize;
    use cubecl::ir::{FloatKind, UIntKind};
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

    fn owned() -> GatedDeltaWalkStateLayout {
        GatedDeltaWalkStateLayout::Owned {
            kept: NonZeroUsize::MIN,
        }
    }

    fn slots(slots: usize) -> GatedDeltaWalkStateLayout {
        GatedDeltaWalkStateLayout::Slots {
            slots: NonZeroUsize::new(slots).unwrap(),
        }
    }

    fn heads() -> GatedDeltaHeads {
        GatedDeltaHeads {
            key_heads: 16,
            group: 3,
            key_features: 128,
            value_features: 128,
        }
    }

    /// The floor is served wherever the problem is admitted, and a device that binds too few
    /// buffers declines it by name.
    #[test]
    fn the_table_serves_what_admission_admits() {
        let h100 = running(devices::h100(), &[F32, F16]);
        for state in [owned(), slots(32)] {
            assert_eq!(serves(&h100, heads(), state, F32), Ok(()));
            assert_eq!(serves(&h100, heads(), state, F16), Ok(()));
            for hw in [
                devices::wgpu_no_subgroups(),
                devices::rdna_range(),
                devices::strix_halo(),
            ] {
                assert_eq!(
                    serves(&running(hw.clone(), &[F32, F16]), heads(), state, F32),
                    Err(GATED_DELTA_WALK_BINDINGS_PAST_DEVICE),
                    "{hw:?}"
                );
            }
        }
    }

    /// A device binding ten buffers serves the walk over states it owns and declines the one in a
    /// store's slots by name, which binds one more.
    #[test]
    fn a_device_one_buffer_short_of_slots_serves_the_owned_walk() {
        let mut hw = running(devices::h100(), &[F32]);
        devices::topology(&mut hw).max_bindings = GATED_DELTA_WALK_SLOTS_BINDING_BUFFERS - 1;
        assert_eq!(serves(&hw, heads(), owned(), F32), Ok(()));
        assert_eq!(
            serves(&hw, heads(), slots(8), F32),
            Err(GATED_DELTA_WALK_BINDINGS_PAST_DEVICE)
        );
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
        let heads = GatedDeltaHeads {
            key_heads: 0,
            ..heads()
        };
        assert_eq!(
            serves(&running(devices::h100(), &[F32]), heads, owned(), F32),
            Err(GATED_DELTA_WALK_HEADS_EMPTY)
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
            Err(GATED_DELTA_WALK_ELEMENT_NOT_FLOAT)
        );
    }
}
