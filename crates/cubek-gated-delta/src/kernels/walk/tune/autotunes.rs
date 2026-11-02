//! The walk's autotune: the cube routine's settings, raced per key, over its floor.

use crate::kernels::walk::tune::GatedDeltaWalkCandidate;
use crate::kernels::walk::tune::key::GatedDeltaWalkKey;
use crate::kernels::walk::{
    CubeGatedDeltaWalkRoutine, CubeGatedDeltaWalkRoutineSettings, GatedDeltaWalkInputs,
    GatedDeltaWalkOutputs, GatedDeltaWalkProblem,
};
use core::num::NonZeroUsize;
use cubecl::ir::ElemType;
use cubecl::tile::launch::Refusal;
use cubecl::tune::{LocalTuner, Tunable, local_tuner};
use cubecl::{CubeTuneId, Device};
use cubek_microkernel::gated_delta::GatedDeltaHeads;
use cubek_std::Hardware;
use cubek_std::autotune::{Autotune, GroupPriority};

/// The walk of `inputs` through the autotune on `device`. `tile` is the priority of the cube
/// routine's group; under it sits the floor, which serves every admitted problem.
pub fn launch_tuned(
    device: &Device,
    inputs: GatedDeltaWalkInputs,
    tile: GroupPriority,
) -> GatedDeltaWalkOutputs {
    let client = inputs.client.clone();
    static TUNER: LocalTuner<GatedDeltaWalkKey, CubeTuneId> = local_tuner!("gated-delta-walk");

    let id = CubeTuneId::new(&client, device);
    let hw = Hardware::from_client(&client);
    let tunables = TUNER.init(&id, move || {
        let mut table =
            Autotune::<GatedDeltaWalkKey, GatedDeltaWalkInputs, GatedDeltaWalkOutputs>::new(
                |inputs: &GatedDeltaWalkInputs| GatedDeltaWalkKey::from(inputs),
            );
        let group = table.group("tile", tile);
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
            // The floor, which every admitted problem prepares: taking `tile` out of the table
            // stops the family being measured, not being served.
            .floor(Tunable::new(
                "cube-gated-delta-walk-floor",
                move |inputs: GatedDeltaWalkInputs| floor.run(inputs),
            ))
            .finish()
    });

    TUNER.execute(&id, &client, tunables, inputs)
}

/// Whether the table serves a layer of `heads` and operands of `element` on `hw`: the element is a
/// plain `f32` float (a packed one is declined, for the caller to unpack) and the problem is
/// admitted, which includes the floor preparing it at one sequence and one position. An `Ok` is the
/// guarantee that [`launch_tuned`] runs the layer, whichever candidate wins, for as many sequences
/// and positions as the device's grid holds.
///
/// # Errors
///
/// What the element or the layer's problem is declined with
/// ([`GatedDeltaWalkProblem::admit_element`], [`GatedDeltaWalkProblem::admit`]).
pub fn serves(hw: &Hardware, heads: GatedDeltaHeads, element: ElemType) -> Result<(), Refusal> {
    GatedDeltaWalkProblem::admit_element(element)?;
    GatedDeltaWalkProblem::dense(1, 1, heads, element, NonZeroUsize::MIN).admit(hw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::refusals::{
        GATED_DELTA_WALK_BINDINGS_PAST_DEVICE, GATED_DELTA_WALK_ELEMENT_NOT_FLOAT,
        GATED_DELTA_WALK_HEADS_EMPTY,
    };
    use cubecl::ir::{FloatKind, UIntKind};
    use cubek_std::devices;

    const F32: ElemType = ElemType::Float(FloatKind::F32);

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
        assert_eq!(serves(&devices::h100(), heads(), F32), Ok(()));
        for hw in [
            devices::wgpu_no_subgroups(),
            devices::rdna_range(),
            devices::strix_halo(),
        ] {
            assert_eq!(
                serves(&hw, heads(), F32),
                Err(GATED_DELTA_WALK_BINDINGS_PAST_DEVICE),
                "{hw:?}"
            );
        }
    }

    /// A layer with no key head is declined, not divided by.
    #[test]
    fn a_layer_with_no_key_head_is_declined() {
        let heads = GatedDeltaHeads {
            key_heads: 0,
            ..heads()
        };
        assert_eq!(
            serves(&devices::h100(), heads, F32),
            Err(GATED_DELTA_WALK_HEADS_EMPTY)
        );
    }

    /// A packed activation is declined by name through `serves`, for the caller to fall back on.
    #[test]
    fn a_packed_operand_is_declined() {
        assert_eq!(
            serves(&devices::h100(), heads(), ElemType::UInt(UIntKind::U32)),
            Err(GATED_DELTA_WALK_ELEMENT_NOT_FLOAT)
        );
    }
}
