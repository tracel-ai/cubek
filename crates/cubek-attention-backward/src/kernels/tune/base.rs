//! Choosing a backward pass's routine by measurement.

use crate::kernels::tune::{AttentionBackwardCandidate, AttentionBackwardKey};
use crate::kernels::{
    AttentionBackwardGradients, AttentionBackwardRoutine, AttentionBackwardTensors,
};
use crate::kernels::{
    CubeStagedAttentionBackwardRoutine, PlaneFragmentAttentionBackwardRoutine,
    UnitAttentionBackwardRoutine,
};
use cubecl::tune::{LocalTuner, Tunable, local_tuner};
use cubecl::{CubeTuneId, Device};
use cubek_std::Hardware;
use cubek_std::autotune::{Autotune, GroupPriority};

type AttentionBackwardAutotune = Autotune<AttentionBackwardKey, AttentionBackwardTensors, ()>;

/// The gradients of the attention `tensors` state, through whichever routine and setting
/// measures fastest for this problem on `device`.
///
/// Autotune races the plane and the cube staged routines where the device commits to a plane
/// width, the unit routine where it does not: elsewhere measuring the unit routine's settings
/// takes longer than the others'. Under them all sits the unit routine's lead, which asks
/// nothing of the hardware and serves any problem on any device.
pub fn attention_backward(
    device: &Device,
    tensors: AttentionBackwardTensors,
) -> AttentionBackwardGradients {
    let client = tensors.client().clone();
    let id = CubeTuneId::new(&client, device);
    let hw = Hardware::from_client(&client);
    // Named rather than `module_path!()`, so the cached winners stay this autotune's wherever the
    // code lives; a cached winner is an index into the candidate list.
    static TUNER: LocalTuner<AttentionBackwardKey, CubeTuneId> = local_tuner!("attention-backward");
    let tunables = TUNER.init(&id, move || {
        let mut autotune = AttentionBackwardAutotune::new(|tensors: &AttentionBackwardTensors| {
            AttentionBackwardKey::from(&tensors.problem())
        });
        let routines = autotune.group("microkernel", GroupPriority::DEFAULT);
        autotune = match hw.plane_dim() {
            Ok(_) => autotune
                .prioritized(
                    &routines,
                    AttentionBackwardCandidate::list(PlaneFragmentAttentionBackwardRoutine, &hw),
                    AttentionBackwardCandidate::launch,
                )
                .prioritized(
                    &routines,
                    AttentionBackwardCandidate::list(CubeStagedAttentionBackwardRoutine, &hw),
                    AttentionBackwardCandidate::launch,
                ),
            Err(_) => autotune.prioritized(
                &routines,
                AttentionBackwardCandidate::list(UnitAttentionBackwardRoutine, &hw),
                AttentionBackwardCandidate::launch,
            ),
        };
        let hw = hw.clone();
        let unit_lead = Tunable::new(
            "unit-attention-backward-lead",
            move |tensors: AttentionBackwardTensors| {
                let routine = UnitAttentionBackwardRoutine;
                let lead = routine.lead(&hw);
                routine
                    .launch_on(&hw, tensors, &lead)
                    .map(|_| ())
                    .map_err(|why| {
                        format!("the unit routine's lead does not serve this problem: {why:?}")
                    })
            },
        );
        autotune.floor(unit_lead).finish()
    });
    TUNER.execute(&id, &client, tunables, tensors.clone());
    tensors.into_gradients()
}
