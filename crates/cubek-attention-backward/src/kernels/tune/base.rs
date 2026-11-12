//! Choosing a backward pass's strategy by measurement.

use crate::kernels::tune::{AttentionBackwardCandidate, AttentionBackwardKey};
use crate::kernels::{
    AttentionBackwardGradients, AttentionBackwardStrategy, AttentionBackwardTensors,
};
use crate::kernels::{
    CubeStagedAttentionBackward, HypercubeAttentionBackward, PlaneFragmentAttentionBackward,
    UnitAttentionBackward,
};
use cubecl::tune::{LocalTuner, Tunable, local_tuner};
use cubecl::{CubeTuneId, Device};
use cubek_std::Hardware;
use cubek_std::autotune::{Autotune, GroupPriority};

type AttentionBackwardAutotune = Autotune<AttentionBackwardKey, AttentionBackwardTensors, ()>;

/// The gradients of the attention `tensors` state, through whichever strategy and setting
/// measures fastest for this problem on `device`.
///
/// Autotune races the plane, the cube staged and the hypercube strategies where the device commits
/// to a plane width, the unit strategy where it does not: elsewhere measuring the unit strategy's
/// settings takes longer than the others'. Under them all sits the unit strategy's lead, which
/// asks nothing of the hardware and serves any problem on any device.
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
        let strategies = autotune.group("microkernel", GroupPriority::DEFAULT);
        autotune = match hw.plane_dim() {
            Ok(_) => autotune
                .prioritized(
                    &strategies,
                    AttentionBackwardCandidate::list(PlaneFragmentAttentionBackward, &hw),
                    AttentionBackwardCandidate::launch,
                )
                .prioritized(
                    &strategies,
                    AttentionBackwardCandidate::list(CubeStagedAttentionBackward, &hw),
                    AttentionBackwardCandidate::launch,
                )
                .prioritized(
                    &strategies,
                    AttentionBackwardCandidate::list(HypercubeAttentionBackward, &hw),
                    AttentionBackwardCandidate::launch,
                ),
            Err(_) => autotune.prioritized(
                &strategies,
                AttentionBackwardCandidate::list(UnitAttentionBackward, &hw),
                AttentionBackwardCandidate::launch,
            ),
        };
        let hw = hw.clone();
        let unit_lead = Tunable::new(
            "unit-attention-backward-lead",
            move |tensors: AttentionBackwardTensors| {
                let strategy = UnitAttentionBackward;
                let lead = strategy.lead(&hw);
                strategy
                    .launch_on(&hw, tensors, &lead)
                    .map(|_| ())
                    .map_err(|why| {
                        format!("the unit strategy's lead does not serve this problem: {why:?}")
                    })
            },
        );
        autotune.floor(unit_lead).finish()
    });
    TUNER.execute(&id, &client, tunables, tensors.clone());
    tensors.into_gradients()
}
