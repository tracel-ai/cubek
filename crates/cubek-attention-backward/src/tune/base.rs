//! Choosing a backward pass's routine by measurement.

use crate::kernels::{
    CubeStagedAttentionBackwardRoutine, PlaneFragmentAttentionBackwardRoutine,
    UnitAttentionBackwardRoutine,
};
use crate::tune::{AttentionBackwardCandidate, AttentionBackwardKey};
use crate::{AttentionBackwardGradients, AttentionBackwardRoutine, AttentionBackwardTensors};
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use cubecl::{CubeTuneId, Device};
use cubek_std::Hardware;
use cubek_std::autotune::{Autotune, GroupPriority};

/// The gradients of the attention `tensors` state, through whichever routine and setting
/// measures fastest for this problem on `device`. Under the routines sits the unit routine's lead,
/// which serves any problem on any device.
pub fn launch_tuned(
    device: &Device,
    tensors: AttentionBackwardTensors,
) -> AttentionBackwardGradients {
    let client = tensors.client().clone();
    let id = CubeTuneId::new(&client, device);
    let hw = Hardware::from_client(&client);
    // Named rather than `module_path!()`, so the cached winners stay this autotune's wherever the
    // code lives; a cached winner is an index into the candidate list.
    static TUNER: LocalTuner<AttentionBackwardKey, CubeTuneId> = local_tuner!("attention-backward");
    let tunables = TUNER.init(&id, move || routines(&hw));
    TUNER.execute(&id, &client, tunables, tensors.clone());
    tensors.into_gradients()
}

type AttentionBackwardAutotune = Autotune<AttentionBackwardKey, AttentionBackwardTensors, ()>;

/// The plane and the cube staged routines where the device commits to a plane width, the unit
/// routine where it does not: elsewhere measuring the unit routine's settings takes longer than
/// the others', and its lead sits under them all.
fn routines(hw: &Hardware) -> TunableSet<AttentionBackwardKey, AttentionBackwardTensors, ()> {
    let mut autotune = AttentionBackwardAutotune::new(|tensors: &AttentionBackwardTensors| {
        AttentionBackwardKey::of(&tensors.problem())
    });
    let routines = autotune.group("microkernel", GroupPriority::DEFAULT);
    autotune = match hw.plane_dim() {
        Ok(_) => autotune
            .prioritized(
                &routines,
                AttentionBackwardCandidate::list(PlaneFragmentAttentionBackwardRoutine, hw),
                launch,
            )
            .prioritized(
                &routines,
                AttentionBackwardCandidate::list(CubeStagedAttentionBackwardRoutine, hw),
                launch,
            ),
        Err(_) => autotune.prioritized(
            &routines,
            AttentionBackwardCandidate::list(UnitAttentionBackwardRoutine, hw),
            launch,
        ),
    };
    autotune.floor(unit_lead(hw)).finish()
}

/// The autotune's floor: the unit routine's lead, which asks nothing of the hardware.
fn unit_lead(hw: &Hardware) -> Tunable<AttentionBackwardKey, AttentionBackwardTensors, ()> {
    let hw = hw.clone();
    Tunable::new(
        "unit-attention-backward-lead",
        move |tensors: AttentionBackwardTensors| {
            let routine = UnitAttentionBackwardRoutine;
            let lead = routine.settings(&hw).swap_remove(0);
            routine
                .launch_on(&hw, tensors, &lead)
                .map(|_| ())
                .map_err(|why| {
                    format!("the unit routine's lead does not serve this problem: {why:?}")
                })
        },
    )
}

/// `tensors`' gradients written by `candidate`'s routine at its setting, or why it did not run.
fn launch<R: AttentionBackwardRoutine + Copy>(
    candidate: &AttentionBackwardCandidate<R>,
    tensors: AttentionBackwardTensors,
) -> Result<(), String> {
    let settings = cubek_std::autotune::TuneCandidate::strategy(candidate);
    let hw = Hardware::from_client(tensors.client());
    candidate
        .routine()
        .launch_on(&hw, tensors, settings)
        .map(|_| ())
        .map_err(|why| format!("{settings} does not serve this problem: {why:?}"))
}
