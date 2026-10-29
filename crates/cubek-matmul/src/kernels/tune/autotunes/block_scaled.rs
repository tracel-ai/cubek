use super::base::{
    MatmulAutotuneGroups, MatmulProduct, MatmulTunables, RoutineIdentity, candidate_list,
    reads_no_routes,
};
use crate::kernels::MatmulRoutine;
use crate::kernels::plan::MatmulCall;
use crate::kernels::tensor::PlainTensor;
use crate::kernels::tune::key::MatmulTuneKey;
use crate::kernels::{CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings, MatmulTensors};
use cubecl::tune::Tunable;
use cubek_microkernel::matmul::cube::CubeStagedMicrokernel;
use cubek_std::Hardware;
use cubek_std::autotune::{GroupPriority, Identity};
use cubek_std::routine::Routine;

/// The block-scaled autotune's candidates on `hw`: the staged routine reading both inputs as stored
/// on the block-scaled tensor cores, at each of its block-scaled settings
/// ([`block_scaled_settings`](CubeStagedMatmulRoutine::block_scaled_settings)), the last, its
/// default, on the floor, since it serves every product the selector sends here.
///
/// Every candidate sums each output in the same order, so the race moves the time and no number.
/// Not bounded: a roofline is a 16-bit product's peak, which a block-scaled instruction runs past, so a
/// bound would end every round at its first candidate. Its one group is its own: the autotune races
/// one kernel's settings.
pub(super) fn candidates(hw: Hardware, _groups: MatmulAutotuneGroups) -> MatmulTunables {
    let mut autotune = candidate_list();
    let group = autotune.group("block-scaled", GroupPriority::DEFAULT);
    let settings = CubeStagedMatmulRoutine.block_scaled_settings(&hw);
    let (&floor, raced) = settings
        .split_last()
        .expect("the block-scaled settings end in the default");
    let autotune = raced.iter().fold(autotune, |autotune, &settings| {
        autotune.candidate(&group, candidate(hw.clone(), settings))
    });
    autotune.floor(candidate(hw, floor)).finish()
}

/// `settings` as a candidate of the block-scaled autotune: both inputs contracted as they are stored.
fn candidate(
    hw: Hardware,
    settings: CubeStagedMatmulRoutineSettings,
) -> Tunable<MatmulTuneKey, MatmulCall, MatmulProduct, RoutineIdentity> {
    RoutineIdentity::named(
        &format!("block-scaled {settings}"),
        move |call: MatmulCall| {
            reads_no_routes(&call)?;
            let declined = |why| format!("{settings} does not serve this product: {why:?}");
            let (lhs, rhs) = (call.lhs.tensor, call.rhs.tensor);
            let tensors = match &call.destination {
                Some(destination) => MatmulTensors::writing_into::<CubeStagedMicrokernel>(
                    &call.client,
                    lhs,
                    rhs,
                    PlainTensor::of_handle(destination.clone()),
                ),
                None => MatmulTensors::written_in(&call.client, lhs, rhs, call.out),
            }
            .map_err(declined)?;
            let (blueprint, grid) = CubeStagedMatmulRoutine
                .prepare(&hw, &tensors.problem(), &settings)
                .map_err(declined)?;
            CubeStagedMatmulRoutine.dispatch_into_buffer(&tensors, blueprint, grid);
            let handle = match call.destination {
                Some(destination) => destination,
                None => tensors
                    .into_out()
                    .expect("a launch allocating its output hands it back"),
            };
            Ok::<_, String>(MatmulProduct {
                client: call.client,
                handle,
            })
        },
    )
}
