use super::{
    MatmulAutotuneGroups, MatmulProduct, MatmulTunables, RoutineIdentity, candidate_list, floor,
};
use crate::kernels::MatmulRoutine;
use crate::kernels::plan::MatmulCall;
use crate::kernels::tensor::PlainTensor;
use crate::kernels::tune::MatmulRoutineSettings;
use crate::kernels::tune::key::MatmulTuneKey;
use crate::kernels::{
    CubeStagedAsyncCopyMatmulRoutine, CubeStagedMatmulRoutine, CubeStagedMatmulRoutineBlueprint,
    CubeStagedMatmulRoutineSettings, CubeStagedTmaMatmulRoutine, CubeStagedTmaStoredMatmulRoutine,
    MatmulRoutineProblem, MatmulTensors,
};
use cubecl::tile::StageStorage;
use cubecl::tune::Tunable;
use cubek_microkernel::matmul::cube::CubeStagedMicrokernel;
use cubek_std::Hardware;
use cubek_std::autotune::{CandidateIdentity, GroupPriority};
use cubek_std::routine::Routine;

/// The block-scaled autotune's candidates on `hw`: the staged routines reading both inputs as
/// stored on the block-scaled tensor cores, at each of their block-scaled settings (the TMA
/// engine's, its store's, the copy engine's), and the units' default, then the floor every
/// autotune ends on, the packed inputs decoded first ([`floor::decoded`]): autotune sends a
/// product here by its formats and its `k`, which no candidate's own binding is asked about
/// first, so a product none of them binds — batched, say — is still answered.
///
/// Every candidate sums each output in the same order, so the race moves the time and no number.
/// Not bounded: a roofline is a 16-bit product's peak, which a block-scaled instruction runs past,
/// so a bound would end every round at its first candidate. Its one group is its own: the autotune
/// races one kernel's settings.
pub(super) fn candidates(hw: Hardware, _groups: MatmulAutotuneGroups) -> MatmulTunables {
    let mut autotune = candidate_list();
    let group = autotune.group("block-scaled", GroupPriority::DEFAULT);
    let tma = CubeStagedTmaMatmulRoutine.block_scaled_settings(&hw);
    let tma_stored = CubeStagedTmaStoredMatmulRoutine.block_scaled_settings(&hw);
    let async_copy = CubeStagedAsyncCopyMatmulRoutine.block_scaled_settings(&hw);
    let autotune = tma.into_iter().fold(autotune, |autotune, settings| {
        autotune.candidate(
            &group,
            candidate(hw.clone(), CubeStagedTmaMatmulRoutine, settings),
        )
    });
    let autotune = tma_stored.into_iter().fold(autotune, |autotune, settings| {
        autotune.candidate(
            &group,
            candidate(hw.clone(), CubeStagedTmaStoredMatmulRoutine, settings),
        )
    });
    let autotune = async_copy.into_iter().fold(autotune, |autotune, settings| {
        autotune.candidate(
            &group,
            candidate(hw.clone(), CubeStagedAsyncCopyMatmulRoutine, settings),
        )
    });
    autotune
        .candidate(
            &group,
            candidate(
                hw.clone(),
                CubeStagedMatmulRoutine,
                CubeStagedMatmulRoutineSettings::DEFAULT,
            ),
        )
        .floor(floor::decoded(&hw))
        .finish()
}

/// `routine` at `settings` as a candidate of the block-scaled autotune: both inputs contracted as
/// they are stored. Identified by its routine and setting, so a caller storing the product through
/// a sink of its own launches the winner again
/// ([`fastest_routine`](super::MatmulAutotune::fastest_routine)).
fn candidate<R>(
    hw: Hardware,
    routine: R,
    settings: R::Settings,
) -> Tunable<MatmulTuneKey, MatmulCall, MatmulProduct, RoutineIdentity>
where
    R: MatmulRoutine<CubeStagedMicrokernel, StageStorage>
        + Routine<
            Problem = MatmulRoutineProblem,
            Microkernel = CubeStagedMicrokernel,
            Blueprint = CubeStagedMatmulRoutineBlueprint,
        > + Copy
        + Send
        + Sync
        + 'static,
    R::Settings: Copy + Send + Sync + Into<MatmulRoutineSettings> + 'static,
{
    Tunable::identified(
        CandidateIdentity::Routine(settings.into()),
        move |call: MatmulCall| {
            let declined = |why| format!("{settings} does not serve this product: {why:?}");
            let (lhs, rhs) = (call.lhs.tensor, call.rhs.tensor);
            let tensors = match &call.destination {
                Some(destination) => MatmulTensors::writing_into(
                    &call.client,
                    lhs,
                    rhs,
                    PlainTensor::of_handle(destination.clone()),
                    R::ROWS_KIND,
                ),
                None => MatmulTensors::written_in(&call.client, lhs, rhs, call.out),
            }
            .map_err(declined)?;
            let (blueprint, grid) = routine
                .prepare(&hw, &tensors.problem(), &settings)
                .map_err(declined)?;
            routine.dispatch_into_buffer(&tensors, blueprint, grid);
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
