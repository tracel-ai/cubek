use super::{CandidateList, MatmulAutotuneGroups, MatmulTunables, candidate_list, floor};
use crate::kernels::tune::key::MatmulTuneKey;
use crate::kernels::tune::{MatmulPackedCandidate, RoutineIdentity};
use crate::kernels::{
    CubeStagedAsyncCopyMatmulRoutine, CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings,
    CubeStagedTmaMatmulRoutine, CubeStagedTmaStoredMatmulRoutine,
};
use cubecl::tune::TuneGroup;
use cubek_std::Hardware;
use cubek_std::autotune::GroupPriority;

/// The candidates of a pin of the block-scaled tensor cores on `hw`: every
/// [block-scaled setting](listed), the staged routine's default, then the floor every autotune ends
/// on, the packed inputs decoded first ([`floor::decoded`]).
///
/// Not bounded: a roofline is a 16-bit product's peak, which a block-scaled instruction runs past.
pub(super) fn candidates(hw: Hardware, _groups: MatmulAutotuneGroups) -> MatmulTunables {
    let mut autotune = candidate_list();
    let group = autotune.group("block-scaled", GroupPriority::DEFAULT);
    listed(autotune, &group, &hw)
        .identified(
            &group,
            MatmulPackedCandidate::of(
                CubeStagedMatmulRoutine,
                [CubeStagedMatmulRoutineSettings::DEFAULT],
                &hw,
            ),
            RoutineIdentity::new,
            MatmulPackedCandidate::launch,
        )
        .floor(floor::decoded(&hw))
        .finish()
}

/// The block-scaled settings on `hw` as candidates of `group`: each input contracted as it is
/// stored on the device's block-scaled instruction, the TMA engine's settings, its store's, then
/// the copy engine's. None on a device with no such instruction.
///
/// Every one sums each output in the same order, so a race of them moves the time and no number.
pub(super) fn listed(
    autotune: CandidateList,
    group: &TuneGroup<MatmulTuneKey>,
    hw: &Hardware,
) -> CandidateList {
    autotune
        .identified(
            group,
            MatmulPackedCandidate::of(
                CubeStagedTmaMatmulRoutine,
                CubeStagedTmaMatmulRoutine.block_scaled_settings(hw),
                hw,
            ),
            RoutineIdentity::new,
            MatmulPackedCandidate::launch,
        )
        .identified(
            group,
            MatmulPackedCandidate::of(
                CubeStagedTmaStoredMatmulRoutine,
                CubeStagedTmaStoredMatmulRoutine.block_scaled_settings(hw),
                hw,
            ),
            RoutineIdentity::new,
            MatmulPackedCandidate::launch,
        )
        .identified(
            group,
            MatmulPackedCandidate::of(
                CubeStagedAsyncCopyMatmulRoutine,
                CubeStagedAsyncCopyMatmulRoutine.block_scaled_settings(hw),
                hw,
            ),
            RoutineIdentity::new,
            MatmulPackedCandidate::launch,
        )
}
