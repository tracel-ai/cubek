use super::{
    MatmulAutotuneGroups, MatmulTunables, RoutineIdentity, candidate_list, floor, roofline,
};
use crate::kernels::tune::MatmulRoutineSettings;
use crate::kernels::tune::{MatmulPackedCandidate, MatmulRoutineCandidate};
use crate::kernels::{
    CubeShuffleMatmulRoutine, CubeShuffleMatmulRoutineSettings, CubeStagedMatmulRoutine,
    CubeStagedMatmulRoutineSettings, PlaneShuffleMatmulRoutine, PlaneShuffleMatmulRoutineSettings,
    UnitWholeKMatmulRoutine,
};
use cubek_std::Hardware;

/// The settings the few-rows autotune races against a packed input, routine by routine: the plane
/// shuffle's, the cube shuffle's, then the packed staged routine's at a decode's settings. A caller
/// that tries settings in order rather than measuring them takes them in this one, so a tie
/// resolves as the autotune's does.
#[derive(Clone, Debug)]
pub struct FewRowsPackedList {
    pub plane: Vec<PlaneShuffleMatmulRoutineSettings>,
    pub cube: Vec<CubeShuffleMatmulRoutineSettings>,
    pub staged: Vec<CubeStagedMatmulRoutineSettings>,
}

impl FewRowsPackedList {
    /// The list on `hw`.
    pub fn new(hw: &Hardware) -> Self {
        Self {
            plane: PlaneShuffleMatmulRoutine.packed_settings(),
            cube: CubeShuffleMatmulRoutine.packed_settings(hw),
            staged: CubeStagedMatmulRoutine.packed_one_step_settings(),
        }
    }

    /// Every setting of the list, in its order.
    pub fn settings(self) -> impl Iterator<Item = MatmulRoutineSettings> {
        let plane = self
            .plane
            .into_iter()
            .map(MatmulRoutineSettings::PlaneShuffle);
        let cube = self
            .cube
            .into_iter()
            .map(MatmulRoutineSettings::CubeShuffle);
        let staged = self
            .staged
            .into_iter()
            .map(MatmulRoutineSettings::CubeStaged);
        plane.chain(cube).chain(staged)
    }
}

/// The few-rows autotune's candidates on `hw`. Against float inputs the plane and cube shuffles and
/// the unit whole-K routine at their own settings; against a packed input the shuffles at the
/// settings worth trying against it, reading it as stored, and the packed staged routine at a
/// decode's settings; against a packed activation, the same again with the activation decoded first
/// ([`launch_activation_decoded`](MatmulPackedCandidate::launch_activation_decoded)).
///
/// Keyed by the power of two at or above the call's rows, as the many-rows autotune is: a routine
/// that does not stage holds that many rows whatever the call's count inside it, so it serves or
/// declines every count of a key alike, and the one kernel it compiles serves them all.
pub(super) fn candidates(hw: Hardware, groups: MatmulAutotuneGroups) -> MatmulTunables {
    let mut autotune = candidate_list();
    let microkernel = autotune.group("microkernel", groups.few_rows);
    let packed = FewRowsPackedList::new(&hw);
    autotune
        .bounded(roofline::few_rows)
        .identified(
            &microkernel,
            MatmulRoutineCandidate::list(PlaneShuffleMatmulRoutine, &hw),
            RoutineIdentity::new,
            MatmulRoutineCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulRoutineCandidate::list(CubeShuffleMatmulRoutine, &hw),
            RoutineIdentity::new,
            MatmulRoutineCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulRoutineCandidate::list(UnitWholeKMatmulRoutine, &hw),
            RoutineIdentity::new,
            MatmulRoutineCandidate::launch,
        )
        // Against a packed input, read as stored, in rows or in its tile order. The plane routine
        // leads, so a tie resolves to what a pin runs.
        .identified(
            &microkernel,
            MatmulPackedCandidate::of(PlaneShuffleMatmulRoutine, packed.plane.clone(), &hw),
            RoutineIdentity::new,
            MatmulPackedCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulPackedCandidate::of(CubeShuffleMatmulRoutine, packed.cube.clone(), &hw),
            RoutineIdentity::new,
            MatmulPackedCandidate::launch,
        )
        // The packed staged routine at a decode's settings: each packed input decoded into its
        // stages and contracted on the tensor cores, which the shuffles' software sums cannot keep
        // up with past a few rows.
        .identified(
            &microkernel,
            MatmulPackedCandidate::of(CubeStagedMatmulRoutine, packed.staged.clone(), &hw),
            RoutineIdentity::new,
            MatmulPackedCandidate::launch,
        )
        // The same, against a packed activation decoded first.
        .identified(
            &microkernel,
            MatmulPackedCandidate::of(PlaneShuffleMatmulRoutine, packed.plane, &hw),
            MatmulPackedCandidate::activation_decoded_identity,
            MatmulPackedCandidate::launch_activation_decoded,
        )
        .identified(
            &microkernel,
            MatmulPackedCandidate::of(CubeShuffleMatmulRoutine, packed.cube, &hw),
            MatmulPackedCandidate::activation_decoded_identity,
            MatmulPackedCandidate::launch_activation_decoded,
        )
        .identified(
            &microkernel,
            MatmulPackedCandidate::of(CubeStagedMatmulRoutine, packed.staged, &hw),
            MatmulPackedCandidate::activation_decoded_identity,
            MatmulPackedCandidate::launch_activation_decoded,
        )
        .floor(floor::decoded(&hw))
        .finish()
}
