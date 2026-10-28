use super::base::{
    MatmulTableGroups, MatmulTunables, RoutineIdentity, decoded_floor, few_rows_roofline,
    measured_table, microkernel_candidate, packed_candidate,
};
use crate::kernels::tune::{MatmulPackedCandidate, MatmulRoutineCandidate};
use crate::kernels::{
    CubeShuffleMatmulRoutine, CubeStagedMatmulRoutine, PlaneShuffleMatmulRoutine,
    UnitWholeKMatmulRoutine,
};
use cubek_std::Hardware;

/// The few-rows table's candidates on `hw`. Against float inputs the plane and cube shuffles and
/// the unit whole-K routine at their own settings; against a packed input the shuffles at the
/// settings worth trying against it, reading it as stored, and the packed staged routine at a
/// decode's settings.
///
/// Keyed by the power of two at or above the call's rows, as the many-rows table is: a routine
/// that does not stage holds that many rows whatever the call's count inside it, so it serves or
/// declines every count of a key alike, and the one kernel it compiles serves them all.
pub(super) fn table(hw: Hardware, groups: MatmulTableGroups) -> MatmulTunables {
    let mut table = measured_table();
    let microkernel = table.group("microkernel", groups.microkernel);
    table
        .bounded(few_rows_roofline)
        .identified(
            &microkernel,
            MatmulRoutineCandidate::list(PlaneShuffleMatmulRoutine, &hw),
            RoutineIdentity::new,
            microkernel_candidate,
        )
        .identified(
            &microkernel,
            MatmulRoutineCandidate::list(CubeShuffleMatmulRoutine, &hw),
            RoutineIdentity::new,
            microkernel_candidate,
        )
        .identified(
            &microkernel,
            MatmulRoutineCandidate::list(UnitWholeKMatmulRoutine, &hw),
            RoutineIdentity::new,
            microkernel_candidate,
        )
        // Against a packed input, read as stored, in rows or in its tile order. The plane routine
        // leads, so a tie resolves to what a pin runs.
        .identified(
            &microkernel,
            MatmulPackedCandidate::of(
                PlaneShuffleMatmulRoutine,
                PlaneShuffleMatmulRoutine.packed_settings(),
                &hw,
            ),
            RoutineIdentity::new,
            packed_candidate,
        )
        .identified(
            &microkernel,
            MatmulPackedCandidate::of(
                CubeShuffleMatmulRoutine,
                CubeShuffleMatmulRoutine.packed_settings(&hw),
                &hw,
            ),
            RoutineIdentity::new,
            packed_candidate,
        )
        // The packed staged routine at a decode's settings: each packed input decoded into its
        // stages and contracted on the tensor cores, which the shuffles' software sums cannot keep
        // up with past a few rows.
        .identified(
            &microkernel,
            MatmulPackedCandidate::of(
                CubeStagedMatmulRoutine,
                CubeStagedMatmulRoutine.packed_one_step_settings(),
                &hw,
            ),
            RoutineIdentity::new,
            packed_candidate,
        )
        .floor(decoded_floor(&hw))
        .finish()
}
