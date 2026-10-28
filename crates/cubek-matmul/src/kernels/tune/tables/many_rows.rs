use super::base::{
    MatmulTableGroups, MatmulTunables, RoutineIdentity, decoded_floor, many_rows_roofline,
    measured_table, microkernel_candidate, packed_candidate,
};
use crate::kernels::tune::{MatmulPackedCandidate, MatmulRoutineCandidate};
use crate::kernels::{
    CubeStagedMatmulRoutine, HypercubeLastArrivalMatmulRoutine, UnitRegistersMatmulRoutine,
};
use cubek_std::Hardware;

/// The many-rows table's candidates on `hw`: the cube staged routine, the last-arrival hypercube
/// routine and the unit routine, in that order; against a packed input the staged routine reading
/// it as stored, in rows or in tiles, decoding it into its stages.
///
/// The unit routine prepares everywhere, though a round orders the candidates by priority and its
/// small block often tiles a key the others mask. The last-arrival routine sizes its grid by the
/// device's workers and declines where it cannot. The relay hypercube routine is on no table: its
/// cubes wait on each other.
pub(super) fn table(hw: Hardware, groups: MatmulTableGroups) -> MatmulTunables {
    let mut table = measured_table();
    let microkernel = table.group("microkernel", groups.microkernel);
    table
        .bounded(many_rows_roofline)
        .identified(
            &microkernel,
            MatmulRoutineCandidate::list(CubeStagedMatmulRoutine, &hw),
            RoutineIdentity::new,
            microkernel_candidate,
        )
        .identified(
            &microkernel,
            MatmulRoutineCandidate::list(HypercubeLastArrivalMatmulRoutine, &hw),
            RoutineIdentity::new,
            microkernel_candidate,
        )
        .identified(
            &microkernel,
            MatmulRoutineCandidate::list(UnitRegistersMatmulRoutine, &hw),
            RoutineIdentity::new,
            microkernel_candidate,
        )
        .identified(
            &microkernel,
            MatmulPackedCandidate::of(
                CubeStagedMatmulRoutine,
                CubeStagedMatmulRoutine.packed_settings(&hw),
                &hw,
            ),
            RoutineIdentity::new,
            packed_candidate,
        )
        .floor(decoded_floor(&hw))
        .finish()
}
