use super::{
    MatmulAutotuneGroups, MatmulTunables, RoutineIdentity, candidate_list, floor, roofline,
};
use crate::kernels::tune::{MatmulPackedCandidate, MatmulRoutineCandidate};
use crate::kernels::{
    CubeStagedMatmulRoutine, HypercubeLastArrivalMatmulRoutine, UnitRegistersMatmulRoutine,
};
use cubek_std::Hardware;

/// The many-rows autotune's candidates on `hw`: the cube staged routine, the last-arrival hypercube
/// routine and the unit routine, in that order; against a packed input the staged routine reading
/// it as stored, in rows or in tiles, decoding it into its stages, the same with a packed
/// activation decoded first
/// ([`launch_activation_decoded`](MatmulPackedCandidate::launch_activation_decoded)), and the
/// last-arrival routine reading it as stored, several cubes sharing each box along `K`.
///
/// The unit routine prepares everywhere, though a round orders the candidates by priority and its
/// small block often tiles a key the others mask. The last-arrival routine sizes its grid by the
/// device's workers and declines where it cannot.
pub(super) fn candidates(hw: Hardware, groups: MatmulAutotuneGroups) -> MatmulTunables {
    let mut autotune = candidate_list();
    let microkernel = autotune.group("microkernel", groups.many_rows);
    autotune
        .bounded(roofline::many_rows)
        .identified(
            &microkernel,
            MatmulRoutineCandidate::list(CubeStagedMatmulRoutine, &hw),
            RoutineIdentity::new,
            MatmulRoutineCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulRoutineCandidate::list(HypercubeLastArrivalMatmulRoutine, &hw),
            RoutineIdentity::new,
            MatmulRoutineCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulRoutineCandidate::list(UnitRegistersMatmulRoutine, &hw),
            RoutineIdentity::new,
            MatmulRoutineCandidate::launch,
        )
        .identified(
            &microkernel,
            MatmulPackedCandidate::of(
                CubeStagedMatmulRoutine,
                CubeStagedMatmulRoutine.packed_settings(&hw),
                &hw,
            ),
            RoutineIdentity::new,
            MatmulPackedCandidate::launch,
        )
        // The same, against a packed activation decoded first.
        .identified(
            &microkernel,
            MatmulPackedCandidate::of(
                CubeStagedMatmulRoutine,
                CubeStagedMatmulRoutine.packed_settings(&hw),
                &hw,
            ),
            MatmulPackedCandidate::activation_decoded_identity,
            MatmulPackedCandidate::launch_activation_decoded,
        )
        // The packed boxes again, several cubes sharing each along `K`: what a product with too
        // few boxes to fill the device and a long `K` wants.
        .identified(
            &microkernel,
            MatmulPackedCandidate::of(
                HypercubeLastArrivalMatmulRoutine,
                HypercubeLastArrivalMatmulRoutine.packed_settings(&hw),
                &hw,
            ),
            RoutineIdentity::new,
            MatmulPackedCandidate::launch,
        )
        .floor(floor::decoded(&hw))
        .finish()
}
