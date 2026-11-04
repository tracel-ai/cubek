use cubecl::ir::ElemType;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::gated_delta::walk::cube::{
    CubeGatedDeltaWalkMicrokernel, StateAccess, cube_gated_delta_walk_microkernel,
};

/// Every cube, walking its runs of value columns through every position over states it
/// owns, the state after each position from `first_kept` on kept.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cube_gated_delta_walk_owned_kernel<E: Float, N: Size>(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    decay: &TileTensor<'_>,
    strength: &TileTensor<'_>,
    state: &TileTensor<'_>,
    out: &TileTensor<'_>,
    kept: &TileTensor<'_>,
    first_kept: usize,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeGatedDeltaWalkMicrokernel,
    #[comptime] runs: usize,
    #[define(E)] _element: ElemType,
) {
    let query = query.tile(&partitioning);
    let key = key.tile(&partitioning);
    let value = value.tile(&partitioning);
    let decay = decay.tile(&partitioning);
    let strength = strength.tile(&partitioning);
    let state = state.tile(&partitioning);
    let out = out.tile(&partitioning);
    let kept = kept.tile(&partitioning);
    for cube in &partitioning {
        let access = StateAccess::owned(&state.at(&cube), &kept.at(&cube), first_kept);
        cube_gated_delta_walk_microkernel::<E, N>(
            &cube,
            &query.at(&cube),
            &key.at(&cube),
            &value.at(&cube),
            &decay.at(&cube),
            &strength.at(&cube),
            &access,
            &out.at(&cube),
            microkernel,
            runs,
        );
    }
}

/// Every cube, walking its runs of value columns through every position in its sequence's
/// slot of the store, and writing the state after position `first_kept` back there.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cube_gated_delta_walk_slots_kernel<E: Float, N: Size>(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    decay: &TileTensor<'_>,
    strength: &TileTensor<'_>,
    store: &TileTensor<'_>,
    out: &TileTensor<'_>,
    positions: &TileTensor<'_>,
    first_kept: usize,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeGatedDeltaWalkMicrokernel,
    #[comptime] runs: usize,
    #[define(E)] _element: ElemType,
) {
    let query = query.tile(&partitioning);
    let key = key.tile(&partitioning);
    let value = value.tile(&partitioning);
    let decay = decay.tile(&partitioning);
    let strength = strength.tile(&partitioning);
    let store = store.tile(&partitioning);
    let out = out.tile(&partitioning);
    let positions = positions.tile(&partitioning);
    for cube in &partitioning {
        let access = StateAccess::in_slot(&store.at(&cube), &positions.at(&cube), first_kept);
        cube_gated_delta_walk_microkernel::<E, N>(
            &cube,
            &query.at(&cube),
            &key.at(&cube),
            &value.at(&cube),
            &decay.at(&cube),
            &strength.at(&cube),
            &access,
            &out.at(&cube),
            microkernel,
            runs,
        );
    }
}
