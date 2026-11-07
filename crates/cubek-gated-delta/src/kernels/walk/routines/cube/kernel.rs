use cubecl::ir::ElemType;
use cubecl::tile::{Partitioning, Tile, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::gated_delta::walk::cube::{
    CubeGatedDeltaWalkMicrokernel, StateAccess, cube_gated_delta_walk_microkernel,
};

/// Every cube, walking its runs of value columns through every position over `state`. The states
/// from position `first_kept` on go to `kept` where the launch binds it and back to `state`
/// otherwise; `lengths`, where bound, says which rows start from zeros.
#[cube(launch_unchecked)]
pub fn cube_gated_delta_walk_kernel<E: Float, N: Size>(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    decay: &TileTensor<'_>,
    strength: &TileTensor<'_>,
    state: &TileTensor<'_>,
    out: &TileTensor<'_>,
    kept: ComptimeOption<&TileTensor<'static>>,
    lengths: ComptimeOption<&TileTensor<'static>>,
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
    let kept: ComptimeOption<Tile> = kept.map(|kept| kept.tile(&partitioning));
    let lengths: ComptimeOption<Tile> = lengths.map(|lengths| lengths.tile(&partitioning));
    for cube in &partitioning {
        let access = StateAccess::new(
            &state.at(&cube),
            &kept.clone().map(|kept| kept.at(&cube)),
            &lengths.clone().map(|lengths| lengths.at(&cube)),
            first_kept,
        );
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
