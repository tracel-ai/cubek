//! The delta walk microkernel on its work item: one sequence of `positions` over every key head
//! and its value heads, each value head's `[key_features, value_features]` state decayed, recalled
//! against the key, corrected toward the value by the strength and read by the query at every
//! position. Every cube one run of a value head's columns, its units each rows of the state; the
//! state is read from `state` and the last position's written back to it.

use crate::linear_recurrence::walk::cube::{
    CubeDeltaWalkMicrokernel, CubeWalkMicrokernelSettings, StateAccess, cube_delta_walk_microkernel,
};
use crate::linear_recurrence::{
    BATCH, HEAD_IN_GROUP, KEY_FEATURE, KEY_HEAD, POSITION, RecurrenceHeads,
    RecurrenceWalkPartitioning, VALUE_FEATURE,
};
use cubecl::ir::FloatKind;
use cubecl::prelude::*;
use cubecl::tile::launch::{Grid, Refusal};
use cubecl::tile::{
    Axis, Launcher, Levels, Partitioning, Space, TileSpec, TileTensor, TileTensorLaunch,
};
use cubecl::{self as cubecl, cube};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

/// The element every operand but the output is stored in.
const F32: ElemType = ElemType::Float(FloatKind::F32);

/// A delta walk microkernel ready to launch over one sequence.
#[derive(Debug, Clone, Copy)]
pub struct WalkLaunch {
    pub microkernel: CubeDeltaWalkMicrokernel,
    pub heads: RecurrenceHeads,
    pub positions: usize,
}

impl WalkLaunch {
    /// The delta walk microkernel `hw` runs at `settings` over `heads`, `positions` long.
    ///
    /// # Errors
    ///
    /// Why the device runs none.
    pub fn new(
        hw: &Hardware,
        heads: RecurrenceHeads,
        positions: usize,
        settings: &CubeWalkMicrokernelSettings,
    ) -> Result<Self, Refusal> {
        let microkernel = CubeDeltaWalkMicrokernel::best(hw, &heads, settings)?;
        Ok(Self {
            microkernel,
            heads,
            positions,
        })
    }

    /// The walk over `query` and `key` `[1, t, key_heads, key_features]`, `value`
    /// `[1, t, key_heads, heads_per_group, value_features]`, `decay` (the log of the factor) and
    /// `strength` `[1, t, key_heads, heads_per_group]` and `state`
    /// `[1, key_heads, heads_per_group, key_features, value_features]`, all `f32`, into `out`, laid
    /// as `value`, of `element`; all row-major.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot bind the launch.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        client: &Client,
        query: TensorArg,
        key: TensorArg,
        value: TensorArg,
        decay: TensorArg,
        strength: TensorArg,
        state: TensorArg,
        out: TensorArg,
        element: ElemType,
    ) -> Result<(), Refusal> {
        let heads = self.heads;
        let microkernel = self.microkernel;
        let space = Space::new(&[
            (BATCH, 1),
            (POSITION, self.positions),
            (KEY_HEAD, heads.key_heads),
            (HEAD_IN_GROUP, heads.heads_per_group),
            (KEY_FEATURE, heads.key_features),
            (VALUE_FEATURE, heads.value_features),
        ]);
        let unit = Levels::leaf(&[
            (POSITION, 1),
            (KEY_HEAD, 1),
            (HEAD_IN_GROUP, 1),
            (KEY_FEATURE, microkernel.rows),
            (VALUE_FEATURE, microkernel.columns),
        ]);
        let partitioning = RecurrenceWalkPartitioning::new(
            space.clone(),
            unit.walk_every(&[POSITION])
                .units(&[(KEY_FEATURE, microkernel.units)])
                .planes(&[(VALUE_FEATURE, 1)]),
        )
        .as_cubek()
        .clone();
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(microkernel.units as u32),
        };
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let bound = |arg, element, axes: &[Axis]| {
            TileTensorLaunch::new(arg, element, 1, TileSpec::direct(axes))
        };
        let keyed = [BATCH, POSITION, KEY_HEAD, KEY_FEATURE];
        let valued = [BATCH, POSITION, KEY_HEAD, HEAD_IN_GROUP, VALUE_FEATURE];
        let scalar = [BATCH, POSITION, KEY_HEAD, HEAD_IN_GROUP];
        let held = [BATCH, KEY_HEAD, HEAD_IN_GROUP, KEY_FEATURE, VALUE_FEATURE];
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles; the shared memory it holds is its
        // own run's partials.
        unsafe {
            cube_delta_walk_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                1,
                bound(query, F32, &keyed),
                bound(key, F32, &keyed),
                bound(value, F32, &valued),
                bound(decay, F32, &scalar),
                bound(strength, F32, &scalar),
                bound(state, F32, &held),
                bound(out, element, &valued),
                self.positions - 1,
                launcher.partitioning_arg(),
                microkernel,
                element,
            );
        }
        Ok(())
    }
}

/// Every cube walking its run of value columns through every position over `state`, the last
/// position's state written back to it.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn cube_delta_walk_kernel<E: Float, N: Size>(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    decay: &TileTensor<'_>,
    strength: &TileTensor<'_>,
    state: &TileTensor<'_>,
    out: &TileTensor<'_>,
    last: usize,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeDeltaWalkMicrokernel,
    #[define(E)] _element: ElemType,
) {
    let query = query.tile(&partitioning);
    let key = key.tile(&partitioning);
    let value = value.tile(&partitioning);
    let decay = decay.tile(&partitioning);
    let strength = strength.tile(&partitioning);
    let state = state.tile(&partitioning);
    let out = out.tile(&partitioning);
    for cube in &partitioning {
        let access = StateAccess::new(
            &state.at(&cube),
            &ComptimeOption::new_None(),
            &ComptimeOption::new_None(),
            last,
        );
        cube_delta_walk_microkernel::<E, N>(
            &cube,
            &query.at(&cube),
            &key.at(&cube),
            &value.at(&cube),
            &decay.at(&cube),
            &strength.at(&cube),
            &access,
            &out.at(&cube),
            microkernel,
            1usize,
        );
    }
}
