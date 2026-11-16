//! The walk microkernel on its work item: one sequence of `positions` over every key head and its
//! value heads, each value head's `[key_features, value_features]` state decayed, written under
//! the update rule and read by the query at every position. Every cube one run of a value head's
//! columns, its units each rows of the state; the state is read from `state` and the last
//! position's written back to it.

use cubecl::ir::FloatKind;
use cubecl::prelude::*;
use cubecl::tile::launch::{Grid, Refusal};
use cubecl::tile::{
    Axis, Launcher, Partitioning, Space, Tile, TileSpec, TileTensor, TileTensorLaunch,
};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::linear_recurrence::walk::cube::{
    CubeWalkMicrokernel, CubeWalkMicrokernelSettings, StateAccess, cube_walk_microkernel,
};
use cubek_microkernel::linear_recurrence::walk::{RecurrenceWalkItem, UpdateRule};
use cubek_microkernel::linear_recurrence::{
    BATCH, HEAD_IN_GROUP, KEY_FEATURE, KEY_HEAD, POSITION, RATE, RecurrenceHeads,
    RecurrenceWalkPartitioning, VALUE_FEATURE,
};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

/// The element every operand but the output is stored in.
const F32: ElemType = ElemType::Float(FloatKind::F32);

/// A walk microkernel ready to launch over one sequence.
#[derive(Debug, Clone, Copy)]
pub struct WalkLaunch {
    pub microkernel: CubeWalkMicrokernel,
    pub heads: RecurrenceHeads,
    pub positions: usize,
}

impl WalkLaunch {
    /// The walk microkernel `hw` runs under `rule` at `settings` over `heads`, `positions` long.
    ///
    /// # Errors
    ///
    /// Why the device runs none.
    pub fn new(
        hw: &Hardware,
        heads: RecurrenceHeads,
        rule: UpdateRule,
        positions: usize,
        settings: &CubeWalkMicrokernelSettings,
    ) -> Result<Self, Refusal> {
        let microkernel =
            CubeWalkMicrokernel::best(hw, &RecurrenceWalkItem { heads, rule }, settings)?;
        Ok(Self {
            microkernel,
            heads,
            positions,
        })
    }

    /// The walk over `query` and `key` `[1, t, key_heads, key_features]`, `value`
    /// `[1, t, key_heads, heads_per_group, value_features]`, `rates` the log of the decay and the
    /// strength `[1, t, key_heads, heads_per_group, 2]`, the additive rule's `skip`
    /// `[key_heads, heads_per_group]` and `state`
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
        rates: TensorArg,
        skip: Option<TensorArg>,
        state: TensorArg,
        out: TensorArg,
        element: ElemType,
    ) -> Result<(), Refusal> {
        let heads = self.heads;
        let microkernel = self.microkernel;
        let space = Space::new(&[
            (BATCH, 1),
            (POSITION, self.positions),
            (RATE, 2),
            (KEY_HEAD, heads.key_heads),
            (HEAD_IN_GROUP, heads.heads_per_group),
            (KEY_FEATURE, heads.key_features),
            (VALUE_FEATURE, heads.value_features),
        ]);
        let partitioning = RecurrenceWalkPartitioning::new(
            space.clone(),
            microkernel.tiling(None).planes(&[(VALUE_FEATURE, 1)]),
        )
        .as_cubek()
        .clone();
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(microkernel.shape.units() as u32),
        };
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let bound = |arg, element, axes: &[Axis]| {
            TileTensorLaunch::new(arg, element, 1, TileSpec::direct(axes))
        };
        let keyed = [BATCH, POSITION, KEY_HEAD, KEY_FEATURE];
        let valued = [BATCH, POSITION, KEY_HEAD, HEAD_IN_GROUP, VALUE_FEATURE];
        let rated = [BATCH, POSITION, KEY_HEAD, HEAD_IN_GROUP, RATE];
        let skipped = [KEY_HEAD, HEAD_IN_GROUP];
        let held = [BATCH, KEY_HEAD, HEAD_IN_GROUP, KEY_FEATURE, VALUE_FEATURE];
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles; the shared memory it holds is its
        // own run's partials.
        unsafe {
            cube_walk_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                1,
                bound(query, F32, &keyed),
                bound(key, F32, &keyed),
                bound(value, F32, &valued),
                bound(rates, F32, &rated),
                bound(state, F32, &held),
                bound(out, element, &valued),
                skip.map(|skip| bound(skip, F32, &skipped)).into(),
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
/// position's state written back to it; the additive rule's `skip` bound where there is one.
#[cube(launch_unchecked)]
fn cube_walk_kernel<E: Float, N: Size>(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    rates: &TileTensor<'_>,
    state: &TileTensor<'_>,
    out: &TileTensor<'_>,
    skip: ComptimeOption<&TileTensor<'static>>,
    last: usize,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeWalkMicrokernel,
    #[define(E)] _element: ElemType,
) {
    let query = query.tile(&partitioning);
    let key = key.tile(&partitioning);
    let value = value.tile(&partitioning);
    let rates = rates.tile(&partitioning);
    let state = state.tile(&partitioning);
    let out = out.tile(&partitioning);
    let skip: ComptimeOption<Tile> = skip.map(|skip| skip.tile(&partitioning));
    for cube in &partitioning {
        let access = StateAccess::new(
            &state.at(&cube),
            &ComptimeOption::new_None(),
            &ComptimeOption::new_None(),
            last,
        );
        cube_walk_microkernel::<E, N>(
            &cube,
            &query.at(&cube),
            &key.at(&cube),
            &value.at(&cube),
            &rates.at(&cube),
            &skip.clone().map(|skip| skip.at(&cube)),
            &access,
            &out.at(&cube),
            microkernel,
            1usize,
        );
    }
}
