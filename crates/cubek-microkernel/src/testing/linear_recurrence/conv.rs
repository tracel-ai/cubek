//! The q|k-normed conv microkernel on its work item, over a stream: one sequence of `positions`,
//! its `q|k|v` stream `[1, positions + taps - 1, conv_dim]` led by `taps - 1` context columns, each
//! channel convolved causally with its filter, then the silu; every key head's queries and keys
//! scaled by their L2 norm, the queries also by the root of their width. Every cube one key head
//! of one position, its channels a line a unit.

use crate::linear_recurrence::conv::QkNormedConvOperand;
use crate::linear_recurrence::conv::cube::{
    CubeQkNormedConvMicrokernel, QkNormedConvSettings, QkNormedConvTiles, QkNormedConvWorkItem,
    cube_qk_normed_conv_microkernel,
};
use crate::linear_recurrence::{
    BATCH, CHANNEL, HEAD_CHANNEL, HEAD_IN_GROUP, KEY_FEATURE, KEY_HEAD, POSITION,
    QkNormedConvPartitioning, RecurrenceHeads, STREAM_POSITION, TAP, VALUE_FEATURE,
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

/// The element the conv writes its queries, keys and values in.
const F32: ElemType = ElemType::Float(FloatKind::F32);

/// One sequence's stream: filters of `taps` over `positions` positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvStream {
    pub taps: usize,
    pub positions: usize,
}

/// A conv microkernel ready to launch over one sequence's stream.
#[derive(Debug, Clone, Copy)]
pub struct ConvLaunch {
    pub microkernel: CubeQkNormedConvMicrokernel,
    pub heads: RecurrenceHeads,
    pub taps: usize,
    pub positions: usize,
}

impl ConvLaunch {
    /// The conv microkernel `hw` runs over `heads` and `stream`, its units capped at the head's
    /// channels, one a unit.
    ///
    /// # Errors
    ///
    /// Why the device runs none.
    pub fn new(hw: &Hardware, heads: RecurrenceHeads, stream: ConvStream) -> Result<Self, Refusal> {
        let ConvStream { taps, positions } = stream;
        let item = QkNormedConvWorkItem {
            heads,
            taps,
            channels: &QkNormedConvOperand::STREAM_CHANNELS,
        };
        let best = CubeQkNormedConvMicrokernel::best(hw, &item, &QkNormedConvSettings)?;
        let microkernel = CubeQkNormedConvMicrokernel {
            units: best.units.min(heads.channels_per_key_head()),
            ..best
        };
        Ok(Self {
            microkernel,
            heads,
            taps,
            positions,
        })
    }

    /// The stream's length: the positions and the context before them.
    pub fn stream_len(&self) -> usize {
        self.positions + self.taps - 1
    }

    /// `qkv` `[1, stream, conv_dim]` and `filters` `[conv_dim, taps]` of `element` convolved into
    /// `query` and `key` `[1, t, key_heads, key_features]` and `value`
    /// `[1, t, key_heads, heads_per_group, value_features]` in `f32`, all row-major.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot bind the launch.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        client: &Client,
        qkv: TensorArg,
        filters: TensorArg,
        query: TensorArg,
        key: TensorArg,
        value: TensorArg,
        element: ElemType,
        epsilon: f32,
    ) -> Result<(), Refusal> {
        let heads = self.heads;
        let space = Space::new(&[
            (BATCH, 1),
            (KEY_HEAD, heads.key_heads),
            (HEAD_IN_GROUP, heads.heads_per_group),
            (KEY_FEATURE, heads.key_features),
            (VALUE_FEATURE, heads.value_features),
            (CHANNEL, heads.conv_dim()),
            (HEAD_CHANNEL, heads.channels_per_key_head()),
            (TAP, self.taps),
            (POSITION, self.positions),
            (STREAM_POSITION, self.stream_len()),
        ]);
        let lines = heads.channels_per_key_head();
        let partitioning = QkNormedConvPartitioning::new(
            space.clone(),
            Levels::leaf(&[
                (POSITION, 1),
                (KEY_HEAD, 1),
                (HEAD_IN_GROUP, heads.heads_per_group),
                (KEY_FEATURE, heads.key_features),
                (VALUE_FEATURE, heads.value_features),
                (CHANNEL, heads.conv_dim()),
                (HEAD_CHANNEL, 1),
                (TAP, self.taps),
            ])
            .units_distributed(HEAD_CHANNEL, lines),
        )
        .as_cubek()
        .clone();
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(self.microkernel.units as u32),
        };
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let bound = |arg, element, axes: &[Axis]| {
            TileTensorLaunch::new(arg, element, 1, TileSpec::direct(axes))
        };
        let keyed = [BATCH, POSITION, KEY_HEAD, KEY_FEATURE];
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles; its shared memory is the head's
        // convolved queries and keys.
        unsafe {
            cube_qk_normed_conv_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                1,
                bound(qkv, element, &[BATCH, STREAM_POSITION, CHANNEL]),
                bound(filters, element, &[CHANNEL, TAP]),
                bound(query, F32, &keyed),
                bound(key, F32, &keyed),
                bound(
                    value,
                    F32,
                    &[BATCH, POSITION, KEY_HEAD, HEAD_IN_GROUP, VALUE_FEATURE],
                ),
                epsilon,
                launcher.partitioning_arg(),
                self.microkernel,
                element,
            );
        }
        Ok(())
    }
}

/// Every cube convolving one key head of one position and normalizing its queries and keys.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn cube_qk_normed_conv_kernel<E: Float, N: Size>(
    qkv: &TileTensor<'_>,
    filters: &TileTensor<'_>,
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    epsilon: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeQkNormedConvMicrokernel,
    #[define(E)] _element: ElemType,
) {
    let tiles = QkNormedConvTiles {
        qkv: qkv.tile(&partitioning),
        filters: filters.tile(&partitioning),
        window: ComptimeOption::new_None(),
        lengths: ComptimeOption::new_None(),
        query: query.tile(&partitioning),
        key: key.tile(&partitioning),
        value: value.tile(&partitioning),
        columns: 0usize,
    };
    for cube in &partitioning {
        let mut tiles = tiles.at(&cube);
        cube_qk_normed_conv_microkernel::<E, N>(&cube, &mut tiles, epsilon, microkernel);
    }
}
