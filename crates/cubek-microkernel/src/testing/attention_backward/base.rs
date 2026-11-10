//! What every microkernel of attention's backward launched alone shares: one batch, one key head
//! of `group` query heads, every key attended, and every operand as a launch argument.

use crate::attention::{B, D, DV, G, KV, Q, S};
use crate::attention_backward::{AttentionBackwardOperand, AttentionBackwardWorkItem};
use core::fmt::Debug;
use cubecl::prelude::*;
use cubecl::tile::launch::{Grid, Refusal};
use cubecl::tile::{Launcher, Level, Partitioning, Space, TileSpec, TileTensorLaunch};

/// Every operand of attention's backward as a launch argument, each laid out as
/// [`AttentionBackwardOperand::axes`](crate::attention_backward::AttentionBackwardOperand::axes)
/// says: the forward's query, keys, values, output and logsumexp, the output's gradient, the
/// dot products of the output with it, and the three gradients. A launch reads and writes the
/// ones its microkernel does.
pub struct AttentionBackwardOperandArgs {
    pub query: TensorArg,
    pub key: TensorArg,
    pub value: TensorArg,
    pub out: TensorArg,
    pub out_gradient: TensorArg,
    pub logsumexp: TensorArg,
    pub delta: TensorArg,
    pub query_gradient: TensorArg,
    pub key_gradient: TensorArg,
    pub value_gradient: TensorArg,
}

/// Which gradients a microkernel writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionBackwardWrites {
    /// The queries' gradient, and the dot products it reads beside it.
    Queries,
    /// The keys' and the values' gradients.
    Keys,
}

/// A gradient microkernel of attention's backward ready to launch alone over one sequence.
pub trait AttentionBackwardLaunch: Debug {
    /// What it attends.
    fn item(&self) -> &AttentionBackwardWorkItem;

    /// The queries and the keys.
    fn seq(&self) -> [usize; 2];

    /// The gradients it writes.
    fn writes(&self) -> AttentionBackwardWrites;

    /// Its gradients from `args` at `scale`.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot bind the launch.
    fn launch(
        &self,
        client: &Client,
        args: AttentionBackwardOperandArgs,
        scale: f32,
    ) -> Result<(), Refusal>;
}

/// A launch over one sequence of `seq` queries and keys of one head.
pub(crate) struct OneSequence {
    pub item: AttentionBackwardWorkItem,
    pub seq: [usize; 2],
}

impl OneSequence {
    /// The launcher of `levels` over the sequence, its planes `plane_width` units wide.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot place it.
    pub(crate) fn launcher(
        &self,
        client: &Client,
        levels: Vec<Level>,
        plane_width: u32,
    ) -> Result<Launcher, Refusal> {
        let [queries, keys] = self.seq;
        let space = Space::new(&[
            (B, 1),
            (KV, 1),
            (G, self.item.group),
            (Q, queries),
            (S, keys),
            (D, self.item.head_dim),
            (DV, self.item.val_dim),
        ]);
        let partitioning = Partitioning::new(space.clone(), levels);
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(plane_width),
        };
        Launcher::new(client, partitioning, &space, grid)
    }

    /// `arg` bound along `operand`'s axes, one element a line.
    pub(crate) fn bound(
        &self,
        arg: TensorArg,
        operand: AttentionBackwardOperand,
    ) -> TileTensorLaunch {
        use AttentionBackwardOperand as Operand;
        let element = match operand {
            Operand::Logsumexp | Operand::Delta => self.item.sum,
            _ => self.item.element,
        };
        TileTensorLaunch::new(arg, element, 1, TileSpec::direct(operand.axes()))
    }
}
