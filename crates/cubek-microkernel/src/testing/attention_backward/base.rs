//! What every gradient microkernel of attention's backward launched alone shares: one batch, one
//! key head of `group` query heads, every key attended, and every operand as a launch argument.

use crate::attention_backward::AttentionBackwardWorkItem;
use core::fmt::Debug;
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;

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
