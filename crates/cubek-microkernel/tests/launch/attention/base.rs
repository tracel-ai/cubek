//! What every attention microkernel launched alone shares: one batch, one head, every key
//! attended, no mask; `out = softmax(scale · query · keyᵀ) · value` and each query's logsumexp.

use core::fmt::Debug;
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::{AttentionOperands, AttentionReading, AttentionWorkItem};

/// The elements of a float attention: what the query, keys and values are stored in, what it
/// sums in and what it writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionElements {
    pub input: ElemType,
    pub sum: ElemType,
    pub out: ElemType,
}

impl AttentionElements {
    /// One head of one group read as stored, with rows of `head_dim` in the query and key and of
    /// `val_dim` in the value.
    pub fn work_item(&self, [head_dim, val_dim]: [usize; 2]) -> AttentionWorkItem {
        AttentionWorkItem {
            elements: AttentionOperands {
                query: self.input,
                key: self.input,
                value: self.input,
                out: self.out,
            },
            sum: self.sum,
            reading: AttentionReading::AsStored,
            page_size: None,
            group: 1,
            head_dim,
            val_dim,
        }
    }
}

/// An attention microkernel ready to launch alone.
pub trait AttentionLaunch: Debug {
    /// What it attends.
    fn item(&self) -> &AttentionWorkItem;

    /// The queries and the keys.
    fn seq(&self) -> [usize; 2];

    /// `out = softmax(scale · query · keyᵀ) · value`, `query` `[1, 1, 1, queries, head_dim]`,
    /// `key` `[1, 1, keys, head_dim]`, `value` `[1, 1, keys, val_dim]` and `out`
    /// `[1, 1, 1, queries, val_dim]`, all row-major (`[B, KV, G, Q, _]` and `[B, KV, S, _]`);
    /// `lengths` holds the one sequence's attended keys, as `u32`. Where `logsumexp` is given,
    /// `[1, 1, 1, queries]` in the sum's element, each query's `ln Σ exp(score)` lands there.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot bind the launch.
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        client: &Client,
        query: TensorBinding,
        key: TensorBinding,
        value: TensorBinding,
        out: TensorBinding,
        logsumexp: Option<TensorBinding>,
        lengths: TensorArg,
        scale: f32,
    ) -> Result<(), Refusal>;
}
