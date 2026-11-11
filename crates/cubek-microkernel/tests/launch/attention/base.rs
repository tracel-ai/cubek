//! What every attention microkernel launched alone shares: one batch, one head, every key
//! attended, no mask; `out = softmax(scale · query · keyᵀ) · value` and each query's logsumexp;
//! the operands bound one value a line or as wide as a routine binds them.

use core::fmt::Debug;
use cubecl::prelude::*;
use cubecl::tile::Space;
use cubecl::tile::launch::{Refusal, Vectorization};
use cubek_microkernel::attention::{
    AttentionOperand, AttentionOperands, AttentionReading, AttentionWorkItem,
};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

use crate::launch::Lines;

/// The elements of a float attention: what the query, keys and values are stored in, what it
/// sums in and what it writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionElements {
    pub input: ElemType,
    pub sum: ElemType,
    pub out: ElemType,
}

impl AttentionElements {
    /// The element each operand is stored in.
    pub fn stored(&self) -> AttentionOperands<ElemType> {
        AttentionOperands {
            query: self.input,
            key: self.input,
            value: self.input,
            out: self.out,
        }
    }

    /// One head of one group read as stored, with rows of `head_dim` in the query and key and of
    /// `val_dim` in the value.
    pub fn work_item(&self, [head_dim, val_dim]: [usize; 2]) -> AttentionWorkItem {
        AttentionWorkItem {
            elements: self.stored(),
            sum: self.sum,
            reading: AttentionReading::AsStored,
            page_size: None,
            group: 1,
            head_dim,
            val_dim,
        }
    }
}

/// How a launch binds the operands of an attention: as `lines` says, each under the widest line
/// the device loads its stored element in.
#[derive(Debug, Clone, Copy)]
pub struct AttentionLines {
    lines: Lines,
    ceilings: AttentionOperands<usize>,
}

impl AttentionLines {
    pub fn new(hw: &Hardware, lines: Lines, item: &AttentionWorkItem) -> Self {
        Self {
            lines,
            ceilings: item.elements.map(|_, element| hw.vector_ceiling(element)),
        }
    }

    /// The width each operand binds at, `microkernel` stating its accesses over a cube's box
    /// `cube_space`, as a routine resolves them.
    pub fn widths<M: Microkernel<Operand = AttentionOperand, CubeSpace = Space>>(
        &self,
        microkernel: &M,
        cube_space: &Space,
    ) -> AttentionOperands<usize> {
        let mut vectorization = Vectorization::new();
        microkernel.vectorize(cube_space, &mut vectorization);
        self.ceilings.map(|operand, _| {
            self.lines
                .width(operand, &vectorization, |member| self.ceilings.get(member))
        })
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
