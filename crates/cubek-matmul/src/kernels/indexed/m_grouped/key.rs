//! The key a caller that measures the grouped product keys a call by ([`MGroupedMatmulKey`]).
//!
//! The product is `out[r] = lhs[r] @ stacked[group(r)]`: any rows grouped by the slab they read
//! are one such product, not only an expert layer's. Its gradient for the stack is the K-grouped
//! product, [`k_grouped`](crate::kernels::indexed::k_grouped).

use crate::kernels::OperandLayout;
use crate::kernels::indexed::m_grouped::MGroupedRowSourceKind;
use crate::kernels::indexed::stack::SlabDimension;
use crate::kernels::select::problem::OperandKind;
use cubecl::ir::ElemType;
use cubecl::{self as cubecl, AutotuneKey};
use serde::{Deserialize, Serialize};

/// What makes two grouped products want the same launch: the stack's shape and layout, as
/// the routed key states them, and the rows, whose count over the groups' is the mean group's
/// size, which decides between holding a group's rows against one read of its slab and staging
/// them.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct MGroupedMatmulKey {
    /// The lhs, the stack and the output all bind at this element.
    elem: ElemType,
    /// The contraction.
    k: usize,
    /// One slab's output columns.
    n: usize,
    /// Slabs in the stack, one per group.
    groups: usize,
    /// Grouped rows, anchored to the power of two above them.
    m: usize,
    /// Whether the stack is float or packed, and how: a packed stack is read packed, by other
    /// kernels than a float one.
    stack: OperandKind,
    /// Which dimension the stack lines along.
    lines: OperandLayout,
    /// Which dimension the slabs lie along.
    slabs: SlabDimension,
    /// Where the rows come from: a candidate is timed with the pass that pads them.
    source: MGroupedRowSourceKind,
}
