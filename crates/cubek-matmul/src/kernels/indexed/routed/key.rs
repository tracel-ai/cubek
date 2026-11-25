//! The key a caller that measures the routed product keys a call by ([`RoutedMatmulKey`]).

use cubecl::ir::ElemType;
use cubecl::{self as cubecl, AutotuneKey};
use serde::{Deserialize, Serialize};

use crate::kernels::OperandLayout;
use crate::kernels::indexed::stack::SlabDimension;
use crate::kernels::select::problem::OperandKind;

/// What makes two routed products want the same launch: the stack's shape and layout, the rows,
/// and the slots each row selects.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct RoutedMatmulKey {
    /// The lhs, the stack and the output all bind at this element.
    elem: ElemType,
    /// The contraction.
    k: usize,
    /// One slab's output columns.
    n: usize,
    /// Slabs in the stack.
    experts: usize,
    /// The lhs's rows, anchored to the power of two above them.
    rows: usize,
    /// Experts each row selects.
    slots: usize,
    /// Whether the stack is float or packed, and how: a packed stack is read packed, by other
    /// kernels than a float one.
    stack: OperandKind,
    /// Which dimension the stack lines along.
    lines: OperandLayout,
    /// Which dimension the slabs lie along.
    slabs: SlabDimension,
}
