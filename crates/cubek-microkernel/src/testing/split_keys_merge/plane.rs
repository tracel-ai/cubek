//! The plane split-keys merge on its work item: the slots `splits` workers left for every query of
//! one head, each a running max, a running sum and an unnormalized output, merged by one plane a
//! query into the output and the query's logsumexp.

use crate::split_keys_merge::plane::plane_split_keys_merge_microkernel;
use crate::split_keys_merge::{SlotBox, SlotShape};
use cubecl::prelude::*;
use cubecl::{self as cubecl, cube};

/// A plane merge ready to launch over `queries` queries of one head, its slots laid out as
/// `slots` says, a plane `slots.units` wide merging each.
#[derive(Debug, Clone, Copy)]
pub struct PlaneMergeLaunch {
    pub slots: SlotShape,
    pub queries: usize,
}

impl PlaneMergeLaunch {
    /// `out` `[1, 1, 1, queries, val_dim]` of `out_element` and `logsumexp` `[1, 1, 1, queries]`
    /// merged from `slot_out` `[1, 1, 1, queries, splits, val_dim]` and `slot_max` and `slot_sum`
    /// `[1, 1, 1, queries, splits]`, all of `sum`, all row-major.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        client: &Client,
        out: TensorArg,
        logsumexp: TensorArg,
        slot_out: TensorArg,
        slot_max: TensorArg,
        slot_sum: TensorArg,
        [sum, out_element]: [ElemType; 2],
    ) {
        // SAFETY: a plane for every query reads that query's slots and writes that query of the
        // output and the logsumexp, at their own strides.
        unsafe {
            plane_merge_kernel::launch_unchecked(
                client,
                CubeCount::new_3d(self.queries as u32, 1, 1),
                CubeDim::new_1d(self.slots.units as u32),
                1,
                out,
                Some(logsumexp).into(),
                slot_out,
                slot_max,
                slot_sum,
                self.slots,
                [sum, out_element],
            );
        }
    }
}

/// A plane for every query merging the query's slots: the query is `q = x`.
#[cube(launch_unchecked)]
fn plane_merge_kernel<EA: Float, E: Float, N: Size>(
    out: &mut Tensor<E>,
    logsumexp: ComptimeOption<&mut Tensor<EA>>,
    slot_out: &mut Tensor<Vector<EA, N>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    #[comptime] slots: SlotShape,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let slot_box = SlotBox::new::<EA>(slot_max, 0, 0, 0, CUBE_POS_X as usize, 1usize);
    plane_split_keys_merge_microkernel::<EA, E, N>(
        slot_out, slot_max, slot_sum, out, logsumexp, &slot_box, slots, 1usize,
    );
}
