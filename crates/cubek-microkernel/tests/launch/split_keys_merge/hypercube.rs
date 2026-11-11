//! The hypercube split-keys merge on its work item: the slots `splits` workers left for every query
//! of one head, the queries in boxes of `merge.seq_q`, every box counted in by `merge.cubes` cubes,
//! the last of which merges it by one plane.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::{self as cubecl, cube};
use cubek_microkernel::split_keys_merge::SlotBox;
use cubek_microkernel::split_keys_merge::hypercube::{
    HypercubeMergeShape, hypercube_split_keys_merge_microkernel,
};

/// A hypercube merge ready to launch over `queries` queries of one head, its slots laid out as
/// `merge.slots` says.
#[derive(Debug, Clone, Copy)]
pub struct HypercubeMergeLaunch {
    pub merge: HypercubeMergeShape,
    pub queries: usize,
}

impl HypercubeMergeLaunch {
    /// `out` `[1, 1, 1, queries, val_dim]` of `out_element` and `logsumexp` `[1, 1, 1, queries]`
    /// merged from `slot_out` `[1, 1, 1, queries, splits, val_dim]` and `slot_max` and `slot_sum`
    /// `[1, 1, 1, queries, splits]`, all of `sum`, all row-major: a cube for every one of a box's
    /// cubes, along `x`, for every box, along `y`.
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
        let boxes = self.queries.div_ceil(self.merge.seq_q);
        let arrivals = TensorHandle::zeros(client, [boxes], u32::elem_type_native());
        // SAFETY: every cube counts itself in at its box's counter, which arrives holding zero, and
        // the box's last merges its queries' slots into those queries of the output and the
        // logsumexp, at their own strides.
        unsafe {
            hypercube_merge_kernel::launch_unchecked(
                client,
                CubeCount::new_3d(self.merge.cubes as u32, boxes as u32, 1),
                CubeDim::new_1d(self.merge.slots.units as u32),
                1,
                out,
                Some(logsumexp).into(),
                slot_out,
                slot_max,
                slot_sum,
                arrivals.into_arg().into_buffer_arg(),
                self.merge,
                [sum, out_element],
            );
        }
    }
}

/// Every cube of a box counting itself in, the last merging the box: the box is `y`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn hypercube_merge_kernel<EA: Float, E: Float, N: Size>(
    out: &mut Tensor<E>,
    logsumexp: ComptimeOption<&mut Tensor<EA>>,
    slot_out: &mut Tensor<Vector<EA, N>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    arrivals: &[Atomic<u32>],
    #[comptime] merge: HypercubeMergeShape,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let first_query = CUBE_POS_Y as usize * merge.seq_q;
    let slot_box = SlotBox::new::<EA>(slot_max, 0, 0, 0, first_query, merge.seq_q);
    hypercube_split_keys_merge_microkernel::<EA, E, N>(
        &slot_box, slot_out, slot_max, slot_sum, out, logsumexp, arrivals, merge,
    );
}
