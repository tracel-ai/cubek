//! The hypercube split-keys merge on its work item: the slots `splits` workers left for every query
//! of one head, the queries in boxes of `merge.seq_q`, every box counted in by `merge.cubes` cubes,
//! the last of which merges it by one plane. Each cube first stores its own share of the slots, the
//! splits it would have walked, so a box's slots are whole only once its last cube has arrived.

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

/// What a cube of every box reads and writes, all row-major: the slots' values each cube stores its
/// share of (`source_*`), the slots it stores them in, and the output and logsumexp the last cube
/// of a box writes. `*_out` is `[1, 1, 1, queries, splits, val_dim]`, `*_max` and `*_sum`
/// `[1, 1, 1, queries, splits]`, `out` `[1, 1, 1, queries, val_dim]`, `logsumexp`
/// `[1, 1, 1, queries]`.
#[derive(Clone)]
pub struct HypercubeMergeTensors {
    pub out: TensorHandle,
    pub logsumexp: TensorHandle,
    pub source_out: TensorHandle,
    pub source_max: TensorHandle,
    pub source_sum: TensorHandle,
    pub slot_out: TensorHandle,
    pub slot_max: TensorHandle,
    pub slot_sum: TensorHandle,
}

impl HypercubeMergeLaunch {
    /// The boxes of queries, the last one short where `merge.seq_q` does not divide them.
    pub fn boxes(&self) -> usize {
        self.queries.div_ceil(self.merge.seq_q)
    }

    /// Cube `cube` of every box, alone: it stores its share of its box's slots and counts itself
    /// in at `arrivals`, one `u32` a box, and merges the box into `tensors`' output and logsumexp
    /// where it arrives last. The cubes of a box launched one after another arrive in that order,
    /// so a merge by any but the last shows in the output of its own launch. `elements` are the
    /// sum's and the output's.
    pub fn launch_cube(
        &self,
        client: &Client,
        tensors: &HypercubeMergeTensors,
        arrivals: &TensorHandle,
        cube: usize,
        [sum, out_element]: [ElemType; 2],
    ) {
        let arg = |tensor: &TensorHandle| tensor.clone().binding().into_tensor_arg();
        // SAFETY: the cube stores its share of its box's slots and counts itself in at its box's
        // counter; the box's last merges the box's slots into those queries of the output and the
        // logsumexp, at their own strides.
        unsafe {
            hypercube_merge_kernel::launch_unchecked(
                client,
                CubeCount::new_3d(1, self.boxes() as u32, 1),
                CubeDim::new_1d(self.merge.slots.units as u32),
                1,
                arg(&tensors.out),
                Some(arg(&tensors.logsumexp)).into(),
                arg(&tensors.source_out),
                arg(&tensors.source_max),
                arg(&tensors.source_sum),
                arg(&tensors.slot_out),
                arg(&tensors.slot_max),
                arg(&tensors.slot_sum),
                arrivals.clone().into_arg().into_buffer_arg(),
                cube as u32,
                self.merge,
                [sum, out_element],
            );
        }
    }
}

/// Cube `cube` of every box storing its share of the box's slots and counting itself in, merging
/// the box where it arrives last: the box is `y`. A cube's share is every split whose index leaves
/// `cube` over the box's cubes.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn hypercube_merge_kernel<EA: Float, E: Float, N: Size>(
    out: &mut Tensor<E>,
    logsumexp: ComptimeOption<&mut Tensor<EA>>,
    source_out: &Tensor<Vector<EA, N>>,
    source_max: &Tensor<EA>,
    source_sum: &Tensor<EA>,
    slot_out: &mut Tensor<Vector<EA, N>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    arrivals: &[Atomic<u32>],
    cube: u32,
    #[comptime] merge: HypercubeMergeShape,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let (splits, val_dim) = (merge.slots.splits, merge.slots.val_dim);
    let first_query = CUBE_POS_Y as usize * merge.seq_q;
    let mut end = first_query + merge.seq_q;
    if end > slot_max.shape(3) {
        end = slot_max.shape(3);
    }
    for query in first_query..end {
        let mut split = cube as usize;
        while split < splits {
            let slot = query * splits + split;
            if UNIT_POS == 0 {
                slot_max[slot] = source_max[slot];
                slot_sum[slot] = source_sum[slot];
            }
            let mut d = UNIT_POS as usize;
            while d < val_dim {
                slot_out[slot * val_dim + d] = source_out[slot * val_dim + d];
                d += CUBE_DIM as usize;
            }
            split += merge.cubes;
        }
    }
    let slot_box = SlotBox::new::<EA>(slot_max, 0, 0, 0, first_query, merge.seq_q);
    hypercube_split_keys_merge_microkernel::<EA, E, N>(
        &slot_box, slot_out, slot_max, slot_sum, out, logsumexp, arrivals, merge,
    );
}
