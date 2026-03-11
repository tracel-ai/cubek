// use cubecl;
// use cubecl::prelude::*;

// use crate::components::softmax::base::SoftmaxConfig;
// use crate::components::softmax::{ReduceOp, Reducer};
// use crate::components::tile::RowWise;
// use crate::components::tile::{SoftmaxRowwise, SoftmaxRowwiseExpand};

// #[derive(CubeType)]
// /// Naive row reducer using shared memory
// #[allow(unused)]
// pub struct NaiveReducer {
//     num_rows_per_unit: usize,
//     plane_dim: usize,
//     num_planes: usize,
// }

// #[cube]
// impl Reducer for NaiveReducer {
//     fn reduce<E: Float, F: SoftmaxRowwise<E>, RO: ReduceOp<E>>(
//         &self,
//         vals: &mut RowWise<E>,
//         data: &F,
//     ) {
//         let num_vals_in_plane = self.num_rows_per_unit * self.plane_dim;
//         let mut smem = SharedMemory::<E>::new((num_vals_in_plane * self.num_planes) as usize);

//         let local_vals = RO::reduce_local::<F>(data);

//         let plane_offset = UNIT_POS_Y * num_vals_in_plane;
//         let unit_offset = UNIT_POS_X;

//         #[unroll]
//         for r in 0..self.num_rows_per_unit as usize {
//             let row_offset = r as u32 * self.plane_dim;
//             let offset = plane_offset + row_offset + unit_offset;

//             smem[offset as usize] = local_vals.index(r);
//         }

//         sync_cube();

//         let num_units_per_row = data.num_units_per_row();

//         #[unroll]
//         for r in 0..self.num_rows_per_unit as usize {
//             let mut val = vals.index(r);

//             let row_offset = r as u32 * self.plane_dim;

//             for c in 0..num_units_per_row {
//                 let unit_offset = (UNIT_POS_X / num_units_per_row) * num_units_per_row;
//                 let offset = plane_offset + row_offset + unit_offset;

//                 val = RO::reduce_step_scalar(val, smem[(offset + c) as usize]);
//             }

//             vals.replace_at(r, val);
//         }

//         sync_cube();
//     }
// }
