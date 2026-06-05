use crate::components::*;
use crate::definition::*;
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn resample_kernel<C: Numeric, Op: GlobalOp>(
    input: &Tensor<C>,
    output: &mut Tensor<C>,
    out_layout: NdLayout,
    in_layout: NdLayout,
    scales: Sequence<f32>, // Sequence of scales for each dimension
) {
    let linear_idx = ABSOLUTE_POS as usize;
    if linear_idx >= output.len() {
        terminate!();
    }

    // 1. Decompose the 1D coalesced index into N-D logical output coordinates
    let out_coord = out_layout.from_linear(linear_idx);

    // 2. Map coordinates through our GlobalOp logic
    // F maps output coordinates to final destination coordinates (usually identity)
    let null_coord = Sequence::<u32>::new(); // empty for now
    let dummy_scales = Sequence::<f32>::new();
    let dest_coord = Op::F::map(out_coord.clone(), null_coord.clone(), dummy_scales);

    // H maps output coordinates to input source coordinates
    let in_coord = Op::H::map(out_coord, null_coord, scales);

    // 3. Use Layout to compute the source memory position
    let in_idx = in_layout.to_source_pos(in_coord);
    let out_idx = out_layout.to_source_pos(dest_coord);

    // Read input - check bounds
    if in_idx < input.len() {
        let x = input[in_idx];

        // Dummy weight read for template completeness
        let w = C::from_int(1); // Would be weights[Op::K::map(...)]

        let combined = Op::Combine::combine::<C>(x, w);

        output[out_idx] = combined; // In a full reduction, this would accumulate
    }
}
