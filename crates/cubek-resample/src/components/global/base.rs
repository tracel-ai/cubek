use crate::components::*;
use crate::definition::*;
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn resample_kernel<
    C: Numeric,
    Op: GlobalOp + Clone + std::hash::Hash + PartialEq + Eq + std::fmt::Debug + Send + Sync + 'static,
>(
    input: &Tensor<C>,
    output: &mut Tensor<C>,
    out_layout: NdLayout,
    in_layout: NdLayout,
    scales: &Sequence<f32>,
    #[comptime] _op: Op,
    #[define(C)] _dtype: StorageType,
) {
    let linear_idx = ABSOLUTE_POS as usize;
    if linear_idx >= output.len() {
        terminate!();
    }

    let out_coord = out_layout.from_linear(linear_idx);

    let null_coord = Sequence::<u32>::new();
    let dummy_scales = Sequence::<f32>::new();
    let dest_coord = Op::F::map(out_coord.clone(), null_coord.clone(), dummy_scales);

    let in_coord = Op::H::map(out_coord, null_coord, scales.clone());

    let in_idx = in_layout.to_source_pos(in_coord);
    let out_idx = out_layout.to_source_pos(dest_coord);

    if in_idx < input.len() {
        let x = input[in_idx];

        let w = C::from_int(1);

        let combined = Op::Combine::combine::<C>(x, w);

        output[out_idx] = combined;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TestOp;

#[cube]
impl GlobalOp for TestOp {
    type F = IdentityMapper;
    type H = NearestMapper;
    type K = IdentityMapper;
    type Combine = IdentityCombine;
    type Reduce = LinearReduction;
}
