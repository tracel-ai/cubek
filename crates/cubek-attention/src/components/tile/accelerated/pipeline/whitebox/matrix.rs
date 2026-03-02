use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::components::tile::{
    AccumulatorRowwise, AccumulatorRowwiseExpand, FragmentMask, RowWise, SoftmaxLayout,
    SoftmaxLayoutExpand, SoftmaxRowwise, SoftmaxRowwiseExpand,
};

#[derive(CubeType)]
pub struct MatrixSoftmaxLayout {}

#[cube]
impl SoftmaxLayout for MatrixSoftmaxLayout {
    fn absolute_pos(&self, local_pos: Coords2d) -> Coords2d {
        todo!()
    }

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        todo!()
    }
}

#[cube]
impl<E: Float> SoftmaxRowwise<E> for cmma::Matrix<E> {
    type Layout = MatrixSoftmaxLayout;

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        todo!()
    }

    fn rowwise_max(&self) -> RowWise<E> {
        todo!()
    }

    fn rowwise_sum(&self) -> RowWise<E> {
        todo!()
    }

    fn scale_and_mask<M: FragmentMask>(this: &mut Self, scale: E, mask: &M) {
        todo!()
    }

    fn exp_diff(&mut self, m: &RowWise<E>) {
        todo!()
    }
}

#[cube]
impl<E: Float> AccumulatorRowwise<E> for cmma::Matrix<E> {
    fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        todo!()
    }
}
