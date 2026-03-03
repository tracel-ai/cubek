use cubecl::{ir::MatrixIdent, prelude::*, std::tensor::layout::Coords2d};

use crate::components::tile::{
    AccumulatorRowwise, AccumulatorRowwiseExpand, FragmentMask, RowWise, SoftmaxLayout,
    SoftmaxLayoutExpand, SoftmaxRowwise, SoftmaxRowwiseExpand,
};

#[derive(CubeType)]
/// Based on cubecl-cpp/cuda/processors: row_index
/// TODO generalize using MmaDefinition
/// Warning: row_index assumes m,n,k = 16,16,16
///
/// Notes:
/// - A and Accumulator share the same **plane-level layout** (same lane/unit placement in the 16×16 tile). B differs.
/// - A and B share the same **unit-level layout** (ordering of elements inside each lane, i.e., local_pos). Accumulator differs.
pub struct RowAwareMatrixLayout {
    #[cube(comptime)]
    pub matrix_ident: MatrixIdent,
}

#[cube]
impl RowAwareMatrixLayout {
    // TODO get this from cubecl's cmma for generality
    fn local_index(&self, #[comptime] row: usize, #[comptime] col: usize) -> comptime_type!(usize) {
        match comptime!(self.matrix_ident) {
            // 0 1 2 3
            // 4 5 6 7
            MatrixIdent::A | MatrixIdent::B => row * 4 + col,
            // 0 1 4 5
            // 2 3 6 7
            MatrixIdent::Accumulator => (row << 1) + (col & 1) + ((col & 2) << 1),
        }
    }
}

#[cube]
impl SoftmaxLayout for RowAwareMatrixLayout {
    fn absolute_pos(&self, local_pos: Coords2d) -> Coords2d {
        let unit_id = UNIT_POS_PLANE;
        match comptime!(self.matrix_ident) {
            MatrixIdent::A | MatrixIdent::Accumulator => (
                unit_id / 4 + local_pos.0 * 8,
                4 * (unit_id % 4) + local_pos.1,
            ),
            MatrixIdent::B => (
                2 * (unit_id / 4) + local_pos.0,
                4 * (unit_id % 4) + local_pos.1,
            ),
        }
    }

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        4
    }
}

#[derive(CubeType)]
pub struct RowAwareMatrixAccumulator<E: Float> {
    pub(crate) fragment: cmma::Matrix<E>,
    pub(crate) layout: RowAwareMatrixLayout,
}

#[cube]
impl<E: Float> SoftmaxRowwise<E> for RowAwareMatrixAccumulator<E> {
    type Layout = RowAwareMatrixLayout;

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        self.layout.num_units_per_row()
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
impl<E: Float> AccumulatorRowwise<E> for RowAwareMatrixAccumulator<E> {
    fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        #[unroll]
        for row in 0..2usize {
            let scale = scale.index(row);
            #[unroll]
            for col in 0..4usize {
                let local_index = self.layout.local_index(row, col);
                let before = self.fragment.read_local(local_index);
                self.fragment.write_local(local_index, before * scale);
            }
        }
    }
}
