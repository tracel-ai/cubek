use cubecl::{cmma::MmaDefinition, ir::MatrixIdent, prelude::*, std::tensor::layout::Coords2d};
use cubek_matmul::{components::tile::StridedTile, definition::TileSize};

use crate::components::tile::{
    AccumulatorRowwise, AccumulatorRowwiseExpand, FragmentMask, FragmentMaskExpand, RowWise,
    SoftmaxLayout, SoftmaxLayoutExpand, SoftmaxRowwise, SoftmaxRowwiseExpand,
};

#[derive(CubeType)]
/// Based on cubecl-cpp/cuda/processors: row_index
/// TODO generalize using MmaDefinition
/// Warning: row_index assumes m,n,k = 16,16,16
///
/// Notes:
/// - A and Accumulator share the same **plane-level layout** (same lane/unit placement in the 16×16 tile). B differs.
/// - A and B share the same **unit-level layout** (ordering of elements inside each lane, i.e., local_pos). Accumulator differs.
pub struct ManualMatrixLayout {
    #[cube(comptime)]
    pub tile_size: TileSize,
    #[cube(comptime)]
    pub matrix_ident: MatrixIdent,
    #[cube(comptime)]
    pub num_lines: usize,
    #[cube(comptime)]
    pub line_size: usize,
}

#[cube]
impl ManualMatrixLayout {
    pub fn new<A: Numeric, B: Numeric, CD: Numeric>(
        #[comptime] tile_size: TileSize,
        #[comptime] matrix_ident: MatrixIdent,
        mma_definition: &MmaDefinition<A, B, CD>,
    ) -> ManualMatrixLayout {
        ManualMatrixLayout {
            tile_size,
            matrix_ident,
            num_lines: mma_definition.lines_per_lane(matrix_ident),
            line_size: mma_definition.line_size(matrix_ident),
        }
    }

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
impl SoftmaxLayout for ManualMatrixLayout {
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
pub struct ManualMatrix<E: Numeric> {
    pub fragment: Array<Line<E>>,
    pub layout: ManualMatrixLayout,
}

#[cube]
impl<E: Numeric> ManualMatrix<E> {
    pub fn new(layout: ManualMatrixLayout) -> ManualMatrix<E> {
        let fragment = Array::lined(layout.num_lines, layout.line_size);
        ManualMatrix::<E> {
            fragment: fragment,
            layout,
        }
    }

    pub fn zero(&mut self) {
        #[unroll]
        for i in 0..self.layout.num_lines {
            let mut reg = self.fragment[i];
            #[unroll]
            for k in 0..self.layout.line_size {
                reg[k] = E::from_int(0);
            }
        }
    }

    pub fn load_from_strided_tile<E2: Numeric>(&mut self, tile: &StridedTile<E2>) {
        #[unroll]
        for i in 0..self.layout.num_lines {
            let mut reg = self.fragment[i];
            #[unroll]
            for k in 0..self.layout.line_size {
                let nth_elem = i * self.layout.line_size + k;
                let (row, col) = def.position_of_nth(lane_id, nth_elem as u32, MatrixIdent::A);
                let value = a[(row * size_k as u32 + col) as usize];
                reg[k] = value;
            }
        }

        // // Assumes line size == 1
        // for r in 0..self.layout.unit_size.0 {
        //     for c in 0..self.layout.unit_size.1 {
        //         let (row, col) = self.layout.absolute_pos((r, c));
        //         self.array[(r * self.layout.unit_size.1 + c) as usize] =
        //             E::cast_from(strided_tile.get_line(row, col))
        //     }
        // }

        // #[unroll]
        // for i in 0..line_count_a {
        //     let mut reg = Line::<A>::empty(line_size_a);
        //     #[unroll]
        //     for k in 0..line_size_a {
        //         let n_elem = i * line_size_a + k;
        //         let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::A);
        //         let value = a[(row * size_k as u32 + col) as usize];
        //         reg[k] = value;
        //     }
        //     registers_a[i] = reg;
        // }
    }

    pub fn store_to_strided_tile<E2: Numeric>(&self, tile: &mut StridedTile<E2, ReadWrite>) {}
}

#[cube]
impl<E: Float> SoftmaxRowwise<E> for ManualMatrix<E> {
    type Layout = ManualMatrixLayout;

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
impl<E: Float> AccumulatorRowwise<E> for ManualMatrix<E> {
    fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        #[unroll]
        for row in 0..2usize {
            let scale = scale.index(row);
            #[unroll]
            for col in 0..4usize {
                let local_index = self.layout.local_index(row, col);
                // let before = self.fragment.local_read(local_index);
                // self.fragment.local_write(local_index, before * scale);
            }
        }
    }
}

#[cube]
impl<E: Numeric> FragmentMask for ManualMatrix<E> {
    type Layout = ManualMatrixLayout;

    fn should_mask(&self, local_pos: Coords2d) -> bool {
        todo!()
    }
}
