use std::marker::PhantomData;

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
pub struct ManualMatrixLayout<MI: MmaIdent<MT>, MT: MmaTypes> {
    #[cube(comptime)]
    pub tile_size: TileSize,
    pub mma_definition: MmaDefinition<MT::A, MT::B, MT::CD>,
    #[cube(comptime)]
    _phantom: PhantomData<MI>,
}

pub trait MmaTypes {
    type A: Numeric;
    type B: Numeric;
    type CD: Numeric;
}
pub trait MmaIdent<M: MmaTypes> {
    type Elem: Numeric;
    const IDENT: MatrixIdent;
}

pub struct IdentA;
impl<M: MmaTypes> MmaIdent<M> for IdentA {
    type Elem = M::A;
    const IDENT: MatrixIdent = MatrixIdent::A;
}
pub struct IdentB;
impl<M: MmaTypes> MmaIdent<M> for IdentB {
    type Elem = M::B;
    const IDENT: MatrixIdent = MatrixIdent::B;
}
pub struct IdentCD;
impl<M: MmaTypes> MmaIdent<M> for IdentCD {
    type Elem = M::CD;
    const IDENT: MatrixIdent = MatrixIdent::Accumulator;
}

#[cube]
pub fn mma_definition<M: MmaTypes>(
    #[comptime] tile_size: TileSize,
) -> MmaDefinition<M::A, M::B, M::CD> {
    MmaDefinition::new(
        tile_size.m as usize,
        tile_size.n as usize,
        tile_size.k as usize,
    )
}

#[cube]
impl<MI: MmaIdent<MT>, MT: MmaTypes> ManualMatrixLayout<MI, MT> {
    pub fn new(#[comptime] tile_size: TileSize) -> ManualMatrixLayout<MI, MT> {
        ManualMatrixLayout::<MI, MT> {
            tile_size,
            mma_definition: mma_definition::<MT>(tile_size),
            _phantom: PhantomData,
        }
    }

    // TODO get this from cubecl's cmma for generality
    fn local_index(&self, #[comptime] row: usize, #[comptime] col: usize) -> comptime_type!(usize) {
        match MI::IDENT {
            // 0 1 2 3
            // 4 5 6 7
            MatrixIdent::A | MatrixIdent::B => row * 4 + col,
            // 0 1 4 5
            // 2 3 6 7
            MatrixIdent::Accumulator => (row << 1) + (col & 1) + ((col & 2) << 1),
        }
    }

    // TODO equivalent of position_of_nth
    fn row_col_from_index(index: usize) -> (usize, usize) {
        let col_high = index / 4;
        let inner = index % 4;

        let row = inner / 2;
        let col_low = inner % 2;

        let col = col_low + 2 * col_high;

        (row, col)
    }

    pub fn create_matrix(self) -> ManualMatrix<MI, MT> {
        ManualMatrix::<MI, MT> {
            fragment: Array::lined(
                self.mma_definition.lines_per_lane(MI::IDENT),
                self.mma_definition.line_size(MI::IDENT),
            ),
            layout: self,
        }
    }
}

#[cube]
impl<MT: MmaTypes> SoftmaxLayout for ManualMatrixLayout<IdentCD, MT> {
    fn absolute_pos(&self, local_pos: Coords2d) -> Coords2d {
        let unit_id = UNIT_POS_PLANE;
        (
            unit_id / 4 + local_pos.0 * 8,
            4 * (unit_id % 4) + local_pos.1,
        )
        // match comptime!(matrix_ident) {
        //     MatrixIdent::A | MatrixIdent::Accumulator => (
        //         unit_id / 4 + local_pos.0 * 8,
        //         4 * (unit_id % 4) + local_pos.1,
        //     ),
        //     MatrixIdent::B => (
        //         2 * (unit_id / 4) + local_pos.0,
        //         4 * (unit_id % 4) + local_pos.1,
        //     ),
        // }
    }

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        4
    }
}

#[derive(CubeType)]
pub struct ManualMatrix<MI: MmaIdent<MT>, MT: MmaTypes> {
    pub fragment: Array<Line<MI::Elem>>,
    pub layout: ManualMatrixLayout<MI, MT>,
}

#[cube]
impl<MI: MmaIdent<MT>, MT: MmaTypes> ManualMatrix<MI, MT> {
    pub fn zero(&mut self) {
        todo!()
        // #[unroll]
        // for i in 0..self.layout.num_lines {
        //     let mut reg = self.fragment[i];
        //     #[unroll]
        //     for k in 0..self.layout.line_size {
        //         reg[k] = E::from_int(0);
        //     }
        // }
    }

    pub fn load_from_strided_tile<E2: Numeric>(&mut self, tile: &StridedTile<E2>) {
        todo!()
        // #[unroll]
        // for i in 0..self.layout.num_lines {
        //     let mut reg = self.fragment[i];
        //     #[unroll]
        //     for k in 0..self.layout.line_size {
        //         let nth_elem = i * self.layout.line_size + k;
        //         let (row, col) = def.position_of_nth(lane_id, nth_elem as u32, MatrixIdent::A);
        //         let value = a[(row * size_k as u32 + col) as usize];
        //         reg[k] = value;
        //     }
        // }

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
impl<MT: MmaTypes<CD: Float>> SoftmaxRowwise<MT::CD> for ManualMatrix<IdentCD, MT> {
    type Layout = ManualMatrixLayout<IdentCD, MT>;

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        self.layout.num_units_per_row()
    }

    fn rowwise_max(&self) -> RowWise<MT::CD> {
        todo!()
    }

    fn rowwise_sum(&self) -> RowWise<MT::CD> {
        todo!()
    }

    fn scale_and_mask<M: FragmentMask>(this: &mut Self, scale: MT::CD, mask: &M) {
        todo!()
    }

    fn exp_diff(&mut self, m: &RowWise<MT::CD>) {
        todo!()
    }
}

#[cube]
impl<MT: MmaTypes<CD: Float>> AccumulatorRowwise<MT::CD> for ManualMatrix<IdentCD, MT> {
    fn rowwise_scale(&mut self, scale: &RowWise<MT::CD>) {
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
impl<MT: MmaTypes> FragmentMask for ManualMatrix<IdentCD, MT> {
    type Layout = ManualMatrixLayout<IdentCD, MT>;

    fn should_mask(&self, local_pos: Coords2d) -> bool {
        todo!()
    }
}
