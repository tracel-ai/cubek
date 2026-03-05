use std::marker::PhantomData;

use cubecl::{
    cmma::MmaDefinition,
    ir::MatrixIdent,
    prelude::*,
    std::tensor::layout::{Coords1d, Coords2d},
};
use cubek_std::{TileSize, tile::StridedTile};

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
    #[cube(comptime)]
    lines_per_lane: usize,
    #[cube(comptime)]
    line_size: usize,
    #[cube(comptime)]
    pub(crate) num_rows: u32,
    #[cube(comptime)]
    pub(crate) num_cols: u32,
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
        let mma_def = mma_definition::<MT>(tile_size);
        let lines_per_lane = mma_def.lines_per_lane(MI::IDENT);
        let line_size = mma_def.line_size(MI::IDENT);

        // Assuming specific layout, TODO generalize
        let num_rows = 2u32;
        let num_cols = 4u32;

        ManualMatrixLayout::<MI, MT> {
            tile_size,
            mma_definition: mma_def,
            _phantom: PhantomData,
            lines_per_lane,
            line_size,
            num_rows,
            num_cols,
        }
    }

    // Assuming specific layout, TODO generalize
    pub fn local_pos_to_nth(&self, local_pos: Coords2d) -> Coords1d {
        let (row, col) = local_pos;

        let nth = match MI::IDENT {
            // 0 1 2 3
            // 4 5 6 7
            MatrixIdent::A | MatrixIdent::B => row * 4 + col,
            // 0 1 4 5
            // 2 3 6 7
            MatrixIdent::Accumulator => (row << 1) + (col & 1) + ((col & 2) << 1),
        };

        nth as usize
    }

    pub fn nth_to_local_pos(&self, nth: Coords1d) -> Coords2d {
        let (row, col) = match MI::IDENT {
            MatrixIdent::A | MatrixIdent::B => {
                let row = nth / 4;
                let col = nth % 4;
                (row, col)
            }
            MatrixIdent::Accumulator => {
                let row = nth >> 2;
                let col_low = nth & 1;
                let col_high = (nth & 4) >> 1;
                let col = col_low | col_high;
                (row, col)
            }
        };

        (row as u32, col as u32)
    }

    pub fn absolute_position_of_nth(&self, nth: Coords1d) -> Coords2d {
        self.mma_definition
            .position_of_nth(UNIT_POS_PLANE, nth as u32, MI::IDENT)
    }

    pub fn local_to_absolute_pos(&self, local_pos: Coords2d) -> Coords2d {
        let nth = self.local_pos_to_nth(local_pos);
        self.absolute_position_of_nth(nth)
    }

    pub fn create_matrix(self) -> ManualMatrix<MI, MT> {
        ManualMatrix::<MI, MT> {
            fragment: Array::lined(self.lines_per_lane, self.line_size),
            layout: self,
        }
    }
}

#[cube]
impl<MT: MmaTypes> SoftmaxLayout for ManualMatrixLayout<IdentCD, MT> {
    fn absolute_pos(&self, local_pos: Coords2d) -> Coords2d {
        self.local_to_absolute_pos(local_pos)
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
        #[unroll]
        for i in 0..self.layout.lines_per_lane {
            let mut reg = self.fragment[i];
            #[unroll]
            for k in 0..self.layout.line_size {
                reg[k] = MI::Elem::from_int(0);
            }
        }
    }

    pub fn load_from_strided_tile<E2: Numeric>(&mut self, _tile: &StridedTile<E2>) {
        todo!()
        // MmaStageReader::<Self::LhsTile>::load_fragment(
        //     tile,
        //     &mut self.fragment,
        //     self.layout.mma_definition,
        //     MI::IDENT,
        //     MatrixLayout::RowMajor,
        //     config,
        // );
    }

    pub fn store_to_strided_tile<E2: Numeric>(&self, _tile: &mut StridedTile<E2, ReadWrite>) {}

    pub fn get_nth(&self, nth: Coords1d) -> MI::Elem {
        let line = nth / self.layout.line_size;
        let within_line = nth % self.layout.line_size;
        self.fragment[line][within_line]
    }

    pub fn set_nth<E2: Numeric>(&mut self, nth: Coords1d, val: E2) {
        let line = nth / self.layout.line_size;
        let within_line = nth % self.layout.line_size;
        self.fragment[line][within_line] = MI::Elem::cast_from(val);
    }
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

    fn scale_and_mask<M: FragmentMask>(_this: &mut Self, _scale: MT::CD, _mask: &M) {
        todo!()
    }

    fn exp_diff(&mut self, _m: &RowWise<MT::CD>) {
        todo!()
    }
}

#[cube]
impl<MT: MmaTypes<CD: Float>> AccumulatorRowwise<MT::CD> for ManualMatrix<IdentCD, MT> {
    fn rowwise_scale(&mut self, scale: &RowWise<MT::CD>) {
        // TODO Do whole lines at once if possible, but not sure
        // if lines match rows
        #[unroll]
        for row in 0..self.layout.num_rows {
            let _scale = scale.index(row as usize);
            #[unroll]
            for col in 0..self.layout.num_cols {
                let _nth = self.layout.local_pos_to_nth((row, col).runtime());
                todo!()
            }
        }
    }
}

#[cube]
impl<MT: MmaTypes> FragmentMask for ManualMatrix<IdentCD, MT> {
    type Layout = ManualMatrixLayout<IdentCD, MT>;

    fn should_mask(&self, _local_pos: Coords2d) -> bool {
        todo!()
    }
}
