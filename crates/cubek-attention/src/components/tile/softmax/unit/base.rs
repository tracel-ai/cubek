use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;
use cubek_matmul::{
    components::tile_matmul::{
        Plane, ProductType, SharedTileConfig, Tile, TileConfig, TileExpand, register_allocate_acc,
    },
    definition::SwizzleModes,
};
use cubek_std::{MatrixLayout, tile::StridedTile};

use crate::{
    components::tile::pipeline::{RowWise, UnitTile, UnitTileLayout},
    components::tile::softmax::base::FragmentMaskExpand,
    components::tile::softmax::unit::UnitSoftmaxConfig,
    components::tile::softmax::{FragmentMask, Softmax, SoftmaxConfig},
    components::tile::{LOGIT_MASKED, MaskTile},
};

#[derive(CubeType)]
pub struct UnitSoftmax<Lhs: Float> {
    #[cube(comptime)]
    _phantom: PhantomData<Lhs>,
}

#[derive(CubeType)]
pub struct UnitSoftmaxWorkspace<Acc: Float, Lhs: Float> {
    max: RowWise<Acc>,
    sum: RowWise<Acc>,
    #[cube(comptime)]
    _phantom: PhantomData<Lhs>,
}

#[cube]
impl<Acc: Float, Lhs: Float> UnitSoftmaxWorkspace<Acc, Lhs> {
    pub fn new(#[comptime] config: UnitSoftmaxConfig) -> Self {
        UnitSoftmaxWorkspace::<Acc, Lhs> {
            max: RowWise::new_min_value(config.num_rows_per_unit()),
            sum: RowWise::new_zero(config.num_rows_per_unit()),
            _phantom: PhantomData,
        }
    }
}

impl UnitSoftmaxConfig {
    pub(crate) fn shared(&self) -> SharedTileConfig {
        SharedTileConfig::new(
            self.tile_size.to_score_matmul_tile_size(),
            1,
            SwizzleModes::default(),
        )
    }
}

#[cube]
impl<Acc: Float, Lhs: Float> Softmax<Acc> for UnitSoftmax<Lhs> {
    type Config = UnitSoftmaxConfig;
    type ScaleColumn = RowWise<Acc>;
    type RunningState = (RowWise<Acc>, RowWise<Acc>);
    type ScoreTile = Tile<Acc, Const<0>, Plane, ReadWrite>;
    type SoftmaxedTile = Tile<Lhs, Const<0>, Plane, ReadWrite>;
    type Workspace = UnitSoftmaxWorkspace<Acc, Lhs>;
    type Mask = UnitTile<Acc>;
    type ScoreLayout = UnitTileLayout;

    fn softmax(
        score_matmul_accumulator: &mut Self::ScoreTile,
        mask: &MaskTile<Acc, Self>,
        value_matmul_lhs: &mut Self::SoftmaxedTile,
        state: &mut Self::RunningState,
        workspace: &mut Self::Workspace,
        head_dim_factor: Acc,
        #[comptime] config: Self::Config,
    ) -> Self::ScaleColumn {
        let num_rows = comptime!(config.tile_size().seq_q);
        let num_cols = comptime!(config.tile_size().seq_kv);

        scale_and_mask_tile::<Acc, MaskTile<Acc, Self>>(
            score_matmul_accumulator,
            head_dim_factor,
            mask,
            num_rows,
            num_cols,
        );

        workspace.max.copy_from(&state.0);
        row_max_into::<Acc>(
            &mut workspace.max,
            score_matmul_accumulator,
            num_rows,
            num_cols,
        );

        exp_diff_tile::<Acc>(score_matmul_accumulator, &workspace.max, num_rows, num_cols);

        workspace.sum.fill(Acc::from_int(0));
        row_sum_into::<Acc>(
            &mut workspace.sum,
            score_matmul_accumulator,
            num_rows,
            num_cols,
        );

        let exp_m_diff = state.0.exp_diff(&workspace.max);

        let new_l = exp_m_diff.mul(&state.1).add(&workspace.sum);

        copy_register_tile::<Acc, Lhs>(
            score_matmul_accumulator,
            value_matmul_lhs,
            num_rows,
            num_cols,
        );

        RowWise::copy_from(&mut state.0, &workspace.max);
        RowWise::copy_from(&mut state.1, &new_l);

        exp_m_diff
    }

    fn init_workspace(#[comptime] config: Self::Config) -> Self::Workspace {
        Self::Workspace::new(config)
    }

    fn init_state(#[comptime] config: Self::Config) -> Self::RunningState {
        (
            RowWise::<Acc>::new_min_value(config.num_rows_per_unit()),
            RowWise::<Acc>::new_zero(config.num_rows_per_unit()),
        )
    }

    fn init_score_tile(#[comptime] config: Self::Config) -> Self::ScoreTile {
        let mut tile = register_allocate_acc::<Acc, Const<0>, Plane>(
            MatrixLayout::RowMajor,
            config.shared(),
            ProductType::Inner,
        );
        Self::zero_score_tile(&mut tile);
        tile
    }

    fn zero_score_tile(score_tile: &mut Self::ScoreTile) {
        zero_register_tile::<Acc>(score_tile);
    }

    fn init_softmax_tile(#[comptime] config: Self::Config) -> Self::SoftmaxedTile {
        register_allocate_acc::<Lhs, Const<0>, Plane>(
            MatrixLayout::RowMajor,
            config.shared(),
            ProductType::Inner,
        )
    }

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask {
        UnitTile::new(<Self as Softmax<Acc>>::layout(config))
    }

    fn load_mask<E: Numeric, ES: Size>(
        tile: &StridedTile<E, ES>,
        fragment: &mut Self::Mask,
        #[comptime] _config: Self::Config,
    ) {
        fragment.load_from_strided_tile(tile);
    }

    fn layout(#[comptime] config: Self::Config) -> Self::ScoreLayout {
        UnitTileLayout {
            num_rows: config.tile_size.seq_q,
            num_cols: config.tile_size.seq_kv,
            transposed_load: false,
        }
    }
}

#[cube]
fn zero_register_tile<E: Numeric>(tile: &mut Tile<E, Const<0>, Plane, ReadWrite>) {
    match tile {
        Tile::Register(t) => {
            let num_elements =
                comptime!(t.config.elements_in_tile_m() * t.config.elements_in_tile_n());
            fill_array_zero::<E>(&mut t.data, num_elements);
        }
        Tile::Cmma(_dummy) => panic!("UnitSoftmax expects Tile::Register"),
        _ => panic!("UnitSoftmax expects Tile::Register"),
    }
}

#[cube]
fn fill_array_zero<E: Numeric>(data: &mut Array<E>, #[comptime] num_elements: u32) {
    for i in 0..num_elements {
        data[i as usize] = E::from_int(0);
    }
}

#[cube]
fn scale_and_mask_tile<Acc: Float, M: FragmentMask>(
    tile: &mut Tile<Acc, Const<0>, Plane, ReadWrite>,
    scale: Acc,
    mask: &M,
    #[comptime] num_rows: u32,
    #[comptime] num_cols: u32,
) {
    match tile {
        Tile::Register(t) => {
            scale_and_mask_array::<Acc, M>(&mut t.data, scale, mask, num_rows, num_cols)
        }
        Tile::Cmma(_dummy) => panic!("UnitSoftmax expects Tile::Register"),
        _ => panic!("UnitSoftmax expects Tile::Register"),
    }
}

#[cube]
fn scale_and_mask_array<E: Float, M: FragmentMask>(
    data: &mut Array<E>,
    scale: E,
    mask: &M,
    #[comptime] num_rows: u32,
    #[comptime] num_cols: u32,
) {
    for r in 0..num_rows {
        let row_offset = r * num_cols;
        for c in 0..num_cols {
            let index = (row_offset + c) as usize;
            data[index] =
                data[index] * scale + E::cast_from(mask.should_mask((r, c))) * E::min_value();
        }
    }
}

#[cube]
fn row_max_into<Acc: Float>(
    acc: &mut RowWise<Acc>,
    tile: &Tile<Acc, Const<0>, Plane, ReadWrite>,
    #[comptime] num_rows: u32,
    #[comptime] num_cols: u32,
) {
    match tile {
        Tile::Register(t) => row_max_array::<Acc>(acc, &t.data, num_rows, num_cols),
        Tile::Cmma(_dummy) => panic!("UnitSoftmax expects Tile::Register"),
        _ => panic!("UnitSoftmax expects Tile::Register"),
    }
}

#[cube]
fn row_max_array<E: Float>(
    acc: &mut RowWise<E>,
    data: &Array<E>,
    #[comptime] num_rows: u32,
    #[comptime] num_cols: u32,
) {
    for r in 0..num_rows as usize {
        let row_offset = r as u32 * num_cols;
        let mut val = E::min_value();
        for c in 0..num_cols {
            val = max(val, data[(row_offset + c) as usize]);
        }
        acc.vals[r] = max(acc.vals[r], val);
    }
}

#[cube]
fn row_sum_into<Acc: Float>(
    acc: &mut RowWise<Acc>,
    tile: &Tile<Acc, Const<0>, Plane, ReadWrite>,
    #[comptime] num_rows: u32,
    #[comptime] num_cols: u32,
) {
    match tile {
        Tile::Register(t) => row_sum_array::<Acc>(acc, &t.data, num_rows, num_cols),
        Tile::Cmma(_dummy) => panic!("UnitSoftmax expects Tile::Register"),
        _ => panic!("UnitSoftmax expects Tile::Register"),
    }
}

#[cube]
fn row_sum_array<E: Float>(
    acc: &mut RowWise<E>,
    data: &Array<E>,
    #[comptime] num_rows: u32,
    #[comptime] num_cols: u32,
) {
    for r in 0..num_rows as usize {
        let row_offset = r as u32 * num_cols;
        let mut val = E::from_int(0);
        for c in 0..num_cols {
            val += data[(row_offset + c) as usize];
        }
        acc.vals[r] += val;
    }
}

#[cube]
fn exp_diff_tile<Acc: Float>(
    tile: &mut Tile<Acc, Const<0>, Plane, ReadWrite>,
    rowwise: &RowWise<Acc>,
    #[comptime] num_rows: u32,
    #[comptime] num_cols: u32,
) {
    match tile {
        Tile::Register(t) => exp_diff_array::<Acc>(&mut t.data, rowwise, num_rows, num_cols),
        Tile::Cmma(_dummy) => panic!("UnitSoftmax expects Tile::Register"),
        _ => panic!("UnitSoftmax expects Tile::Register"),
    }
}

#[cube]
fn exp_diff_array<E: Float>(
    data: &mut Array<E>,
    rowwise: &RowWise<E>,
    #[comptime] num_rows: u32,
    #[comptime] num_cols: u32,
) {
    let threshold = E::new(LOGIT_MASKED);
    for r in 0..num_rows as usize {
        let row_offset = r as u32 * num_cols;
        let val = rowwise.vals[r];
        let safe_val = clamp_min(val, threshold);
        let not_masked = E::cast_from(val >= threshold);
        for c in 0..num_cols {
            let index = (row_offset + c) as usize;
            data[index] = not_masked * (data[index] - safe_val).exp();
        }
    }
}

#[cube]
fn copy_register_tile<SrcE: Numeric, DstE: Numeric>(
    src: &Tile<SrcE, Const<0>, Plane, ReadWrite>,
    dst: &mut Tile<DstE, Const<0>, Plane, ReadWrite>,
    #[comptime] num_rows: u32,
    #[comptime] num_cols: u32,
) {
    match (src, dst) {
        (Tile::Register(s), Tile::Register(d)) => {
            copy_register_arrays::<SrcE, DstE>(&s.data, &mut d.data, num_rows, num_cols)
        }
        _ => panic!("UnitSoftmax expects Tile::Register"),
    }
}

#[cube]
fn copy_register_arrays<SrcE: Numeric, DstE: Numeric>(
    src: &Array<SrcE>,
    dst: &mut Array<DstE>,
    #[comptime] num_rows: u32,
    #[comptime] num_cols: u32,
) {
    for i in 0..num_rows * num_cols {
        dst[i as usize] = DstE::cast_from(src[i as usize]);
    }
}
