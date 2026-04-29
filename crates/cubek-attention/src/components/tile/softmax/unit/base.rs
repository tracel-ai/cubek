use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;
use cubek_std::{
    MatrixLayout, SwizzleModes,
    tile::{
        Plane, ProductType, RegisterMatmul, RowWise, RowwiseTileKind, RowwiseTileWorkspace,
        StridedTile, Tile, TileExpand, UnitTile, UnitTileLayout, register_allocate_acc,
    },
};

use crate::{
    components::tile::MaskTile,
    components::tile::softmax::unit::UnitSoftmaxConfig,
    components::tile::softmax::{Softmax, SoftmaxConfig},
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
    rowwise: RowwiseTileWorkspace<Acc>,
    #[cube(comptime)]
    _phantom: PhantomData<Lhs>,
}

#[cube]
impl<Acc: Float, Lhs: Float> UnitSoftmaxWorkspace<Acc, Lhs> {
    pub fn new(#[comptime] config: UnitSoftmaxConfig) -> Self {
        UnitSoftmaxWorkspace::<Acc, Lhs> {
            max: RowWise::new_min_value(config.num_rows_per_unit()),
            sum: RowWise::new_zero(config.num_rows_per_unit()),
            rowwise: RowwiseTileWorkspace::new(RowwiseTileKind::Direct),
            _phantom: PhantomData,
        }
    }
}

impl UnitSoftmaxConfig {
    pub(crate) fn register(&self) -> RegisterMatmul {
        RegisterMatmul {
            tile_size: self.tile_size.to_score_matmul_tile_size(),
            plane_dim: 1,
            swizzle_modes: SwizzleModes::default(),
            product_type: ProductType::Inner,
        }
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
        #[comptime] _config: Self::Config,
    ) -> Self::ScaleColumn {
        score_matmul_accumulator.scale_and_mask::<MaskTile<Acc, Self>>(
            head_dim_factor,
            mask,
            &mut workspace.rowwise,
        );

        score_matmul_accumulator.row_max(&mut workspace.max, &state.0, &workspace.rowwise);

        score_matmul_accumulator.exp_diff(&workspace.max, &mut workspace.rowwise);

        score_matmul_accumulator.row_sum(&mut workspace.sum, &workspace.rowwise);

        let exp_m_diff = state.0.exp_diff(&workspace.max);

        let new_l = exp_m_diff.mul(&state.1).add(&workspace.sum);

        copy_register_tile::<Acc, Lhs>(score_matmul_accumulator, value_matmul_lhs);

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
            config.register(),
        );
        Self::zero_score_tile(&mut tile);
        tile
    }

    fn zero_score_tile(score_tile: &mut Self::ScoreTile) {
        score_tile.fill_zero();
    }

    fn init_softmax_tile(#[comptime] config: Self::Config) -> Self::SoftmaxedTile {
        register_allocate_acc::<Lhs, Const<0>, Plane>(MatrixLayout::RowMajor, config.register())
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
fn copy_register_tile<SrcE: Float, DstE: Float>(
    src: &Tile<SrcE, Const<0>, Plane, ReadWrite>,
    dst: &mut Tile<DstE, Const<0>, Plane, ReadWrite>,
) {
    match (src, dst) {
        (Tile::Register(s), Tile::Register(d)) => {
            let m = comptime!(s.config.tile_size.m());
            let n = comptime!(s.config.tile_size.n());
            for i in 0..m * n {
                d.data[i as usize] = DstE::cast_from(s.data[i as usize]);
            }
        }
        _ => panic!("UnitSoftmax::copy_register_tile expects Tile::Register"),
    }
}
