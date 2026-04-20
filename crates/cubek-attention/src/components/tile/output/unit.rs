use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;
use cubek_matmul::{
    components::tile::{
        ProductType, SharedTileConfig, Tile, TileExpand, register_allocate_acc, tile_write,
    },
    definition::SwizzleModes,
};
use cubek_std::MatrixLayout;

use crate::{
    components::tile::output::AttentionOutput, components::tile::pipeline::RowWise,
    definition::AttentionTileSize,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnitOutputConfig {
    pub tile_size: AttentionTileSize,
}

impl UnitOutputConfig {
    fn shared(&self) -> SharedTileConfig {
        SharedTileConfig::new(
            self.tile_size.to_value_matmul_tile_size(),
            1,
            SwizzleModes::default(),
        )
    }
}

#[derive(CubeType)]
/// Accumulator tile for Tile Attention
pub struct UnitAttentionOutput<SM: Float, Acc: Float> {
    #[cube(comptime)]
    _phantom: PhantomData<(SM, Acc)>,
}

#[cube]
impl<SM: Float, Acc: Float, VA: Size> AttentionOutput<Acc, VA> for UnitAttentionOutput<SM, Acc> {
    type Config = UnitOutputConfig;
    type ScaleColumn = RowWise<SM>;
    type RunningState = (RowWise<SM>, RowWise<SM>);
    type Workspace = ();

    fn scale_mul(
        tile: &mut Tile<Acc, VA, ReadWrite>,
        scale: &Self::ScaleColumn,
        _workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) {
        let scale_acc = RowWise::<SM>::cast_from::<Acc>(scale);
        apply_rowwise_scale::<Acc, VA>(
            tile,
            &scale_acc,
            config.tile_size.seq_q,
            config.tile_size.val_dim,
        );
    }

    fn scale_div(
        tile: &mut Tile<Acc, VA, ReadWrite>,
        running_state: &Self::RunningState,
        _workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) {
        let mut scale = RowWise::<SM>::cast_from::<Acc>(&running_state.1);
        scale.recip_inplace();

        apply_rowwise_scale::<Acc, VA>(
            tile,
            &scale,
            config.tile_size.seq_q,
            config.tile_size.val_dim,
        );
    }

    fn init_workspace(#[comptime] _config: Self::Config) -> Self::Workspace {}

    fn init_tile(#[comptime] config: Self::Config) -> Tile<Acc, VA, ReadWrite> {
        let mut tile = register_allocate_acc::<Acc, VA>(
            MatrixLayout::RowMajor,
            config.shared(),
            ProductType::Inner,
        );
        zero_register_tile::<Acc, VA>(&mut tile, config.tile_size.seq_q, config.tile_size.val_dim);
        tile
    }

    fn write_results<E: Float, ES: Size>(
        source: &mut Tile<Acc, VA, ReadWrite>,
        dest: &mut Tile<E, ES, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_write::<E, ES, Acc, VA, Acc, Acc>(dest, source);
    }
}

#[cube]
fn apply_rowwise_scale<Acc: Float, VA: Size>(
    tile: &mut Tile<Acc, VA, ReadWrite>,
    scale: &RowWise<Acc>,
    #[comptime] num_rows: u32,
    #[comptime] num_cols: u32,
) {
    match tile {
        Tile::Register(t) => {
            scale_array_rowwise::<Acc>(&mut t.data, scale, num_rows, num_cols);
        }
        Tile::Cmma(_dummy) => panic!("UnitAttentionOutput expects a Tile::Register"),
        _ => panic!("UnitAttentionOutput expects a Tile::Register"),
    }
}

#[cube]
fn scale_array_rowwise<Acc: Float>(
    data: &mut Array<Acc>,
    scale: &RowWise<Acc>,
    #[comptime] num_rows: u32,
    #[comptime] num_cols: u32,
) {
    for r in 0..num_rows {
        let row_offset = r * num_cols;
        for c in 0..num_cols {
            let idx = (row_offset + c) as usize;
            data[idx] = data[idx] * scale.vals[r as usize];
        }
    }
}

#[cube]
fn zero_register_tile<Acc: Float, VA: Size>(
    tile: &mut Tile<Acc, VA, ReadWrite>,
    #[comptime] num_rows: u32,
    #[comptime] num_cols: u32,
) {
    match tile {
        Tile::Register(t) => zero_array::<Acc>(&mut t.data, num_rows * num_cols),
        Tile::Cmma(_dummy) => panic!("UnitAttentionOutput expects a Tile::Register"),
        _ => panic!("UnitAttentionOutput expects a Tile::Register"),
    }
}

#[cube]
fn zero_array<Acc: Float>(data: &mut Array<Acc>, #[comptime] size: u32) {
    for i in 0..size {
        data[i as usize] = Acc::from_int(0);
    }
}
