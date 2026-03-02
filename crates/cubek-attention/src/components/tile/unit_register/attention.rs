use cubecl;
use cubecl::prelude::*;
use cubek_matmul::components::tile::StridedTile;

use crate::components::tile::unit_register::UnitTileLayout;
use crate::components::tile::unit_register::pipeline::UnitTile;
use crate::components::tile::unit_register::setup::UnitTileAttentionConfig;

use crate::components::tile::TileAttention;
use crate::definition::AttentionPrecision;
use crate::definition::attention_types::*;

pub struct UnitRegisterTileAttention;

#[cube]
impl<AP: AttentionPrecision> TileAttention<AP> for UnitRegisterTileAttention {
    type Config = UnitTileAttentionConfig;

    type Query = UnitTile<QT<AP>>;
    type KeyValue = UnitTile<KVT<AP>>;
    type Mask = UnitTile<MSK<AP>>;
    type Softmax = UnitTile<SM<AP>>;
    type SoftmaxRow = UnitTile<SM<AP>>;
    type SoftmaxShared = ();
    type Accumulator = UnitTile<ACC<AP>>;
    type AccumulatorShared = ();
    type SoftmaxLayout = UnitTileLayout;

    fn softmax_layout(#[comptime] config: Self::Config) -> Self::SoftmaxLayout {
        UnitTileLayout {
            num_rows: config.shared.attention_tile_size.seq_q,
            num_cols: config.shared.attention_tile_size.seq_kv,
        }
    }

    fn score_matmul(
        lhs: &Self::Query,
        rhs: &Self::KeyValue,
        out: &mut Self::Softmax,
        #[comptime] config: Self::Config,
    ) {
        let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = config.shared.attention_tile_size.to_score_matmul_tile_size().into(); (m, n, k)};
        unit_inner_matmul(lhs, rhs, out, m, n, k);
    }

    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::KeyValue,
        out: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = config.shared.attention_tile_size.to_value_matmul_tile_size().into(); (m, n, k)};
        unit_inner_matmul(lhs, rhs, out, m, n, k);
    }

    fn allocate_key_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        UnitTile::new(UnitTileLayout::new(
            comptime!(max(
                config.shared.attention_tile_size.head_dim,
                config.shared.attention_tile_size.seq_kv,
            )),
            comptime!(max(
                config.shared.attention_tile_size.seq_kv,
                config.shared.attention_tile_size.val_dim,
            )),
        ))
    }

    fn allocate_key(#[comptime] config: Self::Config) -> Self::KeyValue {
        UnitTile::new(UnitTileLayout::new(
            config.shared.attention_tile_size.head_dim,
            config.shared.attention_tile_size.seq_kv,
        ))
    }

    fn allocate_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        UnitTile::new(UnitTileLayout::new(
            config.shared.attention_tile_size.seq_kv,
            config.shared.attention_tile_size.val_dim,
        ))
    }

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask {
        UnitTile::new(<Self as TileAttention<AP>>::softmax_layout(config))
    }

    fn allocate_softmax_shared(#[comptime] _config: Self::Config) -> Self::SoftmaxShared {}

    fn allocate_accumulator_shared(#[comptime] _config: Self::Config) -> Self::AccumulatorShared {}

    fn allocate_softmax(
        _shared: &mut Self::SoftmaxShared,
        #[comptime] config: Self::Config,
    ) -> Self::Softmax {
        UnitTile::new(<Self as TileAttention<AP>>::softmax_layout(config))
    }

    fn allocate_accumulator(
        _shared: &mut Self::AccumulatorShared,
        #[comptime] config: Self::Config,
    ) -> Self::Accumulator {
        UnitTile::new(UnitTileLayout::new(
            config.shared.attention_tile_size.seq_q,
            config.shared.attention_tile_size.val_dim,
        ))
    }

    fn allocate_query(#[comptime] config: Self::Config) -> Self::Query {
        UnitTile::new(UnitTileLayout::new(
            config.shared.attention_tile_size.seq_q,
            config.shared.attention_tile_size.head_dim,
        ))
    }

    fn load_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query) {
        strided_tile_to_unit_tile(tile, fragment);
    }

    fn load_key_transposed<E: Float>(
        tile: &StridedTile<E>,
        fragment: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        strided_tile_to_transposed_unit_tile(tile, fragment);
    }

    fn load_value<E: Float>(
        tile: &StridedTile<E>,
        fragment: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        strided_tile_to_unit_tile(tile, fragment);
    }

    fn load_mask<E: Numeric>(
        tile: &StridedTile<E>,
        fragment: &mut Self::Mask,
        #[comptime] _config: Self::Config,
    ) {
        strided_tile_to_unit_tile(tile, fragment);
    }

    fn write_results<E: Float>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] _config: Self::Config,
    ) {
        unit_tile_to_slice(out, slice)
    }
}

#[cube]
fn strided_tile_to_unit_tile<E: Numeric, E2: Numeric>(
    strided_tile: &StridedTile<E>,
    unit_tile: &mut UnitTile<E2>,
) {
    let line_size = strided_tile.line_size;
    assert!(unit_tile.layout.num_cols % line_size == 0);

    let col_iterations = comptime!(unit_tile.layout.num_cols / line_size);

    for row in 0..unit_tile.layout.num_rows {
        for col in 0..col_iterations {
            let line_read = strided_tile.get_line(row, col);
            #[unroll]
            for i in 0..line_size {
                unit_tile.data[(row * unit_tile.layout.num_cols + col * line_size + i) as usize] =
                    E2::cast_from(line_read[i as usize]);
            }
        }
    }
}

#[cube]
fn strided_tile_to_transposed_unit_tile<E: Numeric, E2: Numeric>(
    strided_tile: &StridedTile<E>,
    unit_tile: &mut UnitTile<E2>,
) {
    let line_size = strided_tile.line_size;
    assert!(unit_tile.layout.num_cols % line_size == 0);

    let input_num_rows = unit_tile.layout.num_cols.comptime();
    let input_num_cols = unit_tile.layout.num_rows.comptime();
    let line_iterations = input_num_cols / line_size;

    for input_row in 0..input_num_rows {
        for input_col_line in 0..line_iterations {
            let line_read = strided_tile.get_line(input_row, input_col_line);

            #[unroll]
            for i in 0..line_size {
                unit_tile.data[((input_col_line + i) * input_num_rows + input_row) as usize] =
                    E2::cast_from(line_read[i as usize]);
            }
        }
    }
}

#[cube]
fn unit_tile_to_slice<E: Numeric, E2: Numeric>(
    unit_tile: &UnitTile<E>,
    slice: &mut SliceMut<Line<E2>>,
) {
    let line_size = slice.line_size().comptime() as u32;
    assert!(unit_tile.layout.num_cols % line_size == 0);

    let col_iterations = comptime!(unit_tile.layout.num_cols / line_size);

    for row in 0..unit_tile.layout.num_rows {
        for col in 0..col_iterations {
            let mut out_line = Line::empty(line_size as usize);

            #[unroll]
            for i in 0..line_size {
                let index = row * unit_tile.layout.num_cols + col * line_size + i;
                out_line[i as usize] = E2::cast_from(unit_tile.data[index as usize]);
            }

            let line_index = row * col_iterations + col;
            slice[line_index as usize] = out_line;
        }
    }
}

#[cube]
fn unit_inner_matmul<Lhs: Float, Rhs: Float, Acc: Float>(
    lhs: &UnitTile<Lhs>,
    rhs: &UnitTile<Rhs>,
    out: &mut UnitTile<Acc>,
    #[comptime] m: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
) {
    for m_ in 0..m {
        for n_ in 0..n {
            let mut sum = Acc::from_int(0);
            for k_ in 0..k {
                let lhs_val = lhs.get(m_, k_);
                let rhs_val = rhs.get(k_, n_);
                sum += Acc::cast_from(lhs_val) * Acc::cast_from(rhs_val);
            }
            out.accumulate(m_, n_, sum);
        }
    }
}
