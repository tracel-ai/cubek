use cubecl;
use cubecl::prelude::*;
use cubek_std::tile::StridedTile;

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
    type SoftmaxTransit = ();
    type Accumulator = UnitTile<ACC<AP>>;
    type AccumulatorTransit = ();
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

    fn allocate_softmax_transit(#[comptime] _config: Self::Config) -> Self::SoftmaxTransit {}

    fn allocate_accumulator_transit(#[comptime] _config: Self::Config) -> Self::AccumulatorTransit {
    }

    fn allocate_softmax(
        _shared: &mut Self::SoftmaxTransit,
        #[comptime] config: Self::Config,
    ) -> Self::Softmax {
        UnitTile::new(<Self as TileAttention<AP>>::softmax_layout(config))
    }

    fn allocate_accumulator(
        _shared: &mut Self::AccumulatorTransit,
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

    fn load_query<E: Numeric, N: Size>(tile: &StridedTile<E, N>, fragment: &mut Self::Query) {
        strided_tile_to_unit_tile(tile, fragment);
    }

    fn load_key_transposed<E: Float, N: Size>(
        tile: &StridedTile<E, N>,
        fragment: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        strided_tile_to_transposed_unit_tile(tile, fragment);
    }

    fn load_value<E: Float, N: Size>(
        tile: &StridedTile<E, N>,
        fragment: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        strided_tile_to_unit_tile(tile, fragment);
    }

    fn load_mask<E: Numeric, N: Size>(
        tile: &StridedTile<E, N>,
        fragment: &mut Self::Mask,
        #[comptime] _config: Self::Config,
    ) {
        strided_tile_to_unit_tile(tile, fragment);
    }

    fn write_results<E: Float, N: Size>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Vector<E, N>>,
        #[comptime] _config: Self::Config,
    ) {
        unit_tile_to_slice(out, slice)
    }
}

#[cube]
fn strided_tile_to_unit_tile<E: Numeric, N: Size, E2: Numeric>(
    strided_tile: &StridedTile<E, N>,
    unit_tile: &mut UnitTile<E2>,
) {
    let vector_size = N::value().comptime() as u32;
    assert!(unit_tile.layout.num_cols.is_multiple_of(vector_size));

    let col_iterations = comptime!(unit_tile.layout.num_cols / vector_size);

    for row in 0..unit_tile.layout.num_rows {
        for col in 0..col_iterations {
            let vector_read = strided_tile.get_vector(row, col);
            #[unroll]
            for i in 0..vector_size {
                unit_tile.data
                    [(row * unit_tile.layout.num_cols + col * vector_size + i) as usize] =
                    E2::cast_from(vector_read[i as usize]);
            }
        }
    }
}

#[cube]
fn strided_tile_to_transposed_unit_tile<E: Numeric, N: Size, E2: Numeric>(
    strided_tile: &StridedTile<E, N>,
    unit_tile: &mut UnitTile<E2>,
) {
    let vector_size = N::value().comptime() as u32;
    assert!(unit_tile.layout.num_cols.is_multiple_of(vector_size));

    let input_num_rows = unit_tile.layout.num_cols.comptime();
    let input_num_cols = unit_tile.layout.num_rows.comptime();
    let vector_iterations = input_num_cols / vector_size;

    for input_row in 0..input_num_rows {
        for input_col_vector in 0..vector_iterations {
            let vector_read = strided_tile.get_vector(input_row, input_col_vector);

            #[unroll]
            for i in 0..vector_size {
                unit_tile.data[((input_col_vector + i) * input_num_rows + input_row) as usize] =
                    E2::cast_from(vector_read[i as usize]);
            }
        }
    }
}

#[cube]
fn unit_tile_to_slice<E: Numeric, N: Size, E2: Numeric>(
    unit_tile: &UnitTile<E>,
    slice: &mut SliceMut<Vector<E2, N>>,
) {
    let vector_size = N::value().comptime() as u32;
    assert!(unit_tile.layout.num_cols.is_multiple_of(vector_size));

    let col_iterations = comptime!(unit_tile.layout.num_cols / vector_size);

    for row in 0..unit_tile.layout.num_rows {
        for col in 0..col_iterations {
            let mut out_vector = Vector::empty();

            #[unroll]
            for i in 0..vector_size {
                let index = row * unit_tile.layout.num_cols + col * vector_size + i;
                out_vector[i as usize] = E2::cast_from(unit_tile.data[index as usize]);
            }

            let vector_index = row * col_iterations + col;
            slice[vector_index as usize] = out_vector;
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
