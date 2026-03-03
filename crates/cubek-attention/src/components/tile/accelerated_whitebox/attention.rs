use cubecl;
use cubecl::cmma::MmaDefinition;
use cubecl::prelude::*;
use cubecl::std::Swizzle;
use cubek_matmul::components::tile::StridedTile;
use cubek_matmul::definition::MatrixLayout;

use crate::components::tile::accelerated_whitebox::WhiteboxAccumulatorPipeline;
use crate::components::tile::accelerated_whitebox::WhiteboxSmemSoftmaxPipeline;
use crate::components::tile::accelerated_whitebox::WhiteboxSoftmaxPipeline;
use crate::components::tile::accelerated_whitebox::manual_matrix::ManualMatrix;
use crate::components::tile::accelerated_whitebox::manual_matrix::ManualMatrixLayout;
use crate::components::tile::accelerated_whitebox::setup::WhiteboxAcceleratedAttentionMatmulConfig;
use crate::components::tile::{
    AccumulatorPipeline, SoftmaxPipeline, TileAttention, TileAttentionConfig as _,
};
use crate::definition::AttentionPrecision;
use crate::definition::attention_types::*;

/// Uses accelerated instruction, but relies on shared memory for row-dependent computations
/// because the fragment layout is whitebox
pub struct WhiteboxAcceleratedTileAttention;

#[cube]
impl<AP: AttentionPrecision> TileAttention<AP> for WhiteboxAcceleratedTileAttention {
    type Config = WhiteboxAcceleratedAttentionMatmulConfig;

    type Query = ManualMatrix<QT<AP>>;
    type KeyValue = ManualMatrix<KVT<AP>>;
    type Mask = ManualMatrix<MSK<AP>>;

    type Softmax = WhiteboxSmemSoftmaxPipeline<SM<AP>, SML<AP>>;
    type SoftmaxRow = <Self::Softmax as SoftmaxPipeline<SM<AP>>>::Rowwise;
    type SoftmaxTransit = <Self::Softmax as SoftmaxPipeline<SM<AP>>>::Transit;
    type SoftmaxLayout = <Self::Softmax as SoftmaxPipeline<SM<AP>>>::Layout;

    type Accumulator = WhiteboxAccumulatorPipeline<ACC<AP>>;
    type AccumulatorTransit = <Self::Accumulator as AccumulatorPipeline<ACC<AP>>>::Transit;

    fn softmax_layout(#[comptime] config: Self::Config) -> Self::SoftmaxLayout {
        let score_matmul_tile_size = config.attention_tile_size().to_score_matmul_tile_size();
        ManualMatrixLayout::new(
            score_matmul_tile_size,
            cmma::MatrixIdent::Accumulator,
            &MmaDefinition::<QT<AP>, KVT<AP>, SM<AP>>::new(
                score_matmul_tile_size.m as usize,
                score_matmul_tile_size.n as usize,
                score_matmul_tile_size.k as usize,
            ),
        )
    }

    fn score_matmul(
        query: &Self::Query,
        key: &Self::KeyValue,
        softmax: &mut Self::Softmax,
        #[comptime] config: Self::Config,
    ) {
        let score_matmul_tile_size = config.attention_tile_size().to_score_matmul_tile_size();
        MmaDefinition::<QT<AP>, KVT<AP>, SM<AP>>::new(
            score_matmul_tile_size.m as usize,
            score_matmul_tile_size.n as usize,
            score_matmul_tile_size.k as usize,
        )
        .execute_inplace(
            &query.fragment,
            &key.fragment,
            &mut softmax.score_acc.fragment,
        );
    }

    fn value_matmul(
        softmax: &Self::Softmax,
        value: &Self::KeyValue,
        out: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        let value_matmul_tile_size = config.attention_tile_size().to_value_matmul_tile_size();
        MmaDefinition::<SML<AP>, KVT<AP>, ACC<AP>>::new(
            value_matmul_tile_size.m as usize,
            value_matmul_tile_size.n as usize,
            value_matmul_tile_size.k as usize,
        )
        .execute_inplace(
            &softmax.value_lhs.fragment,
            &value.fragment,
            &mut out.accumulator.fragment,
        );
    }

    fn allocate_query(#[comptime] config: Self::Config) -> Self::Query {
        let score_matmul_tile_size = config.attention_tile_size().to_score_matmul_tile_size();
        ManualMatrix::new(ManualMatrixLayout::new::<QT<AP>, KVT<AP>, SM<AP>>(
            score_matmul_tile_size,
            cmma::MatrixIdent::A,
            &MmaDefinition::<QT<AP>, KVT<AP>, SM<AP>>::new(
                score_matmul_tile_size.m as usize,
                score_matmul_tile_size.n as usize,
                score_matmul_tile_size.k as usize,
            ),
        ))
    }

    fn allocate_key_value(#[comptime] _config: Self::Config) -> Self::KeyValue {
        unimplemented!()
    }

    fn allocate_key(#[comptime] config: Self::Config) -> Self::KeyValue {
        let score_matmul_tile_size = config.attention_tile_size().to_score_matmul_tile_size();
        ManualMatrix::new(ManualMatrixLayout::new::<QT<AP>, KVT<AP>, SM<AP>>(
            score_matmul_tile_size,
            cmma::MatrixIdent::B,
            &MmaDefinition::<QT<AP>, KVT<AP>, SM<AP>>::new(
                score_matmul_tile_size.m as usize,
                score_matmul_tile_size.n as usize,
                score_matmul_tile_size.k as usize,
            ),
        ))
    }

    fn allocate_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        let value_matmul_tile_size = config.attention_tile_size().to_value_matmul_tile_size();
        ManualMatrix::new(ManualMatrixLayout::new::<SML<AP>, KVT<AP>, ACC<AP>>(
            value_matmul_tile_size,
            cmma::MatrixIdent::B,
            &MmaDefinition::<SML<AP>, KVT<AP>, ACC<AP>>::new(
                value_matmul_tile_size.m as usize,
                value_matmul_tile_size.n as usize,
                value_matmul_tile_size.k as usize,
            ),
        ))
    }

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask {
        let score_matmul_tile_size = config.attention_tile_size().to_score_matmul_tile_size();
        ManualMatrix::new(ManualMatrixLayout::new::<QT<AP>, KVT<AP>, SM<AP>>(
            score_matmul_tile_size,
            cmma::MatrixIdent::Accumulator,
            &MmaDefinition::<QT<AP>, KVT<AP>, SM<AP>>::new(
                score_matmul_tile_size.m as usize,
                score_matmul_tile_size.n as usize,
                score_matmul_tile_size.k as usize,
            ),
        ))
    }

    fn allocate_softmax_transit(#[comptime] config: Self::Config) -> Self::SoftmaxTransit {
        <Self::Softmax as SoftmaxPipeline<SM<AP>>>::transit(
            config.attention_tile_size(),
            config.num_planes() as usize,
        )
    }

    fn allocate_accumulator_transit(#[comptime] config: Self::Config) -> Self::AccumulatorTransit {
        <Self::Accumulator as AccumulatorPipeline<ACC<AP>>>::transit(
            config.attention_tile_size(),
            config.num_planes() as usize,
        )
    }

    fn allocate_softmax(
        transit: &mut Self::SoftmaxTransit,
        #[comptime] config: Self::Config,
    ) -> Self::Softmax {
        WhiteboxSoftmaxPipeline::new::<QT<AP>, KVT<AP>, KVT<AP>, ACC<AP>>(
            *transit,
            config.attention_tile_size(),
            config,
        )
    }

    fn allocate_accumulator(
        _transit: &mut Self::AccumulatorTransit,
        #[comptime] config: Self::Config,
    ) -> Self::Accumulator {
        WhiteboxAccumulatorPipeline::new::<SM<AP>, KVT<AP>>(config.attention_tile_size())
    }

    fn load_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query) {
        fragment.load_from_strided_tile(tile);
    }

    fn load_key_transposed<E: Float>(
        tile: &StridedTile<E>,
        fragment: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        fragment.load_from_strided_tile(tile);
    }

    fn load_value<E: Float>(
        tile: &StridedTile<E>,
        fragment: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        fragment.load_from_strided_tile(tile);
    }

    fn load_mask<E: Numeric>(
        tile: &StridedTile<E>,
        mask: &mut Self::Mask,
        #[comptime] _config: Self::Config,
    ) {
        mask.load_from_strided_tile(tile)
    }

    fn write_results<E: Float>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
        let mut strided_tile = StridedTile::new_strided_mut(
            *slice,
            0u32.runtime(),
            slice.len() as u32,
            config.attention_tile_size().val_dim,
            Swizzle::none(),
            MatrixLayout::RowMajor,
            config.out_smem_line_size as u32,
        );
        out.accumulator
            .store_to_strided_tile::<E>(&mut strided_tile)
    }
}
