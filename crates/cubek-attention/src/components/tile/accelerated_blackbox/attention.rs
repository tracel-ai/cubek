use cubecl;
use cubecl::prelude::*;
use cubek_std::tile::StridedTile;

use crate::components::tile::accelerated_blackbox::setup::BlackboxAcceleratedAttentionMatmulConfig;
use crate::components::tile::accelerated_blackbox::{
    BlackboxAccumulatorPipeline, BlackboxSoftmaxPipeline, LocalTile, LocalTileLayout,
};
use crate::components::tile::{
    AccumulatorPipeline, SoftmaxPipeline, TileAttention, TileAttentionConfig as _,
};
use crate::definition::AttentionPrecision;
use crate::definition::attention_types::*;

/// Uses accelerated instruction, but relies on shared memory for row-dependent computations
/// because the fragment layout is blackbox
pub struct BlackboxAcceleratedTileAttention;

#[cube]
impl<AP: AttentionPrecision> TileAttention<AP> for BlackboxAcceleratedTileAttention {
    type Config = BlackboxAcceleratedAttentionMatmulConfig;

    type Query = cmma::Matrix<QT<AP>>;
    type KeyValue = cmma::Matrix<KVT<AP>>;
    type Mask = LocalTile<MSK<AP>>;

    type Softmax = BlackboxSoftmaxPipeline<SM<AP>, SML<AP>>;
    type SoftmaxRow = <Self::Softmax as SoftmaxPipeline<SM<AP>>>::Rowwise;
    type SoftmaxTransit = <Self::Softmax as SoftmaxPipeline<SM<AP>>>::Transit;
    type SoftmaxLayout = <Self::Softmax as SoftmaxPipeline<SM<AP>>>::Layout;

    type Accumulator = BlackboxAccumulatorPipeline<ACC<AP>>;
    type AccumulatorTransit = <Self::Accumulator as AccumulatorPipeline<ACC<AP>>>::Transit;

    fn softmax_layout(#[comptime] config: Self::Config) -> LocalTileLayout {
        LocalTileLayout::new(
            (
                config.attention_tile_size().seq_q,
                config.attention_tile_size().seq_kv,
            ),
            config.shared.plane_dim,
            config.inner_layout,
        )
    }

    fn score_matmul(
        query: &Self::Query,
        key: &Self::KeyValue,
        softmax: &mut Self::Softmax,
        #[comptime] _config: Self::Config,
    ) {
        let softmax_frag = &softmax.acc_fragment;
        cmma::execute::<QT<AP>, KVT<AP>, SM<AP>, SM<AP>>(query, key, softmax_frag, softmax_frag);
    }

    fn value_matmul(
        softmax: &Self::Softmax,
        value: &Self::KeyValue,
        out: &mut Self::Accumulator,
        #[comptime] _config: Self::Config,
    ) {
        let softmax_frag = &softmax.lhs_fragment;
        let out = &out.acc_fragment;
        cmma::execute::<SML<AP>, KVT<AP>, ACC<AP>, ACC<AP>>(softmax_frag, value, out, out);
    }

    fn allocate_query(#[comptime] config: Self::Config) -> Self::Query {
        let size = config.attention_tile_size().to_score_matmul_tile_size();

        unsafe {
            cmma::Matrix::<QT<AP>>::uninitialized(
                cmma::MatrixIdent::A,
                size.m() as usize,
                size.n() as usize,
                size.k() as usize,
                cmma::MatrixLayout::RowMajor,
            )
        }
    }

    fn allocate_key_value(#[comptime] _config: Self::Config) -> Self::KeyValue {
        panic!(
            "Can't reuse key/value because the fragment is col major for key and row major for value"
        )
    }

    fn allocate_key(#[comptime] config: Self::Config) -> Self::KeyValue {
        let size = config.attention_tile_size();
        unsafe {
            cmma::Matrix::<KVT<AP>>::uninitialized(
                cmma::MatrixIdent::B,
                size.seq_q as usize,
                size.seq_kv as usize,
                size.head_dim as usize,
                cmma::MatrixLayout::ColMajor,
            )
        }
    }

    fn allocate_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        let size = config.attention_tile_size();
        unsafe {
            cmma::Matrix::<KVT<AP>>::uninitialized(
                cmma::MatrixIdent::B,
                size.seq_q as usize,
                size.val_dim as usize,
                size.seq_kv as usize,
                cmma::MatrixLayout::RowMajor,
            )
        }
    }

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask {
        let size = config.attention_tile_size();
        LocalTile::new(LocalTileLayout::new(
            (size.seq_q, size.seq_kv),
            config.shared.plane_dim,
            config.inner_layout,
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
        BlackboxSoftmaxPipeline::new(
            &mut transit.0,
            &mut transit.1,
            config.attention_tile_size(),
            config,
        )
    }

    fn allocate_accumulator(
        transit: &mut Self::AccumulatorTransit,
        #[comptime] config: Self::Config,
    ) -> Self::Accumulator {
        BlackboxAccumulatorPipeline::new(transit, config.attention_tile_size(), config)
    }

    fn load_query<E: Numeric, N: Size>(tile: &StridedTile<E, N>, fragment: &mut Self::Query) {
        let stride = tile.unvectorized_stride();
        let slice = tile.as_slice();
        cmma::load(fragment, &slice, stride);
    }

    fn load_key_transposed<E: Float, N: Size>(
        tile: &StridedTile<E, N>,
        fragment: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        let stride = tile.unvectorized_stride();
        let slice = tile.as_slice();
        cmma::load(fragment, &slice, stride);
    }

    fn load_value<E: Float, N: Size>(
        tile: &StridedTile<E, N>,
        fragment: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        let stride = tile.unvectorized_stride();
        let slice = tile.as_slice();
        cmma::load(fragment, &slice, stride);
    }

    fn load_mask<E: Numeric, N: Size>(
        tile: &StridedTile<E, N>,
        mask: &mut Self::Mask,
        #[comptime] _config: Self::Config,
    ) {
        mask.load_from_strided_tile(tile)
    }

    fn write_results<E: Float, N: Size>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Vector<E, N>>,
        #[comptime] config: Self::Config,
    ) {
        let acc = cmma::cast::<ACC<AP>, E>(&out.acc_fragment);
        cmma::store(
            slice,
            &acc,
            config.attention_tile_size().val_dim,
            cmma::MatrixLayout::RowMajor,
        );
    }
}
