use cubecl;
use cubecl::prelude::*;
use cubek_matmul::components::tile::StridedTile;

use crate::components::tile::accelerated::accumulator_fragment::AccumulatorHybridFragment;
use crate::components::tile::accelerated::local_tile::LocalTile;
use crate::components::tile::accelerated::local_tile::LocalTileLayout;
use crate::components::tile::accelerated::setup::BlackboxAcceleratedAttentionMatmulConfig;
use crate::components::tile::accelerated::softmax_fragment::SoftmaxHybridFragment;
use crate::components::tile::{TileAttention, TileAttentionConfig as _};
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
    type Softmax = SoftmaxHybridFragment<SM<AP>, SML<AP>>;
    type SoftmaxRow = LocalTile<SM<AP>>;
    type SoftmaxShared = (SharedMemory<SM<AP>>, SharedMemory<SML<AP>>);
    type AccumulatorShared = SharedMemory<ACC<AP>>;
    type Accumulator = AccumulatorHybridFragment<ACC<AP>>;

    type FragmentLayout = LocalTileLayout;

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

    fn allocate_softmax_shared(#[comptime] config: Self::Config) -> Self::SoftmaxShared {
        let size = config.attention_tile_size().to_score_matmul_tile_size();
        let smem_size = (size.m * size.n * config.num_planes()) as usize;
        (SharedMemory::new(smem_size), SharedMemory::new(smem_size))
    }

    fn allocate_accumulator_shared(#[comptime] config: Self::Config) -> Self::AccumulatorShared {
        let size = config.attention_tile_size().to_value_matmul_tile_size();
        SharedMemory::new((size.m * size.n * config.num_planes()) as usize)
    }

    fn allocate_softmax(
        shared: &mut Self::SoftmaxShared,
        #[comptime] config: Self::Config,
    ) -> Self::Softmax {
        let size = config.attention_tile_size();
        SoftmaxHybridFragment::new(&mut shared.0, &mut shared.1, size, config)
    }

    fn allocate_accumulator(
        shared: &mut Self::AccumulatorShared,
        #[comptime] config: Self::Config,
    ) -> Self::Accumulator {
        let size = config.attention_tile_size().to_value_matmul_tile_size();
        AccumulatorHybridFragment::new(shared, size, config)
    }

    fn load_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query) {
        let (slice, stride) = tile.as_unlined();
        cmma::load(fragment, &slice, stride);
    }

    fn load_key_transposed<E: Float>(
        tile: &StridedTile<E>,
        fragment: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        let (slice, stride) = tile.as_unlined();
        cmma::load(fragment, &slice, stride);
    }

    fn load_value<E: Float>(
        tile: &StridedTile<E>,
        fragment: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        let (slice, stride) = tile.as_unlined();
        cmma::load(fragment, &slice, stride);
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
        let acc = cmma::cast::<ACC<AP>, E>(&out.acc_fragment);
        cmma::store(
            slice,
            &acc,
            config.attention_tile_size().val_dim,
            cmma::MatrixLayout::RowMajor,
        );
    }
}
