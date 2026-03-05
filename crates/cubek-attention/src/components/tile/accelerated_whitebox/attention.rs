use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;
use cubecl::std::Swizzle;
use cubek_std::MatrixLayout;
use cubek_std::tile::StridedTile;

use crate::components::tile::accelerated_whitebox::WhiteboxAccumulatorPipeline;
use crate::components::tile::accelerated_whitebox::WhiteboxSmemSoftmaxPipeline;
use crate::components::tile::accelerated_whitebox::WhiteboxSoftmaxPipeline;
use crate::components::tile::accelerated_whitebox::manual_matrix::IdentA;
use crate::components::tile::accelerated_whitebox::manual_matrix::IdentB;
use crate::components::tile::accelerated_whitebox::manual_matrix::IdentCD;
use crate::components::tile::accelerated_whitebox::manual_matrix::ManualMatrix;
use crate::components::tile::accelerated_whitebox::manual_matrix::ManualMatrixLayout;
use crate::components::tile::accelerated_whitebox::manual_matrix::MmaTypes;
use crate::components::tile::accelerated_whitebox::setup::WhiteboxAcceleratedAttentionMatmulConfig;
use crate::components::tile::{
    AccumulatorPipeline, SoftmaxPipeline, TileAttention, TileAttentionConfig as _,
};
use crate::definition::AttentionPrecision;
use crate::definition::attention_types::*;

/// Uses accelerated instruction, but relies on shared memory for row-dependent computations
/// because the fragment layout is whitebox
pub struct WhiteboxAcceleratedTileAttention;

pub struct ScoreMma<AP>(PhantomData<AP>);
impl<AP: AttentionPrecision> MmaTypes for ScoreMma<AP> {
    type A = QT<AP>;
    type B = KVT<AP>;
    type CD = SM<AP>;
}

pub struct ValueMma<AP>(PhantomData<AP>);
impl<AP: AttentionPrecision> MmaTypes for ValueMma<AP> {
    type A = SML<AP>;
    type B = KVT<AP>;
    type CD = ACC<AP>;
}

#[derive(CubeType)]
pub enum KeyValueMatrix<AP: AttentionPrecision> {
    // Only available if ScoreMma == ValueMma
    Reuse(ManualMatrix<IdentB, ScoreMma<AP>>),
    Key(ManualMatrix<IdentB, ScoreMma<AP>>),
    Value(ManualMatrix<IdentB, ValueMma<AP>>),
}

#[cube]
impl<AP: AttentionPrecision> KeyValueMatrix<AP> {
    // fn key(&self) -> &ManualMatrix<IdentB, ScoreMma<AP>> {
    //     match self {
    //         KeyValueMatrix::Reuse(manual_matrix) => manual_matrix,
    //         KeyValueMatrix::Key(manual_matrix) => manual_matrix,
    //         KeyValueMatrix::Value(_) => panic!("Tried to access value on key matrix"),
    //     }
    // }
    // fn value(&self) -> &ManualMatrix<IdentB, ValueMma<AP>> {
    //     match self {
    //         KeyValueMatrix::Reuse(_manual_matrix) => unimplemented!(),
    //         KeyValueMatrix::Key(_) => panic!("Tried to access key on value matrix"),
    //         KeyValueMatrix::Value(manual_matrix) => manual_matrix,
    //     }
    // }

    fn key_mut(&mut self) -> &mut ManualMatrix<IdentB, ScoreMma<AP>> {
        match self {
            KeyValueMatrix::Reuse(manual_matrix) => manual_matrix,
            KeyValueMatrix::Key(manual_matrix) => manual_matrix,
            KeyValueMatrix::Value(_) => panic!("Tried to access value on key matrix"),
        }
    }
    fn value_mut(&mut self) -> &mut ManualMatrix<IdentB, ValueMma<AP>> {
        match self {
            KeyValueMatrix::Reuse(_manual_matrix) => unimplemented!(),
            KeyValueMatrix::Key(_) => panic!("Tried to access key on value matrix"),
            KeyValueMatrix::Value(manual_matrix) => manual_matrix,
        }
    }
}

#[cube]
impl<AP: AttentionPrecision> TileAttention<AP> for WhiteboxAcceleratedTileAttention {
    type Config = WhiteboxAcceleratedAttentionMatmulConfig;

    type Query = ManualMatrix<IdentA, ScoreMma<AP>>;
    type KeyValue = KeyValueMatrix<AP>;
    type Mask = ManualMatrix<IdentCD, ScoreMma<AP>>;

    type Softmax = WhiteboxSmemSoftmaxPipeline<AP>;
    type SoftmaxRow = <Self::Softmax as SoftmaxPipeline<SM<AP>>>::Rowwise;
    type SoftmaxTransit = <Self::Softmax as SoftmaxPipeline<SM<AP>>>::Transit;
    type SoftmaxLayout = <Self::Softmax as SoftmaxPipeline<SM<AP>>>::Layout;

    type Accumulator = WhiteboxAccumulatorPipeline<ValueMma<AP>>;
    type AccumulatorTransit = <Self::Accumulator as AccumulatorPipeline<ACC<AP>>>::Transit;

    fn softmax_layout(#[comptime] config: Self::Config) -> Self::SoftmaxLayout {
        let score_matmul_tile_size = config.attention_tile_size().to_score_matmul_tile_size();
        ManualMatrixLayout::<IdentCD, ScoreMma<AP>>::new(score_matmul_tile_size)
    }

    fn score_matmul(
        _query: &Self::Query,
        _key: &Self::KeyValue,
        _softmax: &mut Self::Softmax,
        #[comptime] _config: Self::Config,
    ) {
        todo!()
        // softmax.softmax_acc.layout.mma_definition.execute_inplace(
        //     &query.fragment,
        //     &key.key().fragment,
        //     &mut softmax.softmax_acc.fragment,
        // );
    }

    fn value_matmul(
        _softmax: &Self::Softmax,
        _value: &Self::KeyValue,
        _out: &mut Self::Accumulator,
        #[comptime] _config: Self::Config,
    ) {
        todo!()
        // softmx.softmax_lhs.layout.mma_definition.execute_inplace(
        //     &softmax.softmax_lhs.fragment,
        //     &value.value().fragment,
        //     &mut out.accumulator.fragment,
        // );
    }

    fn allocate_query(#[comptime] config: Self::Config) -> Self::Query {
        let score_matmul_tile_size = config.attention_tile_size().to_score_matmul_tile_size();
        ManualMatrixLayout::<IdentA, ScoreMma<AP>>::new(score_matmul_tile_size).create_matrix()
    }

    fn allocate_key(#[comptime] config: Self::Config) -> Self::KeyValue {
        let score_matmul_tile_size = config.attention_tile_size().to_score_matmul_tile_size();
        KeyValueMatrix::new_Key(
            ManualMatrixLayout::<IdentB, ScoreMma<AP>>::new(score_matmul_tile_size).create_matrix(),
        )
    }

    fn allocate_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        let value_matmul_tile_size = config.attention_tile_size().to_value_matmul_tile_size();
        KeyValueMatrix::new_Value(
            ManualMatrixLayout::<IdentB, ValueMma<AP>>::new(value_matmul_tile_size).create_matrix(),
        )
    }

    fn allocate_key_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        let value_matmul_tile_size = config.attention_tile_size().to_value_matmul_tile_size();
        KeyValueMatrix::new_Reuse(
            ManualMatrixLayout::<IdentB, ScoreMma<AP>>::new(value_matmul_tile_size).create_matrix(),
        )
    }

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask {
        let score_matmul_tile_size = config.attention_tile_size().to_score_matmul_tile_size();
        ManualMatrixLayout::<IdentCD, ScoreMma<AP>>::new(score_matmul_tile_size).create_matrix()
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
        key: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        key.key_mut().load_from_strided_tile(tile);
    }

    fn load_value<E: Float>(
        tile: &StridedTile<E>,
        value: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        value.value_mut().load_from_strided_tile(tile);
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
