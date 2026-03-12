// use cubecl;
// use cubecl::ir::DeviceProperties;
// use cubecl::prelude::*;
// use cubek_matmul::components::CubeDimResource;
// use cubek_std::tile::StridedTile;

// use crate::components::tile::{
//     AccumulatorPipeline, FragmentMask, SoftmaxLayout, SoftmaxPipeline, SoftmaxRowwise,
// };
// use crate::definition::attention_types::{ACC, SM};
// use crate::definition::{
//     AttentionBlueprint, AttentionElems, AttentionPrecision, AttentionSetupError, AttentionTileSize,
//     InvalidConfigError,
// };

use std::fmt::Debug;
use std::hash::Hash;

use cubecl::ir::DeviceProperties;
use cubek_matmul::components::CubeDimResource;
use cubek_std::InvalidConfigError;

use crate::components::tile::TileAttention;
use crate::definition::{
    AttentionBlueprint, AttentionElems, AttentionPrecision, AttentionSetupError, AttentionTileSize,
};

// #[cube]
// pub trait TileAttentionDeprecated<AP: AttentionPrecision>: Send + Sync + 'static {
//     type Config: TileAttentionConfig;
//     type Query: CubeType;
//     type KeyValue: CubeType;
//     type Mask: FragmentMask<Layout = Self::SoftmaxLayout>;

//     // type Softmax: FragmentSoftmax<SM<AP>, Layout = Self::SoftmaxLayout, SoftmaxRowFormat = Self::SoftmaxRow>;
//     type Softmax: SoftmaxPipeline<SM<AP>, Rowwise = Self::SoftmaxRow, Transit = Self::SoftmaxTransit>;
//     type SoftmaxRow: SoftmaxRowwise<SM<AP>, Layout = Self::SoftmaxLayout>;
//     type SoftmaxLayout: SoftmaxLayout;
//     type SoftmaxTransit: CubeType;

//     type Accumulator: AccumulatorPipeline<ACC<AP>, Transit = Self::AccumulatorTransit>;
//     type AccumulatorTransit: CubeType;

//     fn score_matmul(
//         lhs: &Self::Query,
//         rhs: &Self::KeyValue,
//         out: &mut Self::Softmax,
//         #[comptime] config: Self::Config,
//     );

//     fn value_matmul(
//         lhs: &Self::Softmax,
//         rhs: &Self::KeyValue,
//         out: &mut Self::Accumulator,
//         #[comptime] config: Self::Config,
//     );

//     fn allocate_query(#[comptime] config: Self::Config) -> Self::Query;
//     fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask;

//     fn load_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query);
//     fn allocate_key(#[comptime] config: Self::Config) -> Self::KeyValue;
//     fn allocate_value(#[comptime] config: Self::Config) -> Self::KeyValue;
//     fn allocate_key_value(#[comptime] config: Self::Config) -> Self::KeyValue;

//     fn allocate_softmax_transit(#[comptime] config: Self::Config) -> Self::SoftmaxTransit;
//     fn allocate_accumulator_transit(#[comptime] config: Self::Config) -> Self::AccumulatorTransit;

//     fn allocate_softmax(
//         shared: &mut Self::SoftmaxTransit,
//         #[comptime] config: Self::Config,
//     ) -> Self::Softmax;
//     fn allocate_accumulator(
//         shared: &mut Self::AccumulatorTransit,
//         #[comptime] config: Self::Config,
//     ) -> Self::Accumulator;

//     fn load_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query);

//     fn load_key_transposed<E: Float>(
//         tile: &StridedTile<E>,
//         fragment: &mut Self::KeyValue,
//         #[comptime] config: Self::Config,
//     );
//     fn load_value<E: Float>(
//         tile: &StridedTile<E>,
//         fragment: &mut Self::KeyValue,
//         #[comptime] config: Self::Config,
//     );
//     fn load_mask<E: Numeric>(
//         tile: &StridedTile<E>,
//         fragment: &mut Self::Mask,
//         #[comptime] config: Self::Config,
//     );

//     fn write_results<E: Float>(
//         out: &Self::Accumulator,
//         slice: &mut SliceMut<Line<E>>,
//         #[comptime] config: Self::Config,
//     );
// }
