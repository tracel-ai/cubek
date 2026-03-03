use cubecl;
use cubecl::prelude::*;

use crate::{
    components::tile::{AccumulatorRowwise, SoftmaxLayout, SoftmaxRowwise},
    definition::AttentionTileSize,
};

#[cube]
/// Handles pipelining between the score accumulator, a rowwise intermediate,
/// and the LHS input for the value matmul.
///
/// Converts the score accumulator to a rowwise layout suitable for computations,
/// then to the LHS layout required by the value matmul, performing any necessary
/// type casting between accumulator and LHS fragments.
pub trait SoftmaxPipeline<Acc: Float> {
    /// Format for the score matmul accumulator
    type ScoreAccFormat: CubeType;
    /// Format for the value matmul LHS
    type ValueLhsFormat: CubeType;
    /// Rowwise intermediate (fragment or local tile)
    type Rowwise: SoftmaxRowwise<Acc>;
    /// Should equal Self::Rowwise::Layout
    type Layout: SoftmaxLayout;
    /// Memory used temporarily for casting and/or re-layouting
    /// Can be shared and/or local
    type Transit: CubeType;

    /// Convert accumulator fragment → rowwise intermediate
    fn rowwise_mut(&mut self) -> &mut Self::Rowwise;

    /// Convert rowwise intermediate → LHS fragment
    fn finalize_lhs(&mut self);

    /// Zero out the accumulator
    fn zero(&mut self);

    /// Create the transit component of the pipeline
    fn transit(
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] num_planes: usize,
    ) -> Self::Transit;
}

#[cube]
pub trait AccumulatorPipeline<Acc: Float> {
    /// Format the value matmul uses for accumulator
    type ValueAccFormat: CubeType;
    /// Rowwise intermediate (fragment or local tile)
    type Rowwise: AccumulatorRowwise<Acc>;
    /// Memory used temporarily for casting and/or re-layouting
    /// Can be shared and/or local
    type Transit: CubeType;

    /// Convert accumulator fragment → rowwise intermediate
    fn rowwise_mut(&mut self) -> &mut Self::Rowwise;

    /// Convert rowwise intermediate → back to accumulator fragment
    ///
    /// For “in-place” accumulator computations. May be a no-op or go through smem.
    fn finalize_acc(&mut self);

    /// Zero out the accumulator
    fn zero(&mut self);

    /// Create the transit component of the pipeline
    fn transit(
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] num_planes: usize,
    ) -> Self::Transit;
}
