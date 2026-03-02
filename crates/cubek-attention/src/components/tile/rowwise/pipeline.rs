use cubecl;
use cubecl::prelude::*;

use crate::components::tile::{AccumulatorRowwise, SoftmaxLayout, SoftmaxRowwise};

#[cube]
/// Handles pipelining between the score accumulator, a rowwise intermediate,
/// and the LHS input for the value matmul.
///
/// Converts the score accumulator to a rowwise layout suitable for computations,
/// then to the LHS layout required by the value matmul, performing any necessary
/// type casting between accumulator and LHS fragments.
pub trait SoftmaxPipeline<Acc: Float> {
    /// Original accumulator fragment
    type MatmulAccumulator: CubeType;
    /// Final output for softmax / LHS
    type MatmulLhs: CubeType;
    /// Rowwise intermediate (fragment or local tile)
    type Rowwise: SoftmaxRowwise<Acc>;
    /// Should equal Self::Rowwise::Layout
    type SoftmaxLayout: SoftmaxLayout;

    /// Convert accumulator fragment → rowwise intermediate
    fn rowwise_mut(&mut self) -> &mut Self::Rowwise;

    /// Convert rowwise intermediate → LHS fragment
    fn finalize_lhs(&mut self);

    /// Zero out the accumulator
    fn zero(&mut self);
}

#[cube]
pub trait AccumulatorPipeline<Acc: Float> {
    /// Original accumulator fragment
    type MatmulAccumulator: CubeType;
    /// Rowwise intermediate (fragment or local tile)
    type Rowwise: AccumulatorRowwise<Acc>;

    /// Convert accumulator fragment → rowwise intermediate
    fn rowwise_mut(&mut self) -> &mut Self::Rowwise;

    /// Convert rowwise intermediate → back to accumulator fragment
    ///
    /// For “in-place” accumulator computations. May be a no-op or go through smem.
    fn finalize_acc(&mut self);

    /// Zero out the accumulator
    fn zero(&mut self);
}
