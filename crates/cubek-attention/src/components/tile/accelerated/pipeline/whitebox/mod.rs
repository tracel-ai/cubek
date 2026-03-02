mod fragment_convert;
mod rowaware_matrix;
mod whitebox_accumulator;
mod whitebox_softmax;

pub use whitebox_accumulator::*;
pub use whitebox_softmax::*;

use crate::components::tile::accelerated::pipeline::whitebox::fragment_convert::{
    RegisterFragmentConverter, SmemFragmentConverter,
};

pub type WhiteboxRegisterSoftmaxPipeline<Acc, Lhs> =
    WhiteboxSoftmaxPipeline<Acc, Lhs, RegisterFragmentConverter<Acc, Lhs>>;
pub type WhiteboxSmemSoftmaxPipeline<Acc, Lhs> =
    WhiteboxSoftmaxPipeline<Acc, Lhs, SmemFragmentConverter<Acc, Lhs>>;
