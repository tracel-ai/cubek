mod attention;
mod fragment_convert;
mod manual_matrix;
mod setup;
mod whitebox_accumulator;
mod whitebox_softmax;

pub use attention::*;
pub use whitebox_accumulator::*;
pub use whitebox_softmax::*;

use crate::components::tile::accelerated_whitebox::fragment_convert::{
    RegisterFragmentConverter, SmemFragmentConverter,
};

pub type WhiteboxRegisterSoftmaxPipeline<Acc, Lhs> =
    WhiteboxSoftmaxPipeline<Acc, Lhs, RegisterFragmentConverter<Acc, Lhs>>;
pub type WhiteboxSmemSoftmaxPipeline<Acc, Lhs> =
    WhiteboxSoftmaxPipeline<Acc, Lhs, SmemFragmentConverter<Acc, Lhs>>;
