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

pub type WhiteboxRegisterSoftmaxPipeline<AP> =
    WhiteboxSoftmaxPipeline<AP, RegisterFragmentConverter<AP>>;
pub type WhiteboxSmemSoftmaxPipeline<AP> = WhiteboxSoftmaxPipeline<AP, SmemFragmentConverter<AP>>;
