use crate::components::readers::{GlobalMemoryReader, SharedMemoryReader};
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub enum ReaderType<EA: Float, N: Size> {
    Global(GlobalMemoryReader),
    Shared(SharedMemoryReader<EA, N>),
}

#[cube]
impl<EA: Float, N: Size> ReaderType<EA, N> {
    pub fn read_weighted<EI: Float>(
        &self,
        input: &Tensor<Vector<EI, N>>,
        row: usize,
        col: usize,
        weight: Vector<EA, N>,
    ) -> Vector<EA, N> {
        match self {
            ReaderType::Global(reader) => reader.read_weighted(input, row, col, weight),
            ReaderType::Shared(reader) => reader.read_weighted::<EI>(row, col, weight),
        }
    }
}
