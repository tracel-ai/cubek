use crate::components::readers::{GlobalMemoryReader, SharedMemoryReader};
use cubecl::prelude::*;

#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub enum ReaderType<EI: Float, N: Size> {
    Global(GlobalMemoryReader),
    Shared(SharedMemoryReader<EI, N>),
}

#[cube]
impl<EI: Float, N: Size> ReaderType<EI, N> {
    pub fn read(
        &self,
        input: &Tensor<Vector<EI, N>>,
        row: usize,
        col: usize,
        vector_index: usize,
    ) -> Vector<EI, N> {
        match self {
            ReaderType::Global(reader) => reader.read(input, row, col, vector_index),
            ReaderType::Shared(reader) => reader.read(row, col, vector_index),
        }
    }
}
