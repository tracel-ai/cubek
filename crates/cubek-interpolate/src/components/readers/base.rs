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
    pub fn read_weighted<EA: Float>(
        &self,
        input: &Tensor<Vector<EI, N>>,
        row: usize,
        col: usize,
        weight: Vector<EA, N>,
    ) -> Vector<EA, N> {
        match self {
            ReaderType::Global(reader) => reader.read_weighted(input, row, col, weight),
            ReaderType::Shared(reader) => reader.read_weighted::<EA>(row, col, weight),
        }
    }
}
