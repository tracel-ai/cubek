use crate::{
    ReducePrecision,
    components::{
        instructions::ReduceCoordinate,
        readers::{Reader, ReaderExpand},
    },
};
use cubecl::prelude::*;

#[derive(CubeType)]
pub struct CubeReader<P: ReducePrecision> {
    reader: Reader<P>,
}

#[cube]
#[allow(clippy::len_without_is_empty)]
impl<P: ReducePrecision> CubeReader<P> {
    pub fn new(reader: Reader<P>) -> CubeReader<P> {
        CubeReader::<P> { reader }
    }

    pub fn read(&self, line_index: u32) -> (Line<P::EI>, ReduceCoordinate) {
        match &self.reader {
            Reader::Parallel(reader) => reader.read_cube(line_index),
            Reader::Perpendicular(reader) => reader.read_cube(line_index),
        }
    }

    pub fn length(&self) -> u32 {
        match &self.reader {
            Reader::Parallel(reader) => reader.length_cube(),
            Reader::Perpendicular(reader) => reader.length_cube(),
        }
    }
}
