use crate::{
    ReducePrecision,
    components::{
        instructions::ReduceCoordinate,
        readers::{Reader, ReaderExpand},
    },
};
use cubecl::prelude::*;

#[derive(CubeType)]
pub struct UnitReader<P: ReducePrecision> {
    reader: Reader<P>,
}

#[cube]
#[allow(clippy::len_without_is_empty)]
impl<P: ReducePrecision> UnitReader<P> {
    pub fn new(reader: Reader<P>) -> UnitReader<P> {
        UnitReader::<P> { reader }
    }

    pub fn read(&self, line_index: usize) -> (Line<P::EI>, ReduceCoordinate) {
        match &self.reader {
            Reader::Parallel(reader) => reader.read_unit(line_index),
            Reader::Perpendicular(reader) => reader.read_unit(line_index),
        }
    }

    pub fn length(&self) -> usize {
        match &self.reader {
            Reader::Parallel(reader) => reader.length_unit(),
            Reader::Perpendicular(reader) => reader.length_unit(),
        }
    }
}
