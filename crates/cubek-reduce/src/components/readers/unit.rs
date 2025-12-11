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
impl<P: ReducePrecision> UnitReader<P> {
    pub fn new(reader: Reader<P>) -> UnitReader<P> {
        UnitReader::<P> { reader }
    }

    pub fn read(&self, line_index: u32) -> (Line<P::EI>, ReduceCoordinate) {
        match &self.reader {
            Reader::Parallel(reader) => reader.read_unit(line_index),
            Reader::Perpendicular(reader) => reader.read_unit(line_index),
        }
    }

    pub fn len(&self) -> u32 {
        match &self.reader {
            Reader::Parallel(reader) => reader.len_unit(),
            Reader::Perpendicular(reader) => reader.len_unit(),
        }
    }
}
