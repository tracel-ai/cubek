use crate::{
    ReducePrecision,
    components::{
        instructions::Item,
        readers::{Reader, ReaderExpand},
    },
};
use cubecl::prelude::*;

#[derive(CubeType)]
pub struct UnitReader<'a, P: ReducePrecision> {
    reader: Reader<'a, P>,
}

#[cube]
#[allow(clippy::len_without_is_empty)]
impl<'a, P: ReducePrecision> UnitReader<'a, P> {
    pub fn new(reader: Reader<'a, P>) -> UnitReader<'a, P> {
        UnitReader::<'a, P> { reader }
    }

    pub fn read(&self, vector_index: usize) -> Item<P> {
        match &self.reader {
            Reader::Parallel(reader) => reader.read_unit(vector_index),
            Reader::Perpendicular(reader) => reader.read_unit(vector_index),
        }
    }

    pub fn length(&self) -> usize {
        match &self.reader {
            Reader::Parallel(reader) => reader.length_unit(),
            Reader::Perpendicular(reader) => reader.length_unit(),
        }
    }
}
