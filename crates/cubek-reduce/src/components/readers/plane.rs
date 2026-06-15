use crate::{
    ReducePrecision,
    components::{
        instructions::Item,
        readers::{Reader, ReaderExpand},
    },
};
use cubecl::prelude::*;

#[derive(CubeType)]
pub struct PlaneReader<'a, P: ReducePrecision> {
    reader: Reader<'a, P>,
}

#[cube]
impl<'a, P: ReducePrecision> PlaneReader<'a, P> {
    pub fn new(reader: Reader<'a, P>) -> PlaneReader<'a, P> {
        PlaneReader::<'a, P> { reader }
    }

    pub fn read(&self, vector_index: usize) -> Item<P> {
        match &self.reader {
            Reader::Parallel(reader) => reader.read_plane(vector_index),
            Reader::Perpendicular(reader) => reader.read_plane(vector_index),
        }
    }

    pub fn length(&self) -> usize {
        match &self.reader {
            Reader::Parallel(reader) => reader.length_plane(),
            Reader::Perpendicular(reader) => reader.length_plane(),
        }
    }
}
