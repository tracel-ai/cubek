use crate::{
    ReducePrecision,
    components::{
        instructions::ReduceCoordinate,
        readers::{Reader, ReaderExpand},
    },
};
use cubecl::prelude::*;

#[derive(CubeType)]
pub struct PlaneReader<P: ReducePrecision> {
    reader: Reader<P>,
}

#[cube]
impl<P: ReducePrecision> PlaneReader<P> {
    pub fn new(reader: Reader<P>) -> PlaneReader<P> {
        PlaneReader::<P> { reader }
    }

    pub fn read(&self, line_index: usize) -> (Line<P::EI>, ReduceCoordinate) {
        match &self.reader {
            Reader::Parallel(reader) => reader.read_plane(line_index),
            Reader::Perpendicular(reader) => reader.read_plane(line_index),
        }
    }

    pub fn length(&self) -> usize {
        match &self.reader {
            Reader::Parallel(reader) => reader.length_plane(),
            Reader::Perpendicular(reader) => reader.length_plane(),
        }
    }
}
