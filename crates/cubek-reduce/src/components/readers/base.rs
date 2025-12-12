use crate::{
    BoundChecks, LineMode, ReduceInstruction, ReducePrecision,
    components::{
        instructions::{ReduceCoordinate, ReduceRequirements},
        readers::{parallel::ParallelReader, perpendicular::PerpendicularReader},
    },
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[derive(CubeType)]
pub enum Reader<P: ReducePrecision> {
    Parallel(ParallelReader<P>),
    Perpendicular(PerpendicularReader<P>),
}

#[cube]
impl<P: ReducePrecision> Reader<P> {
    pub fn new<I: ReduceInstruction<P>, Out: Numeric>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        inst: &I,
        reduce_axis: u32,
        reduce_index: u32,
        #[comptime] bound_checks: BoundChecks,
        #[comptime] line_mode: LineMode,
    ) -> Reader<P> {
        match line_mode {
            LineMode::Parallel => Reader::<P>::new_Parallel(ParallelReader::<P>::new::<I, Out>(
                input,
                output,
                inst,
                reduce_axis,
                reduce_index,
                bound_checks,
            )),
            LineMode::Perpendicular => {
                Reader::<P>::new_Perpendicular(PerpendicularReader::<P>::new::<I, Out>(
                    input,
                    output,
                    inst,
                    reduce_axis,
                    reduce_index,
                    bound_checks,
                ))
            }
        }
    }
}

#[cube]
impl ReduceCoordinate {
    pub fn new(
        coordinate: u32,
        requirements: ReduceRequirements,
        #[comptime] line_size: u32,
        #[comptime] line_mode: LineMode,
    ) -> Self {
        if comptime![requirements.coordinates] {
            ReduceCoordinate::new_Required(fill_coordinate_line(coordinate, line_size, line_mode))
        } else {
            ReduceCoordinate::new_NotRequired()
        }
    }
}

// If line mode is parallel, fill a line with `x, x+1, ... x+ line_size - 1` where `x = first`.
// If line mode is perpendicular, fill a line with `x, x, ... x` where `x = first`.
#[cube]
pub(crate) fn fill_coordinate_line(
    first: u32,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
) -> Line<u32> {
    match comptime!(line_mode) {
        LineMode::Parallel => {
            let mut coordinates = Line::empty(line_size);
            #[unroll]
            for j in 0..line_size {
                coordinates[j] = first + j;
            }
            coordinates
        }
        LineMode::Perpendicular => Line::empty(line_size).fill(first),
    }
}
