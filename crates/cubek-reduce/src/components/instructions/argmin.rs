use super::{
    ArgAccumulator, ReduceCoordinate, ReduceCoordinateExpand, ReduceFamily, ReduceInstruction,
    ReduceRequirements, lowest_coordinate_matching,
};
use crate::components::precision::ReducePrecision;
use cubecl::prelude::*;

/// Compute the coordinate of the maximum item returning the smallest coordinate in case of equality.
#[derive(Debug, CubeType, Clone)]
pub struct ArgMin {}

impl ReduceFamily for ArgMin {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ();
}

#[cube]
impl ArgMin {
    /// Compare two pairs of items and coordinates and return a new pair
    /// where each element in the vectors is the minimal item with its coordinate.
    /// In case of equality, the lowest coordinate is selected.
    pub fn choose_argmin<T: Numeric, N: Size>(
        items0: Vector<T, N>,
        coordinates0: Vector<u32, N>,
        items1: Vector<T, N>,
        coordinates1: Vector<u32, N>,
    ) -> (Vector<T, N>, Vector<u32, N>) {
        let to_keep = select_many(
            items0.equal(items1),
            coordinates0.less_than(coordinates1),
            items0.less_than(items1),
        );
        let items = select_many(to_keep, items0, items1);
        let coordinates = select_many(to_keep, coordinates0, coordinates1);
        (items, coordinates)
    }
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for ArgMin {
    type AccumulatorItem = (Vector<P::EA, P::SI>, Vector<u32, P::SI>);
    type SharedAccumulator = ArgAccumulator<P::EA, P::SI>;
    type Config = ();

    fn requirements(_this: &Self) -> ReduceRequirements {
        ReduceRequirements { coordinates: true }
    }
    fn from_config(_config: Self::Config) -> Self {
        ArgMin {}
    }

    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(P::EI::max_value())
    }

    fn null_accumulator(_this: &Self) -> Self::AccumulatorItem {
        (
            Vector::empty().fill(P::EA::max_value()),
            Vector::empty().fill(u32::MAX),
        )
    }

    fn assign_accumulator(
        _this: &Self,
        destination: &mut Self::AccumulatorItem,
        source: &Self::AccumulatorItem,
    ) {
        destination.0 = source.0;
        destination.1 = source.1;
    }

    fn read_accumulator(
        _this: &Self,
        accumulator: &Self::AccumulatorItem,
    ) -> (Vector<P::EI, P::SI>, ReduceCoordinate<P::SI>) {
        (
            Vector::cast_from(accumulator.0),
            ReduceCoordinate::new_Required(accumulator.1),
        )
    }

    fn reduce(
        _this: &Self,
        accumulator: &Self::AccumulatorItem,
        item: Vector<P::EI, P::SI>,
        coordinate: ReduceCoordinate<P::SI>,
        #[comptime] use_planes: bool,
    ) -> Self::AccumulatorItem {
        #[comptime]
        let coordinate = match coordinate {
            ReduceCoordinate::Required(val) => val,
            ReduceCoordinate::NotRequired => {
                comptime! {panic!("Coordinates are required for ArgMin")};
                #[allow(unreachable_code)]
                Vector::new(0)
            }
        };

        let (candidate_item, candidate_coordinate) = if use_planes {
            let candidate_item = plane_min(item);
            let candidate_coordinate = lowest_coordinate_matching(candidate_item, item, coordinate);
            (candidate_item, candidate_coordinate)
        } else {
            (item, coordinate)
        };

        Self::choose_argmin(
            accumulator.0,
            accumulator.1,
            Vector::cast_from(candidate_item),
            candidate_coordinate,
        )
    }

    fn fuse_accumulators(
        _this: &Self,
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem {
        Self::choose_argmin(lhs.0, lhs.1, rhs.0, rhs.1)
    }

    fn merge_vector<Out: Numeric>(
        _this: &Self,
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: usize,
    ) -> Out {
        let vector_size = accumulator.0.size().comptime();
        if vector_size > 1 {
            let mut min = P::EA::max_value();
            let mut coordinate = u32::MAX.runtime();

            #[unroll]
            for k in 0..vector_size {
                let acc_element = accumulator.0[k];
                let acc_coordinate = accumulator.1[k];
                // TODO replace with select
                if acc_element == min && acc_coordinate < coordinate {
                    coordinate = acc_coordinate;
                } else if acc_element < min {
                    min = acc_element;
                    coordinate = acc_coordinate;
                }
            }
            Out::cast_from(coordinate)
        } else {
            Out::cast_from(accumulator.1)
        }
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: usize,
    ) -> Vector<Out, P::SI> {
        Vector::cast_from(accumulator.1)
    }
}
