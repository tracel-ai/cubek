use super::lowest_coordinate_matching;
use crate::components::{
    instructions::{
        Accumulator, AccumulatorFormat, Item, ReduceOutputMode, ReduceRequirements, ReduceStep,
        Value, ValueExpand,
    },
    precision::ReducePrecision,
};
use cubecl::{
    ir::{ElemType, Type, dialect::math::IsNanOp, interfaces::TypedExt},
    prelude::*,
};

// `E: Numeric` can't call the float-only `IsNan` trait even after a comptime type check, so emit
// the same Cube IR operation directly. Callers keep this inside float-only comptime branches.
#[cube]
fn numeric_is_nan<E: Numeric, N: Size>(item: Vector<E, N>) -> Vector<bool, N> {
    intrinsic!(|scope| {
        let item = item.read_value(scope);
        let out_item = Type::Scalar(ElemType::Bool).with_vector_size(item.vector_size(scope.ctx()));
        let is_nan = IsNanOp::new(scope.ctx_mut(), item);
        scope.register_with_result(&is_nan).into()
    })
}

/// Which end of the value range ranks first.
///
/// A NaN outranks every number in both, so neither is the other's reverse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ValueOrder {
    /// Largest value first, as max ranks.
    Descending,
    /// Smallest value first, as min ranks.
    Ascending,
}

/// The value a slot starts from: the worst this `order` can rank, so any
/// candidate replaces it.
#[cube]
pub(crate) fn extremum_identity<E: Numeric>(#[comptime] order: ValueOrder) -> E {
    let elem_type = elem_type_of::<E>();

    match comptime!(order) {
        // WGSL has no infinity literal, so construct it from its IEEE-754 bits.
        ValueOrder::Descending => {
            if comptime!(elem_type.is_float()) {
                E::cast_from(f32::reinterpret(0xff80_0000u32))
            } else {
                E::min_value()
            }
        }
        ValueOrder::Ascending => {
            if comptime!(elem_type.is_float()) {
                E::cast_from(f32::reinterpret(0x7f80_0000u32))
            } else {
                E::max_value()
            }
        }
    }
}

/// Whether `current` ranks ahead of `candidate` on value alone, NaN aside.
#[cube]
pub(crate) fn ranks_ahead<E: Numeric, N: Size>(
    #[comptime] order: ValueOrder,
    current: Vector<E, N>,
    candidate: Vector<E, N>,
) -> Vector<bool, N> {
    match comptime!(order) {
        ValueOrder::Descending => current.greater_than(&candidate),
        ValueOrder::Ascending => current.less_than(&candidate),
    }
}

#[cube]
pub(crate) fn select_extremum<E: Numeric, N: Size>(
    #[comptime] order: ValueOrder,
    current: Vector<E, N>,
    candidate: Vector<E, N>,
) -> Vector<E, N> {
    let elem_type = elem_type_of::<E>();

    let keep_current = if comptime!(elem_type.is_float()) {
        let current_is_nan = numeric_is_nan(current);
        current_is_nan.or(ranks_ahead::<E, N>(order, current, candidate))
    } else {
        ranks_ahead::<E, N>(order, current, candidate)
    };

    select_many(keep_current, current, candidate)
}

/// As [`select_extremum`], carrying the coordinate that goes with the winner.
///
/// The coordinate tie-break is ascending in both orders: among equal values, and
/// among NaNs, the lowest coordinate wins.
#[cube]
pub(crate) fn select_arg_extremum<E: Numeric, N: Size>(
    #[comptime] order: ValueOrder,
    current: Vector<E, N>,
    current_coord: Vector<u32, N>,
    candidate: Vector<E, N>,
    candidate_coord: Vector<u32, N>,
) -> (Vector<E, N>, Vector<u32, N>) {
    let elem_type = elem_type_of::<E>();

    let keep_current = if comptime!(elem_type.is_float()) {
        let current_is_nan = numeric_is_nan(current);
        let candidate_is_nan = numeric_is_nan(candidate);
        let tied = current
            .equal(&candidate)
            .or(current_is_nan.vec_and(candidate_is_nan));

        select_many(
            tied,
            current_coord.less_than(&candidate_coord),
            current_is_nan.or(ranks_ahead::<E, N>(order, current, candidate)),
        )
    } else {
        select_many(
            current.equal(&candidate),
            current_coord.less_than(&candidate_coord),
            ranks_ahead::<E, N>(order, current, candidate),
        )
    };

    (
        select_many(keep_current, current, candidate),
        select_many(keep_current, current_coord, candidate_coord),
    )
}

/// As [`select_arg_extremum`], for a candidate that comes after everything the
/// accumulator has already seen: the coordinate moves only when the candidate
/// takes the slot outright, so a tie keeps the lower one without comparing
/// coordinates.
///
/// Every test is an ordered comparison, since WGSL does not promise how a NaN
/// compares with itself. Merging accumulators or folding lanes cannot use this:
/// there the candidate's coordinate can be the lower one.
#[cube]
pub(crate) fn advance_arg_extremum<E: Numeric, N: Size>(
    #[comptime] order: ValueOrder,
    current: Vector<E, N>,
    current_coord: Vector<u32, N>,
    candidate: Vector<E, N>,
    candidate_coord: Vector<u32, N>,
) -> (Vector<E, N>, Vector<u32, N>) {
    let elem_type = elem_type_of::<E>();

    let keep_current = if comptime!(elem_type.is_float()) {
        numeric_is_nan(current).or(ranks_ahead::<E, N>(order, current, candidate))
    } else {
        ranks_ahead::<E, N>(order, current, candidate)
    };

    // The accumulator starts at the identity with a coordinate above every real
    // one, and the input can hold that identity, so an untouched slot yields even
    // on a tie.
    let untouched = current_coord.equal(&Vector::new(u32::MAX));
    let keep_coord = select_many(
        untouched,
        Vector::new(false),
        keep_current.or(current.equal(&candidate)),
    );

    (
        select_many(keep_current, current, candidate),
        select_many(keep_coord, current_coord, candidate_coord),
    )
}

#[cube]
pub(crate) fn plane_extremum<E: Numeric, N: Size>(
    #[comptime] order: ValueOrder,
    item: Vector<E, N>,
) -> Vector<E, N> {
    match comptime!(order) {
        ValueOrder::Descending => plane_max(item),
        ValueOrder::Ascending => plane_min(item),
    }
}

#[cube]
pub(crate) fn plane_extremum_propagating_nan<E: Numeric, N: Size>(
    #[comptime] order: ValueOrder,
    item: Vector<E, N>,
) -> Vector<E, N> {
    let elem_type = elem_type_of::<E>();
    let ordered_extreme = plane_extremum::<E, N>(order, item);

    if comptime!(elem_type.is_float()) {
        replace_plane_extreme_with_nan(ordered_extreme, item)
    } else {
        ordered_extreme
    }
}

#[cube]
pub(crate) fn plane_arg_extremum_propagating_nan<E: Numeric, N: Size>(
    #[comptime] order: ValueOrder,
    item: Vector<E, N>,
    coordinate: Vector<u32, N>,
) -> (Vector<E, N>, Vector<u32, N>) {
    let ordered_extreme = plane_extremum::<E, N>(order, item);
    let elem_type = elem_type_of::<E>();

    if comptime!(elem_type.is_float()) {
        replace_plane_arg_extreme_with_nan(ordered_extreme, item, coordinate)
    } else {
        let ordered_coordinate = lowest_coordinate_matching(ordered_extreme, item, coordinate);
        (ordered_extreme, ordered_coordinate)
    }
}

#[cube]
fn replace_plane_extreme_with_nan<E: Numeric, N: Size>(
    ordered_extreme: Vector<E, N>,
    item: Vector<E, N>,
) -> Vector<E, N> {
    let is_nan = numeric_is_nan(item);
    let no_lane = Vector::new(u32::MAX);
    let nan_lane = plane_min(select_many(is_nan, Vector::new(UNIT_POS_X), no_lane));
    let has_nan = nan_lane.not_equal(&no_lane);
    let nan_lane = select_many(has_nan, nan_lane, Vector::new(0u32));
    // Preserve an input NaN for each vector component; synthesizing one is not portable on WGPU.
    let nan_item = shuffle_vector(item, nan_lane);

    select_many(has_nan, nan_item, ordered_extreme)
}

#[cube]
fn replace_plane_arg_extreme_with_nan<E: Numeric, N: Size>(
    ordered_extreme: Vector<E, N>,
    item: Vector<E, N>,
    coordinate: Vector<u32, N>,
) -> (Vector<E, N>, Vector<u32, N>) {
    let ordered_coordinate = lowest_coordinate_matching(ordered_extreme, item, coordinate);
    let is_nan = numeric_is_nan(item);
    let no_coordinate = Vector::new(u32::MAX);
    let nan_coordinate = plane_min(select_many(is_nan, coordinate, no_coordinate));
    let has_nan = nan_coordinate.not_equal(&no_coordinate);
    let is_first_nan = is_nan.vec_and(coordinate.equal(&nan_coordinate));
    let no_lane = Vector::new(u32::MAX);
    let nan_lane = plane_min(select_many(is_first_nan, Vector::new(UNIT_POS_X), no_lane));
    let nan_lane = select_many(has_nan, nan_lane, Vector::new(0u32));
    // Different vector components can choose different source lanes.
    let nan_item = shuffle_vector(item, nan_lane);

    (
        select_many(has_nan, nan_item, ordered_extreme),
        select_many(has_nan, nan_coordinate, ordered_coordinate),
    )
}

#[cube]
fn shuffle_vector<E: Numeric, N: Size>(
    item: Vector<E, N>,
    source_lanes: Vector<u32, N>,
) -> Vector<E, N> {
    let mut shuffled = Vector::empty();
    #[unroll]
    for k in 0..N::value() {
        shuffled.insert(k, plane_shuffle(item.extract(k), source_lanes.extract(k)));
    }
    shuffled
}

/// The implementation [`Max`](super::Max) and [`Min`](super::Min) share: one
/// extremum reduction in the [`ValueOrder`] each ranks by.
#[derive(Debug, CubeType, Clone)]
pub(crate) struct Extremum {
    #[cube(comptime)]
    pub(crate) order: ValueOrder,
    #[cube(comptime)]
    pub(crate) output: ReduceOutputMode,
}

#[cube]
impl Extremum {
    fn identity<E: Numeric>(&self) -> E {
        extremum_identity::<E>(self.order)
    }

    /// As [`Self::insert`], for a candidate that comes after everything the
    /// accumulator has seen. Only the per-element path can promise that.
    fn advance<T: Numeric, N: Size>(
        &self,
        elements: &mut Value<Vector<T, N>>,
        coordinates: &mut Value<Vector<u32, N>>,
        candidate: Vector<T, N>,
        candidate_coord: &Value<Vector<u32, N>>,
    ) {
        let acc = elements.item();

        match candidate_coord {
            Value::None => elements.assign(&Value::new_single(select_extremum::<T, N>(
                self.order, acc, candidate,
            ))),
            Value::Single(coord) => {
                let candidate_coord = coord.unwrap();
                let acc_coord = coordinates.item();
                let (selected, selected_coord) = advance_arg_extremum::<T, N>(
                    self.order,
                    acc,
                    acc_coord,
                    candidate,
                    candidate_coord,
                );
                elements.assign(&Value::new_single(selected));
                coordinates.assign(&Value::new_single(selected_coord));
            }
            Value::Multiple(_) => panic!("an extremum candidate carries at most one coordinate"),
        }
    }

    /// Fold `candidate` into the accumulator, keeping whichever item this order
    /// ranks first per vector element, and its coordinate when it carries one.
    ///
    /// Ties break towards the lower coordinate, matching the CPU reference. A
    /// coordinate-less candidate emits no index arithmetic at all.
    fn insert<T: Numeric, N: Size>(
        &self,
        elements: &mut Value<Vector<T, N>>,
        coordinates: &mut Value<Vector<u32, N>>,
        candidate: Vector<T, N>,
        candidate_coord: &Value<Vector<u32, N>>,
    ) {
        let acc = elements.item();

        match candidate_coord {
            Value::None => elements.assign(&Value::new_single(select_extremum::<T, N>(
                self.order, acc, candidate,
            ))),
            Value::Single(coord) => {
                let candidate_coord = coord.unwrap();
                let acc_coord = coordinates.item();
                let (selected, selected_coord) = select_arg_extremum::<T, N>(
                    self.order,
                    acc,
                    acc_coord,
                    candidate,
                    candidate_coord,
                );
                elements.assign(&Value::new_single(selected));
                coordinates.assign(&Value::new_single(selected_coord));
            }
            Value::Multiple(_) => panic!("an extremum candidate carries at most one coordinate"),
        }
    }

    /// Reduce `item` across the plane to one winning candidate, with its lowest
    /// matching coordinate when the item carries one.
    fn plane_candidate<T: Numeric, N: Size>(
        &self,
        item: Vector<T, N>,
        coordinates: &Value<Vector<u32, N>>,
    ) -> (Vector<T, N>, Value<Vector<u32, N>>) {
        match coordinates {
            Value::None => (
                plane_extremum_propagating_nan::<T, N>(self.order, item),
                Value::new_None(),
            ),
            Value::Single(coord) => {
                let (winning, winning_coord) =
                    plane_arg_extremum_propagating_nan::<T, N>(self.order, item, coord.unwrap());
                (winning, Value::new_single(winning_coord))
            }
            Value::Multiple(_) => panic!("an extremum candidate carries at most one coordinate"),
        }
    }

    /// Collapse the vectorized accumulator lanes down to the final winner and its
    /// coordinate, for the parallel layout.
    ///
    /// The accumulator must have been built with coordinate tracking on.
    fn finalize_with_coords<P: ReducePrecision>(
        &self,
        accumulator: &Accumulator<P>,
    ) -> (P::EA, u32) {
        let vector_size = accumulator.elements.item().vector_size().comptime();

        if vector_size > 1 {
            let mut winning = self.identity::<P::EA>();
            let mut coordinate = u32::MAX.runtime();

            #[unroll]
            for k in 0..vector_size {
                let acc_element = accumulator.elements.item().extract(k);
                let acc_coordinate = accumulator.args.item().extract(k);

                let (selected, selected_coordinate) = select_arg_extremum::<P::EA, Const<1>>(
                    self.order,
                    Vector::<P::EA, Const<1>>::new(winning),
                    Vector::<u32, Const<1>>::new(coordinate),
                    Vector::<P::EA, Const<1>>::new(acc_element),
                    Vector::<u32, Const<1>>::new(acc_coordinate),
                );

                winning = selected.extract(0usize);
                coordinate = selected_coordinate.extract(0usize);
            }

            (winning, coordinate)
        } else {
            (
                accumulator.elements.item().extract(0usize),
                accumulator.args.item().extract(0usize),
            )
        }
    }

    pub(crate) fn requirements(&self) -> ReduceRequirements {
        ReduceRequirements {
            coordinates: comptime!(self.output.has_indices()),
        }
    }

    pub(crate) fn accumulator_format(&self) -> comptime_type!(AccumulatorFormat) {
        AccumulatorFormat::Single
    }

    pub(crate) fn null_input<P: ReducePrecision>(&self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(self.identity::<P::EI>())
    }

    pub(crate) fn null_accumulator<P: ReducePrecision>(&self) -> Accumulator<P> {
        let args = if comptime!(self.output.has_indices()) {
            Value::new_single(Vector::empty().fill(u32::MAX))
        } else {
            Value::new_None()
        };

        Accumulator::<P> {
            elements: Value::new_single(Vector::empty().fill(self.identity::<P::EA>())),
            args,
        }
    }

    pub(crate) fn reduce<P: ReducePrecision>(
        &self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        let (candidate, candidate_coord) = match reduce_step {
            ReduceStep::Plane => self.plane_candidate(item.elements, &item.args),
            ReduceStep::Identity => (item.elements, item.args),
        };

        self.advance(
            &mut accumulator.elements,
            &mut accumulator.args,
            Vector::cast_from(candidate),
            &candidate_coord,
        );
    }

    pub(crate) fn plane_reduce_inplace<P: ReducePrecision>(
        &self,
        accumulator: &mut Accumulator<P>,
    ) {
        let (candidate, candidate_coord) =
            self.plane_candidate(accumulator.elements.item(), &accumulator.args);

        self.insert(
            &mut accumulator.elements,
            &mut accumulator.args,
            candidate,
            &candidate_coord,
        );
    }

    pub(crate) fn fuse_accumulators<P: ReducePrecision>(
        &self,
        accumulator: &mut Accumulator<P>,
        other: &Accumulator<P>,
    ) {
        self.insert(
            &mut accumulator.elements,
            &mut accumulator.args,
            other.elements.item(),
            &other.args,
        );
    }

    pub(crate) fn output_mode(&self) -> comptime_type!(ReduceOutputMode) {
        comptime!(self.output)
    }

    pub(crate) fn to_output_parallel<P: ReducePrecision, Out: Numeric, Idx: Numeric>(
        &self,
        accumulator: Accumulator<P>,
    ) -> (Value<Out>, Value<Idx>) {
        match accumulator.args {
            Value::None => {
                let acc = accumulator.elements.item();
                let mut winning = self.identity::<P::EA>();
                #[unroll]
                for k in 0..acc.vector_size() {
                    let candidate = acc.extract(k);
                    winning = select_extremum::<P::EA, Const<1>>(
                        self.order,
                        Vector::<P::EA, Const<1>>::new(candidate),
                        Vector::<P::EA, Const<1>>::new(winning),
                    )
                    .extract(0usize);
                }
                (
                    Value::new_single(Out::cast_from(winning)),
                    Value::new_None(),
                )
            }
            Value::Single(_) => {
                let (winning, coordinate) = self.finalize_with_coords::<P>(&accumulator);
                (
                    Value::new_single(Out::cast_from(winning)),
                    Value::new_single(Idx::cast_from(coordinate)),
                )
            }
            Value::Multiple(_) => {
                panic!("an extremum accumulator holds at most one coordinate vector")
            }
        }
    }

    pub(crate) fn to_output_perpendicular<P: ReducePrecision, Out: Numeric, Idx: Numeric>(
        &self,
        accumulator: Accumulator<P>,
    ) -> (Value<Vector<Out, P::SI>>, Value<Vector<Idx, P::SI>>) {
        let values = Value::new_single(Vector::cast_from(accumulator.elements.item()));
        let indices = match accumulator.args {
            Value::None => Value::new_None(),
            Value::Single(coord) => Value::new_single(Vector::cast_from(coord.unwrap())),
            Value::Multiple(_) => {
                panic!("an extremum accumulator holds at most one coordinate vector")
            }
        };
        (values, indices)
    }
}
