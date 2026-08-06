use super::lowest_coordinate_matching;
use cubecl::{
    ir::{Comparison, ElemType, Instruction, Type, UnaryOperands},
    prelude::*,
};

// `E: Numeric` can't call the float-only `IsNan` trait even after a comptime type check, so emit
// the same Cube IR operation directly. Callers keep this inside float-only comptime branches.
#[cube]
fn numeric_is_nan<E: Numeric, N: Size>(item: Vector<E, N>) -> Vector<bool, N> {
    intrinsic!(|scope| {
        let out_item = Type::scalar(ElemType::Bool).with_vector_size(item.expand.ty.vector_size());
        let out = scope.create_value(out_item);
        scope.register(Instruction::new(
            Comparison::IsNan(UnaryOperands { input: item.expand }),
            out,
        ));
        out.into()
    })
}

#[cube]
pub(crate) fn max_identity<E: Numeric>() -> E {
    let elem_type = type_of::<E>();
    if comptime!(elem_type.is_float()) {
        // WGSL has no infinity literal, so construct it from its IEEE-754 bits.
        E::cast_from(f32::reinterpret(0xff80_0000u32))
    } else {
        E::min_value()
    }
}

#[cube]
pub(crate) fn min_identity<E: Numeric>() -> E {
    let elem_type = type_of::<E>();
    if comptime!(elem_type.is_float()) {
        E::cast_from(f32::reinterpret(0x7f80_0000u32))
    } else {
        E::max_value()
    }
}

#[cube]
pub(crate) fn select_max<E: Numeric, N: Size>(
    current: Vector<E, N>,
    candidate: Vector<E, N>,
) -> Vector<E, N> {
    let elem_type = type_of::<E>();
    if comptime!(elem_type.is_float()) {
        let current_is_nan = numeric_is_nan(current);
        let keep_current = current_is_nan.or(current.greater_than(&candidate));
        select_many(keep_current, current, candidate)
    } else {
        select_many(current.greater_than(&candidate), current, candidate)
    }
}

#[cube]
pub(crate) fn select_min<E: Numeric, N: Size>(
    current: Vector<E, N>,
    candidate: Vector<E, N>,
) -> Vector<E, N> {
    let elem_type = type_of::<E>();
    if comptime!(elem_type.is_float()) {
        let current_is_nan = numeric_is_nan(current);
        let keep_current = current_is_nan.or(current.less_than(&candidate));
        select_many(keep_current, current, candidate)
    } else {
        select_many(current.less_than(&candidate), current, candidate)
    }
}

#[cube]
pub(crate) fn select_argmax<E: Numeric, N: Size>(
    current: Vector<E, N>,
    current_coord: Vector<u32, N>,
    candidate: Vector<E, N>,
    candidate_coord: Vector<u32, N>,
) -> (Vector<E, N>, Vector<u32, N>) {
    let elem_type = type_of::<E>();
    let keep_current = if comptime!(elem_type.is_float()) {
        let current_is_nan = numeric_is_nan(current);
        let candidate_is_nan = numeric_is_nan(candidate);
        let tied = current
            .equal(&candidate)
            .or(current_is_nan.vec_and(candidate_is_nan));

        select_many(
            tied,
            current_coord.less_than(&candidate_coord),
            current_is_nan.or(current.greater_than(&candidate)),
        )
    } else {
        select_many(
            current.equal(&candidate),
            current_coord.less_than(&candidate_coord),
            current.greater_than(&candidate),
        )
    };

    (
        select_many(keep_current, current, candidate),
        select_many(keep_current, current_coord, candidate_coord),
    )
}

#[cube]
pub(crate) fn select_argmin<E: Numeric, N: Size>(
    current: Vector<E, N>,
    current_coord: Vector<u32, N>,
    candidate: Vector<E, N>,
    candidate_coord: Vector<u32, N>,
) -> (Vector<E, N>, Vector<u32, N>) {
    let elem_type = type_of::<E>();
    let keep_current = if comptime!(elem_type.is_float()) {
        let current_is_nan = numeric_is_nan(current);
        let candidate_is_nan = numeric_is_nan(candidate);
        let tied = current
            .equal(&candidate)
            .or(current_is_nan.vec_and(candidate_is_nan));

        select_many(
            tied,
            current_coord.less_than(&candidate_coord),
            current_is_nan.or(current.less_than(&candidate)),
        )
    } else {
        select_many(
            current.equal(&candidate),
            current_coord.less_than(&candidate_coord),
            current.less_than(&candidate),
        )
    };

    (
        select_many(keep_current, current, candidate),
        select_many(keep_current, current_coord, candidate_coord),
    )
}

#[cube]
pub(crate) fn plane_max_propagating_nan<E: Numeric, N: Size>(item: Vector<E, N>) -> Vector<E, N> {
    let elem_type = type_of::<E>();
    if comptime!(elem_type.is_float()) {
        replace_plane_extreme_with_nan(plane_max(item), item)
    } else {
        plane_max(item)
    }
}

#[cube]
pub(crate) fn plane_min_propagating_nan<E: Numeric, N: Size>(item: Vector<E, N>) -> Vector<E, N> {
    let elem_type = type_of::<E>();
    if comptime!(elem_type.is_float()) {
        replace_plane_extreme_with_nan(plane_min(item), item)
    } else {
        plane_min(item)
    }
}

#[cube]
pub(crate) fn plane_argmax_propagating_nan<E: Numeric, N: Size>(
    item: Vector<E, N>,
    coordinate: Vector<u32, N>,
) -> (Vector<E, N>, Vector<u32, N>) {
    let ordered_extreme = plane_max(item);
    let elem_type = type_of::<E>();
    if comptime!(elem_type.is_float()) {
        replace_plane_arg_extreme_with_nan(ordered_extreme, item, coordinate)
    } else {
        let ordered_coordinate = lowest_coordinate_matching(ordered_extreme, item, coordinate);
        (ordered_extreme, ordered_coordinate)
    }
}

#[cube]
pub(crate) fn plane_argmin_propagating_nan<E: Numeric, N: Size>(
    item: Vector<E, N>,
    coordinate: Vector<u32, N>,
) -> (Vector<E, N>, Vector<u32, N>) {
    let ordered_extreme = plane_min(item);
    let elem_type = type_of::<E>();
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
