use cubecl::comptime;
use cubecl::cube;
use cubecl::prelude::*;
use serde::{Deserialize, Serialize};

use crate::components::instructions::{
    Accumulator, AccumulatorExpand, AccumulatorFormat, DynamicSharedAccumulator, Item, Packed,
    Packing, SlotCount, Value, ValueExpand, lowest_coordinate_matching,
};
use crate::{
    ReduceFamily, ReduceInstruction, ReducePrecision,
    components::instructions::{
        ReduceOutputMode, ReduceRequirements, ReduceStep, ReduceWithIndices,
        ReduceWithIndicesFamily,
    },
};
use cubecl::frontend::Numeric;

#[derive_cube_comptime]
#[derive(Serialize, Deserialize)]
pub struct TopKConfig {
    pub k: usize,
    pub output: ReduceOutputMode,
}

#[derive(Debug, CubeType, Clone)]
pub struct TopK {
    #[cube(comptime)]
    pub k: usize,
    #[cube(comptime)]
    pub output: ReduceOutputMode,
}

impl ReduceFamily for TopK {
    type Instruction<P: ReducePrecision> = Self;
    type Config = TopKConfig;
}

impl ReduceWithIndicesFamily for TopK {
    type Instruction<P: ReducePrecision> = Self;
    type Config = TopKConfig;
}

/// Insert `insert_val` into the descending-sorted `elements` (and its
/// coordinate, when it carries one), pushing the smallest slot out.
///
/// Ties break towards the lower coordinate, matching the CPU reference. A
/// coordinate-less candidate emits no index arithmetic at all.
#[cube]
pub(crate) fn topk_insert<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    coordinates: &mut Value<Vector<u32, S>>,
    insert_val: Vector<N, S>,
    insert_coord: &Value<Vector<u32, S>>,
    #[comptime] k: usize,
) {
    let mut insert_val = insert_val;

    match insert_coord {
        Value::None => {
            for j in 0..k {
                let to_keep = elements[j].greater_than(&insert_val);
                let next_val = select_many(to_keep, insert_val, elements[j]);
                elements[j] = select_many(to_keep, elements[j], insert_val);
                insert_val = next_val;
            }
        }
        Value::Single(coord) => {
            let mut insert_coord = coord.unwrap();
            let coords = coordinates.multiple_mut();

            for j in 0..k {
                let to_keep = select_many(
                    elements[j].equal(&insert_val),
                    coords[j].less_than(&insert_coord),
                    elements[j].greater_than(&insert_val),
                );

                let next_val = select_many(to_keep, insert_val, elements[j]);
                elements[j] = select_many(to_keep, elements[j], insert_val);
                insert_val = next_val;

                let next_coord = select_many(to_keep, insert_coord, coords[j]);
                coords[j] = select_many(to_keep, coords[j], insert_coord);
                insert_coord = next_coord;
            }
        }
        Value::Multiple(_) => panic!("a top-k candidate carries at most one coordinate"),
    }
}

#[cube]
impl TopK {
    /// Whether to rank with a [`Packed`], which both call sites below must agree on.
    ///
    /// A single slot is left unpacked: packing pays for itself per slot but builds
    /// its value per element, which one slot never amortizes. On CPU that is worth
    /// 19% against the unpacked insert, and max and min do not get the same choice
    /// because the insert they would fall back to is the dearer of the two.
    fn packs<P: ReducePrecision>(&self) -> comptime_type!(bool) {
        let packs = Packing::packs::<P>(self.output);
        let ranks_several = comptime!(self.k > 1);

        comptime!(packs && ranks_several)
    }
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for TopK {
    type SharedAccumulator = DynamicSharedAccumulator<P>;
    type Config = TopKConfig;

    fn requirements(this: &Self) -> super::ReduceRequirements {
        ReduceRequirements {
            coordinates: comptime!(this.output.has_indices()),
        }
    }

    fn accumulator_format(this: &Self) -> comptime_type!(AccumulatorFormat) {
        let packed = this.packs::<P>();
        let k = comptime!(this.k);

        comptime!(if packed {
            AccumulatorFormat::Packed(SlotCount::Multiple(k))
        } else {
            AccumulatorFormat::Unpacked(SlotCount::Multiple(k))
        })
    }

    fn from_config(#[comptime] config: Self::Config) -> Self {
        TopK {
            k: config.k,
            output: config.output,
        }
    }

    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(P::EI::min_value())
    }

    fn null_accumulator(this: &Self) -> Accumulator<P> {
        let packed = this.packs::<P>();

        if comptime!(packed) {
            let empty =
                Packing::descending().empty::<P::EA, P::SI>(Vector::new(P::EA::min_value()));

            let mut packed = Array::new(comptime!(this.k));
            #[unroll]
            for i in 0..this.k {
                packed[i] = empty;
            }

            Accumulator::new_Packed(Value::new_Multiple(packed))
        } else {
            let mut elements = Array::new(comptime!(this.k));
            #[unroll]
            for i in 0..this.k {
                elements[i] = Vector::new(P::EA::min_value());
            }

            let args = if comptime!(this.output.has_indices()) {
                let mut args = Array::new(comptime!(this.k));
                #[unroll]
                for i in 0..this.k {
                    args[i] = Vector::new(u32::MAX);
                }
                Value::new_Multiple(args)
            } else {
                Value::new_None()
            };

            Accumulator::new_Unpacked(Value::new_Multiple(elements), args)
        }
    }

    fn reduce(
        this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        match accumulator {
            Accumulator::Packed(packed) => {
                let candidate = Packing::descending()
                    .pack::<P::EA, P::SI>(Vector::cast_from(item.elements), item.args.item());
                let packed = packed.multiple_mut();

                match reduce_step {
                    ReduceStep::Plane => {
                        plane_topk_packed_insert::<P::EA, P::SI>(packed, candidate, this.k)
                    }
                    ReduceStep::Identity => {
                        Packing::insert_ranked::<P::SI>(packed, candidate, this.k, false)
                    }
                }
            }
            Accumulator::Unpacked { elements, args } => {
                let slots = elements.multiple_mut();

                match reduce_step {
                    ReduceStep::Plane => {
                        plane_topk_insert::<P::EA, P::SI>(
                            slots,
                            args,
                            Vector::cast_from(item.elements),
                            &item.args,
                            this.k,
                        );
                    }
                    ReduceStep::Identity => {
                        topk_insert::<P::EA, P::SI>(
                            slots,
                            args,
                            Vector::cast_from(item.elements),
                            &item.args,
                            this.k,
                        );
                    }
                }
            }
        }
    }

    fn plane_reduce_inplace(this: &Self, accumulator: &mut Accumulator<P>) {
        match accumulator {
            Accumulator::Packed(packed) => {
                plane_topk_packed_merge::<P::EA, P::SI>(packed.multiple_mut(), this.k)
            }
            Accumulator::Unpacked { elements, args } => {
                plane_topk_merge::<P::EA, P::SI>(elements.multiple_mut(), args, this.k)
            }
        }
    }

    fn fuse_accumulators(this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        match (accumulator, other) {
            (Accumulator::Packed(packed), Accumulator::Packed(other_packed)) => {
                let packed = packed.multiple_mut();
                let other_packed = other_packed.multiple();

                for i in 0..this.k {
                    Packing::insert_ranked::<P::SI>(packed, other_packed[i], this.k, false);
                }
            }
            (
                Accumulator::Unpacked { elements, args },
                Accumulator::Unpacked {
                    elements: other_elements,
                    args: other_args,
                },
            ) => {
                let slots = elements.multiple_mut();
                let other_slots = other_elements.multiple();

                for i in 0..this.k {
                    topk_insert::<P::EA, P::SI>(
                        slots,
                        args,
                        other_slots[i],
                        &other_args.slot(i),
                        this.k,
                    );
                }
            }
            _ => panic!("both accumulators must hold the same representation"),
        }
    }

    fn output_mode(this: &Self) -> comptime_type!(ReduceOutputMode) {
        comptime!(this.output)
    }

    fn to_output_parallel<Out: Numeric, Idx: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> (Value<Out>, Value<Idx>) {
        match accumulator {
            Accumulator::Packed(packed) => {
                let packed = topk_finalize_packed::<P>(packed.multiple(), this.k);

                let mut out_values = Array::new(this.k);
                let mut out_indices = Array::new(this.k);
                #[unroll]
                for i in 0..this.k {
                    let candidate = Vector::<Packed, P::SI>::new(packed[i]);
                    out_values[i] = Out::cast_from(
                        Packing::descending()
                            .value::<P::EA, P::SI>(candidate)
                            .extract(0usize),
                    );
                    out_indices[i] =
                        Idx::cast_from(Packing::coordinate::<P::SI>(candidate).extract(0usize));
                }

                (
                    Value::new_Multiple(out_values),
                    Value::new_Multiple(out_indices),
                )
            }
            Accumulator::Unpacked { elements, args } => match args {
                Value::None => {
                    let values = topk_finalize_values::<P, Out>(&elements, this.k);
                    (Value::new_Multiple(values), Value::new_None())
                }
                Value::Multiple(_) => {
                    let (values, coords) = topk_finalize_with_coords::<P>(&elements, &args, this.k);

                    let mut out_values = Array::new(this.k);
                    let mut out_indices = Array::new(this.k);
                    #[unroll]
                    for i in 0..this.k {
                        out_values[i] = Out::cast_from(values[i]);
                        out_indices[i] = Idx::cast_from(coords[i]);
                    }

                    (
                        Value::new_Multiple(out_values),
                        Value::new_Multiple(out_indices),
                    )
                }
                Value::Single(_) => panic!("top-k accumulator coordinates are one slice per slot"),
            },
        }
    }

    fn to_output_perpendicular<Out: Numeric, Idx: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> (Value<Vector<Out, P::SI>>, Value<Vector<Idx, P::SI>>) {
        match accumulator {
            Accumulator::Packed(packed) => {
                let packed = packed.multiple();

                let mut out_values = Array::new(this.k);
                let mut out_indices = Array::new(this.k);
                #[unroll]
                for i in 0..this.k {
                    out_values[i] =
                        Vector::cast_from(Packing::descending().value::<P::EA, P::SI>(packed[i]));
                    out_indices[i] = Vector::cast_from(Packing::coordinate::<P::SI>(packed[i]));
                }

                (
                    Value::new_Multiple(out_values),
                    Value::new_Multiple(out_indices),
                )
            }
            Accumulator::Unpacked { elements, args } => {
                let acc_values = elements.multiple();
                let mut out_values = Array::new(this.k);
                #[unroll]
                for i in 0..this.k {
                    out_values[i] = Vector::cast_from(acc_values[i]);
                }

                let indices = match &args {
                    Value::None => Value::new_None(),
                    Value::Multiple(acc_args) => {
                        let mut out_indices = Array::new(this.k);
                        #[unroll]
                        for i in 0..this.k {
                            out_indices[i] = Vector::cast_from(acc_args[i]);
                        }
                        Value::new_Multiple(out_indices)
                    }
                    Value::Single(_) => {
                        panic!("top-k accumulator coordinates are one slice per slot")
                    }
                };

                (Value::new_Multiple(out_values), indices)
            }
        }
    }
}

impl<P: ReducePrecision> ReduceWithIndices<P> for TopK {}

/// Collapse the `k * vector_size` accumulator candidates down to the final `k`
/// values, for the parallel (reduce axis is the vectorized axis) layout.
///
/// Coordinates are not tracked, so ties are broken arbitrarily. Use
/// [`topk_finalize_with_coords`] when indices are wanted.
#[cube]
fn topk_finalize_values<P: ReducePrecision, Out: Numeric>(
    elements: &Value<Vector<P::EA, P::SI>>,
    #[comptime] k: usize,
) -> Array<Out> {
    let vals = elements.multiple();
    let vector_size = vals[0].vector_size().comptime();

    let mut topk = Array::new(k);
    #[unroll]
    for slot in 0..k {
        topk[slot] = Out::min_value();
    }

    #[unroll(k * k * vector_size <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
    for i in 0..k {
        #[unroll]
        for j in 0..vector_size {
            let mut element = Out::cast_from(vals[i].extract(j));

            #[unroll(k * k * vector_size <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
            for slot in 0..k {
                let current = topk[slot];
                let keep = current > element;

                topk[slot] = select(keep, current, element);
                element = select(keep, element, current);
            }
        }
    }

    topk
}

/// Collapse the `k * vector_size` accumulator candidates down to the final `k`
/// values *and* their coordinates, for the parallel layout.
///
/// Ties break towards the lower coordinate, matching the CPU reference. The
/// accumulator must have been built with coordinate tracking on.
#[cube]
fn topk_finalize_with_coords<P: ReducePrecision>(
    elements: &Value<Vector<P::EA, P::SI>>,
    args: &Value<Vector<u32, P::SI>>,
    #[comptime] k: usize,
) -> (Array<P::EA>, Array<u32>) {
    let vals = elements.multiple();
    let coords = args.multiple();
    let vector_size = coords[0].vector_size().comptime();

    let mut topk_vals = Array::new(k);
    let mut topk_coords = Array::new(k);

    #[unroll]
    for slot in 0..k {
        topk_vals[slot] = P::EA::min_value();
        topk_coords[slot] = u32::MAX;
    }

    #[unroll(k * k * vector_size <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
    for i in 0..k {
        #[unroll]
        for j in 0..vector_size {
            let mut value = vals[i].extract(j);
            let mut coordinate = coords[i].extract(j);

            #[unroll(k * k * vector_size <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
            for slot in 0..k {
                let current_value = topk_vals[slot];
                let current_coordinate = topk_coords[slot];

                let to_keep = select(
                    current_value == value,
                    current_coordinate < coordinate,
                    current_value > value,
                );

                topk_vals[slot] = select(to_keep, current_value, value);
                topk_coords[slot] = select(to_keep, current_coordinate, coordinate);

                value = select(to_keep, value, current_value);
                coordinate = select(to_keep, coordinate, current_coordinate);
            }
        }
    }

    (topk_vals, topk_coords)
}

/// [`topk_finalize_with_coords`] over packed packed, for the parallel layout.
#[cube]
fn topk_finalize_packed<P: ReducePrecision>(
    packed: &Array<Vector<Packed, P::SI>>,
    #[comptime] k: usize,
) -> Array<Packed> {
    let vector_size = packed[0].vector_size().comptime();

    let empty = Packing::descending()
        .empty::<P::EA, P::SI>(Vector::new(P::EA::min_value()))
        .extract(0usize);

    let mut topk = Array::new(k);
    #[unroll]
    for slot in 0..k {
        topk[slot] = empty;
    }

    #[unroll(k * k * vector_size <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
    for i in 0..k {
        #[unroll]
        for j in 0..vector_size {
            let mut candidate = packed[i].extract(j);

            #[unroll(k * k * vector_size <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
            for slot in 0..k {
                let current = topk[slot];
                let keep = current > candidate;

                topk[slot] = select(keep, current, candidate);
                candidate = select(keep, candidate, current);
            }
        }
    }

    topk
}

/// How much fully-unrolled top-k selection work is worth emitting, counted in
/// copies of a loop body.
///
/// The selection networks below are `k`-by-`k`: every candidate walks all `k`
/// accumulator slots. Unrolling both levels keeps the accumulator in registers
/// with constant slot indices, which is worth a lot — rolled, the slots move to
/// scratch memory and every step becomes a load/store. Measured on a
/// `32x512x4095` `ArgTopK` (device timing, RTX 5090 / Vulkan), unrolling is
/// worth 5.8x at `k = 32` and 2.2x at `k = 64`.
///
/// But the emitted kernel grows with the product, so a flat cap on `k` prices
/// the three nest shapes wrong: `topk_finalize_*` is `k * k * vector_size`, not
/// `k * k`, and is the first to become unaffordable. Budgeting the product
/// instead lets the square nests unroll further than the cubic one, and stops
/// all of them before the backend compiler does: past this budget the same
/// selection runs as a plain runtime loop whose kernel size does not depend on
/// `k` at all.
pub(crate) const TOPK_UNROLL_BUDGET: usize = 1024;

/// Plane-cooperative top-k insertion; the candidate's coordinate decides which
/// algorithm runs, since winners are identified by their coordinate when one
/// rides along and by lane id otherwise.
#[cube]
pub fn plane_topk_insert<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    coordinates: &mut Value<Vector<u32, S>>,
    item: Vector<N, S>,
    coord: &Value<Vector<u32, S>>,
    #[comptime] k: usize,
) {
    match coord {
        Value::None => plane_topk_insert_values(elements, item, k),
        Value::Single(coord) => plane_topk_insert_with_coords(
            elements,
            coordinates.multiple_mut(),
            item,
            coord.unwrap(),
            k,
        ),
        Value::Multiple(_) => panic!("a top-k candidate carries at most one coordinate"),
    }
}

#[cube]
fn plane_topk_insert_with_coords<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    coordinates: &mut Array<Vector<u32, S>>,
    item: Vector<N, S>,
    coord: Vector<u32, S>,
    #[comptime] k: usize,
) {
    let mut local_best_val = item;
    let mut local_best_coord = coord;

    #[unroll(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
    for _i in 0..k {
        let winning_val = plane_max(local_best_val);
        let winning_coord =
            lowest_coordinate_matching(winning_val, local_best_val, local_best_coord);

        let mut insert_val = winning_val;
        let mut insert_coord = winning_coord;

        #[unroll(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
        for j in 0..k {
            let to_keep = select_many(
                elements[j].equal(&insert_val),
                coordinates[j].less_than(&insert_coord),
                elements[j].greater_than(&insert_val),
            );

            let next_val = select_many(to_keep, insert_val, elements[j]);
            elements[j] = select_many(to_keep, elements[j], insert_val);
            insert_val = next_val;

            let next_coord = select_many(to_keep, insert_coord, coordinates[j]);
            coordinates[j] = select_many(to_keep, coordinates[j], insert_coord);
            insert_coord = next_coord;
        }

        // Winner masking logic
        let is_winner = local_best_val
            .equal(&winning_val)
            .vec_and(local_best_coord.equal(&winning_coord));
        local_best_val = select_many(is_winner, Vector::new(N::min_value()), local_best_val);
        local_best_coord = select_many(is_winner, Vector::new(u32::MAX), local_best_coord);
    }
}

#[cube]
fn plane_topk_insert_values<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    item: Vector<N, S>,
    #[comptime] k: usize,
) {
    let mut local_best_val = item;
    let lane_id = Vector::new(UNIT_POS_X);

    #[unroll(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
    for _i in 0..k {
        let winning_val = plane_max(local_best_val);
        let is_match = local_best_val.equal(&winning_val);
        let winning_lane = plane_min(select_many(is_match, lane_id, Vector::new(u32::MAX)));

        let mut insert_val = winning_val;

        #[unroll(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
        for j in 0..k {
            let to_keep = elements[j].greater_than(&insert_val);
            let next_val = select_many(to_keep, insert_val, elements[j]);
            elements[j] = select_many(to_keep, elements[j], insert_val);
            insert_val = next_val;
        }

        // Winner masking logic
        let is_winner = lane_id.equal(&winning_lane);
        local_best_val = select_many(is_winner, Vector::new(N::min_value()), local_best_val);
    }
}

/// Plane-cooperative insertion of one packed-candidate candidate per lane.
///
/// A candidate already carries the tie-break, so the plane's winner is a plain
/// [`plane_max`] and the lane holding it is the lane whose candidate equals it: two
/// lanes cannot hold the same candidate, since the coordinate is part of it.
#[cube]
pub fn plane_topk_packed_insert<N: Numeric, S: Size>(
    packed: &mut Array<Vector<Packed, S>>,
    item: Vector<Packed, S>,
    #[comptime] k: usize,
) {
    let mut local_best = item;

    #[unroll(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
    for _i in 0..k {
        let winning = plane_max(local_best);
        Packing::insert_ranked::<S>(
            packed,
            winning,
            k,
            comptime!(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET),
        );

        let is_winner = local_best.equal(&winning);
        local_best = select_many(
            is_winner,
            Packing::descending().empty::<N, S>(Vector::new(N::min_value())),
            local_best,
        );
    }
}

/// Plane-cooperative merge of per-lane packed-candidate accumulators.
#[cube]
pub fn plane_topk_packed_merge<N: Numeric, S: Size>(
    packed: &mut Array<Vector<Packed, S>>,
    #[comptime] k: usize,
) {
    let mut final_keys = Array::new(k);
    let mut cursor = Vector::new(0u32);
    let lane_id = Vector::new(UNIT_POS_X);

    #[unroll(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
    for i in 0..k {
        let mut local = Packing::descending().empty::<N, S>(Vector::new(N::min_value()));

        #[unroll(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
        for j in 0..k {
            let is_pointed = cursor.equal(&Vector::new(j as u32));
            local = select_many(is_pointed, packed[j], local);
        }

        let winning = plane_max(local);
        final_keys[i] = winning;

        let is_cand = local.equal(&winning);
        let winning_lane = plane_min(select_many(is_cand, lane_id, Vector::new(u32::MAX)));
        let is_winner_thread = lane_id.equal(&winning_lane);
        cursor = select_many(is_winner_thread, cursor + Vector::new(1u32), cursor);
    }

    #[unroll]
    for i in 0..k {
        packed[i] = final_keys[i];
    }
}

/// Plane-cooperative merge of per-lane top-k candidates; the accumulator's
/// coordinates decide which algorithm runs, as in [`plane_topk_insert`].
#[cube]
pub fn plane_topk_merge<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    coordinates: &mut Value<Vector<u32, S>>,
    #[comptime] k: usize,
) {
    match coordinates {
        Value::None => plane_topk_merge_values(elements, k),
        Value::Multiple(coordinates) => plane_topk_merge_with_coords(elements, coordinates, k),
        Value::Single(_) => panic!("top-k accumulator coordinates are one slice per slot"),
    }
}

#[cube]
fn plane_topk_merge_with_coords<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    coordinates: &mut Array<Vector<u32, S>>,
    #[comptime] k: usize,
) {
    let mut final_elements = Array::new(k);
    let mut final_coords = Array::new(k);
    let mut cursor = Vector::new(0u32);
    let lane_id = Vector::new(UNIT_POS_X);

    #[unroll(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
    for i in 0..k {
        let mut local_val = Vector::new(N::min_value());
        let mut local_coord = Vector::new(u32::MAX);

        #[unroll(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
        for j in 0..k {
            let is_pointed = cursor.equal(&Vector::new(j as u32));
            local_val = select_many(is_pointed, elements[j], local_val);
            local_coord = select_many(is_pointed, coordinates[j], local_coord);
        }

        let winning_val = plane_max(local_val);
        let best_c = lowest_coordinate_matching(winning_val, local_val, local_coord);
        final_coords[i] = best_c;
        let is_cand = local_val
            .equal(&winning_val)
            .vec_and(local_coord.equal(&best_c));
        let winning_lane = plane_min(select_many(is_cand, lane_id, Vector::new(u32::MAX)));

        final_elements[i] = winning_val;
        let is_winner_thread = lane_id.equal(&winning_lane);
        cursor = select_many(is_winner_thread, cursor + Vector::new(1u32), cursor);
    }

    #[unroll]
    for i in 0..k {
        elements[i] = final_elements[i];
        coordinates[i] = final_coords[i];
    }
}

#[cube]
fn plane_topk_merge_values<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    #[comptime] k: usize,
) {
    let mut final_elements = Array::new(k);
    let mut cursor = Vector::new(0u32);
    let lane_id = Vector::new(UNIT_POS_X);

    #[unroll(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
    for i in 0..k {
        let mut local_val = Vector::new(N::min_value());

        #[unroll(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
        for j in 0..k {
            let is_pointed = cursor.equal(&Vector::new(j as u32));
            local_val = select_many(is_pointed, elements[j], local_val);
        }

        let winning_val = plane_max(local_val);
        let is_cand = local_val.equal(&winning_val);
        let winning_lane = plane_min(select_many(is_cand, lane_id, Vector::new(u32::MAX)));

        final_elements[i] = winning_val;
        let is_winner_thread = lane_id.equal(&winning_lane);
        cursor = select_many(is_winner_thread, cursor + Vector::new(1u32), cursor);
    }

    #[unroll]
    for i in 0..k {
        elements[i] = final_elements[i];
    }
}
