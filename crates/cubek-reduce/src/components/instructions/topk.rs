use cubecl::comptime;
use cubecl::cube;
use cubecl::prelude::*;
use serde::{Deserialize, Serialize};

use crate::components::instructions::{
    Accumulator, AccumulatorFormat, Item, Value, ValueExpand, lowest_coordinate_matching,
};
use crate::{
    ReduceFamily, ReduceInstruction, ReducePrecision,
    components::instructions::{
        ReduceOutputMode, ReduceRequirements, ReduceStep, ReduceWithIndices,
        ReduceWithIndicesFamily, SharedAccumulator,
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

/// Whether any lane of `item` reaches the list slot `kth`: reaches, not only
/// beats, so a tie still goes through the insertion and its coordinate rule.
///
/// The negation is load-bearing and is not `>=`: this guard only skips what
/// provably cannot enter, so a NaN, which compares unordered against every
/// slot, has to reach the insertion the way it did before the guard existed.
/// There it takes slot 0 (`elements[j] > NaN` is false) and carries its own
/// coordinate out. Under `>=` a NaN would fail the guard instead, and an
/// all-NaN row would insert nothing and emit the `u32::MAX` null-accumulator
/// sentinel as its index.
#[cube]
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn reaches<N: Numeric, S: Size>(item: Vector<N, S>, kth: Vector<N, S>) -> bool {
    let mut any = false;
    #[unroll]
    for i in 0..item.vector_size().comptime() {
        if !(item.extract(i) < kth.extract(i)) {
            any = true;
        }
    }
    any
}

#[cube]
impl TopK {
    /// Insert `insert_val` into the descending-sorted `elements` (and its
    /// coordinate, when it carries one), pushing the smallest slot out.
    ///
    /// Ties break towards the lower coordinate, matching the CPU reference. A
    /// coordinate-less candidate emits no index arithmetic at all.
    ///
    /// A candidate that reaches no lane's last kept slot changes nothing and skips
    /// the `k`-slot walk: over a long row, almost every candidate.
    fn insert<N: Numeric, S: Size>(
        &self,
        elements: &mut Array<Vector<N, S>>,
        coordinates: &mut Value<Vector<u32, S>>,
        insert_val: Vector<N, S>,
        insert_coord: &Value<Vector<u32, S>>,
    ) {
        let k = comptime!(self.k);

        if reaches(insert_val, elements[k - 1]) {
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
    }

    /// Collapse the `k * vector_size` accumulator candidates down to the final `k`
    /// values, for the parallel (reduce axis is the vectorized axis) layout.
    ///
    /// Coordinates are not tracked, so ties are broken arbitrarily. Use
    /// [`Self::finalize_with_coords`] when indices are wanted.
    fn finalize_values<P: ReducePrecision, Out: Numeric>(
        &self,
        accumulator: &Accumulator<P>,
    ) -> Array<Out> {
        let k = comptime!(self.k);
        let vals = accumulator.elements.multiple();
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
    fn finalize_with_coords<P: ReducePrecision>(
        &self,
        accumulator: &Accumulator<P>,
    ) -> (Array<P::EA>, Array<u32>) {
        let k = comptime!(self.k);
        let vals = accumulator.elements.multiple();
        let coords = accumulator.args.multiple();
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

    pub(crate) fn plane_insert<N: Numeric, S: Size>(
        &self,
        elements: &mut Array<Vector<N, S>>,
        coordinates: &mut Value<Vector<u32, S>>,
        item: Vector<N, S>,
        coord: &Value<Vector<u32, S>>,
    ) {
        let k = comptime!(self.k);

        if plane_any(reaches(item, elements[k - 1])) {
            match coord {
                Value::None => self.plane_insert_values(elements, item),
                Value::Single(coord) => self.plane_insert_with_coords(
                    elements,
                    coordinates.multiple_mut(),
                    item,
                    coord.unwrap(),
                ),
                Value::Multiple(_) => panic!("a top-k candidate carries at most one coordinate"),
            }
        }
    }

    fn plane_insert_with_coords<N: Numeric, S: Size>(
        &self,
        elements: &mut Array<Vector<N, S>>,
        coordinates: &mut Array<Vector<u32, S>>,
        item: Vector<N, S>,
        coord: Vector<u32, S>,
    ) {
        let k = comptime!(self.k);
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

    fn plane_insert_values<N: Numeric, S: Size>(
        &self,
        elements: &mut Array<Vector<N, S>>,
        item: Vector<N, S>,
    ) {
        let k = comptime!(self.k);
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

    pub(crate) fn plane_merge<N: Numeric, S: Size>(
        &self,
        elements: &mut Array<Vector<N, S>>,
        coordinates: &mut Value<Vector<u32, S>>,
    ) {
        match coordinates {
            Value::None => self.plane_merge_values(elements),
            Value::Multiple(coordinates) => self.plane_merge_with_coords(elements, coordinates),
            Value::Single(_) => panic!("top-k accumulator coordinates are one slice per slot"),
        }
    }

    fn plane_merge_with_coords<N: Numeric, S: Size>(
        &self,
        elements: &mut Array<Vector<N, S>>,
        coordinates: &mut Array<Vector<u32, S>>,
    ) {
        let k = comptime!(self.k);
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

    fn plane_merge_values<N: Numeric, S: Size>(&self, elements: &mut Array<Vector<N, S>>) {
        let k = comptime!(self.k);
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
}

#[derive(CubeType)]
pub struct TopKSharedAccumulator<P: ReducePrecision> {
    elements: Sequence<Shared<[Vector<P::EA, P::SI>]>>,
    /// Empty unless the instruction tracks coordinates; its length is the single
    /// source of truth for whether coordinates are staged (see `read`/`write`).
    args: Sequence<Shared<[Vector<u32, P::SI>]>>,
    #[cube(comptime)]
    k: usize,
}

#[cube]
impl<P: ReducePrecision> SharedAccumulator<P, TopK> for TopKSharedAccumulator<P> {
    fn allocate(#[comptime] length: usize, #[comptime] _coordinate: bool, inst: &TopK) -> Self {
        let has_coords = comptime!(inst.output.has_indices());

        // Both loops must be unrolled: a `Sequence` is built at expand time, so a
        // runtime loop would run the body once and leave a single slice behind
        // whatever `k` is, and `read`/`write` would then index past the end.
        let mut elements = Sequence::new();
        #[unroll]
        for _ in 0..inst.k {
            elements.push(Shared::new_slice(length));
        }

        let mut args = Sequence::new();
        if has_coords {
            #[unroll]
            for _ in 0..inst.k {
                args.push(Shared::new_slice(length));
            }
        }

        TopKSharedAccumulator::<P> {
            elements,
            args,
            k: inst.k,
        }
    }

    fn read(accumulator: &Self, index: usize) -> Accumulator<P> {
        let mut values = Array::new(accumulator.k);
        #[unroll]
        for i in 0..accumulator.k {
            values[i] = accumulator.elements[i][index];
        }

        let num_args = comptime!(accumulator.args.len());
        let args = if comptime!(num_args != 0) {
            let mut args = Array::new(accumulator.k);
            #[unroll]
            for i in 0..accumulator.k {
                args[i] = accumulator.args[i][index];
            }
            Value::new_Multiple(args)
        } else {
            Value::new_None()
        };

        Accumulator::<P> {
            elements: Value::new_Multiple(values),
            args,
        }
    }

    fn write(accumulator: &mut Self, index: usize, item: Accumulator<P>) {
        let values = item.elements.multiple();
        #[unroll]
        for i in 0..accumulator.k {
            let acc = values[i];
            let shared_acc = &mut accumulator.elements[i];
            shared_acc[index] = acc;
        }

        let num_args = comptime!(accumulator.args.len());
        if comptime!(num_args != 0) {
            let args = item.args.multiple();
            #[unroll]
            for i in 0..accumulator.k {
                let arg = args[i];
                let shared_arg_acc = &mut accumulator.args[i];
                shared_arg_acc[index] = arg;
            }
        }
    }
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for TopK {
    type SharedAccumulator = TopKSharedAccumulator<P>;
    type Config = TopKConfig;

    fn requirements(this: &Self) -> super::ReduceRequirements {
        ReduceRequirements {
            coordinates: comptime!(this.output.has_indices()),
        }
    }

    fn accumulator_format(this: &Self) -> comptime_type!(AccumulatorFormat) {
        comptime!(AccumulatorFormat::Multiple(this.k))
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

        Accumulator::<P> {
            elements: Value::new_Multiple(elements),
            args,
        }
    }

    fn reduce(
        this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        let elements = accumulator.elements.multiple_mut();

        match reduce_step {
            ReduceStep::Plane => {
                this.plane_insert::<P::EA, P::SI>(
                    elements,
                    &mut accumulator.args,
                    Vector::cast_from(item.elements),
                    &item.args,
                );
            }
            ReduceStep::Identity => {
                this.insert::<P::EA, P::SI>(
                    elements,
                    &mut accumulator.args,
                    Vector::cast_from(item.elements),
                    &item.args,
                );
            }
        }
    }

    fn plane_reduce_inplace(this: &Self, accumulator: &mut Accumulator<P>) {
        this.plane_merge::<P::EA, P::SI>(
            accumulator.elements.multiple_mut(),
            &mut accumulator.args,
        );
    }

    fn fuse_accumulators(this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        let elements = accumulator.elements.multiple_mut();
        let other_elements = other.elements.multiple();

        for i in 0..this.k {
            this.insert::<P::EA, P::SI>(
                elements,
                &mut accumulator.args,
                other_elements[i],
                &other.args.slot(i),
            );
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
        match accumulator.args {
            Value::None => {
                let values = this.finalize_values::<P, Out>(&accumulator);
                (Value::new_Multiple(values), Value::new_None())
            }
            Value::Multiple(_) => {
                let (values, coords) = this.finalize_with_coords::<P>(&accumulator);

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
        }
    }

    fn to_output_perpendicular<Out: Numeric, Idx: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> (Value<Vector<Out, P::SI>>, Value<Vector<Idx, P::SI>>) {
        let acc_values = accumulator.elements.multiple();
        let mut out_values = Array::new(this.k);
        #[unroll]
        for i in 0..this.k {
            out_values[i] = Vector::cast_from(acc_values[i]);
        }

        let indices = match &accumulator.args {
            Value::None => Value::new_None(),
            Value::Multiple(acc_args) => {
                let mut out_indices = Array::new(this.k);
                #[unroll]
                for i in 0..this.k {
                    out_indices[i] = Vector::cast_from(acc_args[i]);
                }
                Value::new_Multiple(out_indices)
            }
            Value::Single(_) => panic!("top-k accumulator coordinates are one slice per slot"),
        };

        (Value::new_Multiple(out_values), indices)
    }
}

impl<P: ReducePrecision> ReduceWithIndices<P> for TopK {}

/// How much fully-unrolled top-k selection work is worth emitting, counted in
/// copies of a loop body.
///
/// The selection networks on [`TopK`] are `k`-by-`k`: every candidate walks all
/// `k` accumulator slots. Unrolling both levels keeps the accumulator in registers
/// with constant slot indices, which is worth a lot: rolled, the slots move to
/// scratch memory and every step becomes a load/store. Measured on a
/// `32x512x4095` `ArgTopK` (device timing, RTX 5090 / Vulkan), unrolling is
/// worth 5.8x at `k = 32` and 2.2x at `k = 64`.
///
/// But the emitted kernel grows with the product, so a flat cap on `k` prices
/// the three nest shapes wrong: the finalize nests are `k * k * vector_size`, not
/// `k * k`, and are the first to become unaffordable. Budgeting the product
/// instead lets the square nests unroll further than the cubic one, and stops
/// all of them before the backend compiler does: past this budget the same
/// selection runs as a plain runtime loop whose kernel size does not depend on
/// `k` at all.
pub(crate) const TOPK_UNROLL_BUDGET: usize = 1024;
