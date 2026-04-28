use crate::components::{instructions::lowest_coordinate_matching, precision::ReducePrecision};
use cubecl::prelude::*;

pub trait ReduceFamily: Send + Sync + 'static + std::fmt::Debug {
    type Instruction<P: ReducePrecision>: ReduceInstruction<P, Config = Self::Config>;
    type Config: CubeComptime + Send + Sync;
}

#[derive(CubeType, Clone, Copy)]
/// Whether we keep track of coordinates of items
pub struct ReduceRequirements {
    #[cube(comptime)]
    pub coordinates: bool,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, CubeType)]
pub enum AccumulatorFormat {
    Multiple(usize),
    Single,
}

impl AccumulatorFormat {
    pub fn len(&self) -> usize {
        match self {
            AccumulatorFormat::Multiple(k) => *k,
            AccumulatorFormat::Single => 1,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(CubeType)]
/// Whether the accumulator has zero, one or more vectors
pub enum Value<X: CubePrimitive> {
    Multiple(Array<X>),
    /// Wrap the item to be able to modify it as a field
    Single(ValueWrapper<X>),
    None,
}

#[derive(CubeType)]
/// Wrap the item to be able to modify it as a field
pub struct ValueWrapper<X: CubePrimitive> {
    val: X,
}

#[cube]
impl<X: CubePrimitive> ValueWrapper<X> {
    pub fn unwrap(&self) -> X {
        self.val
    }
}

#[cube]
impl<X: CubePrimitive> Value<X> {
    pub fn new_single(val: X) -> Value<X> {
        Value::new_Single(ValueWrapper::<X> { val })
    }

    pub fn item(&self) -> X {
        match self {
            Value::Multiple(_) => panic!("Tried item on Multiple"),
            Value::Single(item) => item.val,
            Value::None => panic!("Tried item on None"),
        }
    }

    pub fn multiple(&self) -> &Array<X> {
        match self {
            Value::Multiple(array) => array,
            Value::Single(_) => panic!("Tried multiple on Single"),
            Value::None => panic!("Tried multiple on None"),
        }
    }

    pub fn multiple_mut(&mut self) -> &mut Array<X> {
        match self {
            Value::Multiple(array) => array,
            Value::Single(_) => panic!("Tried multiple on Single"),
            Value::None => panic!("Tried multiple on None"),
        }
    }

    pub fn assign(&mut self, other: &Value<X>) {
        match (self, other) {
            (Value::Multiple(this), Value::Multiple(other)) => {
                for i in 0..this.len() {
                    this[i] = other[i];
                }
            }
            (Value::Single(this), Value::Single(other)) => {
                this.val = other.val;
            }
            (Value::None, Value::None) => {}
            _ => panic!("Tried assigning different accumulator kinds"),
        }
    }
}

#[cube]
pub fn plane_topk_insert<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    coordinates: &mut Value<Vector<u32, S>>,
    item: Vector<N, S>,
    coord: &Value<Vector<u32, S>>,
    #[comptime] k: usize,
    #[comptime] has_coords: bool,
) {
    let mut local_best_val = item;
    let lane_id = Vector::new(UNIT_POS_X);

    let mut local_best_coord = if has_coords {
        coord.item()
    } else {
        Vector::new(u32::MAX)
    };

    #[unroll]
    for _i in 0..k {
        let winning_val = plane_max(local_best_val);

        let winning_coord = if has_coords {
            lowest_coordinate_matching(winning_val, local_best_val, local_best_coord)
        } else {
            let is_match = local_best_val.equal(winning_val);
            let claim = select_many(is_match, lane_id, Vector::new(u32::MAX));
            plane_min(claim)
        };

        let mut insert_val = winning_val;
        let mut insert_coord = winning_coord;

        if has_coords {
            let coordinates = coordinates.multiple_mut();
            #[unroll]
            for j in 0..k {
                let to_keep = select_many(
                    elements[j].equal(insert_val),
                    coordinates[j].less_than(insert_coord),
                    elements[j].greater_than(insert_val),
                );

                let next_val = select_many(to_keep, insert_val, elements[j]);
                elements[j] = select_many(to_keep, elements[j], insert_val);
                insert_val = next_val;

                let next_coord = select_many(to_keep, insert_coord, coordinates[j]);
                coordinates[j] = select_many(to_keep, coordinates[j], insert_coord);
                insert_coord = next_coord;
            }
        } else {
            #[unroll]
            for j in 0..k {
                let to_keep = elements[j].greater_than(insert_val);
                let next_val = select_many(to_keep, insert_val, elements[j]);
                elements[j] = select_many(to_keep, elements[j], insert_val);
                insert_val = next_val;
            }
        }

        // Winner masking logic
        let is_winner = if has_coords {
            local_best_val
                .equal(winning_val)
                .and(local_best_coord.equal(winning_coord))
        } else {
            lane_id.equal(winning_coord)
        };

        local_best_val = select_many(is_winner, Vector::new(N::min_value()), local_best_val);
        if has_coords {
            local_best_coord = select_many(is_winner, Vector::new(u32::MAX), local_best_coord);
        }
    }
}

#[cube]
pub fn plane_topk_merge<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    coordinates: &mut Value<Vector<u32, S>>,
    #[comptime] k: usize,
    #[comptime] has_coords: bool,
) {
    let mut final_elements = Array::new(k);
    let mut final_coords = Array::new(k);
    let mut cursor = Vector::new(0u32);
    let lane_id = Vector::new(UNIT_POS_X);

    #[unroll]
    for i in 0..k {
        let mut local_val = Vector::new(N::min_value());
        let mut local_coord = Vector::new(u32::MAX);

        #[unroll]
        for j in 0..k {
            let is_pointed = cursor.equal(Vector::new(j as u32));
            local_val = select_many(is_pointed, elements[j], local_val);
            if has_coords {
                let coords = coordinates.multiple_mut();
                local_coord = select_many(is_pointed, coords[j], local_coord);
            }
        }

        let winning_val = plane_max(local_val);
        let winning_lane = if has_coords {
            let best_c = lowest_coordinate_matching(winning_val, local_val, local_coord);
            final_coords[i] = best_c;
            let is_cand = local_val.equal(winning_val).and(local_coord.equal(best_c));
            plane_min(select_many(is_cand, lane_id, Vector::new(u32::MAX)))
        } else {
            let is_cand = local_val.equal(winning_val);
            plane_min(select_many(is_cand, lane_id, Vector::new(u32::MAX)))
        };

        final_elements[i] = winning_val;
        let is_winner_thread = lane_id.equal(winning_lane);
        cursor = select_many(is_winner_thread, cursor + Vector::new(1u32), cursor);
    }

    #[unroll]
    for i in 0..k {
        elements[i] = final_elements[i];
        if has_coords {
            let coords = coordinates.multiple_mut();
            coords[i] = final_coords[i];
        }
    }
}

#[derive(CubeType)]
/// Whether the accumulator has zero, one or more vectors
/// This should be the same variant as AccumulatorKind for an instruction
pub enum SharedAccumulatorKind<X: CubePrimitive> {
    Multiple(Sequence<SharedMemory<X>>),
    Single(SharedMemory<X>),
    None,
}

#[cube]
impl<X: CubePrimitive> SharedAccumulatorKind<X> {
    pub fn get(&self, i: usize) -> Value<X> {
        match self {
            SharedAccumulatorKind::Multiple(sequence) => {
                let mut array = Array::new(sequence.len());
                #[unroll]
                for k_iter in 0..sequence.len() {
                    array[k_iter] = sequence[k_iter][i];
                }
                Value::new_Multiple(array)
            }
            SharedAccumulatorKind::Single(shared_memory) => Value::new_single(shared_memory[i]),
            SharedAccumulatorKind::None => Value::new_None(),
        }
    }

    pub fn set(&mut self, i: usize, value: Value<X>) {
        match self {
            SharedAccumulatorKind::Multiple(sequence) =>
            {
                #[unroll]
                for k_iter in 0..sequence.len() {
                    let mut shared_acc = sequence[k_iter];
                    shared_acc[i] = value.multiple()[k_iter];
                }
            }
            SharedAccumulatorKind::Single(shared_memory) => shared_memory[i] = value.item(),
            SharedAccumulatorKind::None => {}
        }
    }
}

/// An instruction for a reduce algorithm that works with [`Vector`].
///
/// See a provided implementation, such as [`Sum`](super::Sum) or [`ArgMax`](super::ArgMax) for an example how to implement
/// this trait for a custom instruction.
///
/// A reduction works at three levels. First, it takes input data of type `In` and reduce them
/// with their coordinate into an `AccumulatorItem`. Then, multiple `AccumulatorItem` are possibly fused
/// together into a single accumulator that is converted to the expected output type.
#[cube]
pub trait ReduceInstruction<P: ReducePrecision>:
    Send + Sync + 'static + std::fmt::Debug + CubeType + Sized
{
    type Config: CubeComptime + Send + Sync;

    /// When multiple agents are collaborating to reduce a single slice,
    /// we need a share accumulator to store multiple `AccumulatorItem`.
    /// This is most likely a `SharedMemory<Vector<T>>` or a struct or tuple of vectorized shared memories.
    type SharedAccumulator: SharedAccumulator<P, Self>;

    /// Requirements of the reduce.
    fn requirements(this: &Self) -> ReduceRequirements;
    fn accumulator_format(this: &Self) -> comptime_type!(AccumulatorFormat);

    fn from_config(#[comptime] config: Self::Config) -> Self;
    /// A input such that `Self::reduce(accumulator, Self::null_input(), coordinate, use_planes)`
    /// is guaranteed to return `accumulator` unchanged for any choice of `coordinate`.
    fn null_input(this: &Self) -> Vector<P::EI, P::SI>;

    /// A accumulator such that `Self::fuse_accumulators(accumulator, Self::null_accumulator()` always returns
    /// is guaranteed to return `accumulator` unchanged.
    fn null_accumulator(this: &Self) -> Accumulator<P>;

    /// If `ReduceStep` is `Plane`, reduce all the `item` and `coordinate` within the `accumulator`.
    /// if `ReduceStep` is `Identity`, reduce the given `item` and `coordinate` into the accumulator.
    fn reduce(
        this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    );

    fn plane_reduce_inplace(this: &Self, accumulator: &mut Accumulator<P>);

    /// Reduce a whole accumulator (other) in accumulator.
    fn fuse_accumulators(this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>);

    /// Reduce all elements of the accumulator into a single output element of type `Out`.
    fn to_output_parallel<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        shape_axis_reduce: usize,
    ) -> Value<Out>;

    /// Convert each element of the accumulator into the expected output element of type `Out`.
    fn to_output_perpendicular<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        shape_axis_reduce: usize,
    ) -> Value<Vector<Out, P::SI>>;
}

#[derive(CubeType)]
pub struct Item<P: ReducePrecision> {
    pub elements: Vector<P::EI, P::SI>,
    // Warning: should not be Multiple
    pub args: Value<Vector<u32, P::SI>>,
}

#[derive(CubeType)]
pub struct Accumulator<P: ReducePrecision> {
    pub elements: Value<Vector<P::EA, P::SI>>,
    pub args: Value<Vector<u32, P::SI>>,
}

/// A simple trait that abstract over a single or multiple shared memory.
#[cube]
pub trait SharedAccumulator<P: ReducePrecision, I: ReduceInstruction<P>>:
    CubeType + Send + Sync + 'static
{
    fn allocate(#[comptime] length: usize, #[comptime] _coordinate: bool, inst: &I) -> Self;

    fn read(accumulator: &Self, index: usize) -> Accumulator<P>;

    fn write(accumulator: &mut Self, index: usize, item: Accumulator<P>);
}

#[cube]
impl<P: ReducePrecision, I: ReduceInstruction<P>> SharedAccumulator<P, I>
    for SharedMemory<Vector<P::EA, P::SI>>
{
    fn allocate(#[comptime] length: usize, #[comptime] _coordinate: bool, _inst: &I) -> Self {
        SharedMemory::new(length)
    }

    fn read(accumulator: &Self, index: usize) -> Accumulator<P> {
        Accumulator::<P> {
            elements: Value::new_single(accumulator[index]),
            args: Value::new_None(),
        }
    }

    fn write(accumulator: &mut Self, index: usize, item: Accumulator<P>) {
        accumulator[index] = item.elements.item();
    }
}

/// A pair of shared memory used for [`ArgMax`](super::ArgMax) and [`ArgMin`](super::ArgMin).
#[derive(CubeType)]
pub struct ArgAccumulator<P: ReducePrecision> {
    pub elements: SharedMemory<Vector<P::EA, P::SI>>,
    pub args: SharedMemory<Vector<u32, P::SI>>,
}

/// For a single reduce step whether we need to do plane reduction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceStep {
    /// Just keep the current value
    Identity,
    /// reduce across the plane
    Plane,
}

#[cube]
impl<P: ReducePrecision, I: ReduceInstruction<P>> SharedAccumulator<P, I> for ArgAccumulator<P> {
    fn allocate(#[comptime] length: usize, #[comptime] _coordinate: bool, _inst: &I) -> Self {
        ArgAccumulator::<P> {
            elements: SharedMemory::new(length),
            args: SharedMemory::new(length),
        }
    }

    fn read(accumulator: &Self, index: usize) -> Accumulator<P> {
        Accumulator::<P> {
            elements: Value::new_single(accumulator.elements[index]),
            args: Value::new_single(accumulator.args[index]),
        }
    }

    fn write(accumulator: &mut Self, index: usize, item: Accumulator<P>) {
        accumulator.elements[index] = item.elements.item();
        accumulator.args[index] = item.args.item();
    }
}

#[cube]
pub fn reduce_inplace<P: ReducePrecision, R: ReduceInstruction<P>>(
    inst: &R,
    accumulator: &mut Accumulator<P>,
    item: Item<P>,
    #[comptime] reduce_step: ReduceStep,
) {
    R::reduce(inst, accumulator, item, reduce_step)
}

#[cube]
pub fn reduce_shared_inplace<P: ReducePrecision, R: ReduceInstruction<P>>(
    inst: &R,
    accumulator: &mut R::SharedAccumulator,
    index: usize,
    item: Item<P>,
    #[comptime] reduce_step: ReduceStep,
) {
    let mut acc_item = R::SharedAccumulator::read(accumulator, index);
    R::reduce(inst, &mut acc_item, item, reduce_step);
    R::SharedAccumulator::write(accumulator, index, acc_item);
}

#[cube]
pub fn fuse_accumulator_inplace<P: ReducePrecision, R: ReduceInstruction<P>>(
    inst: &R,
    accumulator: &mut R::SharedAccumulator,
    destination: usize,
    origin: usize,
) {
    let mut acc = R::SharedAccumulator::read(accumulator, destination);
    R::fuse_accumulators(
        inst,
        &mut acc,
        &R::SharedAccumulator::read(accumulator, origin),
    );
    R::SharedAccumulator::write(accumulator, destination, acc);
}
