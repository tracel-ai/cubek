use crate::components::precision::ReducePrecision;
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

#[derive(CubeType)]
/// Whether the accumulator has zero, one or more vectors
pub enum AccumulatorKind<X: CubePrimitive> {
    Multiple(Array<X>),
    /// Wrap the item to be able to modify it as a field
    Single(ItemWrapper<X>),
    None,
}

#[derive(CubeType)]
/// Wrap the item to be able to modify it as a field
pub struct ItemWrapper<X: CubePrimitive> {
    item: X,
}

#[cube]
impl<X: CubePrimitive> AccumulatorKind<X> {
    pub fn new_single(item: X) -> AccumulatorKind<X> {
        AccumulatorKind::new_Single(ItemWrapper::<X> { item })
    }

    pub fn item(&self) -> X {
        match self {
            AccumulatorKind::Multiple(_) => panic!("Tried item on Multiple"),
            AccumulatorKind::Single(item) => item.item,
            AccumulatorKind::None => panic!("Tried item on None"),
        }
    }

    pub fn multiple(self) -> Array<X> {
        match self {
            AccumulatorKind::Multiple(array) => array,
            AccumulatorKind::Single(_) => panic!("Tried multiple on Single"),
            AccumulatorKind::None => panic!("Tried multiple on None"),
        }
    }

    pub fn assign(&mut self, other: &AccumulatorKind<X>) {
        match (self, other) {
            (AccumulatorKind::Multiple(this), AccumulatorKind::Multiple(other)) => {
                for i in 0..this.len() {
                    this[i] = other[i];
                }
            }
            (AccumulatorKind::Single(this), AccumulatorKind::Single(other)) => {
                this.item = other.item;
            }
            (AccumulatorKind::None, AccumulatorKind::None) => {}
            _ => panic!("Tried assigning different accumulator kinds"),
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
    pub fn get(&self, i: usize) -> AccumulatorKind<X> {
        match self {
            SharedAccumulatorKind::Multiple(_sequence) => todo!(),
            SharedAccumulatorKind::Single(shared_memory) => {
                AccumulatorKind::new_single(shared_memory[i])
            }
            SharedAccumulatorKind::None => AccumulatorKind::new_None(),
        }
    }

    pub fn set(&mut self, i: usize, value: AccumulatorKind<X>) {
        match self {
            SharedAccumulatorKind::Multiple(_sequence) => todo!(),
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
    Send + Sync + 'static + std::fmt::Debug + CubeType
{
    type Config: CubeComptime + Send + Sync;

    /// When multiple agents are collaborating to reduce a single slice,
    /// we need a share accumulator to store multiple `AccumulatorItem`.
    /// This is most likely a `SharedMemory<Vector<T>>` or a struct or tuple of vectorized shared memories.
    type SharedAccumulator: SharedAccumulator<P>;

    /// Requirements of the reduce.
    fn requirements(this: &Self) -> ReduceRequirements;

    fn from_config(#[comptime] config: Self::Config) -> Self;
    /// A input such that `Self::reduce(accumulator, Self::null_input(), coordinate, use_planes)`
    /// is guaranteed to return `accumulator` unchanged for any choice of `coordinate`.
    fn null_input(this: &Self) -> Vector<P::EI, P::SI>;

    /// A accumulator such that `Self::fuse_accumulators(accumulator, Self::null_accumulator()` always returns
    /// is guaranteed to return `accumulator` unchanged.
    fn null_accumulator(this: &Self) -> Accumulator<P>;

    /// Assign the value of `source` into `destination`.
    /// In spirit, this is equivalent to `destination = source;`,
    /// but this syntax is not currently supported by CubeCL.
    fn assign_accumulator(this: &Self, destination: &mut Accumulator<P>, source: &Accumulator<P>);

    /// If `ReduceStep` is `Plane`, reduce all the `item` and `coordinate` within the `accumulator`.
    /// if `ReduceStep` is `Identity`, reduce the given `item` and `coordinate` into the accumulator.
    fn reduce(
        this: &Self,
        accumulator: &Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) -> Accumulator<P>;

    fn plane_reduce_inplace(this: &Self, accumulator: &mut Accumulator<P>);

    /// Reduce two accumulators into a single accumulator.
    fn fuse_accumulators(this: &Self, lhs: &Accumulator<P>, rhs: &Accumulator<P>)
    -> Accumulator<P>;

    /// Reduce all elements of the accumulator into a single output element of type `Out`.
    fn merge_vector<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        shape_axis_reduce: usize,
    ) -> AccumulatorKind<Out>;

    /// Convert each element of the accumulator into the expected output element of type `Out`.
    fn to_output_perpendicular<Out: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        shape_axis_reduce: usize,
    ) -> AccumulatorKind<Vector<Out, P::SI>>;
}

#[derive(CubeType)]
pub struct Item<P: ReducePrecision> {
    pub elements: Vector<P::EI, P::SI>,
    // Warning: should not be Multiple
    pub args: AccumulatorKind<Vector<u32, P::SI>>,
}

#[derive(CubeType)]
pub struct Accumulator<P: ReducePrecision> {
    pub(crate) elements: AccumulatorKind<Vector<P::EA, P::SI>>,
    pub args: AccumulatorKind<Vector<u32, P::SI>>,
}

/// A simple trait that abstract over a single or multiple shared memory.
#[cube]
pub trait SharedAccumulator<P: ReducePrecision>: CubeType + Send + Sync + 'static {
    fn allocate(#[comptime] length: usize, #[comptime] _coordinate: bool) -> Self;

    fn read(accumulator: &Self, index: usize) -> Accumulator<P>;

    fn write(accumulator: &mut Self, index: usize, item: Accumulator<P>);
}

#[cube]
impl<P: ReducePrecision> SharedAccumulator<P> for SharedMemory<Vector<P::EA, P::SI>> {
    fn allocate(#[comptime] length: usize, #[comptime] _coordinate: bool) -> Self {
        SharedMemory::new(length)
    }

    fn read(accumulator: &Self, index: usize) -> Accumulator<P> {
        Accumulator::<P> {
            elements: AccumulatorKind::new_single(accumulator[index]),
            args: AccumulatorKind::new_None(),
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
impl<P: ReducePrecision> SharedAccumulator<P> for ArgAccumulator<P> {
    fn allocate(#[comptime] length: usize, #[comptime] _coordinate: bool) -> Self {
        ArgAccumulator::<P> {
            elements: SharedMemory::new(length),
            args: SharedMemory::new(length),
        }
    }

    fn read(accumulator: &Self, index: usize) -> Accumulator<P> {
        Accumulator::<P> {
            elements: AccumulatorKind::new_single(accumulator.elements[index]),
            args: AccumulatorKind::new_single(accumulator.args[index]),
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
    let reduction = &R::reduce(inst, accumulator, item, reduce_step);
    R::assign_accumulator(inst, accumulator, reduction);
}

#[cube]
pub fn reduce_shared_inplace<P: ReducePrecision, R: ReduceInstruction<P>>(
    inst: &R,
    accumulator: &mut R::SharedAccumulator,
    index: usize,
    item: Item<P>,
    #[comptime] reduce_step: ReduceStep,
) {
    let acc_item = R::SharedAccumulator::read(accumulator, index);
    let reduction = R::reduce(inst, &acc_item, item, reduce_step);
    R::SharedAccumulator::write(accumulator, index, reduction);
}

#[cube]
pub fn fuse_accumulator_inplace<P: ReducePrecision, R: ReduceInstruction<P>>(
    inst: &R,
    accumulator: &mut R::SharedAccumulator,
    destination: usize,
    origin: usize,
) {
    let fused = R::fuse_accumulators(
        inst,
        &R::SharedAccumulator::read(accumulator, destination),
        &R::SharedAccumulator::read(accumulator, origin),
    );
    R::SharedAccumulator::write(accumulator, destination, fused);
}
