use crate::components::precision::ReducePrecision;
use cubecl::prelude::*;

pub trait ReduceFamily: Send + Sync + 'static + std::fmt::Debug {
    type Instruction<P: ReducePrecision>: ReduceInstruction<P, Config = Self::Config>;
    type Config: CubeComptime + Send + Sync;
}

#[derive(CubeType, Clone, Copy)]
pub struct ReduceRequirements {
    #[cube(comptime)]
    pub coordinates: bool,
}

#[derive(CubeType)]
pub enum AccumulatorKind<X: CubePrimitive> {
    Multiple(Array<X>),
    Single(AccumulatorWrapper<X>),
    None,
}

#[derive(CubeType)]
pub struct AccumulatorWrapper<X: CubePrimitive> {
    value: X,
}

#[cube]
impl<X: CubePrimitive> AccumulatorKind<X> {
    pub fn new_single(value: X) -> AccumulatorKind<X> {
        AccumulatorKind::new_Single(AccumulatorWrapper::<X> { value })
    }

    pub fn item(&self) -> X {
        match self {
            AccumulatorKind::Multiple(_) => panic!("Tried item on Multiple"),
            AccumulatorKind::Single(item) => item.value,
            AccumulatorKind::None => panic!("Tried item on None"),
        }
    }

    pub fn multiple(&self) -> &Array<X> {
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
                this.value = other.value;
            }
            (AccumulatorKind::None, AccumulatorKind::None) => {}
            _ => panic!("Tried assigning different accumulator kinds"),
        }
    }
}

#[derive(CubeType)]
pub enum SharedAccumulatorKind<X: CubePrimitive> {
    Multiple(Sequence<SharedMemory<X>>),
    Single(SharedMemory<X>),
    None,
}

#[cube]
impl<X: CubePrimitive> SharedAccumulatorKind<X> {
    pub fn get(&self, i: usize) -> AccumulatorKind<X> {
        match self {
            SharedAccumulatorKind::Multiple(sequence) => todo!(),
            SharedAccumulatorKind::Single(shared_memory) => {
                AccumulatorKind::new_single(shared_memory[i])
            }
            SharedAccumulatorKind::None => AccumulatorKind::new_None(),
        }
    }

    pub fn set(&mut self, i: usize, value: AccumulatorKind<X>) {
        match self {
            SharedAccumulatorKind::Multiple(sequence) => todo!(),
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

    /// The intermediate state into which we accumulate new input elements.
    /// This is most likely a `Vector<T>` or a struct or tuple of vectors.
    type Accumulator: CubeType;

    /// When multiple agents are collaborating to reduce a single slice,
    /// we need a share accumulator to store multiple `AccumulatorItem`.
    /// This is most likely a `SharedMemory<Vector<T>>` or a struct or tuple of vectorized shared memories.
    type SharedAccumulator: SharedAccumulator<Item = Self::Accumulator>;

    /// Requirements of the reduce.
    fn requirements(this: &Self) -> ReduceRequirements;

    fn from_config(#[comptime] config: Self::Config) -> Self;
    /// A input such that `Self::reduce(accumulator, Self::null_input(), coordinate, use_planes)`
    /// is guaranteed to return `accumulator` unchanged for any choice of `coordinate`.
    fn null_input(this: &Self) -> Vector<P::EI, P::SI>;

    /// A accumulator such that `Self::fuse_accumulators(accumulator, Self::null_accumulator()` always returns
    /// is guaranteed to return `accumulator` unchanged.
    fn null_accumulator(this: &Self) -> Self::Accumulator;

    /// Assign the value of `source` into `destination`.
    /// In spirit, this is equivalent to `destination = source;`,
    /// but this syntax is not currently supported by CubeCL.
    fn assign_accumulator(
        this: &Self,
        destination: &mut Self::Accumulator,
        source: &Self::Accumulator,
    );

    /// Splits the accumulator between its values and coordinates, if they're tracked
    fn split_accumulator(
        this: &Self,
        accumulator: &Self::Accumulator,
    ) -> (
        AccumulatorKind<Vector<P::EI, P::SI>>,
        ReduceCoordinate<P::SI>,
    );

    /// If `use_planes` is `true`, reduce all the `item` and `coordinate` within the `accumulator`.
    /// Else, reduce the given `item` and `coordinate` into the accumulator.
    fn reduce(
        this: &Self,
        accumulator: &Self::Accumulator,
        item: Vector<P::EI, P::SI>,
        coordinate: ReduceCoordinate<P::SI>,
        #[comptime] plane_reduce: ReduceStep,
    ) -> Self::Accumulator;

    /// Reduce two accumulators into a single accumulator.
    fn fuse_accumulators(
        this: &Self,
        lhs: Self::Accumulator,
        rhs: Self::Accumulator,
    ) -> Self::Accumulator;

    // Self::Accumulator -> Out

    /// Reduce all elements of the accumulator into a single output element of type `Out`.
    fn merge_vector<Out: Numeric>(
        this: &Self,
        accumulator: Self::Accumulator,
        shape_axis_reduce: usize,
    ) -> AccumulatorKind<Out>;

    /// Convert each element of the accumulator into the expected output element of type `Out`.
    fn to_output_perpendicular<Out: Numeric>(
        this: &Self,
        accumulator: Self::Accumulator,
        shape_axis_reduce: usize,
    ) -> AccumulatorKind<Vector<Out, P::SI>>;
}

#[derive(CubeType)]
pub enum ReduceCoordinate<N: Size> {
    Required(AccumulatorKind<Vector<u32, N>>),
    NotRequired,
}

/// A simple trait that abstract over a single or multiple shared memory.
#[cube]
pub trait SharedAccumulator: CubeType + Send + Sync + 'static {
    type Item: CubeType;

    fn allocate(#[comptime] length: usize, #[comptime] _coordinate: bool) -> Self;

    fn read(accumulator: &Self, index: usize) -> Self::Item;

    fn write(accumulator: &mut Self, index: usize, item: Self::Item);
}

#[cube]
impl<In: Numeric, N: Size> SharedAccumulator for SharedMemory<Vector<In, N>> {
    type Item = Vector<In, N>;

    fn allocate(#[comptime] length: usize, #[comptime] _coordinate: bool) -> Self {
        SharedMemory::new(length)
    }

    fn read(accumulator: &Self, index: usize) -> Self::Item {
        accumulator[index]
    }

    fn write(accumulator: &mut Self, index: usize, item: Self::Item) {
        accumulator[index] = item;
    }
}

/// A pair of shared memory used for [`ArgMax`](super::ArgMax) and [`ArgMin`](super::ArgMin).
#[derive(CubeType)]
pub struct ArgAccumulator<T: Numeric, N: Size> {
    pub elements: SharedMemory<Vector<T, N>>,
    pub args: SharedMemory<Vector<u32, N>>,
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
impl<In: Numeric, N: Size> SharedAccumulator for ArgAccumulator<In, N> {
    type Item = (Vector<In, N>, Vector<u32, N>);

    fn allocate(#[comptime] length: usize, #[comptime] _coordinate: bool) -> Self {
        ArgAccumulator::<In, N> {
            elements: SharedMemory::new(length),
            args: SharedMemory::new(length),
        }
    }

    fn read(accumulator: &Self, index: usize) -> Self::Item {
        (accumulator.elements[index], accumulator.args[index])
    }

    fn write(accumulator: &mut Self, index: usize, item: Self::Item) {
        accumulator.elements[index] = item.0;
        accumulator.args[index] = item.1;
    }
}

#[cube]
pub fn reduce_inplace<P: ReducePrecision, R: ReduceInstruction<P>>(
    inst: &R,
    accumulator: &mut R::Accumulator,
    item: Vector<P::EI, P::SI>,
    coordinate: ReduceCoordinate<P::SI>,
    #[comptime] plane_reduce: ReduceStep,
) {
    let reduction = &R::reduce(inst, accumulator, item, coordinate, plane_reduce);
    R::assign_accumulator(inst, accumulator, reduction);
}

#[cube]
pub fn reduce_shared_inplace<P: ReducePrecision, R: ReduceInstruction<P>>(
    inst: &R,
    accumulator: &mut R::SharedAccumulator,
    index: usize,
    item: Vector<P::EI, P::SI>,
    coordinate: ReduceCoordinate<P::SI>,
    #[comptime] use_planes: ReduceStep,
) {
    let acc_item = R::SharedAccumulator::read(accumulator, index);
    let reduction = R::reduce(inst, &acc_item, item, coordinate, use_planes);
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
        R::SharedAccumulator::read(accumulator, destination),
        R::SharedAccumulator::read(accumulator, origin),
    );
    R::SharedAccumulator::write(accumulator, destination, fused);
}
