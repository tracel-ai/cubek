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

    /// Requirements of the reduce.
    fn requirements(this: &Self) -> ReduceRequirements;

    /// The intermediate state into which we accumulate new input elements.
    /// This is most likely a `Vector<T>` or a struct or tuple of vectors.
    type AccumulatorItem: CubeType;

    /// When multiple agents are collaborating to reduce a single slice,
    /// we need a share accumulator to store multiple `AccumulatorItem`.
    /// This is most likely a `SharedMemory<Vector<T>>` or a struct or tuple of vectorized shared memories.
    type SharedAccumulator: SharedAccumulator<Item = Self::AccumulatorItem>;

    fn from_config(#[comptime] config: Self::Config) -> Self;
    /// A input such that `Self::reduce(accumulator, Self::null_input(), coordinate, use_planes)`
    /// is guaranteed to return `accumulator` unchanged for any choice of `coordinate`.
    fn null_input(this: &Self) -> Vector<P::EI, P::SI>;

    /// A accumulator such that `Self::fuse_accumulators(accumulator, Self::null_accumulator()` always returns
    /// is guaranteed to return `accumulator` unchanged.
    fn null_accumulator(this: &Self) -> Self::AccumulatorItem;

    /// Assign the value of `source` into `destination`.
    /// In spirit, this is equivalent to `destination = source;`,
    /// but this syntax is not currently supported by CubeCL.
    fn assign_accumulator(
        this: &Self,
        destination: &mut Self::AccumulatorItem,
        source: &Self::AccumulatorItem,
    );

    fn read_accumulator(
        this: &Self,
        accumulator: &Self::AccumulatorItem,
    ) -> (Vector<P::EI, P::SI>, ReduceCoordinate<P::SI>);

    /// If `ReduceStep` is `Plane`, reduce all the `item` and `coordinate` within the `accumulator`.
    /// if `ReduceStep` is `Identity`, reduce the given `item` and `coordinate` into the accumulator.
    fn reduce(
        this: &Self,
        accumulator: &Self::AccumulatorItem,
        item: Vector<P::EI, P::SI>,
        coordinate: ReduceCoordinate<P::SI>,
        #[comptime] reduce_step: ReduceStep,
    ) -> Self::AccumulatorItem;

    /// Reduce two accumulators into a single accumulator.
    fn fuse_accumulators(
        this: &Self,
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem;

    /// Reduce all elements of the accumulator into a single output element of type `Out`.
    fn merge_vector<Out: Numeric>(
        this: &Self,
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: usize,
    ) -> Out;

    /// Convert each element of the accumulator into the expected output element of type `Out`.
    fn to_output_perpendicular<Out: Numeric>(
        this: &Self,
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: usize,
    ) -> Vector<Out, P::SI>;
}

#[derive(CubeType)]
pub enum ReduceCoordinate<N: Size> {
    Required(Vector<u32, N>),
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
    accumulator: &mut R::AccumulatorItem,
    item: Vector<P::EI, P::SI>,
    coordinate: ReduceCoordinate<P::SI>,
    #[comptime] reduce_step: ReduceStep,
) {
    let reduction = &R::reduce(inst, accumulator, item, coordinate, reduce_step);
    R::assign_accumulator(inst, accumulator, reduction);
}

#[cube]
pub fn reduce_shared_inplace<P: ReducePrecision, R: ReduceInstruction<P>>(
    inst: &R,
    accumulator: &mut R::SharedAccumulator,
    index: usize,
    item: Vector<P::EI, P::SI>,
    coordinate: ReduceCoordinate<P::SI>,
    #[comptime] reduce_step: ReduceStep,
) {
    let acc_item = R::SharedAccumulator::read(accumulator, index);
    let reduction = R::reduce(inst, &acc_item, item, coordinate, reduce_step);
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
