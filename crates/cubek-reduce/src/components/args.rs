use cubecl::prelude::*;
use cubecl::std::tensor::r#virtual::{
    VirtualTensor, VirtualTensorOperations, VirtualTensorOperationsExpand,
};
use std::marker::PhantomData;

pub trait ReduceDType {
    type In: Numeric;
    type SizeIn: Size;
    type Out: Numeric;
    type SizeOut: Size;
}

impl<In: Numeric, SizeIn: Size, Out: Numeric, SizeOut: Size> ReduceDType
    for ((In, SizeIn), (Out, SizeOut))
{
    type In = In;
    type SizeIn = SizeIn;
    type Out = Out;
    type SizeOut = SizeOut;
}

pub trait NumericVector {
    type T: Numeric;
    type N: Size;
}

impl<T: Numeric, N: Size> NumericVector for (T, N) {
    type T = T;
    type N = N;
}

#[cube]
#[allow(dead_code)]
pub trait ReduceArgs: Send + Sync + 'static + Clone {
    type Input<E: Numeric, S: Size>: LaunchArg + CubeType;

    type Output<E: Numeric, S: Size>: LaunchArg + CubeType;

    type State<P: ReduceDType>: Clone + CubeType<ExpandType: Clone>;

    fn init_state<P: ReduceDType>(
        input: &Self::Input<P::In, P::SizeIn>,
        output: &mut Self::Output<P::Out, P::SizeOut>,
    ) -> Self::State<P>;

    fn read_input<P: ReduceDType>(state: &Self::State<P>, index: usize)
    -> Vector<P::In, P::SizeIn>;
    fn read_output<P: ReduceDType>(
        state: &Self::State<P>,
        index: usize,
    ) -> Vector<P::Out, P::SizeOut>;

    fn write_output<P: ReduceDType>(
        state: &mut Self::State<P>,
        index: usize,
        value: Vector<P::Out, P::SizeOut>,
    );

    fn len_input<P: ReduceDType>(state: &Self::State<P>) -> usize;
    fn len_output<P: ReduceDType>(state: &Self::State<P>) -> usize;

    fn buffer_len_input<P: ReduceDType>(state: &Self::State<P>) -> usize;
    fn buffer_len_output<P: ReduceDType>(state: &Self::State<P>) -> usize;

    fn rank_input<P: ReduceDType>(state: &Self::State<P>) -> usize;
    fn rank_output<P: ReduceDType>(state: &Self::State<P>) -> usize;

    fn shape_input<P: ReduceDType>(state: &Self::State<P>, dim: usize) -> usize;
    fn shape_output<P: ReduceDType>(state: &Self::State<P>, dim: usize) -> usize;

    fn stride_input<P: ReduceDType>(state: &Self::State<P>, dim: usize) -> usize;
    fn stride_output<P: ReduceDType>(state: &Self::State<P>, dim: usize) -> usize;

    fn vector_size_input<P: ReduceDType>(state: &Self::State<P>) -> comptime_type!(VectorSize);
    fn vector_size_output<P: ReduceDType>(state: &Self::State<P>) -> comptime_type!(VectorSize);
}

#[cube]
pub fn init_tensors<RA: ReduceArgs, In: Numeric, InSize: Size, Out: Numeric, OutSize: Size>(
    input: &RA::Input<In, InSize>,
    output: &mut RA::Output<Out, OutSize>,
) -> (
    VirtualTensor<In, InSize>,
    VirtualTensor<Out, OutSize, ReadWrite>,
) {
    let mut state = RA::init_state::<((In, InSize), (Out, OutSize))>(input, output);

    let input = TensorArg::new_input(&state);
    let output = TensorArg::new_output(&mut state);

    let input = VirtualTensor::<In, InSize>::new::<
        TensorArg<((In, InSize), (Out, OutSize)), RA, Input>,
    >(input);
    let output = VirtualTensor::<Out, OutSize, ReadWrite>::new::<
        TensorArg<((In, InSize), (Out, OutSize)), RA, Output>,
    >(output);

    (input, output)
}

#[derive(Clone)]
pub struct TensorArgs;

#[cube]
impl ReduceArgs for TensorArgs {
    type Input<EG: Numeric, N: Size> = OwnedTensor<Vector<EG, N>>;
    type Output<EG: Numeric, N: Size> = OwnedTensor<Vector<EG, N>>;
    type State<P: ReduceDType> = (
        OwnedTensor<Vector<P::In, P::SizeIn>>,
        OwnedTensor<Vector<P::Out, P::SizeOut>>,
    );

    fn init_state<P: ReduceDType>(
        input: &Self::Input<P::In, P::SizeIn>,
        output: &mut Self::Output<P::Out, P::SizeOut>,
    ) -> Self::State<P> {
        (input.clone(), output.clone())
    }

    fn read_input<P: ReduceDType>(
        state: &Self::State<P>,
        index: usize,
    ) -> Vector<P::In, P::SizeIn> {
        state.0[index]
    }

    fn read_output<P: ReduceDType>(
        state: &Self::State<P>,
        index: usize,
    ) -> Vector<P::Out, P::SizeOut> {
        state.1[index]
    }

    fn write_output<P: ReduceDType>(
        state: &mut Self::State<P>,
        index: usize,
        value: Vector<P::Out, P::SizeOut>,
    ) {
        state.1[index] = value;
    }

    fn buffer_len_input<P: ReduceDType>(state: &Self::State<P>) -> usize {
        state.0.buffer_len()
    }

    fn buffer_len_output<P: ReduceDType>(state: &Self::State<P>) -> usize {
        state.1.buffer_len()
    }

    fn len_input<P: ReduceDType>(state: &Self::State<P>) -> usize {
        state.0.len()
    }

    fn len_output<P: ReduceDType>(state: &Self::State<P>) -> usize {
        state.1.len()
    }
    fn rank_input<P: ReduceDType>(state: &Self::State<P>) -> usize {
        state.0.rank()
    }

    fn rank_output<P: ReduceDType>(state: &Self::State<P>) -> usize {
        state.1.rank()
    }

    fn shape_input<P: ReduceDType>(state: &Self::State<P>, dim: usize) -> usize {
        state.0.shape(dim)
    }

    fn shape_output<P: ReduceDType>(state: &Self::State<P>, dim: usize) -> usize {
        state.1.shape(dim)
    }

    fn stride_input<P: ReduceDType>(state: &Self::State<P>, dim: usize) -> usize {
        state.0.stride(dim)
    }

    fn stride_output<P: ReduceDType>(state: &Self::State<P>, dim: usize) -> usize {
        state.1.stride(dim)
    }

    fn vector_size_input<P: ReduceDType>(state: &Self::State<P>) -> comptime_type!(VectorSize) {
        state.0.vector_size()
    }

    fn vector_size_output<P: ReduceDType>(state: &Self::State<P>) -> comptime_type!(VectorSize) {
        state.1.vector_size()
    }
}

pub struct Input;
pub struct Output;

#[derive(CubeType)]
pub struct TensorArg<P: ReduceDType, RA: ReduceArgs, Tag> {
    #[allow(unused, reason = "Used in expand")]
    state: RA::State<P>,
    #[cube(comptime)]
    tag: PhantomData<Tag>,
}

#[cube]
impl<P: ReduceDType, RA: ReduceArgs> TensorArg<P, RA, Input> {
    pub fn new_input(state: &RA::State<P>) -> Self {
        TensorArg::<P, RA, Input> {
            state: state.clone(),
            tag: PhantomData::<Input>,
        }
    }
}

#[cube]
impl<P: ReduceDType, RA: ReduceArgs> TensorArg<P, RA, Output> {
    pub fn new_output(state: &mut RA::State<P>) -> Self {
        TensorArg::<P, RA, Output> {
            state: state.clone(),
            tag: PhantomData::<Output>,
        }
    }
}

impl<P: ReduceDType, RA: ReduceArgs> VirtualTensorOperations<P::Out, P::SizeOut>
    for TensorArg<P, RA, Output>
{
}
impl<P: ReduceDType, RA: ReduceArgs> VirtualTensorOperations<P::In, P::SizeIn>
    for TensorArg<P, RA, Input>
{
}

impl<P: ReduceDType, RA: ReduceArgs> VirtualTensorOperationsExpand<P::In, P::SizeIn>
    for TensorArgExpand<P, RA, Input>
{
    fn __expand_read_method(
        &self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> NativeExpand<Vector<P::In, P::SizeIn>> {
        RA::__expand_read_input(scope, &self.state, index)
    }

    fn __expand_write_method(
        &self,
        _scope: &Scope,
        _index: NativeExpand<usize>,
        _value: NativeExpand<Vector<P::In, P::SizeIn>>,
    ) {
        unreachable!("Can't write to input")
    }

    fn __expand_shape_method(
        &self,
        scope: &Scope,
        axis: NativeExpand<usize>,
    ) -> NativeExpand<usize> {
        RA::__expand_shape_input(scope, &self.state, axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &Scope,
        axis: NativeExpand<usize>,
    ) -> NativeExpand<usize> {
        RA::__expand_stride_input(scope, &self.state, axis)
    }

    fn __expand_rank_method(&self, scope: &Scope) -> NativeExpand<usize> {
        RA::__expand_rank_input(scope, &self.state)
    }
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        RA::__expand_len_input(scope, &self.state)
    }
    fn __expand_buffer_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        RA::__expand_buffer_len_input(scope, &self.state)
    }

    fn __expand_read_window_method(
        &self,
        _context: &Scope,
        _start: NativeExpand<usize>,
        _end: NativeExpand<usize>,
    ) -> &SliceExpand<Vector<P::In, P::SizeIn>> {
        panic!("Unsupported")
    }

    fn __expand_as_tensor_map_method(
        &self,
        scope: &Scope,
    ) -> ComptimeOptionExpand<TensorMap<P::In, Tiled>> {
        ComptimeOption::__expand_new_None(scope)
    }
}

impl<P: ReduceDType, RA: ReduceArgs> Vectorized for TensorArg<P, RA, Input> {}
impl<P: ReduceDType, RA: ReduceArgs> VectorizedExpand for TensorArgExpand<P, RA, Input> {
    fn vector_size(&self) -> usize {
        let scope = Scope::root(false);
        RA::__expand_vector_size_input(&scope, &self.state)
    }
}

impl<P: ReduceDType, RA: ReduceArgs> VirtualTensorOperationsExpand<P::Out, P::SizeOut>
    for TensorArgExpand<P, RA, Output>
{
    fn __expand_read_method(
        &self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> NativeExpand<Vector<P::Out, P::SizeOut>> {
        RA::__expand_read_output(scope, &self.state, index)
    }

    fn __expand_write_method(
        &self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: NativeExpand<Vector<P::Out, P::SizeOut>>,
    ) {
        let mut state = self.state.clone();
        RA::__expand_write_output(scope, &mut state, index, value);
    }

    fn __expand_shape_method(
        &self,
        scope: &Scope,
        axis: NativeExpand<usize>,
    ) -> NativeExpand<usize> {
        RA::__expand_shape_output(scope, &self.state, axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &Scope,
        axis: NativeExpand<usize>,
    ) -> NativeExpand<usize> {
        RA::__expand_stride_output(scope, &self.state, axis)
    }

    fn __expand_rank_method(&self, scope: &Scope) -> NativeExpand<usize> {
        RA::__expand_rank_output(scope, &self.state)
    }

    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        RA::__expand_len_output(scope, &self.state)
    }
    fn __expand_buffer_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        RA::__expand_buffer_len_output(scope, &self.state)
    }

    fn __expand_read_window_method(
        &self,
        _context: &Scope,
        _start: NativeExpand<usize>,
        _end: NativeExpand<usize>,
    ) -> &SliceExpand<Vector<P::Out, P::SizeOut>> {
        panic!("Unsupported")
    }

    fn __expand_as_tensor_map_method(
        &self,
        scope: &Scope,
    ) -> ComptimeOptionExpand<TensorMap<P::Out, Tiled>> {
        ComptimeOption::__expand_new_None(scope)
    }
}

impl<P: ReduceDType, RA: ReduceArgs> Vectorized for TensorArg<P, RA, Output> {}
impl<P: ReduceDType, RA: ReduceArgs> VectorizedExpand for TensorArgExpand<P, RA, Output> {
    fn vector_size(&self) -> usize {
        let scope = Scope::root(false);
        RA::__expand_vector_size_output(&scope, &self.state)
    }
}
