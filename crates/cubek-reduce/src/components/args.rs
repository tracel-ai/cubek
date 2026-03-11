use cubecl::prelude::*;
use cubecl::std::tensor::r#virtual::{
    VirtualTensor, VirtualTensorOperations, VirtualTensorOperationsExpand,
};
use cubecl::unexpanded;
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

pub trait NumericLine {
    type T: Numeric;
    type N: Size;
}

impl<T: Numeric, N: Size> NumericLine for (T, N) {
    type T = T;
    type N = N;
}

#[cube]
#[allow(dead_code)]
pub trait ReduceArgs: Send + Sync + 'static + Clone {
    type Input<E: Numeric, S: Size>: LaunchArg + CubeType;

    type Output<E: Numeric, S: Size>: LaunchArg + CubeType;

    type State<P: ReduceDType>: CubeType;

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
    let mut output = TensorArg::new_output(&mut state);

    let input = VirtualTensor::<In, InSize>::new::<
        TensorArg<((In, InSize), (Out, OutSize)), RA, Input>,
    >(&input);
    let output = VirtualTensor::<Out, OutSize, ReadWrite>::new::<
        TensorArg<((In, InSize), (Out, OutSize)), RA, Output>,
    >(&mut output);

    (input, output)
}

#[derive(Clone)]
pub struct TensorArgs;

#[cube]
impl ReduceArgs for TensorArgs {
    type Input<EG: Numeric, N: Size> = Tensor<Vector<EG, N>>;
    type Output<EG: Numeric, N: Size> = Tensor<Vector<EG, N>>;
    type State<P: ReduceDType> = (
        *const Tensor<Vector<P::In, P::SizeIn>>,
        *mut Tensor<Vector<P::Out, P::SizeOut>>,
    );

    fn init_state<P: ReduceDType>(
        input: &Self::Input<P::In, P::SizeIn>,
        output: &mut Self::Output<P::Out, P::SizeOut>,
    ) -> Self::State<P> {
        (input, output)
    }

    fn read_input<P: ReduceDType>(
        state: &Self::State<P>,
        index: usize,
    ) -> Vector<P::In, P::SizeIn> {
        unsafe { (*state.0)[index] }
    }

    fn read_output<P: ReduceDType>(
        state: &Self::State<P>,
        index: usize,
    ) -> Vector<P::Out, P::SizeOut> {
        unsafe { (*state.1)[index] }
    }

    fn write_output<P: ReduceDType>(
        state: &mut Self::State<P>,
        index: usize,
        value: Vector<P::Out, P::SizeOut>,
    ) {
        unsafe { (*state.1)[index] = value }
    }

    fn buffer_len_input<P: ReduceDType>(state: &Self::State<P>) -> usize {
        unsafe { (*state.0).buffer_len() }
    }

    fn buffer_len_output<P: ReduceDType>(state: &Self::State<P>) -> usize {
        unsafe { (*state.1).buffer_len() }
    }

    fn len_input<P: ReduceDType>(state: &Self::State<P>) -> usize {
        unsafe { (*state.0).len() }
    }

    fn len_output<P: ReduceDType>(state: &Self::State<P>) -> usize {
        unsafe { (*state.1).len() }
    }
    fn rank_input<P: ReduceDType>(state: &Self::State<P>) -> usize {
        unsafe { (*state.0).rank() }
    }

    fn rank_output<P: ReduceDType>(state: &Self::State<P>) -> usize {
        unsafe { (*state.1).rank() }
    }

    fn shape_input<P: ReduceDType>(state: &Self::State<P>, dim: usize) -> usize {
        unsafe { (*state.0).shape(dim) }
    }

    fn shape_output<P: ReduceDType>(state: &Self::State<P>, dim: usize) -> usize {
        unsafe { (*state.1).shape(dim) }
    }

    fn stride_input<P: ReduceDType>(state: &Self::State<P>, dim: usize) -> usize {
        unsafe { (*state.0).stride(dim) }
    }

    fn stride_output<P: ReduceDType>(state: &Self::State<P>, dim: usize) -> usize {
        unsafe { (*state.1).stride(dim) }
    }

    fn vector_size_input<P: ReduceDType>(state: &Self::State<P>) -> comptime_type!(VectorSize) {
        unsafe { (*state.0).vector_size() }
    }

    fn vector_size_output<P: ReduceDType>(state: &Self::State<P>) -> comptime_type!(VectorSize) {
        unsafe { (*state.1).vector_size() }
    }
}

pub struct Input;
pub struct Output;

pub struct TensorArg<P: ReduceDType, RA: ReduceArgs, Tag> {
    _state: *mut RA::State<P>,
    tag: PhantomData<Tag>,
}

pub struct TensorArgExpand<P: ReduceDType, RA: ReduceArgs, Tag> {
    state: <RA::State<P> as CubeType>::ExpandType,
    tag: PhantomData<Tag>,
}

impl<P: ReduceDType, RA: ReduceArgs> TensorArg<P, RA, Input> {
    pub fn new_input(_state: &RA::State<P>) -> Self {
        unexpanded!()
    }
    pub fn __expand_new_input(
        _scope: &mut Scope,
        state: <RA::State<P> as CubeType>::ExpandType,
    ) -> TensorArgExpand<P, RA, Input> {
        TensorArgExpand {
            state,
            tag: PhantomData,
        }
    }
}

impl<P: ReduceDType, RA: ReduceArgs> TensorArg<P, RA, Output> {
    pub fn new_output(_state: &mut RA::State<P>) -> Self {
        unexpanded!()
    }
    pub fn __expand_new_output(
        _scope: &mut Scope,
        state: <RA::State<P> as CubeType>::ExpandType,
    ) -> TensorArgExpand<P, RA, Output> {
        TensorArgExpand {
            state,
            tag: PhantomData,
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
        scope: &mut Scope,
        index: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<Vector<P::In, P::SizeIn>> {
        RA::__expand_read_input(scope, self.state.clone(), index)
    }

    fn __expand_write_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<usize>,
        _value: ExpandElementTyped<Vector<P::In, P::SizeIn>>,
    ) {
        unreachable!("Can't write to input")
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<usize> {
        RA::__expand_shape_input(scope, self.state.clone(), axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<usize> {
        RA::__expand_stride_input(scope, self.state.clone(), axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        RA::__expand_rank_input(scope, self.state.clone())
    }
    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        RA::__expand_len_input(scope, self.state.clone())
    }
    fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        RA::__expand_buffer_len_input(scope, self.state.clone())
    }

    fn __expand_read_window_method(
        &self,
        _context: &mut Scope,
        _start: ExpandElementTyped<usize>,
        _end: ExpandElementTyped<usize>,
    ) -> SliceExpand<Vector<P::In, P::SizeIn>, ReadOnly> {
        panic!("Unsupported")
    }

    fn __expand_as_tensor_map_method(
        &self,
        scope: &mut Scope,
    ) -> ComptimeOptionExpand<TensorMap<P::In, Tiled>> {
        ComptimeOption::__expand_new_None(scope)
    }
}

impl<P: ReduceDType, RA: ReduceArgs> Vectorized for TensorArg<P, RA, Input> {}
impl<P: ReduceDType, RA: ReduceArgs> VectorizedExpand for TensorArgExpand<P, RA, Input> {
    fn vector_size(&self) -> usize {
        let mut scope = Scope::root(false);
        RA::__expand_vector_size_input(&mut scope, self.state.clone())
    }
}

impl<P: ReduceDType, RA: ReduceArgs> VirtualTensorOperationsExpand<P::Out, P::SizeOut>
    for TensorArgExpand<P, RA, Output>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<Vector<P::Out, P::SizeOut>> {
        RA::__expand_read_output(scope, self.state.clone(), index)
    }

    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<usize>,
        value: ExpandElementTyped<Vector<P::Out, P::SizeOut>>,
    ) {
        RA::__expand_write_output(scope, self.state.clone(), index, value)
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<usize> {
        RA::__expand_shape_output(scope, self.state.clone(), axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<usize> {
        RA::__expand_stride_output(scope, self.state.clone(), axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        RA::__expand_rank_output(scope, self.state.clone())
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        RA::__expand_len_output(scope, self.state.clone())
    }
    fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        RA::__expand_buffer_len_output(scope, self.state.clone())
    }

    fn __expand_read_window_method(
        &self,
        _context: &mut Scope,
        _start: ExpandElementTyped<usize>,
        _end: ExpandElementTyped<usize>,
    ) -> SliceExpand<Vector<P::Out, P::SizeOut>, ReadOnly> {
        panic!("Unsupported")
    }

    fn __expand_as_tensor_map_method(
        &self,
        scope: &mut Scope,
    ) -> ComptimeOptionExpand<TensorMap<P::Out, Tiled>> {
        ComptimeOption::__expand_new_None(scope)
    }
}

impl<P: ReduceDType, RA: ReduceArgs> Vectorized for TensorArg<P, RA, Output> {}
impl<P: ReduceDType, RA: ReduceArgs> VectorizedExpand for TensorArgExpand<P, RA, Output> {
    fn vector_size(&self) -> usize {
        let mut scope = Scope::root(false);
        RA::__expand_vector_size_output(&mut scope, self.state.clone())
    }
}

mod __tensor_arg {
    use super::*;

    impl<P: ReduceDType, RA: ReduceArgs, Tag> CubeType for TensorArg<P, RA, Tag> {
        type ExpandType = TensorArgExpand<P, RA, Tag>;
    }

    impl<P: ReduceDType, RA: ReduceArgs, Tag> IntoMut for TensorArgExpand<P, RA, Tag> {
        fn into_mut(self, _scope: &mut Scope) -> Self {
            self
        }
    }

    impl<P: ReduceDType, RA: ReduceArgs, Tag> CubeDebug for TensorArgExpand<P, RA, Tag> {}
    impl<P: ReduceDType, RA: ReduceArgs, Tag> Clone for TensorArgExpand<P, RA, Tag> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
                tag: self.tag,
            }
        }
    }
}
