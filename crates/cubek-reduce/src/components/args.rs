use cubecl::prelude::*;
use cubecl::std::{
    CubeOption, CubeOptionExpand,
    tensor::r#virtual::{VirtualTensor, VirtualTensorOperations, VirtualTensorOperationsExpand},
};
use cubecl::unexpanded;
use std::marker::PhantomData;

pub trait ReduceDType {
    type In: Numeric;
    type Out: Numeric;
}

impl<In: Numeric, Out: Numeric> ReduceDType for (In, Out) {
    type In = In;
    type Out = Out;
}

#[cube]
#[allow(dead_code)]
pub trait ReduceArgs: Send + Sync + 'static + Clone {
    type Input<E: Numeric>: LaunchArg + CubeType;

    type Output<E: Numeric>: LaunchArg + CubeType;

    type State<P: ReduceDType>: CubeType;

    fn init_state<P: ReduceDType>(
        input: &Self::Input<P::In>,
        output: &mut Self::Output<P::Out>,
    ) -> Self::State<P>;

    fn read_input<P: ReduceDType>(state: &Self::State<P>, index: usize) -> Line<P::In>;
    fn read_output<P: ReduceDType>(state: &Self::State<P>, index: usize) -> Line<P::Out>;

    fn write_output<P: ReduceDType>(state: &mut Self::State<P>, index: usize, value: Line<P::Out>);

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

    fn line_size_input<P: ReduceDType>(state: &Self::State<P>) -> comptime_type!(LineSize);
    fn line_size_output<P: ReduceDType>(state: &Self::State<P>) -> comptime_type!(LineSize);
}

#[cube]
pub fn init_tensors<RA: ReduceArgs, In: Numeric, Out: Numeric>(
    input: &RA::Input<In>,
    output: &mut RA::Output<Out>,
) -> (VirtualTensor<In>, VirtualTensor<Out, ReadWrite>) {
    let mut state = RA::init_state::<(In, Out)>(input, output);

    let input = TensorArg::new_input(&state);
    let mut output = TensorArg::new_output(&mut state);

    let input = VirtualTensor::<In>::new::<TensorArg<(In, Out), RA, Input>>(&input);
    let output =
        VirtualTensor::<Out, ReadWrite>::new::<TensorArg<(In, Out), RA, Output>>(&mut output);

    (input, output)
}

#[derive(Clone)]
pub struct TensorArgs;

#[cube]
impl ReduceArgs for TensorArgs {
    type Input<EG: Numeric> = Tensor<Line<EG>>;
    type Output<EG: Numeric> = Tensor<Line<EG>>;
    type State<P: ReduceDType> = (*const Tensor<Line<P::In>>, *mut Tensor<Line<P::Out>>);

    fn init_state<P: ReduceDType>(
        input: &Self::Input<P::In>,
        output: &mut Self::Output<P::Out>,
    ) -> Self::State<P> {
        (input, output)
    }

    fn read_input<P: ReduceDType>(state: &Self::State<P>, index: usize) -> Line<P::In> {
        unsafe { (*state.0)[index] }
    }

    fn read_output<P: ReduceDType>(state: &Self::State<P>, index: usize) -> Line<P::Out> {
        unsafe { (*state.1)[index] }
    }

    fn write_output<P: ReduceDType>(state: &mut Self::State<P>, index: usize, value: Line<P::Out>) {
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

    fn line_size_input<P: ReduceDType>(state: &Self::State<P>) -> comptime_type!(LineSize) {
        unsafe { (*state.0).line_size() }
    }

    fn line_size_output<P: ReduceDType>(state: &Self::State<P>) -> comptime_type!(LineSize) {
        unsafe { (*state.1).line_size() }
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

impl<P: ReduceDType, RA: ReduceArgs> VirtualTensorOperations<P::Out> for TensorArg<P, RA, Output> {}
impl<P: ReduceDType, RA: ReduceArgs> VirtualTensorOperations<P::In> for TensorArg<P, RA, Input> {}

impl<P: ReduceDType, RA: ReduceArgs> VirtualTensorOperationsExpand<P::In>
    for TensorArgExpand<P, RA, Input>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<Line<P::In>> {
        RA::__expand_read_input(scope, self.state.clone(), index)
    }

    fn __expand_write_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<usize>,
        _value: ExpandElementTyped<Line<P::In>>,
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
    ) -> SliceExpand<Line<P::In>, ReadOnly> {
        panic!("Unsupported")
    }

    fn __expand_as_tensor_map_method(
        &self,
        scope: &mut Scope,
    ) -> CubeOptionExpand<TensorMap<P::In, Tiled>> {
        CubeOption::__expand_new_None(scope)
    }
}

impl<P: ReduceDType, RA: ReduceArgs> Lined for TensorArg<P, RA, Input> {}
impl<P: ReduceDType, RA: ReduceArgs> LinedExpand for TensorArgExpand<P, RA, Input> {
    fn line_size(&self) -> usize {
        let mut scope = Scope::root(false);
        RA::__expand_line_size_input(&mut scope, self.state.clone())
    }
}

impl<P: ReduceDType, RA: ReduceArgs> VirtualTensorOperationsExpand<P::Out>
    for TensorArgExpand<P, RA, Output>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<Line<P::Out>> {
        RA::__expand_read_output(scope, self.state.clone(), index)
    }

    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<usize>,
        value: ExpandElementTyped<Line<P::Out>>,
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
    ) -> SliceExpand<Line<P::Out>, ReadOnly> {
        panic!("Unsupported")
    }

    fn __expand_as_tensor_map_method(
        &self,
        scope: &mut Scope,
    ) -> CubeOptionExpand<TensorMap<P::Out, Tiled>> {
        CubeOption::__expand_new_None(scope)
    }
}

impl<P: ReduceDType, RA: ReduceArgs> Lined for TensorArg<P, RA, Output> {}
impl<P: ReduceDType, RA: ReduceArgs> LinedExpand for TensorArgExpand<P, RA, Output> {
    fn line_size(&self) -> usize {
        let mut scope = Scope::root(false);
        RA::__expand_line_size_output(&mut scope, self.state.clone())
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
