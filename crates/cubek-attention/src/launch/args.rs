use cubecl::prelude::*;
use cubecl::std::tensor::r#virtual::{VirtualTensorOperations, VirtualTensorOperationsExpand};
use cubecl::{self as cubecl};

use crate::definition::{AttentionBlueprint, AttentionProblem};

/// Create the input runtime arguments for a attention kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteInputsFactory: LaunchArg {
    fn create<R: Runtime>(
        query: TensorBinding<R>,
        key: TensorBinding<R>,
        value: TensorBinding<R>,
        mask: Option<TensorBinding<R>>,
        selection: &AttentionBlueprint,
        problem: &AttentionProblem,
    ) -> Self::RuntimeArg<R>;
}

/// Create the output runtime argument for a attention kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteOutputFactory: LaunchArg {
    fn create<R: Runtime>(
        out: TensorBinding<R>,
        selection: &AttentionBlueprint,
        problem: &AttentionProblem,
    ) -> Self::RuntimeArg<R>;
}

pub trait FloatLine: 'static {
    type T: Float;
    type N: Size;
}

impl<T: Float, N: Size> FloatLine for (T, N) {
    type T = T;
    type N = N;
}

pub trait NumericLine: 'static {
    type T: Numeric;
    type N: Size;
}

impl<T: Numeric, N: Size> NumericLine for (T, N) {
    type T = T;
    type N = N;
}

#[cube]
/// Arguments for the attention algorithm.
pub trait AttentionArgs: Send + Sync + 'static + Clone {
    /// Type used for the input.
    type Input<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine>: LaunchArg + CubeType;
    /// Type used for the output.
    type Output<O: FloatLine>: LaunchArg + CubeType;
    /// Inner state that is used to create tensor inputs and
    /// [tensor outputs](TensorOutput) .
    type State<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>: CubeType;

    /// Init the state.
    fn init_state<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        input: &Self::Input<Q, K, V, M>,
        output: &mut Self::Output<O>,
    ) -> Self::State<Q, K, V, M, O>;

    /// Whether the mask argument is present. Returns `Option` to allow matching at
    /// comptime
    fn has_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> ComptimeOption<()>;

    /// Read the vector of the query tensor using the state at the given coordinate.
    fn read_query<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        coordinate: usize,
    ) -> Vector<Q::T, Q::N>;
    /// Read the vector of the key tensor using the state at the given coordinate.
    fn read_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        coordinate: usize,
    ) -> Vector<K::T, K::N>;
    /// Read the vector of the value tensor using the state at the given coordinate.
    fn read_value<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        coordinate: usize,
    ) -> Vector<V::T, V::N>;
    /// Read the vector of the mask tensor using the state at the given coordinate.
    fn read_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        coordinate: usize,
    ) -> Vector<M::T, M::N>;

    /// Read the vector of the query tensor using the state at the given coordinate.
    fn read_window_query<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        start: usize,
        end: usize,
    ) -> Slice<Vector<Q::T, Q::N>>;
    /// Read the vector of the key tensor using the state at the given coordinate.
    fn read_window_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        start: usize,
        end: usize,
    ) -> Slice<Vector<K::T, K::N>>;
    /// Read the vector of the value tensor using the state at the given coordinate.
    fn read_window_value<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        start: usize,
        end: usize,
    ) -> Slice<Vector<V::T, V::N>>;
    /// Read the vector of the mask tensor using the state at the given coordinate.
    fn read_window_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        start: usize,
        end: usize,
    ) -> Slice<Vector<M::T, M::N>>;

    /// Reinterpret query as tensor map
    fn as_tensor_map_query<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> ComptimeOption<TensorMap<Q::T, Tiled>>;
    /// Reinterpret key as tensor map
    fn as_tensor_map_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> ComptimeOption<TensorMap<K::T, Tiled>>;
    /// Reinterpret value as tensor map
    fn as_tensor_map_value<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> ComptimeOption<TensorMap<V::T, Tiled>>;
    /// Reinterpret mask as tensor map
    fn as_tensor_map_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> ComptimeOption<TensorMap<M::T, Tiled>>;

    /// Write the vector to the output at the given coordinate using the state.
    fn write_out<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &mut Self::State<Q, K, V, M, O>,
        coordinate: usize,
        val: Vector<O::T, O::N>,
    );

    /// Get the rank of the query tensor using the state.
    fn rank_query<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize;
    /// Get the rank of the key tensor using the state.
    fn rank_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize;
    /// Get the rank of the value tensor using the state.
    fn rank_value<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize;
    /// Get the rank of the mask tensor using the state.
    fn rank_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize;
    /// Get the rank of the out tensor using the state.
    fn rank_out<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize;

    /// Get the length of the query tensor using the state.
    fn len_query<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize;
    /// Get the length of the key tensor using the state.
    fn len_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize;
    /// Get the length of the value tensor using the state.
    fn len_value<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize;
    /// Get the length of the mask tensor using the state.
    fn len_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize;
    /// Get the length of the out tensor using the state.
    fn len_out<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize;

    /// Get the buffer length of the query tensor using the state.
    fn buffer_len_query<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize;
    /// Get the buffer length of the key tensor using the state.
    fn buffer_len_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize;
    /// Get the buffer length of the value tensor using the state.
    fn buffer_len_value<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize;
    /// Get the buffer length of the mask tensor using the state.
    fn buffer_len_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize;
    /// Get the buffer length of the out tensor using the state.
    fn buffer_len_out<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize;

    /// Get the shape of the query tensor using the state.
    fn shape_query<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        axis: usize,
    ) -> usize;
    /// Get the shape of the key tensor using the state.
    fn shape_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        axis: usize,
    ) -> usize;
    /// Get the shape of the value tensor using the state.
    fn shape_value<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        axis: usize,
    ) -> usize;
    /// Get the shape of the mask tensor using the state.
    fn shape_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        axis: usize,
    ) -> usize;
    /// Get the shape of the out tensor using the state.
    fn shape_out<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        axis: usize,
    ) -> usize;

    /// Get the stride of the query tensor using the state.
    fn stride_query<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        axis: usize,
    ) -> usize;
    /// Get the stride of the key tensor using the state.
    fn stride_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        axis: usize,
    ) -> usize;
    /// Get the stride of the value tensor using the state.
    fn stride_value<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        axis: usize,
    ) -> usize;
    /// Get the stride of the mask tensor using the state.
    fn stride_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        axis: usize,
    ) -> usize;
    /// Get the stride of the out tensor using the state.
    fn stride_out<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        axis: usize,
    ) -> usize;

    /// Get the vector size of the query tensor using the state.
    fn vector_size_query<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> comptime_type!(VectorSize);
    /// Get the vector size of the key tensor using the state.
    fn vector_size_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> comptime_type!(VectorSize);
    /// Get the vector size of the value tensor using the state.
    fn vector_size_value<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> comptime_type!(VectorSize);
    /// Get the vector size of the mask tensor using the state.
    fn vector_size_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> comptime_type!(VectorSize);
    /// Get the vector size of the out tensor using the state.
    fn vector_size_out<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> comptime_type!(VectorSize);
}

/// Tensor input representation.
///
/// You can use the tensor input as if it was a pointer to the actually tensor.
pub struct TensorQuery<
    Q: FloatLine,
    K: FloatLine,
    V: FloatLine,
    M: NumericLine,
    O: FloatLine,
    GA: AttentionArgs,
> {
    state: *const GA::State<Q, K, V, M, O>,
}

pub struct TensorKey<
    Q: FloatLine,
    K: FloatLine,
    V: FloatLine,
    M: NumericLine,
    O: FloatLine,
    GA: AttentionArgs,
> {
    state: *const GA::State<Q, K, V, M, O>,
}

pub struct TensorValue<
    Q: FloatLine,
    K: FloatLine,
    V: FloatLine,
    M: NumericLine,
    O: FloatLine,
    GA: AttentionArgs,
> {
    state: *const GA::State<Q, K, V, M, O>,
}

pub struct TensorMask<
    Q: FloatLine,
    K: FloatLine,
    V: FloatLine,
    M: NumericLine,
    O: FloatLine,
    GA: AttentionArgs,
> {
    state: *const GA::State<Q, K, V, M, O>,
}

impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    VirtualTensorOperations<Q::T, Q::N> for TensorQuery<Q, K, V, M, O, MA>
{
}

impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    VirtualTensorOperations<K::T, K::N> for TensorKey<Q, K, V, M, O, MA>
{
}

impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    VirtualTensorOperations<V::T, V::N> for TensorValue<Q, K, V, M, O, MA>
{
}

impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    VirtualTensorOperations<M::T, M::N> for TensorMask<Q, K, V, M, O, MA>
{
}

impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    VirtualTensorOperations<O::T, O::N> for TensorOutput<Q, K, V, M, O, MA>
{
}

impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    VirtualTensorOperationsExpand<O::T, O::N> for TensorOutputExpand<Q, K, V, M, O, MA>
{
    fn __expand_read_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<Vector<O::T, O::N>> {
        panic!("Can't read output tensor");
    }

    fn __expand_read_window_method(
        &self,
        _context: &mut Scope,
        _start: ExpandElementTyped<usize>,
        _end: ExpandElementTyped<usize>,
    ) -> SliceExpand<Vector<O::T, O::N>, ReadOnly> {
        panic!("Can't read output tensor");
    }

    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<usize>,
        val: ExpandElementTyped<Vector<O::T, O::N>>,
    ) {
        TensorOutputExpand::__expand_write_method(self.clone(), scope, index, val)
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<usize> {
        TensorOutputExpand::__expand_shape_method(self.clone(), scope, axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<usize> {
        TensorOutputExpand::__expand_stride_method(self.clone(), scope, axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        TensorOutputExpand::__expand_rank_method(self.clone(), scope)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        TensorOutputExpand::__expand_len_method(self.clone(), scope)
    }

    fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        TensorOutputExpand::__expand_buffer_len_method(self.clone(), scope)
    }

    fn __expand_as_tensor_map_method(
        &self,
        scope: &mut Scope,
    ) -> ComptimeOptionExpand<TensorMap<O::T, Tiled>> {
        ComptimeOption::__expand_new_None(scope)
    }
}

impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    Vectorized for TensorOutput<Q, K, V, M, O, MA>
{
}
impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    VectorizedExpand for TensorOutputExpand<Q, K, V, M, O, MA>
{
    fn vector_size(&self) -> VectorSize {
        let mut scope = Scope::root(false);
        TensorOutputExpand::__expand_vector_size_method(self.clone(), &mut scope)
    }
}

impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    VirtualTensorOperationsExpand<Q::T, Q::N> for TensorQueryExpand<Q, K, V, M, O, MA>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<Vector<Q::T, Q::N>> {
        TensorQueryExpand::__expand_read_method(self.clone(), scope, index)
    }
    fn __expand_read_window_method(
        &self,
        context: &mut Scope,
        start: ExpandElementTyped<usize>,
        end: ExpandElementTyped<usize>,
    ) -> SliceExpand<Vector<Q::T, Q::N>, ReadOnly> {
        TensorQueryExpand::__expand_read_window_method(self.clone(), context, start, end)
    }

    fn __expand_write_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<usize>,
        _val: ExpandElementTyped<Vector<Q::T, Q::N>>,
    ) {
        panic!("Can't write to input tensor");
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<usize> {
        TensorQueryExpand::__expand_shape_method(self.clone(), scope, axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<usize> {
        TensorQueryExpand::__expand_stride_method(self.clone(), scope, axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        TensorQueryExpand::__expand_rank_method(self.clone(), scope)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        TensorQueryExpand::__expand_len_method(self.clone(), scope)
    }

    fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        TensorQueryExpand::__expand_buffer_len_method(self.clone(), scope)
    }

    fn __expand_as_tensor_map_method(
        &self,
        scope: &mut Scope,
    ) -> ComptimeOptionExpand<TensorMap<Q::T, Tiled>> {
        TensorQueryExpand::__expand_as_tensor_map_method(self.clone(), scope)
    }
}

impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    Vectorized for TensorQuery<Q, K, V, M, O, MA>
{
}
impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    VectorizedExpand for TensorQueryExpand<Q, K, V, M, O, MA>
{
    fn vector_size(&self) -> VectorSize {
        let mut scope = Scope::root(false);
        TensorQueryExpand::__expand_vector_size_method(self.clone(), &mut scope)
    }
}

impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    VirtualTensorOperationsExpand<K::T, K::N> for TensorKeyExpand<Q, K, V, M, O, MA>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<Vector<K::T, K::N>> {
        TensorKeyExpand::__expand_read_method(self.clone(), scope, index)
    }
    fn __expand_read_window_method(
        &self,
        context: &mut Scope,
        start: ExpandElementTyped<usize>,
        end: ExpandElementTyped<usize>,
    ) -> SliceExpand<Vector<K::T, K::N>, ReadOnly> {
        TensorKeyExpand::__expand_read_window_method(self.clone(), context, start, end)
    }

    fn __expand_write_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<usize>,
        _val: ExpandElementTyped<Vector<K::T, K::N>>,
    ) {
        panic!("Can't write to input tensor");
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<usize> {
        TensorKeyExpand::__expand_shape_method(self.clone(), scope, axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<usize> {
        TensorKeyExpand::__expand_stride_method(self.clone(), scope, axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        TensorKeyExpand::__expand_rank_method(self.clone(), scope)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        TensorKeyExpand::__expand_len_method(self.clone(), scope)
    }

    fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        TensorKeyExpand::__expand_buffer_len_method(self.clone(), scope)
    }

    fn __expand_as_tensor_map_method(
        &self,
        scope: &mut Scope,
    ) -> ComptimeOptionExpand<TensorMap<K::T, Tiled>> {
        TensorKeyExpand::__expand_as_tensor_map_method(self.clone(), scope)
    }
}

impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    Vectorized for TensorKey<Q, K, V, M, O, MA>
{
}
impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    VectorizedExpand for TensorKeyExpand<Q, K, V, M, O, MA>
{
    fn vector_size(&self) -> VectorSize {
        let mut scope = Scope::root(false);
        TensorKeyExpand::__expand_vector_size_method(self.clone(), &mut scope)
    }
}

impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    VirtualTensorOperationsExpand<V::T, V::N> for TensorValueExpand<Q, K, V, M, O, MA>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<Vector<V::T, V::N>> {
        TensorValueExpand::__expand_read_method(self.clone(), scope, index)
    }
    fn __expand_read_window_method(
        &self,
        context: &mut Scope,
        start: ExpandElementTyped<usize>,
        end: ExpandElementTyped<usize>,
    ) -> SliceExpand<Vector<V::T, V::N>, ReadOnly> {
        TensorValueExpand::__expand_read_window_method(self.clone(), context, start, end)
    }

    fn __expand_write_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<usize>,
        _val: ExpandElementTyped<Vector<V::T, V::N>>,
    ) {
        panic!("Can't write to input tensor");
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<usize> {
        TensorValueExpand::__expand_shape_method(self.clone(), scope, axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<usize> {
        TensorValueExpand::__expand_stride_method(self.clone(), scope, axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        TensorValueExpand::__expand_rank_method(self.clone(), scope)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        TensorValueExpand::__expand_len_method(self.clone(), scope)
    }

    fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        TensorValueExpand::__expand_buffer_len_method(self.clone(), scope)
    }

    fn __expand_as_tensor_map_method(
        &self,
        scope: &mut Scope,
    ) -> ComptimeOptionExpand<TensorMap<V::T, Tiled>> {
        TensorValueExpand::__expand_as_tensor_map_method(self.clone(), scope)
    }
}

impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    Vectorized for TensorValue<Q, K, V, M, O, MA>
{
}
impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    VectorizedExpand for TensorValueExpand<Q, K, V, M, O, MA>
{
    fn vector_size(&self) -> VectorSize {
        let mut scope = Scope::root(false);
        TensorValueExpand::__expand_vector_size_method(self.clone(), &mut scope)
    }
}

impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    VirtualTensorOperationsExpand<M::T, M::N> for TensorMaskExpand<Q, K, V, M, O, MA>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<Vector<M::T, M::N>> {
        TensorMaskExpand::__expand_read_method(self.clone(), scope, index)
    }
    fn __expand_read_window_method(
        &self,
        context: &mut Scope,
        start: ExpandElementTyped<usize>,
        end: ExpandElementTyped<usize>,
    ) -> SliceExpand<Vector<M::T, M::N>, ReadOnly> {
        TensorMaskExpand::__expand_read_window_method(self.clone(), context, start, end)
    }

    fn __expand_write_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<usize>,
        _val: ExpandElementTyped<Vector<M::T, M::N>>,
    ) {
        panic!("Can't write to input tensor");
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<usize> {
        TensorMaskExpand::__expand_shape_method(self.clone(), scope, axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<usize> {
        TensorMaskExpand::__expand_stride_method(self.clone(), scope, axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        TensorMaskExpand::__expand_rank_method(self.clone(), scope)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        TensorMaskExpand::__expand_len_method(self.clone(), scope)
    }

    fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        TensorMaskExpand::__expand_buffer_len_method(self.clone(), scope)
    }

    fn __expand_as_tensor_map_method(
        &self,
        scope: &mut Scope,
    ) -> ComptimeOptionExpand<TensorMap<M::T, Tiled>> {
        TensorMaskExpand::__expand_as_tensor_map_method(self.clone(), scope)
    }
}

impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    Vectorized for TensorMask<Q, K, V, M, O, MA>
{
}
impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    VectorizedExpand for TensorMaskExpand<Q, K, V, M, O, MA>
{
    fn vector_size(&self) -> VectorSize {
        let mut scope = Scope::root(false);
        TensorMaskExpand::__expand_vector_size_method(self.clone(), &mut scope)
    }
}

/// Tensor output representation.
///
/// You can use the tensor output as if it was a pointer to the actually tensor.
///
/// # Warning
/// # Warning
///
/// There is no mutability guarantee.
pub struct TensorOutput<
    Q: FloatLine,
    K: FloatLine,
    V: FloatLine,
    M: NumericLine,
    O: FloatLine,
    GA: AttentionArgs,
> {
    state: *mut GA::State<Q, K, V, M, O>,
}

/// Expand type for tensor input.
pub struct TensorQueryExpand<
    Q: FloatLine,
    K: FloatLine,
    V: FloatLine,
    M: NumericLine,
    O: FloatLine,
    GA: AttentionArgs,
> {
    state: <GA::State<Q, K, V, M, O> as CubeType>::ExpandType,
}

pub struct TensorKeyExpand<
    Q: FloatLine,
    K: FloatLine,
    V: FloatLine,
    M: NumericLine,
    O: FloatLine,
    GA: AttentionArgs,
> {
    state: <GA::State<Q, K, V, M, O> as CubeType>::ExpandType,
}

pub struct TensorValueExpand<
    Q: FloatLine,
    K: FloatLine,
    V: FloatLine,
    M: NumericLine,
    O: FloatLine,
    GA: AttentionArgs,
> {
    state: <GA::State<Q, K, V, M, O> as CubeType>::ExpandType,
}

pub struct TensorMaskExpand<
    Q: FloatLine,
    K: FloatLine,
    V: FloatLine,
    M: NumericLine,
    O: FloatLine,
    GA: AttentionArgs,
> {
    state: <GA::State<Q, K, V, M, O> as CubeType>::ExpandType,
}

/// Expand type for [tensor output](TensorOutput).
pub struct TensorOutputExpand<
    Q: FloatLine,
    K: FloatLine,
    V: FloatLine,
    M: NumericLine,
    O: FloatLine,
    GA: AttentionArgs,
> {
    state: <GA::State<Q, K, V, M, O> as CubeType>::ExpandType,
}

#[cube]
impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    TensorQuery<Q, K, V, M, O, MA>
{
    /// Create a tensor input from the state and the ident.
    pub fn new(state: &MA::State<Q, K, V, M, O>) -> TensorQuery<Q, K, V, M, O, MA> {
        TensorQuery::<Q, K, V, M, O, MA> { state }
    }

    //// Read the tensor at the given coordinate.
    pub fn read_window(&self, start: usize, end: usize) -> Slice<Vector<Q::T, Q::N>> {
        unsafe { MA::read_window_query(&(*self.state), start, end) }
    }

    /// Read the tensor at the given coordinate.
    pub fn read(&self, coordinate: usize) -> Vector<Q::T, Q::N> {
        unsafe { MA::read_query(&(*self.state), coordinate) }
    }

    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: usize) -> usize {
        unsafe { MA::shape_query(&(*self.state), axis) }
    }

    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, axis: usize) -> usize {
        unsafe { MA::stride_query(&(*self.state), axis) }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> usize {
        unsafe { MA::rank_query(&(*self.state)) }
    }

    /// Get the length of the tensor.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        unsafe { MA::len_query(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn buffer_len(&self) -> usize {
        unsafe { MA::buffer_len_query(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn as_tensor_map(&self) -> ComptimeOption<TensorMap<Q::T, Tiled>> {
        unsafe { MA::as_tensor_map_query(&(*self.state)) }
    }

    /// Get the vector size of the tensor.
    pub fn vector_size(&self) -> comptime_type!(VectorSize) {
        unsafe { MA::vector_size_query(&(*self.state)) }
    }
}

#[cube]
impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    TensorKey<Q, K, V, M, O, MA>
{
    /// Create a tensor input from the state and the ident.
    pub fn new(state: &MA::State<Q, K, V, M, O>) -> TensorKey<Q, K, V, M, O, MA> {
        TensorKey::<Q, K, V, M, O, MA> { state }
    }

    //// Read the tensor at the given coordinate.
    pub fn read_window(&self, start: usize, end: usize) -> Slice<Vector<K::T, K::N>> {
        unsafe { MA::read_window_key(&(*self.state), start, end) }
    }

    /// Read the tensor at the given coordinate.
    pub fn read(&self, coordinate: usize) -> Vector<K::T, K::N> {
        unsafe { MA::read_key(&(*self.state), coordinate) }
    }

    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: usize) -> usize {
        unsafe { MA::shape_key(&(*self.state), axis) }
    }

    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, axis: usize) -> usize {
        unsafe { MA::stride_key(&(*self.state), axis) }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> usize {
        unsafe { MA::rank_key(&(*self.state)) }
    }

    /// Get the length of the tensor.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        unsafe { MA::len_key(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn buffer_len(&self) -> usize {
        unsafe { MA::buffer_len_key(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn as_tensor_map(&self) -> ComptimeOption<TensorMap<K::T, Tiled>> {
        unsafe { MA::as_tensor_map_key(&(*self.state)) }
    }

    /// Get the vector size of the tensor.
    pub fn vector_size(&self) -> comptime_type!(VectorSize) {
        unsafe { MA::vector_size_key(&(*self.state)) }
    }
}

#[cube]
impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    TensorValue<Q, K, V, M, O, MA>
{
    /// Create a tensor input from the state and the ident.
    pub fn new(state: &MA::State<Q, K, V, M, O>) -> TensorValue<Q, K, V, M, O, MA> {
        TensorValue::<Q, K, V, M, O, MA> { state }
    }

    //// Read the tensor at the given coordinate.
    pub fn read_window(&self, start: usize, end: usize) -> Slice<Vector<V::T, V::N>> {
        unsafe { MA::read_window_value(&(*self.state), start, end) }
    }

    /// Read the tensor at the given coordinate.
    pub fn read(&self, coordinate: usize) -> Vector<V::T, V::N> {
        unsafe { MA::read_value(&(*self.state), coordinate) }
    }

    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: usize) -> usize {
        unsafe { MA::shape_value(&(*self.state), axis) }
    }

    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, axis: usize) -> usize {
        unsafe { MA::stride_value(&(*self.state), axis) }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> usize {
        unsafe { MA::rank_value(&(*self.state)) }
    }

    /// Get the length of the tensor.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        unsafe { MA::len_value(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn buffer_len(&self) -> usize {
        unsafe { MA::buffer_len_value(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn as_tensor_map(&self) -> ComptimeOption<TensorMap<V::T, Tiled>> {
        unsafe { MA::as_tensor_map_value(&(*self.state)) }
    }

    /// Get the vector size of the tensor.
    pub fn vector_size(&self) -> comptime_type!(VectorSize) {
        unsafe { MA::vector_size_value(&(*self.state)) }
    }
}

#[cube]
impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, MA: AttentionArgs>
    TensorMask<Q, K, V, M, O, MA>
{
    /// Create a tensor input from the state and the ident.
    pub fn new(state: &MA::State<Q, K, V, M, O>) -> TensorMask<Q, K, V, M, O, MA> {
        TensorMask::<Q, K, V, M, O, MA> { state }
    }

    //// Read the tensor at the given coordinate.
    pub fn read_window(&self, start: usize, end: usize) -> Slice<Vector<M::T, M::N>> {
        unsafe { MA::read_window_mask(&(*self.state), start, end) }
    }

    /// Read the tensor at the given coordinate.
    pub fn read(&self, coordinate: usize) -> Vector<M::T, M::N> {
        unsafe { MA::read_mask(&(*self.state), coordinate) }
    }

    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: usize) -> usize {
        unsafe { MA::shape_mask(&(*self.state), axis) }
    }

    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, axis: usize) -> usize {
        unsafe { MA::stride_mask(&(*self.state), axis) }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> usize {
        unsafe { MA::rank_mask(&(*self.state)) }
    }

    /// Get the length of the tensor.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        unsafe { MA::len_mask(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn buffer_len(&self) -> usize {
        unsafe { MA::buffer_len_mask(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn as_tensor_map(&self) -> ComptimeOption<TensorMap<M::T, Tiled>> {
        unsafe { MA::as_tensor_map_mask(&(*self.state)) }
    }

    /// Get the vector size of the tensor.
    pub fn vector_size(&self) -> comptime_type!(VectorSize) {
        unsafe { MA::vector_size_mask(&(*self.state)) }
    }
}

#[cube]
impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
    TensorOutput<Q, K, V, M, O, GA>
{
    /// Create a [tensor output](TensorOutput) from the state.
    pub fn new(state: &mut GA::State<Q, K, V, M, O>) -> TensorOutput<Q, K, V, M, O, GA> {
        TensorOutput::<Q, K, V, M, O, GA> { state }
    }

    /// Write the val to tensor at the given coordinate.
    pub fn write(&self, coordinate: usize, val: Vector<O::T, O::N>) {
        unsafe { GA::write_out(&mut (*self.state), coordinate, val) }
    }

    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: usize) -> usize {
        unsafe { GA::shape_out(&(*self.state), axis) }
    }

    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, dim: usize) -> usize {
        unsafe { GA::stride_out(&(*self.state), dim) }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> usize {
        unsafe { GA::rank_out(&(*self.state)) }
    }

    /// Get the length of the tensor.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        unsafe { GA::len_out(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn buffer_len(&self) -> usize {
        unsafe { GA::buffer_len_out(&(*self.state)) }
    }

    /// Get the vector size of the tensor.
    pub fn vector_size(&self) -> comptime_type!(VectorSize) {
        unsafe { GA::vector_size_out(&(*self.state)) }
    }
}

#[derive(Clone)]
/// Type implementing [AttentionArgs] where all inputs and the output are materialized tensors.
///
/// Other types might implement [AttentionArgs] for fused matrix multiplication kernels.
pub struct TensorArgs;

#[derive(CubeLaunch, CubeType)]
#[launch(skip_bounds)]
/// Input representation for [TensorArgs] implementing [AttentionArgs].
pub struct TensorInputs<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine> {
    pub query: Tensor<Vector<Q::T, Q::N>>,
    pub key: Tensor<Vector<K::T, K::N>>,
    pub value: Tensor<Vector<V::T, V::N>>,
    pub mask: ComptimeOption<Tensor<Vector<M::T, M::N>>>,
}

impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine> ConcreteInputsFactory
    for TensorInputs<Q, K, V, M>
{
    fn create<R: Runtime>(
        query: TensorBinding<R>,
        key: TensorBinding<R>,
        value: TensorBinding<R>,
        mask: Option<TensorBinding<R>>,
        _selection: &AttentionBlueprint,
        _problem: &AttentionProblem,
    ) -> Self::RuntimeArg<R> {
        TensorInputsLaunch::new(
            query.into_tensor_arg(),
            key.into_tensor_arg(),
            value.into_tensor_arg(),
            match mask {
                Some(mask) => ComptimeOptionArgs::Some(mask.into_tensor_arg()),
                None => ComptimeOptionArgs::None,
            },
        )
    }
}

impl<EG: Numeric, EGS: Size> ConcreteOutputFactory for Tensor<Vector<EG, EGS>> {
    fn create<'a, R: Runtime>(
        out: TensorBinding<R>,
        _selection: &AttentionBlueprint,
        _problem: &AttentionProblem,
    ) -> Self::RuntimeArg<R> {
        out.into_tensor_arg()
    }
}

#[derive(CubeType)]
pub struct AttentionState<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine> {
    pub query: *const Tensor<Vector<Q::T, Q::N>>,
    pub key: *const Tensor<Vector<K::T, K::N>>,
    pub value: *const Tensor<Vector<V::T, V::N>>,
    pub mask: ComptimeOption<*const Tensor<Vector<M::T, M::N>>>,
    pub output: *mut Tensor<Vector<O::T, O::N>>,
}

#[cube]
impl AttentionArgs for TensorArgs {
    type Input<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine> = TensorInputs<Q, K, V, M>;
    type Output<O: FloatLine> = Tensor<Vector<O::T, O::N>>;
    type State<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine> =
        AttentionState<Q, K, V, M, O>;

    fn init_state<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        input: &Self::Input<Q, K, V, M>,
        output: &mut Self::Output<O>,
    ) -> Self::State<Q, K, V, M, O> {
        let mask = input.mask.as_ref().map(|mask| {
            let ptr: *const Tensor<Vector<M::T, M::N>> = mask;
            ptr
        });

        AttentionState::<Q, K, V, M, O> {
            query: &input.query,
            key: &input.key,
            value: &input.value,
            mask,
            output,
        }
    }

    fn has_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> ComptimeOption<()> {
        state.mask.as_ref().map(|_| ())
    }

    fn read_query<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        coordinate: usize,
    ) -> Vector<Q::T, Q::N> {
        unsafe { (*state.query)[coordinate] }
    }

    fn read_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        coordinate: usize,
    ) -> Vector<K::T, K::N> {
        unsafe { (*state.key)[coordinate] }
    }

    fn read_value<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        coordinate: usize,
    ) -> Vector<V::T, V::N> {
        unsafe { (*state.value)[coordinate] }
    }

    fn read_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        coordinate: usize,
    ) -> Vector<M::T, M::N> {
        unsafe { (*state.mask.unwrap())[coordinate] }
    }

    fn read_window_query<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        start: usize,
        end: usize,
    ) -> Slice<Vector<Q::T, Q::N>> {
        unsafe { (*state.query).slice(start, end) }
    }

    fn read_window_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        start: usize,
        end: usize,
    ) -> Slice<Vector<K::T, K::N>> {
        unsafe { (*state.key).slice(start, end) }
    }

    fn read_window_value<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        start: usize,
        end: usize,
    ) -> Slice<Vector<V::T, V::N>> {
        unsafe { (*state.value).slice(start, end) }
    }

    fn read_window_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        start: usize,
        end: usize,
    ) -> Slice<Vector<M::T, M::N>> {
        unsafe { (*state.mask.unwrap()).slice(start, end) }
    }

    fn as_tensor_map_query<
        Q: FloatLine,
        K: FloatLine,
        V: FloatLine,
        M: NumericLine,
        O: FloatLine,
    >(
        _state: &Self::State<Q, K, V, M, O>,
    ) -> ComptimeOption<TensorMap<Q::T, Tiled>> {
        ComptimeOption::new_None()
    }

    fn as_tensor_map_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        _state: &Self::State<Q, K, V, M, O>,
    ) -> ComptimeOption<TensorMap<K::T, Tiled>> {
        ComptimeOption::new_None()
    }

    fn as_tensor_map_value<
        Q: FloatLine,
        K: FloatLine,
        V: FloatLine,
        M: NumericLine,
        O: FloatLine,
    >(
        _state: &Self::State<Q, K, V, M, O>,
    ) -> ComptimeOption<TensorMap<V::T, Tiled>> {
        ComptimeOption::new_None()
    }

    fn as_tensor_map_mask<
        Q: FloatLine,
        K: FloatLine,
        V: FloatLine,
        M: NumericLine,
        O: FloatLine,
    >(
        _state: &Self::State<Q, K, V, M, O>,
    ) -> ComptimeOption<TensorMap<M::T, Tiled>> {
        ComptimeOption::new_None()
    }

    fn shape_query<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        dim: usize,
    ) -> usize {
        unsafe { (*state.query).shape(dim) }
    }

    fn shape_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        dim: usize,
    ) -> usize {
        unsafe { (*state.key).shape(dim) }
    }

    fn shape_value<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        dim: usize,
    ) -> usize {
        unsafe { (*state.value).shape(dim) }
    }

    fn shape_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        dim: usize,
    ) -> usize {
        unsafe { (*state.mask.unwrap()).shape(dim) }
    }

    fn shape_out<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        dim: usize,
    ) -> usize {
        unsafe { (*state.output).shape(dim) }
    }

    fn stride_query<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        dim: usize,
    ) -> usize {
        unsafe { (*state.query).stride(dim) }
    }

    fn stride_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        dim: usize,
    ) -> usize {
        unsafe { (*state.key).stride(dim) }
    }

    fn stride_value<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        dim: usize,
    ) -> usize {
        unsafe { (*state.value).stride(dim) }
    }

    fn stride_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        dim: usize,
    ) -> usize {
        unsafe { (*state.mask.unwrap()).stride(dim) }
    }

    fn stride_out<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
        dim: usize,
    ) -> usize {
        unsafe { (*state.output).stride(dim) }
    }

    fn write_out<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &mut Self::State<Q, K, V, M, O>,
        coordinate: usize,
        val: Vector<O::T, O::N>,
    ) {
        unsafe { (*state.output)[coordinate] = val }
    }

    fn rank_query<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize {
        unsafe { (*state.query).rank() }
    }

    fn rank_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize {
        unsafe { (*state.key).rank() }
    }

    fn rank_value<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize {
        unsafe { (*state.value).rank() }
    }

    fn rank_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize {
        unsafe { (*state.mask.unwrap()).rank() }
    }

    fn rank_out<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize {
        unsafe { (*state.output).rank() }
    }

    fn len_query<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize {
        unsafe { (*state.query).len() }
    }

    fn len_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize {
        unsafe { (*state.key).len() }
    }

    fn len_value<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize {
        unsafe { (*state.value).len() }
    }

    fn len_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize {
        unsafe { (*state.mask.unwrap()).len() }
    }

    fn len_out<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize {
        unsafe { (*state.output).len() }
    }

    fn buffer_len_query<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize {
        unsafe { (*state.query).buffer_len() }
    }

    fn buffer_len_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize {
        unsafe { (*state.key).buffer_len() }
    }

    fn buffer_len_value<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize {
        unsafe { (*state.value).buffer_len() }
    }

    fn buffer_len_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize {
        unsafe { (*state.mask.unwrap()).buffer_len() }
    }

    fn buffer_len_out<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> usize {
        unsafe { (*state.output).buffer_len() }
    }

    fn vector_size_query<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> comptime_type!(usize) {
        unsafe { (*state.query).vector_size() }
    }

    fn vector_size_key<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> comptime_type!(usize) {
        unsafe { (*state.key).vector_size() }
    }

    fn vector_size_value<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> comptime_type!(usize) {
        unsafe { (*state.value).vector_size() }
    }

    fn vector_size_mask<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> comptime_type!(usize) {
        unsafe { (*state.mask.unwrap()).vector_size() }
    }

    fn vector_size_out<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine>(
        state: &Self::State<Q, K, V, M, O>,
    ) -> comptime_type!(usize) {
        unsafe { (*state.output).vector_size() }
    }
}

mod __query {
    use super::*;

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        CubeType for TensorQuery<Q, K, V, M, O, GA>
    {
        type ExpandType = TensorQueryExpand<Q, K, V, M, O, GA>;
    }

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        Clone for TensorQueryExpand<Q, K, V, M, O, GA>
    {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        IntoMut for TensorQueryExpand<Q, K, V, M, O, GA>
    {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }
    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        CubeDebug for TensorQueryExpand<Q, K, V, M, O, GA>
    {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }
    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        Clone for TensorQuery<Q, K, V, M, O, GA>
    {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        Copy for TensorQuery<Q, K, V, M, O, GA>
    {
    }
}

mod __key {
    use super::*;

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        CubeType for TensorKey<Q, K, V, M, O, GA>
    {
        type ExpandType = TensorKeyExpand<Q, K, V, M, O, GA>;
    }

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        Clone for TensorKeyExpand<Q, K, V, M, O, GA>
    {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        IntoMut for TensorKeyExpand<Q, K, V, M, O, GA>
    {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }
    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        CubeDebug for TensorKeyExpand<Q, K, V, M, O, GA>
    {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }
    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        Clone for TensorKey<Q, K, V, M, O, GA>
    {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        Copy for TensorKey<Q, K, V, M, O, GA>
    {
    }
}

mod __value {
    use super::*;

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        CubeType for TensorValue<Q, K, V, M, O, GA>
    {
        type ExpandType = TensorValueExpand<Q, K, V, M, O, GA>;
    }

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        Clone for TensorValueExpand<Q, K, V, M, O, GA>
    {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        IntoMut for TensorValueExpand<Q, K, V, M, O, GA>
    {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }
    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        CubeDebug for TensorValueExpand<Q, K, V, M, O, GA>
    {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }
    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        Clone for TensorValue<Q, K, V, M, O, GA>
    {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        Copy for TensorValue<Q, K, V, M, O, GA>
    {
    }
}

mod __mask {
    use super::*;

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        CubeType for TensorMask<Q, K, V, M, O, GA>
    {
        type ExpandType = TensorMaskExpand<Q, K, V, M, O, GA>;
    }

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        Clone for TensorMaskExpand<Q, K, V, M, O, GA>
    {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        IntoMut for TensorMaskExpand<Q, K, V, M, O, GA>
    {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }
    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        CubeDebug for TensorMaskExpand<Q, K, V, M, O, GA>
    {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }
    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        Clone for TensorMask<Q, K, V, M, O, GA>
    {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        Copy for TensorMask<Q, K, V, M, O, GA>
    {
    }
}

mod __output {
    use super::*;

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        CubeType for TensorOutput<Q, K, V, M, O, GA>
    {
        type ExpandType = TensorOutputExpand<Q, K, V, M, O, GA>;
    }

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        Clone for TensorOutput<Q, K, V, M, O, GA>
    {
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        Clone for TensorOutputExpand<Q, K, V, M, O, GA>
    {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        IntoMut for TensorOutputExpand<Q, K, V, M, O, GA>
    {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        CubeDebug for TensorOutputExpand<Q, K, V, M, O, GA>
    {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }

    impl<Q: FloatLine, K: FloatLine, V: FloatLine, M: NumericLine, O: FloatLine, GA: AttentionArgs>
        Copy for TensorOutput<Q, K, V, M, O, GA>
    {
    }
}
