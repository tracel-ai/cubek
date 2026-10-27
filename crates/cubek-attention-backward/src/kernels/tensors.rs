//! The tensors of one backward pass, and how a walk binds them.

use crate::kernels::{
    AttentionBackwardMicrokernel, AttentionBackwardProblem, AttentionBackwardWalk,
};
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::Launcher;
use cubecl::tile::launch::{Bound, Grid as LaunchGrid, Refusal};
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, PlainTensor};
use cubek_microkernel::attention::AttentionVisibility;
use cubek_microkernel::attention_backward::AttentionBackwardOperands;
use cubek_std::routine::ADMITTED_PLAN_RUNS;

/// What a backward pass declines before preparing anything: an element the routines do not
/// take, inputs of two elements, or a logsumexp not stored in `f32`.
pub const ATTENTION_BACKWARD_ELEMENT_UNVERIFIED: Refusal = Refusal::Declined {
    why: "attention's backward pass takes f32, f16 and bf16 inputs of one element, and an f32 \
          logsumexp",
};

/// The tensors of one backward pass on one client: the forward's inputs, its output and
/// logsumexp, the loss's gradient against the output, and the three gradients it computes, with
/// the dot product of the output and its gradient the first walk leaves for the second.
///
/// Every operand is read where it lies, by its strides: the query's heads, and the output's, its
/// gradient's and the logsumexp's, are viewed as a group per key head, `[b, kv, g, q, ·]`.
/// Nothing is copied. The problem the tensors state is read off them.
#[derive(Clone)]
pub struct AttentionBackwardTensors {
    client: Client,
    /// The tensors as the kernels read and write them, `[b, kv, g, q, ·]` for a query's,
    /// `[b, kv, s, ·]` for a key's.
    views: AttentionBackwardOperands<PlainTensor>,
    gradients: AttentionBackwardGradients,
    scale: f32,
    visibility: AttentionVisibility,
}

impl core::fmt::Debug for AttentionBackwardTensors {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The client says nothing about the backward pass.
        f.debug_struct("AttentionBackwardTensors")
            .field("views", &self.views)
            .field("gradients", &self.gradients)
            .field("scale", &self.scale)
            .field("visibility", &self.visibility)
            .finish()
    }
}

/// The gradients a backward pass computes, each shaped as its input.
#[derive(Debug, Clone)]
pub struct AttentionBackwardGradients {
    /// `[b, h, q, d]`.
    pub query: TensorHandle,
    /// `[b, kv, s, d]`.
    pub key: TensorHandle,
    /// `[b, kv, s, dv]`.
    pub value: TensorHandle,
}

impl AttentionBackwardTensors {
    /// The elements the routines are verified at.
    pub(crate) const VERIFIED: [ElemType; 3] = [
        ElemType::Float(FloatKind::F32),
        ElemType::Float(FloatKind::F16),
        ElemType::Float(FloatKind::BF16),
    ];

    /// The backward pass of the attention of `query` (`[b, h, q, d]`) over `key` (`[b, kv, s, d]`)
    /// and `value` (`[b, kv, s, dv]`), the queries the last `q` positions of the keys, each seeing
    /// them as `visibility` says, each score taking `scale`: `out` (`[b, h, q, dv]`) and
    /// `logsumexp` (`[b, h, q]`, `f32`) as the forward left them, and `out_gradient` shaped as
    /// `out`. The gradients are allocated contiguous.
    ///
    /// # Errors
    ///
    /// [`ATTENTION_BACKWARD_ELEMENT_UNVERIFIED`] and [`ATTENTION_BACKWARD_GROUP_UNEVEN`]: what the
    /// routines decline whatever the device, and [`ELEMENT_NOT_ON_DEVICE`]: an element this
    /// device's runtime does not run.
    ///
    /// [`ATTENTION_BACKWARD_GROUP_UNEVEN`]: crate::kernels::ATTENTION_BACKWARD_GROUP_UNEVEN
    ///
    /// # Panics
    ///
    /// Operands whose ranks or dims do not agree: no attention at all.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: &Client,
        query: PlainTensor,
        key: PlainTensor,
        value: PlainTensor,
        out: PlainTensor,
        out_gradient: PlainTensor,
        logsumexp: PlainTensor,
        scale: f32,
        visibility: AttentionVisibility,
    ) -> Result<Self, Refusal> {
        let element = query.elem;
        let inputs = [&key, &value, &out, &out_gradient];
        if !Self::VERIFIED.contains(&element)
            || inputs.iter().any(|input| input.elem != element)
            || logsumexp.elem != AttentionBackwardProblem::SUM
        {
            return Err(ATTENTION_BACKWARD_ELEMENT_UNVERIFIED);
        }
        if !query.runs_on(client) {
            return Err(ELEMENT_NOT_ON_DEVICE);
        }
        let (q_shape, k_shape, v_shape) = (query.shape(), key.shape(), value.shape());
        assert!(
            q_shape.num_dims() == 4 && k_shape.num_dims() == 4 && v_shape.num_dims() == 4,
            "attention's backward pass reads [b, h, q, d], [b, kv, s, d] and [b, kv, s, dv]"
        );
        let [batch, heads, queries, head] = q_shape.dims::<4>();
        let [_, kv_heads, keys, _] = k_shape.dims::<4>();
        let value_dim = v_shape[3];
        assert!(
            k_shape[0] == batch
                && k_shape[3] == head
                && v_shape.dims::<4>()[..3] == k_shape.dims::<4>()[..3],
            "the keys hold the query's head dim, and the values a row per key"
        );
        assert!(
            out.shape().dims::<4>() == [batch, heads, queries, value_dim]
                && out_gradient.shape() == out.shape()
                && logsumexp.shape().dims::<3>() == [batch, heads, queries],
            "the output and its gradient are [b, h, q, dv], the logsumexp [b, h, q]"
        );
        let group = AttentionBackwardProblem::group_of(heads, kv_heads)?;
        let gradient =
            |shape: [usize; 4]| PlainTensor::empty_contiguous(client, shape.into(), element);
        let (query_gradient, query_gradient_view) = gradient([batch, heads, queries, head]);
        let (key_gradient, key_gradient_view) = gradient([batch, kv_heads, keys, head]);
        let (value_gradient, value_gradient_view) = gradient([batch, kv_heads, keys, value_dim]);
        let (_, delta) = PlainTensor::empty_contiguous(
            client,
            [batch, heads, queries].into(),
            AttentionBackwardProblem::SUM,
        );
        let grouped = |tensor: &PlainTensor| Self::grouped(tensor, group);
        Ok(Self {
            client: client.clone(),
            views: AttentionBackwardOperands {
                query: grouped(&query),
                key,
                value,
                out: grouped(&out),
                out_gradient: grouped(&out_gradient),
                logsumexp: grouped(&logsumexp),
                delta: grouped(&delta),
                query_gradient: grouped(&query_gradient_view),
                key_gradient: key_gradient_view,
                value_gradient: value_gradient_view,
            },
            gradients: AttentionBackwardGradients {
                query: query_gradient,
                key: key_gradient,
                value: value_gradient,
            },
            scale,
            visibility,
        })
    }

    /// `tensor`'s heads, its second dim, split into `group` heads a key head over the same buffer:
    /// `[b, h, ·]` viewed as `[b, h / group, group, ·]`.
    fn grouped(tensor: &PlainTensor, group: usize) -> PlainTensor {
        let (shape, strides) = (&tensor.binding.shape, &tensor.binding.strides);
        let mut view_shape = vec![shape[0], shape[1] / group, group];
        let mut view_strides = vec![strides[0], strides[1] * group, strides[1]];
        view_shape.extend(shape.iter().skip(2));
        view_strides.extend(strides.iter().skip(2));
        tensor.viewed(Metadata::new(view_shape, view_strides))
    }

    /// The problem the tensors state.
    pub(crate) fn problem(&self) -> AttentionBackwardProblem {
        let views = &self.views;
        let [batch, kv_heads, group, queries, head] = views.query.shape().dims::<5>();
        let [_, _, keys, value] = views.value.shape().dims::<4>();
        AttentionBackwardProblem {
            batch,
            num_kv_heads: kv_heads,
            group,
            seq_q: queries,
            seq_kv: keys,
            head_dim: head,
            val_dim: value,
            visibility: self.visibility,
            element: views.query.elem,
            elem_alignments: views.map(|_, view| view.elem_alignment()),
            start_bytes: views.map(|_, view| view.start_bytes()),
        }
    }

    /// The client the operands live on, which a launch submits to.
    pub(crate) fn client(&self) -> &Client {
        &self.client
    }

    /// The scale each dot product takes before the softmax.
    pub(crate) fn scale(&self) -> f32 {
        self.scale
    }

    /// The gradients, once both walks are launched.
    pub(crate) fn into_gradients(self) -> AttentionBackwardGradients {
        self.gradients
    }

    /// Every operand bound as `walk` states it: under its partitioning, at its widths, on its
    /// grid. A walk passes its kernel the ones it reads and writes.
    pub(crate) fn bind<M, SM>(
        &self,
        walk: &AttentionBackwardWalk<M, SM>,
    ) -> AttentionBackwardBinding
    where
        M: AttentionBackwardMicrokernel,
    {
        let launcher = Launcher::new(
            &self.client,
            walk.kernel_partitioning(),
            walk.partitioning.space(),
            LaunchGrid::Stated {
                cube_count: walk.grid.launch.cube_count.clone(),
                cube_dim: walk.grid.launch.cube_dim,
            },
        )
        .expect(ADMITTED_PLAN_RUNS);
        let operands = self.views.map(|operand, view| {
            launcher
                .arg(view.binding.clone(), view.elem)
                .axes(operand.axes())
                .vectorize(walk.widths.get(operand))
                .build()
                .expect(ADMITTED_PLAN_RUNS)
        });
        AttentionBackwardBinding { launcher, operands }
    }
}

/// The arguments of one walk's launch, its operands bound under its launcher. Not `Debug`: the
/// tile DSL's `Launcher` is not.
pub(crate) struct AttentionBackwardBinding {
    pub launcher: Launcher,
    pub operands: AttentionBackwardOperands<Bound>,
}
