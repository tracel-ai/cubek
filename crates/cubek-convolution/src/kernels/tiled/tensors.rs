use super::{B, CI, CO, ConvBinding, ConvBlueprint, ConvOperation, ConvProblem, ConvWindow};
use cubecl::client::Client;
use cubecl::prelude::TensorBinding;
use cubecl::tile::Axis;
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::{BoundaryPolicy, Refusal};
use cubecl::tile::layout::ElemAlignment;
use cubek_matmul::definition::MatmulElems;
use cubek_microkernel::matmul::MatmulOperands;
use cubek_std::InputBinding;
use cubek_std::routine::Grid;

/// What a launch declines before preparing anything: a bias, which is not part of the contraction.
pub const CONV_BIAS_UNREAD: Refusal = Refusal::Declined {
    why: "the tiled forward convolution adds no bias",
};

/// What a launch declines before preparing anything: a weight or weight gradient whose channels
/// are not the input's.
pub const CONV_GROUPS_UNREAD: Refusal = Refusal::Declined {
    why: "the tiled convolution contracts every input channel, so it has no groups",
};

/// What a launch declines before preparing anything: operands that are not 2D NHWC tensors and a
/// `[co, kh, kw, ci]` weight or weight gradient, or arguments of another dimension.
pub const CONV_NOT_2D: Refusal = Refusal::Declined {
    why: "the tiled convolution reads 2D NHWC tensors and a [co, kh, kw, ci] weight",
};

/// What a launch declines before preparing anything: an operand that is quantized, whose scales
/// the binding would drop.
pub const CONV_QUANTIZED_UNREAD: Refusal = Refusal::Declined {
    why: "the tiled convolution reads float operands, a quantized one would lose its scales",
};

/// What a launch declines before preparing anything: the gradient of the input at a stride above 1,
/// which reads the output gradient only at every stride-th position.
pub const CONV_DATA_BACKWARD_STRIDED: Refusal = Refusal::Declined {
    why: "the tiled data backward convolution has a stride of 1, the gradient read at every stride-th position is not stated",
};

/// What a launch declines before preparing anything: an output of other channels than the
/// weight's filters.
pub const CONV_OUT_CHANNELS_MISMATCH: Refusal = Refusal::Declined {
    why: "the tiled convolution's output channels are its weight's filters",
};

/// What a launch declines before preparing anything: operands of different batches.
pub const CONV_BATCH_MISMATCH: Refusal = Refusal::Declined {
    why: "the tiled convolution's operands share one batch",
};

/// The tensors of one convolution on one client, held by the role they play in the contraction:
/// the gathered input, the rhs (the weight, or the output gradient) and the output (the
/// activation, or the weight gradient).
#[derive(Clone)]
pub struct ConvTensors {
    operation: ConvOperation,
    client: Client,
    input: TensorBinding,
    rhs: TensorBinding,
    out: TensorBinding,
    window: ConvWindow,
    dtypes: MatmulElems,
}

impl ConvTensors {
    /// The forward convolution of `input` and `weight` into `out` over `window`, each element as
    /// `dtypes` says.
    ///
    /// # Errors
    ///
    /// The refusals above: what the routines decline whatever the device.
    pub fn forward(
        client: &Client,
        input: InputBinding,
        weight: InputBinding,
        bias: Option<InputBinding>,
        out: TensorBinding,
        window: ConvWindow,
        dtypes: MatmulElems,
    ) -> Result<Self, Refusal> {
        if bias.is_some() {
            return Err(CONV_BIAS_UNREAD);
        }
        let operation = ConvOperation::Forward;
        Self::checked(operation, client, [input, weight], out, window, dtypes)
    }

    /// The weight gradient of `input` and `out_grad` into `weight_grad` over `window`, each
    /// element as `dtypes` says.
    ///
    /// # Errors
    ///
    /// The refusals above: what the routines decline whatever the device.
    pub fn backward_weight(
        client: &Client,
        input: InputBinding,
        out_grad: InputBinding,
        weight_grad: TensorBinding,
        window: ConvWindow,
        dtypes: MatmulElems,
    ) -> Result<Self, Refusal> {
        let operation = ConvOperation::BackwardWeight;
        let operands = [input, out_grad];
        Self::checked(operation, client, operands, weight_grad, window, dtypes)
    }

    /// The data gradient of `out_grad` and `weights` into `in_grad` over `window`, each element
    /// as `dtypes` says.
    ///
    /// # Errors
    ///
    /// [`CONV_DATA_BACKWARD_STRIDED`] and the refusals above.
    pub fn backward_data(
        client: &Client,
        out_grad: InputBinding,
        weights: InputBinding,
        in_grad: TensorBinding,
        window: ConvWindow,
        dtypes: MatmulElems,
    ) -> Result<Self, Refusal> {
        if window.stride.iter().any(|&stride| stride > 1) {
            return Err(CONV_DATA_BACKWARD_STRIDED);
        }
        let operation = ConvOperation::BackwardData;
        let operands = [out_grad, weights];
        Self::checked(operation, client, operands, in_grad, window, dtypes)
    }

    /// The tensors of `operation`, `[input, rhs]` and `out`, once the roles agree on every extent
    /// they share: the batch, the input channels and the output channels.
    fn checked(
        operation: ConvOperation,
        client: &Client,
        [input, rhs]: [InputBinding; 2],
        out: TensorBinding,
        window: ConvWindow,
        dtypes: MatmulElems,
    ) -> Result<Self, Refusal> {
        let (InputBinding::Normal(input, _), InputBinding::Normal(rhs, _)) = (input, rhs) else {
            return Err(CONV_QUANTIZED_UNREAD);
        };
        if [&input, &rhs, &out]
            .iter()
            .any(|tensor| tensor.shape.len() != 4)
        {
            return Err(CONV_NOT_2D);
        }
        let roles = operation.roles();
        let (rhs, out) = (roles.rhs_view(rhs)?, roles.out_view(out)?);
        let named = [(&roles.rhs[..], &rhs), (&roles.free[..], &out)];
        let agree = |axis: Axis, input: Option<usize>| {
            let mut extents = input
                .into_iter()
                .chain(named.iter().filter_map(|(axes, tensor)| {
                    let dim = axes.iter().position(|&named| named == axis)?;
                    Some(tensor.shape[dim])
                }));
            let first = extents.next();
            extents.all(|extent| Some(extent) == first)
        };
        let channels = |axis| (axis == roles.inner.lhs).then_some(input.shape[3]);
        if !agree(CI, channels(CI)) {
            return Err(CONV_GROUPS_UNREAD);
        }
        if !agree(CO, channels(CO)) {
            return Err(CONV_OUT_CHANNELS_MISMATCH);
        }
        if !agree(B, Some(input.shape[0])) {
            return Err(CONV_BATCH_MISMATCH);
        }
        Ok(Self {
            operation,
            client: client.clone(),
            input,
            rhs,
            out,
            window,
            dtypes,
        })
    }

    /// The problem these tensors state.
    pub fn problem(&self) -> ConvProblem {
        let (input, rhs, out) = (&self.input.shape, &self.rhs.shape, &self.out.shape);
        let (channels_in, channels_out, kernel, out_extent) = match self.operation {
            ConvOperation::Forward => (input[3], rhs[0], [rhs[1], rhs[2]], [out[1], out[2]]),
            ConvOperation::BackwardWeight => (input[3], out[3], [out[0], out[1]], [rhs[1], rhs[2]]),
            ConvOperation::BackwardData => (out[3], input[3], [rhs[0], rhs[1]], [out[1], out[2]]),
        };
        let start = |tensor: &TensorBinding| tensor.handle.offset_start.unwrap_or(0);
        ConvProblem {
            operation: self.operation,
            batch: input[0],
            channels_in,
            gathered: [input[1], input[2]],
            channels_out,
            kernel,
            out: out_extent,
            window: self.window,
            elements: MatmulOperands {
                lhs: self.dtypes.lhs_global,
                rhs: self.dtypes.rhs_global,
                out: self.dtypes.acc_global,
            },
            sum: self.dtypes.acc_register,
            alignments: MatmulOperands {
                lhs: ElemAlignment::from(&self.input),
                rhs: ElemAlignment::from(&self.rhs),
                out: ElemAlignment::from(&self.out),
            },
            start_bytes: MatmulOperands {
                lhs: start(&self.input),
                rhs: start(&self.rhs),
                out: start(&self.out),
            },
        }
    }

    /// These tensors of `problem` bound under the launcher `grid` spawns for `blueprint`, each
    /// masked where the box overhangs it.
    ///
    /// # Errors
    ///
    /// A width the binding cannot serve one of the operands at.
    pub fn bind<M>(
        self,
        problem: &ConvProblem,
        blueprint: &ConvBlueprint<M>,
        grid: &Grid,
    ) -> Result<ConvBinding, Refusal> {
        let partitioning = &blueprint.partitioning;
        let launcher = grid.launcher(&self.client, || {
            (partitioning.as_tile().clone(), partitioning.space().clone())
        });
        let widths = blueprint.widths;
        let roles = problem.operation.roles();
        let input = launcher
            .arg(self.input, problem.elements.lhs)
            .gathered(self.window.input(problem.operation, roles.lhs))
            .vectorize(widths.lhs);
        // A box that overhangs the across axis gathers past the input even where every real
        // output position reads inside it.
        let inside = problem.reads_inside() && !partitioning.overhangs(roles.across());
        let input = match inside {
            true => input,
            false => input.boundary(BoundaryPolicy::Every(Boundary::Zero)),
        };
        let rhs = launcher
            .arg(self.rhs, problem.elements.rhs)
            .axes(&roles.rhs)
            .vectorize(widths.rhs);
        let mut out = launcher
            .arg(self.out, problem.elements.out)
            .axes(&roles.free)
            .vectorize(widths.out)
            .build()?;
        if blueprint.masks_out && !out.spec.is_checked() {
            let masked = roles
                .free
                .map(|axis| (axis == roles.axes().m).then_some(Boundary::Zero));
            out.spec = out.spec.boundaries(&masked);
        }
        let (input, rhs) = (input.build()?, rhs.build()?);
        Ok(ConvBinding {
            launcher,
            input: input.arg(),
            rhs: rhs.arg(),
            out: out.arg(),
        })
    }
}
