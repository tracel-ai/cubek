use super::axes;
use crate::components::ConvolutionOperation;
use crate::kernels::tiled::{
    B, CO, ChannelAxis, ConvBias, ConvBinding, ConvChannelOperands, ConvDecline, ConvGrouping,
    ConvPhasing, ConvProblem, ConvRoles, SpatialDim, TileConvBlueprint,
};
use crate::launch::{SpatialExtent, SpatialWindow, TapCount};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::TensorBinding;
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::{BoundaryPolicy, Decline};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tune::Work;
use cubek_matmul::definition::MatmulElems;
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};
use cubek_std::InputBinding;
use cubek_std::routine::{EstimatedProblem, Grid};

/// The tensors of one convolution on one client, held by the role they play in the contraction:
/// the gathered input, the rhs (the weight, or the output gradient) and the output (the
/// activation, or the weight gradient), and a forward's bias. A grouped convolution's tensors have
/// their channels split by group.
#[derive(Clone)]
pub struct ConvTensors {
    operation: ConvolutionOperation,
    grouping: ConvGrouping,
    client: Client,
    input: TensorBinding,
    rhs: TensorBinding,
    out: TensorBinding,
    bias: Option<TensorBinding>,
    window: Vec<SpatialWindow>,
    dtypes: MatmulElems,
}

impl core::fmt::Debug for ConvTensors {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ConvTensors").finish_non_exhaustive()
    }
}

impl ConvTensors {
    /// The forward convolution of `input` and `weight` into `out` over `window`, each element as
    /// `dtypes` says.
    ///
    /// # Errors
    ///
    /// A [`ConvDecline`] reason, whatever the device.
    pub fn forward(
        client: &Client,
        input: InputBinding,
        weight: InputBinding,
        bias: Option<InputBinding>,
        out: TensorBinding,
        window: &[SpatialWindow],
        dtypes: MatmulElems,
    ) -> Result<Self, Decline> {
        let operation = ConvolutionOperation::Forward;
        let tensors = Self::checked(operation, client, [input, weight], out, window, dtypes)?;
        match bias {
            Some(bias) => tensors.with_bias(bias),
            None => Ok(tensors),
        }
    }

    /// The weight gradient of `input` and `out_grad` into `weight_grad` over `window`, each
    /// element as `dtypes` says.
    ///
    /// # Errors
    ///
    /// A [`ConvDecline`] reason, whatever the device.
    pub fn backward_weight(
        client: &Client,
        input: InputBinding,
        out_grad: InputBinding,
        weight_grad: TensorBinding,
        window: &[SpatialWindow],
        dtypes: MatmulElems,
    ) -> Result<Self, Decline> {
        let operation = ConvolutionOperation::BackwardWeight;
        let operands = [input, out_grad];
        Self::checked(operation, client, operands, weight_grad, window, dtypes)
    }

    /// The data gradient of `out_grad` and `weights` into `in_grad` over `window`, each element
    /// as `dtypes` says.
    ///
    /// # Errors
    ///
    /// [`ConvDecline::DataBackwardStridedDilated`], or another [`ConvDecline`] reason.
    pub fn backward_data(
        client: &Client,
        out_grad: InputBinding,
        weights: InputBinding,
        in_grad: TensorBinding,
        window: &[SpatialWindow],
        dtypes: MatmulElems,
    ) -> Result<Self, Decline> {
        if (window.iter()).any(|window| window.stride > 1 && window.dilation > 1) {
            return Err(ConvDecline::DataBackwardStridedDilated.into());
        }
        let operation = ConvolutionOperation::BackwardData;
        let operands = [out_grad, weights];
        Self::checked(operation, client, operands, in_grad, window, dtypes)
    }

    /// Whether the output holds no element, so a launch has nothing to compute.
    pub fn is_empty(&self) -> bool {
        self.out.shape.iter().any(|&extent| extent == 0)
    }

    /// The client these tensors live on.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// What the convolution these tensors state asks for, whichever routine runs it.
    pub fn demand(&self) -> Work {
        self.problem().demand()
    }

    /// The problem these tensors state. Their shapes are already the roles' views, as the checks
    /// that built them read each dim through those views.
    pub fn problem(&self) -> ConvProblem {
        let start = |tensor: &TensorBinding| tensor.handle.offset_start.unwrap_or(0);
        let shape = |tensor: &TensorBinding| tensor.shape.to_vec();
        let bias = match self.bias {
            Some(_) => ConvBias::PerChannel,
            None => ConvBias::Absent,
        };
        let roles = ConvRoles::of(self.operation, self.grouping, &self.window)
            .expect("tensors are checked when they are built");
        ConvProblem::from_views(
            &roles,
            &self.window,
            MatmulOperands {
                lhs: shape(&self.input),
                rhs: shape(&self.rhs),
                out: shape(&self.out),
            },
            &self.dtypes,
            MatmulOperands {
                lhs: ElemAlignment::from(&self.input),
                rhs: ElemAlignment::from(&self.rhs),
                out: ElemAlignment::from(&self.out),
            },
            MatmulOperands {
                lhs: start(&self.input),
                rhs: start(&self.rhs),
                out: start(&self.out),
            },
            bias,
        )
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
        blueprint: &TileConvBlueprint<M>,
        grid: &Grid,
    ) -> Result<ConvBinding, Decline> {
        let partitioning = &blueprint.partitioning;
        let roles = problem.roles();
        let launcher = grid.launcher(&self.client, || {
            (partitioning.kernel(&roles), partitioning.space().clone())
        })?;
        let widths = blueprint.widths;
        let input = launcher
            .arg(self.input, problem.elements.lhs)
            .gathered(problem.input_projection())
            .vectorize(widths.lhs);
        // A box that overhangs a gathered dimension reads past the input even where every real
        // output position reads inside it.
        let inside = problem.reads_inside() && !partitioning.overhangs_gathered(&roles);
        let input = match inside {
            true => input,
            false => input.boundary(BoundaryPolicy::Every(Boundary::Zero)),
        };
        let phased = roles.phasing != ConvPhasing::Whole;
        let rhs = launcher.arg(self.rhs, problem.elements.rhs);
        let rhs = match phased {
            true => rhs.gathered(problem.phased_weight_projection()),
            false => rhs.dimensions(&roles.rhs),
        }
        .vectorize(widths.rhs);
        // A tap past the weight's end reads zero, which only a checked axis does.
        let rhs = match problem.phased_tap_strides().is_empty() {
            true => rhs,
            false => rhs.boundary(BoundaryPolicy::Every(Boundary::Zero)),
        };
        let free = roles.free.in_order();
        let out = launcher.arg(self.out, problem.elements.out);
        let mut out = match phased {
            true => out.gathered(problem.phased_output_projection()),
            false => out.dimensions(&free),
        }
        .vectorize(widths.out)
        .build()?;
        let masked: Vec<_> = (roles.out_axes().into_iter())
            .map(|dimension| (dimension == roles.dimensions().m).then_some(Boundary::Zero))
            .collect();
        out.spec = match blueprint.masks_out && !out.spec.is_checked() {
            true => out.spec.boundaries(&masked),
            false => out.spec,
        };
        let (input, rhs) = (input.build()?, rhs.build()?);
        let bias = self
            .bias
            .map(|bias| {
                launcher
                    .arg(bias, problem.elements.out)
                    .dimensions(&roles.bias_dimensions())
                    .vectorize(1)
                    .build()
            })
            .transpose()?;
        Ok(ConvBinding {
            launcher,
            input: input.arg(),
            rhs: rhs
                .arg()
                .runtime_map(problem.phased_tap_strides(), Vec::new()),
            out: out.arg(),
            bias: bias.map(|bias| bias.arg()),
        })
    }
}

impl ConvTensors {
    /// The tensors of `operation`, `[input, rhs]` and `out`, once the roles agree on every extent
    /// they share: the batch, the input channels and the output channels, and the spatial extents
    /// the window states.
    fn checked(
        operation: ConvolutionOperation,
        client: &Client,
        [input, rhs]: [InputBinding; 2],
        out: TensorBinding,
        window: &[SpatialWindow],
        dtypes: MatmulElems,
    ) -> Result<Self, Decline> {
        let (InputBinding::Normal(input, _), InputBinding::Normal(rhs, _)) = (input, rhs) else {
            return Err(ConvDecline::Quantized.into());
        };
        if !matches!(dtypes.acc_global, ElemType::Float(_)) {
            return Err(ConvDecline::OutputElement.into());
        }
        let rank = window.len() + 2;
        if [&input, &rhs, &out]
            .iter()
            .any(|tensor| tensor.shape.len() != rank)
        {
            return Err(ConvDecline::RankMismatch.into());
        }
        let channels = ConvChannelOperands::of(operation)?;
        let shapes = MatmulOperands {
            lhs: &input.shape[..],
            rhs: &rhs.shape[..],
            out: &out.shape[..],
        };
        let grouping = ConvGrouping::of(channels.groups(&shapes)?);
        let roles = ConvRoles::of(operation, grouping, window)?;
        let input = axes::split(&roles, MatmulOperand::Lhs, input)?;
        let rhs = axes::split(&roles, MatmulOperand::Rhs, rhs)?;
        let out = axes::split(&roles, MatmulOperand::Out, out)?;
        let (rhs, out) = (axes::rhs_view(&roles, rhs)?, axes::out_view(&roles, out)?);
        let extents = &roles.extents;
        let channels = |dimension| {
            (dimension == roles.inner.lhs).then_some(
                extents
                    .lhs_channels(roles.inner.lhs)
                    .read_shape(&input.shape),
            )
        };
        let checks = [
            (CO, ConvDecline::OutChannels, channels(CO)),
            (
                B,
                ConvDecline::Batch,
                Some(extents.batch.read_shape(&input.shape)),
            ),
        ];
        if let Some(&(_, declined, _)) = checks.iter().find(|&&(dimension, _, extent)| {
            !axes::agree_on(&roles, dimension, extent, [&rhs, &out])
        }) {
            return Err(declined.into());
        }
        let tensors = Self {
            operation,
            grouping,
            client: client.clone(),
            input,
            rhs,
            out,
            bias: None,
            window: window.to_vec(),
            dtypes,
        };
        tensors.check_window()?;
        if !tensors.is_empty() && tensors.problem().contracts_nothing() {
            return Err(ConvDecline::ContractionEmpty.into());
        }
        Ok(tensors)
    }

    /// These tensors with `bias` added to every output channel.
    fn with_bias(self, bias: InputBinding) -> Result<Self, Decline> {
        let InputBinding::Normal(bias, element) = bias else {
            return Err(ConvDecline::Quantized.into());
        };
        let problem = self.problem();
        if bias.shape.len() != 1 || bias.shape[0] != problem.groups * problem.channels_out {
            return Err(ConvDecline::BiasShape.into());
        }
        if element != self.dtypes.acc_global {
            return Err(ConvDecline::BiasElement.into());
        }
        let bias =
            axes::split_channels(ChannelAxis::First, bias, ConvGrouping::of(problem.groups))?;
        Ok(Self {
            bias: Some(bias),
            ..self
        })
    }

    /// Whether the window is one a convolution steps by and the spatial extents are the ones it
    /// yields.
    fn check_window(&self) -> Result<(), Decline> {
        let problem = self.problem();
        if problem
            .spatial
            .iter()
            .any(|spatial_dim| spatial_dim.kernel == 0)
        {
            return Err(ConvDecline::KernelEmpty.into());
        }
        let steps_zero = |spatial_dim: &SpatialDim| {
            spatial_dim.window.stride == 0 || spatial_dim.window.dilation == 0
        };
        if problem.spatial.iter().any(steps_zero) {
            return Err(ConvDecline::WindowStepZero.into());
        }
        let roles = problem.roles();
        let states_output = |spatial_dim: &SpatialDim| {
            let (input, output) = roles.gather.window_ends(spatial_dim);
            spatial_dim
                .window
                .output(SpatialExtent(input), TapCount(spatial_dim.kernel))
                == output
        };
        match problem.spatial.iter().all(states_output) {
            true => Ok(()),
            false => Err(roles.extent_mismatch.into()),
        }
    }
}
