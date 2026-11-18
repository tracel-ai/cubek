use super::{
    B, CI, CO, ConvBinding, ConvDecline, ConvDim, ConvProblem, ConvRoles, TileConvBlueprint,
};
use crate::components::ConvolutionOperation;
use crate::launch::SpatialWindow;
use cubecl::client::Client;
use cubecl::prelude::TensorBinding;
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::{BoundaryPolicy, Refusal};
use cubecl::tile::layout::ElemAlignment;
use cubek_matmul::definition::MatmulElems;
use cubek_microkernel::matmul::MatmulOperands;
use cubek_std::InputBinding;
use cubek_std::routine::Grid;

/// The tensors of one convolution on one client, held by the role they play in the contraction:
/// the gathered input, the rhs (the weight, or the output gradient) and the output (the
/// activation, or the weight gradient).
#[derive(Clone)]
pub struct ConvTensors {
    operation: ConvolutionOperation,
    client: Client,
    input: TensorBinding,
    rhs: TensorBinding,
    out: TensorBinding,
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
    ) -> Result<Self, Refusal> {
        if bias.is_some() {
            return Err(ConvDecline::Bias.into());
        }
        let operation = ConvolutionOperation::Forward;
        Self::checked(operation, client, [input, weight], out, window, dtypes)
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
    ) -> Result<Self, Refusal> {
        let operation = ConvolutionOperation::BackwardWeight;
        let operands = [input, out_grad];
        Self::checked(operation, client, operands, weight_grad, window, dtypes)
    }

    /// The data gradient of `out_grad` and `weights` into `in_grad` over `window`, each element
    /// as `dtypes` says.
    ///
    /// # Errors
    ///
    /// [`ConvDecline::DataBackwardStrided`], or another [`ConvDecline`] reason.
    pub fn backward_data(
        client: &Client,
        out_grad: InputBinding,
        weights: InputBinding,
        in_grad: TensorBinding,
        window: &[SpatialWindow],
        dtypes: MatmulElems,
    ) -> Result<Self, Refusal> {
        if window.iter().any(|window| window.stride > 1) {
            return Err(ConvDecline::DataBackwardStrided.into());
        }
        let operation = ConvolutionOperation::BackwardData;
        let operands = [out_grad, weights];
        Self::checked(operation, client, operands, in_grad, window, dtypes)
    }

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
    ) -> Result<Self, Refusal> {
        let (InputBinding::Normal(input, _), InputBinding::Normal(rhs, _)) = (input, rhs) else {
            return Err(ConvDecline::Quantized.into());
        };
        let rank = window.len() + 2;
        if [&input, &rhs, &out]
            .iter()
            .any(|tensor| tensor.shape.len() != rank)
        {
            return Err(ConvDecline::RankMismatch.into());
        }
        let roles = ConvRoles::of(operation, window.len())?;
        let (rhs, out) = (roles.rhs_view(rhs)?, roles.out_view(out)?);
        let channels =
            |axis| (axis == roles.inner.lhs).then_some(input.shape[roles.spatial_dims + 1]);
        let checks = [
            (CI, ConvDecline::Groups, channels(CI)),
            (CO, ConvDecline::OutChannels, channels(CO)),
            (B, ConvDecline::Batch, Some(input.shape[0])),
        ];
        if let Some(&(_, declined, _)) = checks
            .iter()
            .find(|&&(axis, _, extent)| !roles.agree_on(axis, extent, [&rhs, &out]))
        {
            return Err(declined.into());
        }
        let tensors = Self {
            operation,
            client: client.clone(),
            input,
            rhs,
            out,
            window: window.to_vec(),
            dtypes,
        };
        tensors.check_window()?;
        if !tensors.is_empty() && tensors.problem().contracts_nothing() {
            return Err(ConvDecline::ContractionEmpty.into());
        }
        Ok(tensors)
    }

    /// Whether the window is one a convolution steps by and the spatial extents are the ones it
    /// yields.
    fn check_window(&self) -> Result<(), Refusal> {
        let problem = self.problem();
        if problem.spatial.iter().any(|conv_dim| conv_dim.kernel == 0) {
            return Err(ConvDecline::KernelEmpty.into());
        }
        let steps_zero =
            |conv_dim: &ConvDim| conv_dim.window.stride == 0 || conv_dim.window.dilation == 0;
        if problem.spatial.iter().any(steps_zero) {
            return Err(ConvDecline::WindowStepZero.into());
        }
        let roles = ConvRoles::of(self.operation, self.window.len())?;
        let states_output = |conv_dim: &ConvDim| {
            let (input, output) = roles.gather.window_ends(conv_dim);
            conv_dim.window.output(input, conv_dim.kernel) == output
        };
        match problem.spatial.iter().all(states_output) {
            true => Ok(()),
            false => Err(roles.extent_mismatch.into()),
        }
    }

    /// Whether the output holds no element, so a launch has nothing to compute.
    pub fn is_empty(&self) -> bool {
        self.out.shape.iter().any(|&extent| extent == 0)
    }

    /// The client these tensors live on.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The problem these tensors state. Their shapes are already the roles' views, as the checks
    /// that built them read each dim through those views.
    pub fn problem(&self) -> ConvProblem {
        let start = |tensor: &TensorBinding| tensor.handle.offset_start.unwrap_or(0);
        let shape = |tensor: &TensorBinding| tensor.shape.to_vec();
        ConvProblem::from_views(
            self.operation,
            &self.window,
            MatmulOperands {
                lhs: shape(&self.input),
                rhs: shape(&self.rhs),
                out: shape(&self.out),
            },
            MatmulOperands {
                lhs: self.dtypes.lhs_global,
                rhs: self.dtypes.rhs_global,
                out: self.dtypes.acc_global,
            },
            self.dtypes.acc_register,
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
    ) -> Result<ConvBinding, Refusal> {
        let partitioning = &blueprint.partitioning;
        let roles = problem.roles();
        let launcher = grid.launcher(&self.client, || {
            (partitioning.kernel(&roles), partitioning.space().clone())
        });
        let widths = blueprint.widths;
        let input = launcher
            .arg(self.input, problem.elements.lhs)
            .gathered(problem.input_projection())
            .vectorize(widths.lhs);
        // A box that overhangs a gathered axis reads past the input even where every real output
        // position reads inside it.
        let inside = problem.reads_inside() && !partitioning.overhangs_gathered(&roles);
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
            .axes(&roles.free.in_order())
            .vectorize(widths.out)
            .build()?;
        let masked: Vec<_> = (roles.free.in_order().into_iter())
            .map(|axis| (axis == roles.axes().m).then_some(Boundary::Zero))
            .collect();
        out.spec = match blueprint.masks_out && !out.spec.is_checked() {
            true => out.spec.boundaries(&masked),
            false => out.spec,
        };
        let (input, rhs) = (input.build()?, rhs.build()?);
        Ok(ConvBinding {
            launcher,
            input: input.arg(),
            rhs: rhs.arg(),
            out: out.arg(),
        })
    }
}
