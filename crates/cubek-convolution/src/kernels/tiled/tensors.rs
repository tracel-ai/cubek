use super::{B, CI, CO, ConvBinding, ConvDeclined, ConvProblem, ConvRoles, TileConvBlueprint};
use crate::components::ConvolutionOperation;
use crate::launch::ConvolutionArgs;
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
    window: ConvolutionArgs<2>,
    dtypes: MatmulElems,
}

impl ConvTensors {
    /// The forward convolution of `input` and `weight` into `out` over `window`, each element as
    /// `dtypes` says.
    ///
    /// # Errors
    ///
    /// A [`ConvDeclined`] reason, whatever the device.
    pub fn forward(
        client: &Client,
        input: InputBinding,
        weight: InputBinding,
        bias: Option<InputBinding>,
        out: TensorBinding,
        window: ConvolutionArgs<2>,
        dtypes: MatmulElems,
    ) -> Result<Self, Refusal> {
        if bias.is_some() {
            return Err(ConvDeclined::Bias.into());
        }
        let operation = ConvolutionOperation::Forward;
        Self::checked(operation, client, [input, weight], out, window, dtypes)
    }

    /// The weight gradient of `input` and `out_grad` into `weight_grad` over `window`, each
    /// element as `dtypes` says.
    ///
    /// # Errors
    ///
    /// A [`ConvDeclined`] reason, whatever the device.
    pub fn backward_weight(
        client: &Client,
        input: InputBinding,
        out_grad: InputBinding,
        weight_grad: TensorBinding,
        window: ConvolutionArgs<2>,
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
    /// [`ConvDeclined::DataBackwardStrided`], or another [`ConvDeclined`] reason.
    pub fn backward_data(
        client: &Client,
        out_grad: InputBinding,
        weights: InputBinding,
        in_grad: TensorBinding,
        window: ConvolutionArgs<2>,
        dtypes: MatmulElems,
    ) -> Result<Self, Refusal> {
        if window.stride.iter().any(|&stride| stride > 1) {
            return Err(ConvDeclined::DataBackwardStrided.into());
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
        window: ConvolutionArgs<2>,
        dtypes: MatmulElems,
    ) -> Result<Self, Refusal> {
        let (InputBinding::Normal(input, _), InputBinding::Normal(rhs, _)) = (input, rhs) else {
            return Err(ConvDeclined::Quantized.into());
        };
        if [&input, &rhs, &out]
            .iter()
            .any(|tensor| tensor.shape.len() != 4)
        {
            return Err(ConvDeclined::NotTwoDimensional.into());
        }
        let roles = ConvRoles::of(operation)?;
        let (rhs, out) = (roles.rhs_view(rhs)?, roles.out_view(out)?);
        let channels = |axis| (axis == roles.inner.lhs).then_some(input.shape[3]);
        let checks = [
            (CI, ConvDeclined::Groups, channels(CI)),
            (CO, ConvDeclined::OutChannels, channels(CO)),
            (B, ConvDeclined::Batch, Some(input.shape[0])),
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
            window,
            dtypes,
        };
        tensors.check_window()?;
        if !tensors.is_empty() && tensors.problem().contracts_nothing() {
            return Err(ConvDeclined::ContractionEmpty.into());
        }
        Ok(tensors)
    }

    /// Whether the window is one a convolution steps by and the spatial extents are the ones it
    /// yields.
    fn check_window(&self) -> Result<(), Refusal> {
        let problem = self.problem();
        if problem.kernel.contains(&0) {
            return Err(ConvDeclined::KernelEmpty.into());
        }
        let window = &self.window;
        if window.stride.contains(&0) || window.dilation.contains(&0) {
            return Err(ConvDeclined::WindowStepZero.into());
        }
        let (input, output, mismatch) = match self.operation {
            ConvolutionOperation::Forward => {
                (problem.gathered, problem.out, ConvDeclined::ForwardExtent)
            }
            ConvolutionOperation::BackwardWeight => (
                problem.gathered,
                problem.out,
                ConvDeclined::WeightGradExtent,
            ),
            ConvolutionOperation::BackwardData | ConvolutionOperation::ForwardTransposed => {
                (problem.out, problem.gathered, ConvDeclined::DataGradExtent)
            }
        };
        match window.output(input, problem.kernel) == output {
            true => Ok(()),
            false => Err(mismatch.into()),
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

    /// The problem these tensors state.
    pub fn problem(&self) -> ConvProblem {
        let start = |tensor: &TensorBinding| tensor.handle.offset_start.unwrap_or(0);
        let shape = |tensor: &TensorBinding| {
            let dims: &[usize] = &tensor.shape;
            [dims[0], dims[1], dims[2], dims[3]]
        };
        ConvProblem::from_views(
            self.operation,
            self.window,
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
            (
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
            ),
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
        let inside = problem.reads_inside() && !partitioning.overhangs_gathered();
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
        if blueprint.masks_out && !out.spec.is_checked() {
            let masked = roles
                .free
                .in_order()
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
