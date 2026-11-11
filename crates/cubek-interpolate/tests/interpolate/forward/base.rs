//! What every forward interpolation computes, against the crate's host reference: an `NHWC` input,
//! uniform in `[-3, 3)`, resampled to an output size under a filter and a coordinate transform.

use cubecl::client::Client;
use cubecl::tile::launch::Refusal;
use cubek_interpolate::definition::{
    InterpolateError, InterpolateForwardProblem, InterpolateMode, InterpolateOptions, NearestMode,
};
use cubek_interpolate::eval::cpu_reference::cpu_reference_interpolate_from_host;
use cubek_interpolate::kernels::CubeInterpolateRoutineSettings;
use cubek_interpolate::{InterpolateKernel, InterpolateStrategy, interpolate};
use cubek_test_utils::{HostData, HostDataType, ScopeUnderTest, TestInput, Written};

/// The geometry the tests hold the routines to: the shape the plane-derived split produced before
/// every choice became the caller's.
pub(crate) const BASELINE: CubeInterpolateRoutineSettings =
    CubeInterpolateRoutineSettings::new(4, 2, 1);

/// Bilinear without aligned corners, the configuration tests' filter.
pub(crate) const BILINEAR: InterpolateOptions = InterpolateOptions {
    mode: InterpolateMode::Bilinear,
    align_corners: false,
};

/// Every filter.
pub(crate) const MODES: [InterpolateMode; 5] = [
    InterpolateMode::Nearest(NearestMode::Floor),
    InterpolateMode::Nearest(NearestMode::Exact),
    InterpolateMode::Bilinear,
    InterpolateMode::Bicubic,
    InterpolateMode::Lanczos3,
];

/// An `NHWC` input of `input` resampled to `output` under `options`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Resampling {
    pub options: InterpolateOptions,
    pub input: [usize; 4],
    pub output: [usize; 2],
}

impl Resampling {
    /// `options` on the tests' usual problem: two images of `8 × 9`, four channels, upsampled to
    /// `13 × 15`.
    pub(crate) const fn usual(options: InterpolateOptions) -> Self {
        Self {
            options,
            input: [2, 8, 9, 4],
            output: [13, 15],
        }
    }

    /// The same resampling over `channels` channels.
    pub(crate) const fn with_channels(self, channels: usize) -> Self {
        let [batch, height, width, _] = self.input;
        Self {
            input: [batch, height, width, channels],
            ..self
        }
    }

    fn problem(&self) -> InterpolateForwardProblem {
        InterpolateForwardProblem::from_input_output_shapes(
            &self.input.into(),
            &self.output,
            self.options,
        )
    }

    /// The input on `client`, the same values every time, with them on the host.
    fn input(&self, client: &Client) -> (cubecl::std::tensor::TensorHandle, HostData) {
        TestInput::builder(client.clone(), self.problem().input_shape())
            .uniform(123, -3.0, 3.0)
            .generate_with_f32_host_data()
    }
}

/// A strategy on a resampling.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ForwardCase {
    pub strategy: InterpolateStrategy,
    pub resampling: Resampling,
}

impl ForwardCase {
    /// `kernel` named outright on `resampling`.
    pub(crate) const fn specific(kernel: InterpolateKernel, resampling: Resampling) -> Self {
        Self {
            strategy: InterpolateStrategy::Specific(kernel),
            resampling,
        }
    }

    /// The in-place routine at `settings` on `resampling`.
    pub(crate) const fn cube(
        settings: CubeInterpolateRoutineSettings,
        resampling: Resampling,
    ) -> Self {
        Self::specific(InterpolateKernel::Cube(settings), resampling)
    }

    /// The staged routine at `settings` on `resampling`.
    pub(crate) const fn staged(
        settings: CubeInterpolateRoutineSettings,
        resampling: Resampling,
    ) -> Self {
        Self::specific(InterpolateKernel::CubeStaged(settings), resampling)
    }
}

/// The forward interpolation on the test device.
pub(crate) struct Forward;

impl ScopeUnderTest for Forward {
    type Case = ForwardCase;

    /// # Panics
    ///
    /// Where the launch fails for a reason other than a refusal.
    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let resampling = case.resampling;
        let (input, _) = resampling.input(client);
        let output = TestInput::builder(client.clone(), resampling.problem().output_shape())
            .dtype(input.dtype)
            .zeros()
            .generate_without_host_data();
        match interpolate(
            client,
            input.binding(),
            output.clone().binding(),
            resampling.options,
            case.strategy,
            output.dtype,
        ) {
            Ok(()) => Ok(vec![Written {
                name: "output",
                tensor: output,
                host: HostDataType::F32,
            }]),
            Err(InterpolateError::Refused(why)) => Err(why),
            Err(error) => panic!("{case:?}: {error}"),
        }
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        let resampling = case.resampling;
        let (_, input) = resampling.input(&cubecl::test_device().client());
        vec![cpu_reference_interpolate_from_host(
            &input,
            &resampling.problem().output_shape(),
            &resampling.options,
        )]
    }

    fn tolerance(&self, _case: &Self::Case) -> f32 {
        1e-4
    }
}
