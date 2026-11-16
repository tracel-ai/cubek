//! What every backward interpolation computes, against the crate's host reference: an output
//! gradient, uniform over its case's range, gathered back onto the input it was interpolated from.

use cubecl::client::Client;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubek_interpolate::definition::{InterpolateError, InterpolateMode, InterpolateOptions};
use cubek_interpolate::eval::cpu_reference::cpu_reference_interpolate_backward_from_host;
use cubek_interpolate::interpolate_backward;
use cubek_test_utils::{HostData, HostDataType, ScopeUnderTest, TestInput, Written};

/// An output gradient of `out_gradient` (`NHWC`) taken back to an input of `input` (`H × W`)
/// under `options`, its values drawn from `seed` over `range`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BackwardCase {
    pub options: InterpolateOptions,
    pub input: [usize; 2],
    pub out_gradient: [usize; 4],
    pub seed: u64,
    pub range: [f32; 2],
}

impl BackwardCase {
    /// The same geometry and values under each of `options`.
    pub(crate) fn under<const N: usize>(self, options: [InterpolateOptions; N]) -> [Self; N] {
        options.map(|options| Self { options, ..self })
    }

    /// The input's shape, which the gradient against it takes.
    fn input_shape(&self) -> Vec<usize> {
        let [batch, _, _, channels] = self.out_gradient;
        let [height, width] = self.input;
        vec![batch, height, width, channels]
    }

    /// `shape` drawn from `seed` over the case's range, on `client` and on the host.
    fn uniform(&self, client: &Client, shape: Vec<usize>, seed: u64) -> (TensorHandle, HostData) {
        let [low, high] = self.range;
        TestInput::builder(client.clone(), shape)
            .uniform(seed, low, high)
            .generate_with_f32_host_data()
    }
}

/// The backward interpolation on the test device.
pub(crate) struct Backward;

impl ScopeUnderTest for Backward {
    type Case = BackwardCase;

    /// # Panics
    ///
    /// Where the launch fails for a reason other than a refusal.
    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let (out_gradient, _) = case.uniform(client, case.out_gradient.to_vec(), case.seed);
        let (input, _) = case.uniform(client, case.input_shape(), case.seed.wrapping_add(1));
        let gradient = TestInput::builder(client.clone(), case.input_shape())
            .dtype(out_gradient.dtype)
            .zeros()
            .generate_without_host_data();
        match interpolate_backward(
            client,
            input.binding(),
            out_gradient.clone().binding(),
            gradient.clone().binding(),
            case.options,
            out_gradient.dtype,
        ) {
            Ok(()) => Ok(vec![Written {
                name: "input_gradient",
                tensor: gradient,
                host: HostDataType::F32,
            }]),
            Err(InterpolateError::Refused(why)) => Err(why),
            Err(error) => panic!("{case:?}: {error}"),
        }
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        let client = cubecl::test_device().client();
        let (_, out_gradient) = case.uniform(&client, case.out_gradient.to_vec(), case.seed);
        vec![cpu_reference_interpolate_backward_from_host(
            &out_gradient,
            &case.input_shape(),
            &case.options,
        )]
    }

    /// A nearest gradient is a sum of whole cells, exact; a bilinear one weighs them.
    fn tolerance(&self, case: &Self::Case) -> f32 {
        match case.options.mode {
            InterpolateMode::Nearest(_) => 0.0,
            _ => 1e-4,
        }
    }
}
