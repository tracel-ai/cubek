use std::marker::PhantomData;

use cubecl::{
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::Client,
    future,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_matmul::{
    definition::MatmulElems,
    definition::{MatmulPrecision, MatrixPrecision},
};
use cubek_std::InputBinding;
use cubek_test_utils::{RunSamples, TestInput};

use crate::{ConvolutionInputs, Strategy, eval::benchmarks::problem::Conv2dProblem, launch_ref};

type LhsG<MP> = <<MP as MatmulPrecision>::Lhs as MatrixPrecision>::Global;
type LhsS<MP> = <<MP as MatmulPrecision>::Lhs as MatrixPrecision>::Stage;
type RhsG<MP> = <<MP as MatmulPrecision>::Rhs as MatrixPrecision>::Global;
type AccG<MP> = <<MP as MatmulPrecision>::Acc as MatrixPrecision>::Global;
type AccR<MP> = <<MP as MatmulPrecision>::Acc as MatrixPrecision>::Register;

pub fn bench(
    strategy: &Strategy,
    problem: &Conv2dProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = cubecl::test_device();
    let client = device.client();

    let bench = Conv2dBench::<half::f16> {
        problem: problem.clone(),
        strategy: strategy.clone(),
        device,
        client,
        samples: num_samples,
        _phantom: PhantomData,
    };

    let durations = bench
        .run(cubek_test_utils::timing_method(TimingMethod::System))
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
}

struct Conv2dBench<MP> {
    problem: Conv2dProblem,
    strategy: Strategy,
    device: cubecl::Device,
    client: Client,
    samples: usize,
    _phantom: PhantomData<MP>,
}

/// The input, filter and output tensors a run allocates, in the layout the
/// kernel reads: NHWC maps and an OHWI filter.
///
/// `launch_with_routine` takes the channel count off `shape[rank - 1]` and the
/// spatial extent off `shape[1..rank - 1]`, while [`Conv2dProblem`] states its
/// shapes `[N, C, H, W]` and `[O, I, KH, KW]`. Handing those arrays to the
/// launcher unchanged therefore describes a different convolution — and one
/// whose gemm operands outrun the tensors backing them, so the run reads and
/// writes past them and still launches.
fn tensor_shapes(problem: &Conv2dProblem) -> ([usize; 4], [usize; 4], [usize; 4]) {
    let [n, c_in, h_in, w_in] = problem.input_shape;
    let [c_out, _, k_h, k_w] = problem.weight_shape;
    let [s_h, s_w] = problem.args.stride;
    let [p_h, p_w] = problem.args.padding;
    let [d_h, d_w] = problem.args.dilation;

    let h_out = (h_in + 2 * p_h - d_h * (k_h - 1) - 1) / s_h + 1;
    let w_out = (w_in + 2 * p_w - d_w * (k_w - 1) - 1) / s_w + 1;

    (
        [n, h_in, w_in, c_in],
        [c_out, k_h, k_w, c_in],
        [n, h_out, w_out, c_out],
    )
}

fn make_uniform_4d(client: &Client, shape: [usize; 4], dtype: ElemType, seed: u64) -> TensorHandle {
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(dtype)
        .uniform(seed, 0.0, 1.0)
        .generate_without_host_data()
}

impl<MP: MatmulPrecision> Benchmark for Conv2dBench<MP> {
    type Input = (TensorHandle, TensorHandle, TensorHandle);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = self.device.client();
        let (input_shape, weight_shape, _) = tensor_shapes(&self.problem);

        let input = make_uniform_4d(&client, input_shape, LhsG::<MP>::elem_type_native(), 0);
        let weight = make_uniform_4d(&client, weight_shape, RhsG::<MP>::elem_type_native(), 1);
        let bias = TestInput::builder(client.clone(), Shape::from(vec![self.problem.bias_shape]))
            .dtype(AccG::<MP>::elem_type_native())
            .layout(cubek_test_utils::StridedLayout::Explicit(vec![1]))
            .uniform(2, 0.0, 1.0)
            .generate_without_host_data();

        (input, weight, bias)
    }

    fn execute(&self, (input, weight, bias): Self::Input) -> Result<(), String> {
        let client = self.device.client();
        let (_, _, out_shape) = tensor_shapes(&self.problem);
        let elems = MatmulElems::new_deprecated::<MP>();

        let out: TensorHandle = TensorHandle::empty(&client, out_shape.to_vec(), elems.acc_global);

        launch_ref::<2>(
            &self.strategy,
            &self.client,
            ConvolutionInputs::Forward {
                input: InputBinding::Normal(input.binding(), elems.lhs_global),
                weight: InputBinding::Normal(weight.binding(), elems.rhs_global),
                bias: Some(InputBinding::Normal(bias.binding(), elems.acc_global)),
                out: out.binding(),
            },
            self.problem.args.clone(),
            elems,
        )
        .map_err(|it| format!("{it:?}"))?;
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        let client = self.device.client();
        format!(
            "{}-conv2d-{}-{}-{}-{}",
            client.name(),
            LhsG::<MP>::elem_type_native(),
            LhsS::<MP>::elem_type_native(),
            AccR::<MP>::elem_type_native(),
            AccG::<MP>::elem_type_native(),
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        cubek_test_utils::profile_launch(&self.client, "conv-bench", || self.execute(args))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::benchmarks::problem::problems;

    fn shapes_of(id: &str) -> ([usize; 4], [usize; 4], [usize; 4]) {
        let problem = problems()
            .into_iter()
            .find(|e| e.id == id)
            .unwrap_or_else(|| panic!("no such problem: {id}"))
            .value;

        tensor_shapes(&problem)
    }

    /// The layout each catalogue entry allocates, as literals — so the
    /// expectation does not come from [`tensor_shapes`] itself.
    ///
    /// The regression this guards: `prepare` handed the declared
    /// `[N, C, H, W]` and `[O, I, KH, KW]` arrays to the launcher unchanged,
    /// so `alexnet_like` contracted over k = 227·3·11 = 7491 instead of
    /// 3·11·11 = 363, reading 719 136 filter elements out of a 34 848-element
    /// tensor and writing 8 110 080 into a 4 646 400-element output. Comparing
    /// element *counts* would not catch it: a permutation preserves the
    /// product.
    #[test]
    fn allocates_nhwc_maps_and_an_ohwi_filter() {
        assert_eq!(
            shapes_of("alexnet_like"),
            ([16, 227, 227, 3], [96, 11, 11, 3], [16, 55, 55, 96]),
        );
        assert_eq!(
            shapes_of("large_kernel"),
            ([16, 256, 256, 4], [64, 8, 8, 4], [16, 249, 249, 64]),
        );
    }
}
