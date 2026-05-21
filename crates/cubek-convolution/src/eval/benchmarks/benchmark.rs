use std::marker::PhantomData;

use cubecl::{
    Runtime, TestRuntime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_matmul::definition::{MatmulElems, MatmulPrecision, MatrixPrecision};
use cubek_std::InputBinding;
use cubek_test_utils::{RunSamples, TestInput};

use crate::eval::benchmarks::problem::Conv2dProblem;
use crate::{ConvolutionInputs, Strategy, launch_ref};

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
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);

    let bench = Conv2dBench::<half::f16> {
        problem: problem.clone(),
        strategy: strategy.clone(),
        device,
        client,
        samples: num_samples,
        _phantom: PhantomData,
    };

    let durations = bench
        .run(TimingMethod::System)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
}

struct Conv2dBench<MP> {
    problem: Conv2dProblem,
    strategy: Strategy,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    samples: usize,
    _phantom: PhantomData<MP>,
}

fn make_uniform_4d(
    client: &ComputeClient<TestRuntime>,
    shape: [usize; 4],
    dtype: StorageType,
    seed: u64,
) -> TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(dtype)
        .uniform(seed, 0.0, 1.0)
        .generate_without_host_data()
}

impl<MP: MatmulPrecision> Benchmark for Conv2dBench<MP> {
    type Input = (
        TensorHandle<TestRuntime>,
        TensorHandle<TestRuntime>,
        TensorHandle<TestRuntime>,
    );
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = <TestRuntime as Runtime>::client(&self.device);

        let input = make_uniform_4d(
            &client,
            self.problem.input_shape,
            LhsG::<MP>::as_type_native_unchecked().storage_type(),
            0,
        );
        let weight = make_uniform_4d(
            &client,
            self.problem.weight_shape,
            RhsG::<MP>::as_type_native_unchecked().storage_type(),
            1,
        );
        let bias = TestInput::builder(client.clone(), Shape::from(vec![self.problem.bias_shape]))
            .dtype(AccG::<MP>::as_type_native_unchecked().storage_type())
            .layout(cubek_test_utils::StridedLayout::Explicit(vec![1]))
            .uniform(2, 0.0, 1.0)
            .generate_without_host_data();

        (input, weight, bias)
    }

    fn execute(&self, (input, weight, bias): Self::Input) -> Result<(), String> {
        let client = <TestRuntime as Runtime>::client(&self.device);
        let [n, _, h_in, w_in] = self.problem.input_shape;
        let [c_out, _, k_h, k_w] = self.problem.weight_shape;
        let [s_h, s_w] = self.problem.args.stride;
        let [p_h, p_w] = self.problem.args.padding;
        let [d_h, d_w] = self.problem.args.dilation;

        let h_out = (h_in + 2 * p_h - d_h * (k_h - 1) - 1) / s_h + 1;
        let w_out = (w_in + 2 * p_w - d_w * (k_w - 1) - 1) / s_w + 1;

        let elems = MatmulElems::new_deprecated::<MP>();

        let out: TensorHandle<TestRuntime> =
            TensorHandle::empty(&client, vec![n, c_out, h_out, w_out], elems.acc_global);

        launch_ref::<TestRuntime, 2>(
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
        let client = <TestRuntime as Runtime>::client(&self.device);
        format!(
            "{}-conv2d-{}-{}-{}-{}",
            <TestRuntime as Runtime>::name(&client),
            LhsG::<MP>::as_type_native_unchecked(),
            LhsS::<MP>::as_type_native_unchecked(),
            AccR::<MP>::as_type_native_unchecked(),
            AccG::<MP>::as_type_native_unchecked(),
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "conv-bench")
            .map(|it| it.1)
            .map_err(|it| format!("{it:?}"))
    }
}
