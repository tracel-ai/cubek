use std::marker::PhantomData;

use cubecl::{
    Runtime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek::{
    convolution::{self, ConvolutionInputs, Strategy},
    matmul::definition::{MatmulElems, MatmulPrecision, MatrixPrecision},
    random::random_uniform,
    std::InputBinding,
};

use crate::{
    conv2d::{
        problem::{Conv2dProblem, problem_for},
        strategy::strategy_for,
    },
    registry::RunSamples,
};

type LhsG<MP> = <<MP as MatmulPrecision>::Lhs as MatrixPrecision>::Global;
type LhsS<MP> = <<MP as MatmulPrecision>::Lhs as MatrixPrecision>::Stage;
type RhsG<MP> = <<MP as MatmulPrecision>::Rhs as MatrixPrecision>::Global;
type AccG<MP> = <<MP as MatmulPrecision>::Acc as MatrixPrecision>::Global;
type AccR<MP> = <<MP as MatmulPrecision>::Acc as MatrixPrecision>::Register;

pub fn run(strategy_id: &str, problem_id: &str, num_samples: usize) -> Result<RunSamples, String> {
    run_on::<cubecl::TestRuntime, half::f16>(
        Default::default(),
        strategy_id,
        problem_id,
        num_samples,
    )
}

pub fn run_on<R: Runtime, MP: MatmulPrecision>(
    device: R::Device,
    strategy_id: &str,
    problem_id: &str,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let client = R::client(&device);
    let problem =
        problem_for(problem_id).ok_or_else(|| format!("unknown problem: {problem_id}"))?;
    let strategy =
        strategy_for(strategy_id).ok_or_else(|| format!("unknown strategy: {strategy_id}"))?;

    let bench = Conv2dBench::<R, MP> {
        problem,
        strategy,
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

struct Conv2dBench<R: Runtime, MP> {
    problem: Conv2dProblem,
    strategy: Strategy,
    device: R::Device,
    client: ComputeClient<R>,
    samples: usize,
    _phantom: PhantomData<MP>,
}

impl<R: Runtime, MP: MatmulPrecision> Benchmark for Conv2dBench<R, MP> {
    type Input = (TensorHandle<R>, TensorHandle<R>, TensorHandle<R>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);

        let input = TensorHandle::empty(
            &client,
            self.problem.input_shape.to_vec(),
            LhsG::<MP>::as_type_native_unchecked(),
        );
        random_uniform(
            &client,
            0.0,
            1.0,
            input.clone().binding(),
            LhsG::<MP>::as_type_native_unchecked().storage_type(),
        )
        .unwrap();
        let weight = TensorHandle::empty(
            &client,
            self.problem.weight_shape.to_vec(),
            RhsG::<MP>::as_type_native_unchecked(),
        );
        random_uniform(
            &client,
            0.0,
            1.0,
            weight.clone().binding(),
            RhsG::<MP>::as_type_native_unchecked().storage_type(),
        )
        .unwrap();
        let bias = TensorHandle::empty(
            &client,
            vec![self.problem.bias_shape],
            AccG::<MP>::as_type_native_unchecked(),
        );
        random_uniform(
            &client,
            0.0,
            1.0,
            bias.clone().binding(),
            AccG::<MP>::as_type_native_unchecked().storage_type(),
        )
        .unwrap();

        (input, weight, bias)
    }

    fn execute(&self, (input, weight, bias): Self::Input) -> Result<(), String> {
        let client = R::client(&self.device);
        let [n, _, h_in, w_in] = self.problem.input_shape;
        let [c_out, _, k_h, k_w] = self.problem.weight_shape;
        let [s_h, s_w] = self.problem.args.stride;
        let [p_h, p_w] = self.problem.args.padding;
        let [d_h, d_w] = self.problem.args.dilation;

        let h_out = (h_in + 2 * p_h - d_h * (k_h - 1) - 1) / s_h + 1;
        let w_out = (w_in + 2 * p_w - d_w * (k_w - 1) - 1) / s_w + 1;

        let elems = MatmulElems::new_deprecated::<MP>();

        let out: TensorHandle<R> =
            TensorHandle::empty(&client, vec![n, c_out, h_out, w_out], elems.acc_global);

        convolution::launch_ref::<R, 2>(
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
        let client = R::client(&self.device);
        format!(
            "{}-conv2d-{}-{}-{}-{}",
            R::name(&client),
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
