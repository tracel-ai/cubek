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

use crate::{
    ConvolutionInputs, Strategy,
    components::ConvolutionOperation,
    eval::benchmarks::problem::{Conv2dElement, Conv2dProblem},
    launch_ref,
};

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
    match problem.element {
        Conv2dElement::F16 => bench_as::<half::f16>(strategy, problem, num_samples),
        Conv2dElement::F32 => bench_as::<f32>(strategy, problem, num_samples),
    }
}

fn bench_as<MP: MatmulPrecision>(
    strategy: &Strategy,
    problem: &Conv2dProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = cubecl::test_device();
    let client = device.client();

    let bench = Conv2dBench::<MP> {
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

fn make_uniform_4d(client: &Client, shape: [usize; 4], dtype: ElemType, seed: u64) -> TensorHandle {
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(dtype)
        .uniform(seed, 0.0, 1.0)
        .generate_without_host_data()
}

impl<MP: MatmulPrecision> Benchmark for Conv2dBench<MP> {
    type Input = (TensorHandle, TensorHandle);
    type Output = ();

    /// NHWC input/out_grad and `[co, kh, kw, ci]` weight, the layout `launch_ref` reads.
    fn prepare(&self) -> Self::Input {
        let client = self.device.client();
        let [n, c_in, h_in, w_in] = self.problem.input_shape;
        let [c_out, _, k_h, k_w] = self.problem.weight_shape;
        let [h_out, w_out] = self.problem.out_extent();

        match self.problem.operation {
            ConvolutionOperation::Forward => {
                let input = make_uniform_4d(
                    &client,
                    [n, h_in, w_in, c_in],
                    LhsG::<MP>::elem_type_native(),
                    0,
                );
                let weight = make_uniform_4d(
                    &client,
                    [c_out, k_h, k_w, c_in],
                    RhsG::<MP>::elem_type_native(),
                    1,
                );
                (input, weight)
            }
            ConvolutionOperation::BackwardData => {
                let out_grad = make_uniform_4d(
                    &client,
                    [n, h_out, w_out, c_out],
                    LhsG::<MP>::elem_type_native(),
                    0,
                );
                let weight = make_uniform_4d(
                    &client,
                    [c_out, k_h, k_w, c_in],
                    RhsG::<MP>::elem_type_native(),
                    1,
                );
                (out_grad, weight)
            }
            ConvolutionOperation::BackwardWeight => {
                let input = make_uniform_4d(
                    &client,
                    [n, h_in, w_in, c_in],
                    LhsG::<MP>::elem_type_native(),
                    0,
                );
                let out_grad = make_uniform_4d(
                    &client,
                    [n, h_out, w_out, c_out],
                    RhsG::<MP>::elem_type_native(),
                    1,
                );
                (input, out_grad)
            }
            _ => unimplemented!("unsupported convolution operation"),
        }
    }

    fn execute(&self, (lhs, rhs): Self::Input) -> Result<(), String> {
        let client = self.device.client();
        let [n, c_in, h_in, w_in] = self.problem.input_shape;
        let [c_out, _, k_h, k_w] = self.problem.weight_shape;
        let [h_out, w_out] = self.problem.out_extent();

        let elems = MatmulElems::new_deprecated::<MP>();

        let inputs = match self.problem.operation {
            ConvolutionOperation::Forward => {
                let out: TensorHandle =
                    TensorHandle::empty(&client, vec![n, h_out, w_out, c_out], elems.acc_global);
                ConvolutionInputs::Forward {
                    input: InputBinding::Normal(lhs.binding(), elems.lhs_global),
                    weight: InputBinding::Normal(rhs.binding(), elems.rhs_global),
                    bias: None,
                    out: out.binding(),
                }
            }
            ConvolutionOperation::BackwardData => {
                let in_grad: TensorHandle =
                    TensorHandle::empty(&client, vec![n, h_in, w_in, c_in], elems.acc_global);
                ConvolutionInputs::BackwardData {
                    out_grad: InputBinding::Normal(lhs.binding(), elems.lhs_global),
                    weights: InputBinding::Normal(rhs.binding(), elems.rhs_global),
                    in_grad: in_grad.binding(),
                }
            }
            ConvolutionOperation::BackwardWeight => {
                let weight_grad: TensorHandle =
                    TensorHandle::empty(&client, vec![c_out, k_h, k_w, c_in], elems.acc_global);
                ConvolutionInputs::BackwardWeight {
                    input: InputBinding::Normal(lhs.binding(), elems.lhs_global),
                    out_grad: InputBinding::Normal(rhs.binding(), elems.rhs_global),
                    weight_grad: weight_grad.binding(),
                }
            }
            _ => unimplemented!("unsupported convolution operation"),
        };

        launch_ref::<2>(
            &self.strategy,
            &self.client,
            inputs,
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
        let op = match self.problem.operation {
            ConvolutionOperation::Forward => "conv2d",
            ConvolutionOperation::BackwardData => "conv2d-bwd-data",
            ConvolutionOperation::BackwardWeight => "conv2d-bwd-weight",
            _ => "conv2d",
        };
        format!(
            "{}-{}-{}-{}-{}-{}",
            client.name(),
            op,
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
