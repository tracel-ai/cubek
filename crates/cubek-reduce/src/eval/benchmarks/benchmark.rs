use cubecl::{
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::Client,
    future,
    ir::ElemType,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_test_utils::{RunSamples, TestInput};

use crate::ReduceStrategy;
use crate::components::instructions::ReduceOperationConfig;
use crate::eval::benchmarks::correctness::ReduceCorrectness;
use crate::eval::benchmarks::problem::{ReduceBenchKind, ReduceProblem};

pub fn bench(
    strategy: &ReduceStrategy,
    problem: &ReduceProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    ReduceCorrectness::verify(strategy, problem)?;

    let device = cubecl::test_device();
    let client = device.client();

    let bench = ReduceBench {
        shape: problem.shape.clone(),
        axis: problem.axis,
        config: problem.config,
        kind: problem.kind,
        value_dtype: problem.precision.dtype(),
        strategy: strategy.clone(),
        device,
        client,
        samples: num_samples,
    };

    // Device timing (hardware timestamps) rather than system timing: the reduce
    // problems are ~270 MB, and wall-clock timing of launch+sync picked up enough
    // host-side noise that identical kernels varied by over 10x between runs,
    // which made fused-vs-two-launch comparisons meaningless.
    let durations = bench
        .run(cubek_test_utils::timing_method(TimingMethod::Device))
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
}

/// The pair of single-output configs a caller runs today to get both halves of
/// `config`.
fn two_launch_configs(
    config: ReduceOperationConfig,
) -> (ReduceOperationConfig, ReduceOperationConfig) {
    match config {
        ReduceOperationConfig::TopK(k) | ReduceOperationConfig::ArgTopK(k) => (
            ReduceOperationConfig::TopK(k),
            ReduceOperationConfig::ArgTopK(k),
        ),
        ReduceOperationConfig::Max | ReduceOperationConfig::ArgMax => {
            (ReduceOperationConfig::Max, ReduceOperationConfig::ArgMax)
        }
        ReduceOperationConfig::Min | ReduceOperationConfig::ArgMin => {
            (ReduceOperationConfig::Min, ReduceOperationConfig::ArgMin)
        }
        other => panic!("{other:?} has no values/indices pair to compare"),
    }
}

/// How the benchmark orders its input along the reduce axis, from
/// `CUBEK_BENCH_INPUT`.
///
/// Top-k rejects a candidate that cannot reach its weakest slot, so its cost
/// depends on the order it meets values in, and the two sorted orders are the
/// bounds either side of the random one. Neither is a workload: code that knows
/// its input is sorted takes the last `k` and never calls a reduction.
#[derive(Copy, Clone, PartialEq)]
enum InputOrder {
    Uniform,
    Ascending,
    Descending,
}

impl InputOrder {
    fn from_env() -> Self {
        match std::env::var("CUBEK_BENCH_INPUT").as_deref() {
            Ok("ascending") => InputOrder::Ascending,
            Ok("descending") => InputOrder::Descending,
            Ok("uniform") | Err(_) => InputOrder::Uniform,
            Ok(other) => {
                panic!("CUBEK_BENCH_INPUT takes uniform, ascending or descending, got {other:?}")
            }
        }
    }

    /// The coordinate along `axis`, or its mirror, as the value at every
    /// position.
    ///
    /// Built from the coordinate rather than from the linear index so that the
    /// values stay whole numbers a f32 holds exactly: a 67 M element `arange`
    /// runs past 2^24, where consecutive integers collide and a row meant to be
    /// sorted is mostly ties.
    fn data(self, shape: &[usize], axis: usize) -> Vec<f32> {
        let axis_len = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let last = (axis_len - 1) as f32;

        (0..shape.iter().product::<usize>())
            .map(|linear| {
                let coordinate = (linear / inner) % axis_len;
                match self {
                    InputOrder::Ascending => coordinate as f32,
                    InputOrder::Descending => last - coordinate as f32,
                    InputOrder::Uniform => unreachable!("uniform does not build its data here"),
                }
            })
            .collect()
    }
}

struct ReduceBench {
    shape: Vec<usize>,
    axis: usize,
    config: ReduceOperationConfig,
    kind: ReduceBenchKind,
    value_dtype: ElemType,
    strategy: ReduceStrategy,
    device: cubecl::Device,
    client: Client,
    samples: usize,
}

impl Benchmark for ReduceBench {
    /// `(input, values, indices)`. The index tensor is allocated for every kind so
    /// that allocation never lands inside the timed section, but only the
    /// two-launch and fused kinds write to it.
    type Input = (TensorHandle, TensorHandle, TensorHandle);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = self.device.client();
        let elem = self.value_dtype;
        let output_elem = crate::eval::cpu_reference::output_dtype_for(&self.config, elem);

        let builder =
            TestInput::builder(client.clone(), Shape::from(self.shape.clone())).dtype(elem);
        let input = match InputOrder::from_env() {
            InputOrder::Uniform => builder.uniform(0, 0., 1.),
            order => builder.custom(order.data(&self.shape, self.axis)),
        }
        .generate_without_host_data();
        let mut shape_out = self.shape.clone();
        let reduce_len = match self.config {
            ReduceOperationConfig::ArgTopK(len) => len,
            ReduceOperationConfig::TopK(len) => len,
            _ => 1,
        };
        shape_out[self.axis] = reduce_len;
        let out = TensorHandle::empty(&client, shape_out.clone(), output_elem);
        let indices = TensorHandle::empty(&client, shape_out, u32::elem_type_native());

        (input, out, indices)
    }

    fn execute(&self, (input, out, indices): Self::Input) -> Result<(), String> {
        let value_dtype = self.value_dtype;
        let index_dtype = u32::elem_type_native();
        let acc_dtype = crate::eval::cpu_reference::accumulation_dtype();

        match self.kind {
            ReduceBenchKind::Single => {
                let output_dtype =
                    crate::eval::cpu_reference::output_dtype_for(&self.config, value_dtype);
                crate::reduce(
                    &self.client,
                    input.binding(),
                    out.binding(),
                    self.axis,
                    self.strategy.clone(),
                    self.config,
                    crate::ReduceDtypes {
                        input: value_dtype,
                        output: output_dtype,
                        accumulation: acc_dtype,
                    },
                )
                .map_err(|err| format!("{err}"))?;
            }
            // What a caller needing both halves does today: run the whole
            // reduction twice, discarding half of each result.
            ReduceBenchKind::TwoLaunch => {
                let (values_config, indices_config) = two_launch_configs(self.config);
                crate::reduce(
                    &self.client,
                    input.clone().binding(),
                    out.binding(),
                    self.axis,
                    self.strategy.clone(),
                    values_config,
                    crate::ReduceDtypes {
                        input: value_dtype,
                        output: value_dtype,
                        accumulation: acc_dtype,
                    },
                )
                .map_err(|err| format!("{err}"))?;
                crate::reduce(
                    &self.client,
                    input.binding(),
                    indices.binding(),
                    self.axis,
                    self.strategy.clone(),
                    indices_config,
                    crate::ReduceDtypes {
                        input: value_dtype,
                        output: index_dtype,
                        accumulation: acc_dtype,
                    },
                )
                .map_err(|err| format!("{err}"))?;
            }
            ReduceBenchKind::Fused => {
                crate::reduce_with_indices(
                    &self.client,
                    input.binding(),
                    out.binding(),
                    indices.binding(),
                    self.axis,
                    self.strategy.clone(),
                    self.config,
                    crate::ReduceWithIndicesDtypes {
                        input: value_dtype,
                        values: value_dtype,
                        indices: index_dtype,
                        accumulation: acc_dtype,
                    },
                )
                .map_err(|err| format!("{err}"))?;
            }
        }

        Ok(())
    }

    /// Measure with device timestamps around the launch, so the reported duration
    /// is the kernel's, not the host's view of launch+sync.
    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        cubek_test_utils::profile_launch(&self.client, "reduce-bench", || self.execute(args))
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "reduce-axis({})-{}-{:?}-{:?}-{:?}-{:?}",
            self.axis, self.value_dtype, self.shape, self.strategy, self.config, self.kind,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::InputOrder;

    /// Rows along the reduce axis, for a shape whose reduce axis is not the
    /// innermost, so a generator that walked the linear index would fail here.
    fn rows(order: InputOrder, shape: &[usize], axis: usize) -> Vec<Vec<f32>> {
        let data = order.data(shape, axis);
        let inner: usize = shape[axis + 1..].iter().product();
        let outer: usize = shape[..axis].iter().product();

        let mut rows = Vec::new();
        for o in 0..outer {
            for i in 0..inner {
                let start = o * shape[axis] * inner + i;
                rows.push((0..shape[axis]).map(|c| data[start + c * inner]).collect());
            }
        }
        rows
    }

    #[test]
    fn ascending_rises_along_the_reduce_axis() {
        for row in rows(InputOrder::Ascending, &[2, 3, 4], 1) {
            assert_eq!(row, vec![0.0, 1.0, 2.0]);
        }
    }

    #[test]
    fn descending_falls_along_the_reduce_axis() {
        for row in rows(InputOrder::Descending, &[2, 3, 4], 1) {
            assert_eq!(row, vec![2.0, 1.0, 0.0]);
        }
    }

    #[test]
    fn every_value_in_a_row_is_distinct_at_the_benchmark_shape() {
        let shape = [32, 512, 4095];
        let data = InputOrder::Ascending.data(&shape, 2);
        let row: Vec<f32> = data[shape[2] * 700..shape[2] * 701].to_vec();

        assert!(row.windows(2).all(|w| w[1] > w[0]));
        assert_eq!(row[0], 0.0);
        assert_eq!(row[shape[2] - 1], (shape[2] - 1) as f32);
    }
}
