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

use crate::ReduceOperation;
use crate::eval::benchmarks::correctness::ReduceCorrectness;
use crate::eval::benchmarks::problem::{ReduceBenchKind, ReduceProblem};
use crate::eval::cpu_reference::{accumulation_dtype, output_dtype_for};
use crate::eval::workspace::workspace_for;
use crate::kernels::{self, ReduceElements, ReduceOpStrategy, ReduceWorkspace};

/// `problem` reduced by `strategy` `num_samples` times on the test device, once its result is
/// checked against the reference.
///
/// # Errors
///
/// Where the result differs from the reference, or the strategy's kernel cannot run the problem.
///
/// # Panics
///
/// Where `problem` compares a values launch with an indices launch for an operation that has no
/// such pair (a sum, a mean, a product, a max of absolute values, any or all).
pub fn bench(
    strategy: &ReduceOpStrategy,
    problem: &ReduceProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    ReduceCorrectness::verify(strategy, problem)?;

    let device = cubecl::test_device();
    let client = device.client();
    let value_dtype = problem.precision.to_dtype();
    // The two launches of a values and indices pair, and the fused one, keep indices beside their
    // values: a workspace for the indexed carrier serves both.
    let workspace = workspace_for(
        &client,
        &problem.shape,
        problem.axis,
        problem.operation,
        ReduceElements {
            input: value_dtype,
            accumulator: accumulation_dtype(),
            out: output_dtype_for(problem.operation, value_dtype),
        },
        problem.kind != ReduceBenchKind::Single,
    );

    let bench = ReduceBenchmark {
        shape: problem.shape.clone(),
        axis: problem.axis,
        operation: problem.operation,
        kind: problem.kind,
        value_dtype,
        strategy: *strategy,
        workspace,
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
/// `operation`.
pub(super) fn two_launch_operations(
    operation: ReduceOperation,
) -> (ReduceOperation, ReduceOperation) {
    match operation {
        ReduceOperation::TopK { k } | ReduceOperation::ArgTopK { k } => {
            (ReduceOperation::TopK { k }, ReduceOperation::ArgTopK { k })
        }
        ReduceOperation::Max | ReduceOperation::ArgMax => {
            (ReduceOperation::Max, ReduceOperation::ArgMax)
        }
        ReduceOperation::Min | ReduceOperation::ArgMin => {
            (ReduceOperation::Min, ReduceOperation::ArgMin)
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
#[derive(Debug, Copy, Clone, PartialEq)]
enum InputOrder {
    Uniform,
    Ascending,
    Descending,
}

impl InputOrder {
    fn from_env() -> Self {
        Self::named(std::env::var("CUBEK_BENCH_INPUT").ok().as_deref())
    }

    /// The order `name` names, uniform where there is none.
    ///
    /// # Panics
    ///
    /// Where `name` names no order.
    fn named(name: Option<&str>) -> Self {
        match name {
            Some("ascending") => InputOrder::Ascending,
            Some("descending") => InputOrder::Descending,
            Some("uniform") | None => InputOrder::Uniform,
            Some(other) => {
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

/// The range a timed input is drawn from.
const UNIFORM_LOW: f32 = 0.0;
const UNIFORM_HIGH: f32 = 1.0;

struct ReduceBenchmark {
    /// The timed input's extent along each axis, read by axis (Position Usage).
    shape: Vec<usize>,
    axis: usize,
    operation: ReduceOperation,
    kind: ReduceBenchKind,
    value_dtype: ElemType,
    strategy: ReduceOpStrategy,
    /// The scratch every launch is given, allocated before any is timed.
    workspace: ReduceWorkspace,
    device: cubecl::Device,
    client: Client,
    samples: usize,
}

impl ReduceBenchmark {
    /// `input` reduced into `out` by `operation`, in `elements`, at this bench's strategy.
    fn reduce(
        &self,
        input: TensorHandle,
        out: TensorHandle,
        operation: ReduceOperation,
        elements: ReduceElements,
    ) -> Result<(), String> {
        kernels::Reduce::new(self.strategy)
            .with_workspace(self.workspace.clone())
            .execute(
                &self.client,
                &self.device,
                input.binding(),
                out.binding(),
                self.axis,
                operation,
                elements,
            )
            .map_err(|err| format!("{err}"))
    }

    /// What a caller needing both halves does without the fused launch: `input` reduced twice,
    /// once for the values into `values` in `elements[0]`, once for the indices into `indices` in
    /// `elements[1]`, each discarding the other half.
    fn reduce_twice(
        &self,
        input: TensorHandle,
        [values, indices]: [TensorHandle; 2],
        [value_elements, index_elements]: [ReduceElements; 2],
    ) -> Result<(), String> {
        let (values_operation, indices_operation) = two_launch_operations(self.operation);
        self.reduce(input.clone(), values, values_operation, value_elements)?;
        self.reduce(input, indices, indices_operation, index_elements)
    }
}

impl Benchmark for ReduceBenchmark {
    /// `(input, values, indices)`. The index tensor is allocated for every kind so
    /// that allocation never lands inside the timed section, but only the
    /// two-launch and fused kinds write to it.
    type Input = (TensorHandle, TensorHandle, TensorHandle);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = self.device.client();
        let elem = self.value_dtype;
        let output_elem = match self.kind {
            ReduceBenchKind::Single => output_dtype_for(self.operation, elem),
            ReduceBenchKind::TwoLaunch | ReduceBenchKind::Fused => elem,
        };

        let builder =
            TestInput::builder(client.clone(), Shape::from(self.shape.clone())).dtype(elem);
        let input = match InputOrder::from_env() {
            InputOrder::Uniform => builder.uniform(0, UNIFORM_LOW, UNIFORM_HIGH),
            order => builder.custom(order.data(&self.shape, self.axis)),
        }
        .generate_without_host_data();
        let mut shape_out = self.shape.clone();
        shape_out[self.axis] = self.operation.written_per_row();
        let out = TensorHandle::empty(&client, shape_out.clone(), output_elem);
        let indices = TensorHandle::empty(&client, shape_out, u32::elem_type_native());

        (input, out, indices)
    }

    fn execute(&self, (input, out, indices): Self::Input) -> Result<(), String> {
        let value_dtype = self.value_dtype;
        let index_dtype = u32::elem_type_native();
        let elements = |out: ElemType| ReduceElements {
            input: value_dtype,
            accumulator: accumulation_dtype(),
            out,
        };

        match self.kind {
            ReduceBenchKind::Single => self.reduce(
                input,
                out,
                self.operation,
                elements(output_dtype_for(self.operation, value_dtype)),
            )?,
            ReduceBenchKind::TwoLaunch => self.reduce_twice(
                input,
                [out, indices],
                [elements(value_dtype), elements(index_dtype)],
            )?,
            ReduceBenchKind::Fused => kernels::Reduce::new(self.strategy)
                .with_workspace(self.workspace.clone())
                .execute_with_indices(
                    &self.client,
                    &self.device,
                    input.binding(),
                    out.binding(),
                    indices.binding(),
                    self.axis,
                    self.operation,
                    elements(value_dtype),
                    index_dtype,
                )
                .map_err(|err| format!("{err}"))?,
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
            "reduce-axis({})-{}-{:?}-{}-{:?}-{:?}",
            self.axis,
            self.value_dtype,
            self.shape,
            self.strategy.name(),
            self.operation,
            self.kind,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::{InputOrder, two_launch_operations};
    use crate::ReduceOperation;

    #[test]
    fn ascending_is_read_by_its_name() {
        assert_eq!(InputOrder::named(Some("ascending")), InputOrder::Ascending);
    }

    #[test]
    fn descending_is_read_by_its_name() {
        assert_eq!(
            InputOrder::named(Some("descending")),
            InputOrder::Descending
        );
    }

    #[test]
    fn no_order_named_is_uniform() {
        assert_eq!(InputOrder::named(None), InputOrder::Uniform);
    }

    #[test]
    #[should_panic]
    fn an_unknown_order_panics() {
        InputOrder::named(Some("shuffled"));
    }

    /// An arg max runs as a max, then its indices.
    #[test]
    fn an_arg_max_runs_as_its_values_then_its_indices() {
        assert_eq!(
            two_launch_operations(ReduceOperation::ArgMax),
            (ReduceOperation::Max, ReduceOperation::ArgMax)
        );
    }

    /// A top-k runs as its values, then its indices.
    #[test]
    fn a_top_k_runs_as_its_values_then_its_indices() {
        const K: usize = 3;
        assert_eq!(
            two_launch_operations(ReduceOperation::TopK { k: K }),
            (
                ReduceOperation::TopK { k: K },
                ReduceOperation::ArgTopK { k: K }
            )
        );
    }

    #[test]
    #[should_panic]
    fn a_sum_has_no_values_indices_pair() {
        two_launch_operations(ReduceOperation::Sum);
    }

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
