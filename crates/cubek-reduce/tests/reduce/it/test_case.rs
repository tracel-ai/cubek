use cubecl::client::Client;
use cubecl::tile::launch::Refusal;
use cubecl::{
    ir::{ElemType, FloatKind},
    prelude::*,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};
use cubek_reduce::{
    ReduceDtypes, ReduceError, ReducePrecision, ReduceStrategy, ReduceWithIndicesDtypes,
    components::instructions::ReduceOperationConfig,
};
use cubek_test_utils::{
    Distribution, ExecutionOutcome, HostData, HostDataType, HostDataVec, ScopeUnderTest,
    StridedLayout, TestInput, Written,
};

use cubek_reduce::eval::cpu_reference::{
    ReduceEntry, contiguous_strides, reference_all, reference_any, reference_argmax,
    reference_argmin, reference_argtopk, reference_max, reference_max_abs, reference_mean,
    reference_min, reference_prod, reference_sum, reference_topk,
};

#[derive(Clone)]
pub struct TestCase {
    pub shape: Shape,
    pub stride: Strides,
    pub axis: Option<usize>,
    pub strategy: ReduceStrategy,
    pub input_dtype: ElemType,
    pub accumulation_dtype: ElemType,
    custom_input: Option<Vec<f32>>,
}

impl core::fmt::Debug for TestCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TestCase")
            .field("shape", &self.shape)
            .field("stride", &self.stride)
            .field("axis", &self.axis)
            .field("strategy", &self.strategy)
            .field("input_dtype", &self.input_dtype)
            .field("accumulation_dtype", &self.accumulation_dtype)
            .finish()
    }
}

impl TestCase {
    pub fn new<P: ReducePrecision>(
        shape: Shape,
        stride: Strides,
        axis: Option<usize>,
        strategy: ReduceStrategy,
    ) -> Self
    where
        P::EI: Scalar,
        P::EA: Scalar,
    {
        Self {
            shape,
            stride,
            axis,
            strategy,
            input_dtype: <P::EI as Scalar>::elem_type_native(),
            accumulation_dtype: <P::EA as Scalar>::elem_type_native(),
            custom_input: None,
        }
    }

    /// Use explicit regression data. Custom cases must execute even when the default matrix-test
    /// policy permits unsupported generated configurations to fail compilation.
    pub fn with_data(mut self, data: Vec<f32>) -> Self {
        self.custom_input = Some(data);
        self
    }

    /// This case reduced by `op`, against the host.
    fn check(&self, op: Op) {
        Reduces.check([Reduction {
            case: self.clone(),
            op,
        }]);
    }

    pub fn test_sum(&self) {
        self.check(Op::Sum);
    }

    pub fn test_mean(&self) {
        self.check(Op::Mean);
    }

    pub fn test_prod(&self) {
        self.check(Op::Prod);
    }

    pub fn test_min(&self) {
        self.check(Op::Min);
    }

    pub fn test_max(&self) {
        self.check(Op::Max);
    }

    pub fn test_max_abs(&self) {
        self.check(Op::MaxAbs);
    }

    pub fn test_any(&self) {
        self.check(Op::Any);
    }

    pub fn test_all(&self) {
        self.check(Op::All);
    }

    pub fn test_argmax(&self) {
        self.check(Op::ArgMax);
    }

    pub fn test_argmin(&self) {
        self.check(Op::ArgMin);
    }

    pub fn test_argtopk(&self, k: usize) {
        self.check(Op::ArgTopK(k));
    }

    pub fn test_topk(&self, k: usize) {
        self.check(Op::TopK(k));
    }

    /// The fused top-k, both outputs against the very references the separate `TopK` and
    /// `ArgTopK` launches are checked against, so one launch has to agree with what two launches
    /// produce.
    pub fn test_topk_with_indices(&self, k: usize) {
        self.check(Op::TopKWithIndices(k));
    }

    pub fn test_min_with_indices(&self) {
        self.check(Op::MinWithIndices);
    }

    pub fn test_max_with_indices(&self) {
        self.check(Op::MaxWithIndices);
    }

    /// Per-element probability of `1` that yields a genuine mix of 0/1 reduction outputs for the
    /// current reduce axis length.
    fn flag_probability(&self) -> f32 {
        let axis_len = self.shape[self.axis.unwrap()].max(1) as f32;
        (1.5 / axis_len).clamp(0.0, 1.0)
    }

    /// The input, as the case states it or seeded from `distribution`: its tensor, and its values
    /// on the host.
    fn input(&self, distribution: Distribution) -> (TensorHandle, HostData) {
        let input = TestInput::builder(cubecl::test_device().client(), self.shape.clone())
            .dtype(self.input_dtype)
            .layout(StridedLayout::Explicit(
                self.stride.iter().copied().collect(),
            ));
        let input = match &self.custom_input {
            Some(data) => input.custom(data.clone()),
            None => input.random(1234, distribution),
        };
        input.generate_with_f32_host_data()
    }

    fn build_output_tensor(
        &self,
        client: &cubecl::client::Client,
        output_dtype: ElemType,
        output_shape: &Shape,
        config: &ReduceOperationConfig,
    ) -> TensorHandle {
        let axis = self.axis.unwrap();
        let is_parallel = self.stride[axis] == 1;
        let strides = match config {
            ReduceOperationConfig::ArgTopK(k) | ReduceOperationConfig::TopK(k) if is_parallel => {
                parallel_multiple_output_strides(self.shape.as_slice(), &self.stride, axis, *k)
            }
            _ => contiguous_strides(output_shape.as_slice()),
        };
        // Poisoned: NaN where the output holds values, an index no row reaches where it holds
        // indices or flags, so a cell the kernel never writes fails the comparison.
        let poison = match output_dtype {
            ElemType::Float(_) => f32::NAN,
            _ => 77_777.0,
        };
        TestInput::builder(client.clone(), output_shape.clone())
            .dtype(output_dtype)
            .layout(StridedLayout::Explicit(strides.iter().copied().collect()))
            .custom(vec![poison; output_shape.num_elements()])
            .generate_without_host_data()
    }
}

/// A reduction a case is run through: one output, or a value and its index fused in one launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Sum,
    Mean,
    Prod,
    Min,
    Max,
    MaxAbs,
    Any,
    All,
    ArgMax,
    ArgMin,
    ArgTopK(usize),
    TopK(usize),
    TopKWithIndices(usize),
    MinWithIndices,
    MaxWithIndices,
}

impl Op {
    /// What the kernel is told to compute.
    fn config(self) -> ReduceOperationConfig {
        match self {
            Op::Sum => ReduceOperationConfig::Sum,
            Op::Mean => ReduceOperationConfig::Mean,
            Op::Prod => ReduceOperationConfig::Prod,
            Op::Min | Op::MinWithIndices => ReduceOperationConfig::Min,
            Op::Max | Op::MaxWithIndices => ReduceOperationConfig::Max,
            Op::MaxAbs => ReduceOperationConfig::MaxAbs,
            Op::Any => ReduceOperationConfig::Any,
            Op::All => ReduceOperationConfig::All,
            Op::ArgMax => ReduceOperationConfig::ArgMax,
            Op::ArgMin => ReduceOperationConfig::ArgMin,
            Op::ArgTopK(k) => ReduceOperationConfig::ArgTopK(k),
            Op::TopK(k) | Op::TopKWithIndices(k) => ReduceOperationConfig::TopK(k),
        }
    }

    /// Whether one launch writes the values and their indices.
    fn with_indices(self) -> bool {
        matches!(
            self,
            Op::TopKWithIndices(_) | Op::MinWithIndices | Op::MaxWithIndices
        )
    }

    /// The element each output is written in: the input's for values, `u32` for indices and for
    /// the flags of `any` and `all`, the one flag storage every test runtime supports.
    fn output_dtypes(self, input: ElemType) -> Vec<ElemType> {
        let index = u32::elem_type_native();
        match self {
            Op::Any | Op::All | Op::ArgMax | Op::ArgMin | Op::ArgTopK(_) => vec![index],
            op if op.with_indices() => vec![input, index],
            _ => vec![input],
        }
    }

    /// The input's values. `any` and `all` read Bernoulli flags at `flag_probability`, so about a
    /// fifth of the reduced slices take each merge outcome rather than the trivial one a uniform
    /// input would give.
    fn distribution(self, flag_probability: f32) -> Distribution {
        match self {
            Op::Any => Distribution::Bernoulli(flag_probability),
            Op::All => Distribution::Bernoulli(1.0 - flag_probability),
            // Around one, so a long row's product neither vanishes nor overflows, and a product
            // the kernel never wrote cannot pass as one too small to tell from zero.
            Op::Prod => Distribution::Uniform(0.9, 1.1),
            _ => Distribution::Uniform(-1., 1.),
        }
    }

    /// What each output holds, from the input's values on the host.
    fn references(self, input: &HostData, axis: usize) -> Vec<HostData> {
        match self {
            Op::Sum => vec![reference_sum(input, axis, None)],
            Op::Mean => vec![reference_mean(input, axis, None)],
            Op::Prod => vec![reference_prod(input, axis, None)],
            Op::Min => vec![reference_min(input, axis, None)],
            Op::Max => vec![reference_max(input, axis, None)],
            Op::MaxAbs => vec![reference_max_abs(input, axis, None)],
            Op::Any => vec![reference_any(input, axis, None)],
            Op::All => vec![reference_all(input, axis, None)],
            Op::ArgMax => vec![reference_argmax(input, axis, None)],
            Op::ArgMin => vec![reference_argmin(input, axis, None)],
            Op::ArgTopK(k) => vec![reference_argtopk(input, axis, k, None)],
            Op::TopK(k) => vec![reference_topk(input, axis, k, None)],
            Op::TopKWithIndices(k) => vec![
                reference_topk(input, axis, k, None),
                reference_argtopk(input, axis, k, None),
            ],
            Op::MinWithIndices => vec![
                reference_min(input, axis, None),
                reference_argmin(input, axis, None),
            ],
            Op::MaxWithIndices => vec![
                reference_max(input, axis, None),
                reference_argmax(input, axis, None),
            ],
        }
    }

    /// The tolerance of the values: nothing for what a reduction selects or counts, and for a
    /// sum, a mean or a product, the rounding of the element it is written in, or of a 16-bit
    /// accumulator, whose error grows with the row.
    fn tolerance(self, output: ElemType, accumulation: ElemType) -> f32 {
        let rounding = |element: ElemType| match element {
            ElemType::Float(FloatKind::F64) => 1e-6,
            ElemType::Float(FloatKind::F16) => 2e-3,
            ElemType::Float(FloatKind::BF16) => 1.6e-2,
            _ => 1e-4,
        };
        match self {
            Op::Sum | Op::Mean | Op::Prod => match accumulation {
                ElemType::Float(FloatKind::F16 | FloatKind::BF16) => 0.0625,
                _ => rounding(output),
            },
            Op::Min
            | Op::Max
            | Op::MaxAbs
            | Op::TopK(_)
            | Op::TopKWithIndices(_)
            | Op::MinWithIndices
            | Op::MaxWithIndices
            | Op::Any
            | Op::All
            | Op::ArgMax
            | Op::ArgMin
            | Op::ArgTopK(_) => 0.,
        }
    }
}

/// One case reduced by one operation.
#[derive(Debug, Clone)]
pub struct Reduction {
    pub case: TestCase,
    pub op: Op,
}

impl Reduction {
    fn input(&self) -> (TensorHandle, HostData) {
        self.case
            .input(self.op.distribution(self.case.flag_probability()))
    }
}

/// The reduce entry the environment names, against the host.
pub struct Reduces;

impl ScopeUnderTest for Reduces {
    type Case = Reduction;

    /// # Panics
    ///
    /// Where a case of explicit regression data is refused: it must execute.
    fn launch(&self, client: &Client, reduction: &Reduction) -> Result<Vec<Written>, Refusal> {
        let case = &reduction.case;
        let (op, axis, config) = (reduction.op, case.axis.unwrap(), reduction.op.config());
        let (input, host) = reduction.input();
        let outputs: Vec<(ElemType, TensorHandle)> = op
            .output_dtypes(case.input_dtype)
            .into_iter()
            .zip(op.references(&host, axis))
            .map(|(dtype, reference)| {
                let output = case.build_output_tensor(client, dtype, &reference.shape, &config);
                (dtype, output)
            })
            .collect();
        let launched = match op.with_indices() {
            true => ReduceEntry::from_env().reduce_with_indices()(
                client,
                input.binding(),
                outputs[0].1.clone().binding(),
                outputs[1].1.clone().binding(),
                axis,
                case.strategy.clone(),
                config,
                ReduceWithIndicesDtypes {
                    input: case.input_dtype,
                    values: outputs[0].0,
                    indices: outputs[1].0,
                    accumulation: case.accumulation_dtype,
                },
            ),
            false => ReduceEntry::from_env().reduce()(
                client,
                input.binding(),
                outputs[0].1.clone().binding(),
                axis,
                case.strategy.clone(),
                config,
                ReduceDtypes {
                    input: case.input_dtype,
                    output: outputs[0].0,
                    accumulation: case.accumulation_dtype,
                },
            ),
        };
        // A device declines only by what it lacks; any other error is the test's or the kernel's,
        // and a case of explicit data must execute whatever the device.
        match launched {
            Ok(()) => {}
            Err(why) if case.custom_input.is_some() => {
                panic!("{reduction:?}: a case of explicit data must execute, refused: {why:?}")
            }
            Err(
                ReduceError::PlanesUnavailable
                | ReduceError::ImprecisePlaneDim
                | ReduceError::CubeCountTooLarge
                // A cube forced to a plane width the device does not have.
                | ReduceError::Validation {
                    details: "`cube_dim.x` must match `plane_size_max`",
                },
            ) => {
                return Err(Refusal::Declined {
                    why: "the reduce entry runs no such configuration on this device",
                });
            }
            Err(why) => panic!("{reduction:?}: {why}"),
        }
        if case.custom_input.is_some() {
            let handles: Vec<_> = outputs.iter().map(|(_, tensor)| &tensor.handle).collect();
            if let ExecutionOutcome::CompileError(why) =
                ExecutionOutcome::of_outputs(client, &handles)
            {
                panic!("{reduction:?}: a case of explicit data must execute: {why}");
            }
        }
        let names = match op.with_indices() {
            true => ["values", "indices"].as_slice(),
            false => ["output"].as_slice(),
        };
        Ok(names
            .iter()
            .zip(outputs)
            .map(|(&name, (_, tensor))| Written {
                name,
                tensor,
                host: HostDataType::F32,
            })
            .collect())
    }

    /// Each reference cast through its output's element, so the comparison accounts for what the
    /// kernel loses storing to a narrower type, as an `f32` accumulator overflowing an `f16`
    /// output.
    fn reference(&self, reduction: &Reduction) -> Vec<HostData> {
        let (op, case) = (reduction.op, &reduction.case);
        let (_, host) = reduction.input();
        op.references(&host, case.axis.unwrap())
            .into_iter()
            .zip(op.output_dtypes(case.input_dtype))
            .map(|(reference, dtype)| cast_host_through_dtype(reference, dtype))
            .collect()
    }

    fn tolerance(&self, reduction: &Reduction) -> f32 {
        let case = &reduction.case;
        let output = reduction.op.output_dtypes(case.input_dtype)[0];
        reduction.op.tolerance(output, case.accumulation_dtype)
    }

    /// The indices a fused reduction writes are exact.
    fn tolerance_of(&self, reduction: &Reduction, name: &'static str) -> f32 {
        match name {
            "indices" => 0.,
            _ => self.tolerance(reduction),
        }
    }
}

/// Output strides for multiple accumulators (like argtopk) in parallel mode.
fn parallel_multiple_output_strides(
    input_shape: &[usize],
    input_strides: &[usize],
    reduce_axis: usize,
    k: usize,
) -> Strides {
    let rank = input_shape.len();
    let size_r = input_shape[reduce_axis];

    let v = (0..rank)
        .filter(|&d| d != reduce_axis)
        .min_by_key(|&d| input_strides[d])
        .unwrap_or(0);
    let size_v = input_shape[v];

    let mut out = vec![0usize; rank];
    out[reduce_axis] = size_v;
    out[v] = 1;
    for d in 0..rank {
        if d == reduce_axis || d == v {
            continue;
        }
        out[d] = input_strides[d] * k / size_r;
    }
    Strides::new(&out)
}

/// Cast expected values through the GPU output dtype so comparisons account for
/// the precision loss that occurs when the kernel stores to a narrower type
/// (e.g. an f32 accumulator overflows once written to an f16 output).
fn cast_host_through_dtype(mut host: HostData, dtype: ElemType) -> HostData {
    if let HostDataVec::F32(values) = &host.data {
        let casted = match dtype {
            ElemType::Float(FloatKind::F16) => values
                .iter()
                .map(|&x| half::f16::from_f32(x).to_f32())
                .collect(),
            ElemType::Float(FloatKind::BF16) => values
                .iter()
                .map(|&x| half::bf16::from_f32(x).to_f32())
                .collect(),
            _ => return host,
        };
        host.data = HostDataVec::F32(casted);
    }
    host
}
