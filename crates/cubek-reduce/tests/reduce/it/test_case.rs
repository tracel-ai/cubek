use cubecl::client::Client;
use cubecl::tile::Geometry;
use cubecl::tile::launch::Decline;
use cubecl::{
    ir::{ElemType, FloatKind},
    prelude::*,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};
use cubek_reduce::kernels::{
    Reduce, ReduceElements, ReduceKernel, ReduceKernelHeuristic, ReduceOpStrategy,
};
use cubek_reduce::{ReduceError, ReduceOperation};
use cubek_test_utils::{
    Distribution, HostData, HostDataType, HostDataVec, ScopeChecker, StridedLayout, TestInput,
    TestOutcome, Written, assert_equals_approx, supports_f16_arithmetic,
};

use cubek_reduce::eval::EvalWorkspace;
use cubek_reduce::eval::cpu_reference::ReduceReference;

/// The unit kernel: a unit reduces each row.
pub const UNIT: ReduceOpStrategy = ReduceOpStrategy::Specific(ReduceKernel::Unit);
/// The plane kernel: a plane reduces each row, its lanes meeting by shuffles.
pub const PLANE: ReduceOpStrategy = ReduceOpStrategy::Specific(ReduceKernel::PlaneShuffle);
/// The cube kernel: a cube reduces each row, its planes meeting through shared memory.
pub const CUBE: ReduceOpStrategy = ReduceOpStrategy::Specific(ReduceKernel::CubeStaged);
/// The last-arrival hypercube: a row split across cubes, the last cube to arrive merging the parts.
pub const HYPERCUBE: ReduceOpStrategy =
    ReduceOpStrategy::Specific(ReduceKernel::HypercubeLastArrival);
/// The rule, which picks among the kernels by the problem and the device.
pub const RULE: ReduceOpStrategy = ReduceOpStrategy::Heuristic(ReduceKernelHeuristic::Tile);

/// The input of a reduction, and the kernel that reduces it.
#[derive(Clone)]
pub struct TestCase {
    /// The input's extent along each axis, read by axis (Position Usage).
    pub shape: Shape,
    /// The input's stride along each axis, read by axis (Position Usage).
    pub stride: Strides,
    pub axis: Option<usize>,
    pub strategy: ReduceOpStrategy,
    pub input_dtype: ElemType,
    /// The input's values in row-major order (Arrival Usage), where they are not drawn.
    pub custom_input: Option<Vec<f32>>,
}

impl core::fmt::Debug for TestCase {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TestCase")
            .field("shape", &self.shape)
            .field("stride", &self.stride)
            .field("axis", &self.axis)
            .field("strategy", &self.strategy)
            .field("input_dtype", &self.input_dtype)
            .finish()
    }
}

impl TestCase {
    pub fn new<E: Scalar>(
        shape: Shape,
        stride: Strides,
        axis: Option<usize>,
        strategy: ReduceOpStrategy,
    ) -> Self {
        let input_dtype = E::elem_type_native();
        Self {
            shape,
            stride,
            axis,
            strategy,
            input_dtype,
            custom_input: None,
        }
    }

    /// This case holding `data`.
    pub fn with_data(&self, data: Vec<f32>) -> Self {
        Self {
            custom_input: Some(data),
            ..self.clone()
        }
    }

    /// This case reduced by `op`.
    pub fn reduced_by(&self, op: Op) -> Reduction {
        Reduction {
            case: self.clone(),
            op,
        }
    }
}

/// Where a value lies: its coordinate along the reduced axis, the row it reduces into, and how many
/// values that row holds.
#[derive(Debug, Clone, Copy)]
pub struct ValuePlace {
    pub coordinate: usize,
    pub row: usize,
    pub row_value_count: usize,
}

/// The values `value` states of each place of a row-major `shape` reduced along `axis`.
pub fn values_by_place(shape: &Shape, axis: usize, value: impl Fn(ValuePlace) -> f32) -> Vec<f32> {
    let row_value_count = shape[axis];
    let inner = shape.iter().skip(axis + 1).product::<usize>();
    (0..shape.iter().product::<usize>())
        .map(|linear| {
            let outer = linear / (inner * row_value_count);
            value(ValuePlace {
                coordinate: (linear / inner) % row_value_count,
                row: outer * inner + linear % inner,
                row_value_count,
            })
        })
        .collect()
}

/// Checks that `written` holds on `client` what the reference reduces `input` to along `axis` by
/// `operation`, read back as `host`.
pub fn is_the_references(
    client: &Client,
    written: TensorHandle,
    host: HostDataType,
    input: &HostData,
    axis: usize,
    operation: ReduceOperation,
) {
    let written = HostData::from_tensor_handle(client, written, host);
    let reference = ReduceReference::new(input, axis, operation, None);
    assert_equals_approx(&written, &reference.output, 0.0)
        .as_test_outcome()
        .enforce();
}

/// Whether `launched` ran: where it declined, a skip, or a failure where the decline is
/// [`Decline::Broken`].
///
/// # Panics
///
/// Where `launched` failed for a reason other than a decline: the call is outside the operation's
/// Domain, which a test never makes.
pub fn ran(launched: Result<(), ReduceError>) -> bool {
    let declined = match launched {
        Ok(()) => return true,
        Err(ReduceError::Declined(why)) => why,
        Err(why) => panic!("{why}"),
    };
    TestOutcome::refused(&"the call", &declined).enforce();
    false
}

/// Per-element probability of `1` that yields a genuine mix of 0/1 reduction outputs for `case`'s
/// reduce axis length.
fn flag_probability(case: &TestCase) -> f32 {
    let axis_len = case.shape[case.axis.unwrap()].max(1) as f32;
    (1.5 / axis_len).clamp(0.0, 1.0)
}

/// The input of `case`, as it states it or seeded from `distribution`: its tensor, and its values
/// on the host.
fn input_of(case: &TestCase, distribution: Distribution) -> (TensorHandle, HostData) {
    let input = TestInput::builder(cubecl::test_device().client(), case.shape.clone())
        .dtype(case.input_dtype)
        .layout(StridedLayout::Explicit(
            case.stride.iter().copied().collect(),
        ));
    let input = match &case.custom_input {
        Some(data) => input.custom(data.clone()),
        None => input.random(1234, distribution),
    };
    input.generate_with_f32_host_data()
}

/// The output of `case` reduced by `operation`, of `output_shape` in `output_dtype` on `client`.
fn output_of(
    case: &TestCase,
    client: &Client,
    output_dtype: ElemType,
    output_shape: &Shape,
    operation: ReduceOperation,
) -> TensorHandle {
    let axis = case.axis.unwrap();
    let is_parallel = case.stride[axis] == 1;
    let strides = match operation {
        ReduceOperation::ArgTopK { k } | ReduceOperation::TopK { k } if is_parallel => {
            parallel_multiple_output_strides(case.shape.as_slice(), &case.stride, axis, k)
        }
        _ => Strides::new(Geometry::row_major(output_shape.as_slice()).strides()),
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
    fn operation(self) -> ReduceOperation {
        match self {
            Op::Sum => ReduceOperation::Sum,
            Op::Mean => ReduceOperation::Mean,
            Op::Prod => ReduceOperation::Prod,
            Op::Min | Op::MinWithIndices => ReduceOperation::Min,
            Op::Max | Op::MaxWithIndices => ReduceOperation::Max,
            Op::MaxAbs => ReduceOperation::MaxAbs,
            Op::Any => ReduceOperation::Any,
            Op::All => ReduceOperation::All,
            Op::ArgMax => ReduceOperation::ArgMax,
            Op::ArgMin => ReduceOperation::ArgMin,
            Op::ArgTopK(k) => ReduceOperation::ArgTopK { k },
            Op::TopK(k) | Op::TopKWithIndices(k) => ReduceOperation::TopK { k },
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
    fn output_dtypes(self, input: ElemType) -> impl Iterator<Item = ElemType> {
        let index = u32::elem_type_native();
        let first = match self {
            Op::Any | Op::All | Op::ArgMax | Op::ArgMin | Op::ArgTopK(_) => index,
            _ => input,
        };
        core::iter::once(first).chain(self.with_indices().then_some(index))
    }

    /// The input's values. `any` and `all` read Bernoulli flags at `flag_probability`, so about a
    /// fifth of the reduced slices take each merge outcome rather than the trivial one a uniform
    /// input would give.
    fn distribution_at_flag_probability(self, flag_probability: f32) -> Distribution {
        match self {
            Op::Any => Distribution::Bernoulli(flag_probability),
            Op::All => Distribution::Bernoulli(1.0 - flag_probability),
            // Around one, so a long row's product neither vanishes nor overflows, and a product
            // the kernel never wrote cannot pass as one too small to tell from zero.
            Op::Prod => Distribution::Uniform(0.9, 1.1),
            _ => Distribution::Uniform(-1., 1.),
        }
    }

    /// What each output holds, from the input's values on the host: the operation's values, and
    /// beside them, for a reduction that writes its indices too, the indices it picked.
    fn references(self, input: &HostData, axis: usize) -> Vec<HostData> {
        let operation = self.operation();
        let picked = match self {
            Op::TopKWithIndices(k) => Some(ReduceOperation::ArgTopK { k }),
            Op::MinWithIndices => Some(ReduceOperation::ArgMin),
            Op::MaxWithIndices => Some(ReduceOperation::ArgMax),
            _ => None,
        };
        core::iter::once(operation)
            .chain(picked)
            .map(|operation| ReduceReference::new(input, axis, operation, None).output)
            .collect()
    }

    /// The tolerance of the values: nothing for what a reduction selects or counts, and for a
    /// sum, a mean or a product, the rounding of the element it is written in.
    fn tolerance(self, output: ElemType) -> f32 {
        match self {
            Op::Sum | Op::Mean | Op::Prod => match output {
                ElemType::Float(FloatKind::F16) => 2e-3,
                _ => 1e-4,
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

/// Why a case in f16 does not run on a device that computes none.
const NO_F16_ARITHMETIC: Decline = Decline::Unserved {
    why: "the device computes no f16",
};

/// One case reduced by one operation.
#[derive(Debug, Clone)]
pub struct Reduction {
    pub case: TestCase,
    pub op: Op,
}

/// `reduction`'s input: its tensor, and its values on the host.
fn input_reduced(reduction: &Reduction) -> (TensorHandle, HostData) {
    let probability = flag_probability(&reduction.case);
    let distribution = reduction.op.distribution_at_flag_probability(probability);
    input_of(&reduction.case, distribution)
}

/// Checks [`Reduce`] at its case's strategy against the host. A case in f16 is a skip where the device
/// computes none.
pub struct ReduceScopeChecker;

impl ScopeChecker for ReduceScopeChecker {
    type Case = Reduction;

    /// # Panics
    ///
    /// Where the case is outside the operation's Domain.
    fn launch(&self, client: &Client, reduction: &Reduction) -> Result<Vec<Written>, Decline> {
        let case = &reduction.case;
        let runs =
            case.input_dtype != half::f16::elem_type_native() || supports_f16_arithmetic(client);
        if !runs {
            return Err(NO_F16_ARITHMETIC);
        }
        let (op, axis, operation) = (reduction.op, case.axis.unwrap(), reduction.op.operation());
        let (input, host) = input_reduced(reduction);
        let outputs: Vec<(ElemType, TensorHandle)> = op
            .output_dtypes(case.input_dtype)
            .zip(op.references(&host, axis))
            .map(|(dtype, reference)| {
                let output = output_of(case, client, dtype, &reference.shape, operation);
                (dtype, output)
            })
            .collect();
        let elements =
            ReduceElements::for_operation(operation, case.input_dtype, Some(outputs[0].0));
        let workspace = EvalWorkspace::new(
            client,
            &case.shape,
            axis,
            operation,
            elements,
            op.with_indices(),
        )
        .workspace;
        let reduce = Reduce::new(case.strategy).with_workspace(workspace);
        let launched = match op.with_indices() {
            true => reduce.execute_with_indices(
                client,
                input.binding(),
                outputs[0].1.clone().binding(),
                outputs[1].1.clone().binding(),
                axis,
                operation,
                elements,
                outputs[1].0,
            ),
            false => reduce.execute(
                client,
                input.binding(),
                outputs[0].1.clone().binding(),
                axis,
                operation,
                elements,
            ),
        };
        // A kernel refuses only by what the device lacks or the problem asks beyond it, which the
        // case reports as a skip; any other error is the test's.
        match launched {
            Ok(()) => {}
            Err(ReduceError::Declined(decline)) => return Err(decline),
            Err(why) => panic!("{reduction:?}: {why}"),
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
        let (_, host) = input_reduced(reduction);
        op.references(&host, case.axis.unwrap())
            .into_iter()
            .zip(op.output_dtypes(case.input_dtype))
            .map(|(reference, dtype)| cast_host_through_dtype(reference, dtype))
            .collect()
    }

    fn tolerance(&self, reduction: &Reduction) -> f32 {
        let case = &reduction.case;
        let written = reduction.op.output_dtypes(case.input_dtype).next();
        let output = written.expect("a reduction writes one output at least");
        reduction.op.tolerance(output)
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
    for d in (0..rank).filter(|&d| d != reduce_axis && d != v) {
        out[d] = input_strides[d] * k / size_r;
    }
    Strides::new(&out)
}

/// Cast expected values through the GPU output dtype so comparisons account for
/// the precision loss that occurs when the kernel stores to a narrower type
/// (e.g. an f32 accumulator overflows once written to an f16 output).
fn cast_host_through_dtype(host: HostData, dtype: ElemType) -> HostData {
    let data = match (host.data, dtype) {
        (HostDataVec::F32(values), ElemType::Float(FloatKind::F16)) => {
            HostDataVec::F32(values.into_iter().map(value_through_f16).collect())
        }
        (data, _) => data,
    };
    HostData { data, ..host }
}

/// `value` as f16 holds it.
fn value_through_f16(value: f32) -> f32 {
    half::f16::from_f32(value).to_f32()
}
