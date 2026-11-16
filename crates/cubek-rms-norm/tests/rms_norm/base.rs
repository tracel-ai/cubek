//! What every RMS-norm routine computes: each row, or each group of a row, divided by its root mean
//! square and scaled by a weight row, `out = x / sqrt(mean(x²) + ε) · γ`, gated by `silu(z)` before
//! or after the norm where asked, at shapes chosen to be awkward: a decode step's model row, a
//! prompt's head rows, a short row, a prime row that binds one element at a time, a row one line
//! past a power of two, a row longer than a plane's registers hold, a weight with a row for each
//! row, groups, and operands that are slices of a wider projection.

use cubecl::client::Client;
use cubecl::features::TypeUsage;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::metadata::Metadata;
use cubecl::zspace::shape;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::rms_norm::GateOrder;
use cubek_rms_norm::kernels::{RmsNormCall, RmsNormRoutine, RmsNormRoutineCandidate};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, Written};

pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);
pub(crate) const BF16: ElemType = ElemType::Float(FloatKind::BF16);

/// The `ε` every case adds to the mean of squares.
pub(crate) const EPSILON: f32 = 1e-6;

/// A value no case holds, which a read from the wrong place would show.
const POISON: f32 = 123.0;

/// Which refusals a routine may answer a case with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Declines {
    /// None: the autotune's floor, which a decline would leave with nothing to fall back on.
    Never,
    /// Its microkernel's own, where the device runs none at the case.
    AsItsMicrokernelStates,
}

/// One launch's elements, and where its gate enters.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Run {
    pub(crate) element: ElemType,
    pub(crate) output_element: ElemType,
    pub(crate) order: Option<GateOrder>,
}

impl Run {
    /// The runs of `pairs`, each under every order of `orders`.
    fn each(orders: &[Option<GateOrder>], pairs: &[(ElemType, ElemType)]) -> Vec<Self> {
        orders
            .iter()
            .flat_map(|&order| {
                pairs.iter().map(move |&(element, output_element)| Self {
                    element,
                    output_element,
                    order,
                })
            })
            .collect()
    }

    /// Whether the device reads, writes and computes both elements.
    pub(crate) fn runs_on(&self, client: &Client) -> bool {
        let full = TypeUsage::Arithmetic | TypeUsage::Buffer;
        [self.element, self.output_element]
            .into_iter()
            .all(|element| {
                client
                    .properties()
                    .features
                    .type_usage(element)
                    .is_superset(full)
            })
    }
}

/// The input and the gate as views of wider buffers whose rows are `pitch` elements apart, each
/// starting that many elements in.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Slices {
    pitch: usize,
    input_start: usize,
    gate_start: usize,
}

/// `rows` rows of `dim` channels, scaled by one weight row or one per row.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Rows {
    rows: usize,
    dim: usize,
    weight: WeightRows,
    group_size: Option<usize>,
    slices: Option<Slices>,
    /// What the input's values are multiplied by.
    magnitude: f32,
}

/// How many rows the weight holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WeightRows {
    One,
    EachRow,
}

impl Rows {
    pub(crate) const fn new(rows: usize, dim: usize) -> Self {
        Self {
            rows,
            dim,
            weight: WeightRows::One,
            group_size: None,
            slices: None,
            magnitude: 1.0,
        }
    }

    const fn weight_a_row(self) -> Self {
        Self {
            weight: WeightRows::EachRow,
            ..self
        }
    }

    pub(crate) const fn grouped(self, group_size: usize) -> Self {
        Self {
            group_size: Some(group_size),
            ..self
        }
    }

    const fn sliced(self, pitch: usize, input_start: usize, gate_start: usize) -> Self {
        Self {
            slices: Some(Slices {
                pitch,
                input_start,
                gate_start,
            }),
            ..self
        }
    }

    const fn scaled(self, magnitude: f32) -> Self {
        Self { magnitude, ..self }
    }

    fn input(&self) -> Vec<f32> {
        (0..self.rows * self.dim)
            .map(|i| Self::wobble(i, 1) * self.magnitude)
            .collect()
    }

    fn gate(&self) -> Vec<f32> {
        (0..self.rows * self.dim)
            .map(|i| Self::wobble(i, 3))
            .collect()
    }

    fn weight(&self) -> Vec<f32> {
        let rows = match self.weight {
            WeightRows::One => 1,
            WeightRows::EachRow => self.rows,
        };
        (0..rows * self.dim)
            .map(|i| Self::wobble(i, 2) / 2.0)
            .collect()
    }

    /// The norm of the input's groups scaled by the weight, gated as `order` says, with every
    /// operand as `element` holds it, in `f64`.
    fn reference(&self, element: ElemType, order: Option<GateOrder>) -> HostData {
        let held = |values: Vec<f32>| -> Vec<f64> {
            values
                .into_iter()
                .map(|x| Self::round_to(x, element))
                .collect()
        };
        let (input, gate, weight) = (held(self.input()), held(self.gate()), held(self.weight()));
        let group = self.group_size.unwrap_or(self.dim);
        let mut out = Vec::with_capacity(input.len());
        for row in 0..self.rows {
            let weights = match self.weight {
                WeightRows::One => 0,
                WeightRows::EachRow => row * self.dim,
            };
            for first in (0..self.dim).step_by(group) {
                let at = row * self.dim + first;
                let gates = &gate[at..at + group];
                let mut values = input[at..at + group].to_vec();
                if order == Some(GateOrder::GateThenNorm) {
                    for (x, &z) in values.iter_mut().zip(gates) {
                        *x *= Self::silu(z);
                    }
                }
                let mean = values.iter().map(|x| x * x).sum::<f64>() / group as f64;
                let scale = 1.0 / (mean + EPSILON as f64).sqrt();
                for (d, x) in values.iter().enumerate() {
                    let after = match order {
                        Some(GateOrder::NormThenGate) => Self::silu(gates[d]),
                        _ => 1.0,
                    };
                    out.push((x * scale * weight[weights + first + d] * after) as f32);
                }
            }
        }
        let shape = match self.group_size {
            Some(group) => shape![self.rows, self.dim / group, group],
            None => shape![self.rows, self.dim],
        };
        HostData::row_major(HostDataVec::F32(out), shape)
    }

    /// The norm `run` states of these rows, its operands uploaded.
    pub(crate) fn call(&self, client: &Client, run: &Run) -> RmsNormCall {
        let element = run.element;
        let weight = self.weight();
        let operand = |values: Vec<f32>, start: fn(Slices) -> usize| {
            let (pitch, start) = match self.slices {
                Some(slices) => (slices.pitch, start(slices)),
                None => (self.dim, 0),
            };
            view(
                client,
                &values,
                &[self.rows, self.dim],
                (pitch, start),
                element,
            )
        };
        let call = RmsNormCall {
            client: client.clone(),
            input: operand(self.input(), |slices| slices.input_start),
            weight: view(client, &weight, &[weight.len()], (weight.len(), 0), element),
            epsilon: EPSILON,
            output_element: run.output_element,
            gate: None,
        };
        let call = match self.group_size {
            Some(group_size) => call.grouped(group_size),
            None => call,
        };
        match run.order {
            Some(order) => call.gated(operand(self.gate(), |slices| slices.gate_start), order),
            None => call,
        }
    }

    /// `value` as `element` holds it.
    fn round_to(value: f32, element: ElemType) -> f64 {
        match element {
            F16 => half::f16::from_f32(value).to_f32() as f64,
            BF16 => half::bf16::from_f32(value).to_f32() as f64,
            _ => value as f64,
        }
    }

    /// Values in `[-2, 2)`, not periodic in any extent a case uses.
    fn wobble(i: usize, salt: usize) -> f32 {
        ((i.wrapping_mul(2654435761) + salt * 40503) % 2048) as f32 / 512.0 - 2.0
    }

    /// `z · sigmoid(z)`.
    fn silu(z: f64) -> f64 {
        z / (1.0 + (-z).exp())
    }
}

/// Runs over rows, at every setting a routine lists or at its default one.
struct Block {
    every_setting: bool,
    runs: Vec<Run>,
    rows: Vec<Rows>,
}

impl Block {
    /// What each setting runs, and what the default one runs.
    fn every() -> [Self; 3] {
        let orders = [
            None,
            Some(GateOrder::GateThenNorm),
            Some(GateOrder::NormThenGate),
        ];
        [
            // Every setting's partitioning: decode steps (the model's row, a head norm's rows), a
            // prompt's rows of the model and of its heads, a short row, a prime one that binds one
            // element at a time, one a line past a power of two, one longer than a plane's
            // registers hold, and a weight a row, as the query and key heads of a decode step share
            // their norms' weights.
            Self {
                every_setting: true,
                runs: Run::each(&[None], &[(F32, F32), (F16, F16)]),
                rows: vec![
                    Rows::new(1, 2560),
                    Rows::new(16, 128),
                    Rows::new(37, 1024),
                    Rows::new(300, 128),
                    Rows::new(3, 96),
                    Rows::new(2, 111),
                    Rows::new(1, 4100),
                    Rows::new(2, 8192),
                    Rows::new(6, 128).weight_a_row(),
                ],
            },
            // Each order at each element pair, plain and grouped.
            Self {
                every_setting: false,
                runs: Run::each(
                    &orders,
                    &[
                        (F32, F32),
                        (F16, F16),
                        (BF16, BF16),
                        (F16, F32),
                        (BF16, F32),
                    ],
                ),
                rows: vec![Rows::new(16, 128), Rows::new(5, 8192).grouped(1024)],
            },
            // Edge cases, under the orders and elements that stress them.
            Self {
                every_setting: false,
                runs: vec![
                    Run {
                        element: F16,
                        output_element: F32,
                        order: Some(GateOrder::GateThenNorm),
                    },
                    Run {
                        element: BF16,
                        output_element: BF16,
                        order: Some(GateOrder::NormThenGate),
                    },
                ],
                rows: vec![
                    // One group, a decode row in groups, and a group size that narrows the width.
                    Rows::new(1, 4096).grouped(4096),
                    Rows::new(1, 8192).grouped(1024),
                    Rows::new(2, 333).grouped(111),
                    // A gated product that leaves `f16`'s range, held in `f32`.
                    Rows::new(4, 1024).grouped(256).scaled(25_000.0),
                    // The input and the gate as slices of a projection three channels wide:
                    // aligned, and at an offset and pitch no wide read divides, which take the
                    // dense fallback.
                    Rows::new(5, 1024).grouped(128).sliced(3 * 1024, 1024, 2048),
                    Rows::new(5, 1024)
                        .grouped(128)
                        .sliced(3 * 1024 + 3, 1, 1024 + 2),
                ],
            },
        ]
    }
}

/// A routine at one of its settings, on one case, in one run.
#[derive(Clone)]
pub(crate) struct RmsNormCase<R: RmsNormRoutine<M>, M: Microkernel> {
    candidate: RmsNormRoutineCandidate<R>,
    run: Run,
    rows: Rows,
    microkernel: core::marker::PhantomData<M>,
}

impl<R: RmsNormRoutine<M>, M: Microkernel> core::fmt::Debug for RmsNormCase<R, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} {:?} at {:?}",
            self.candidate.settings(),
            self.rows,
            self.run
        )
    }
}

/// An RMS-norm routine against the reference, refusing only as `declines` allows.
pub(crate) struct RmsNorm<R, M> {
    routine: R,
    declines: Declines,
    microkernel: core::marker::PhantomData<M>,
}

impl<R: RmsNormRoutine<M> + Copy, M: Microkernel> RmsNorm<R, M> {
    pub(crate) fn new(routine: R, declines: Declines) -> Self {
        Self {
            routine,
            declines,
            microkernel: core::marker::PhantomData,
        }
    }

    /// Every block on the test device: each at the settings the routine lists or its default one,
    /// at each run the device supports, on each of its rows.
    pub(crate) fn every_case(&self) -> Vec<RmsNormCase<R, M>> {
        let client = cubecl::test_device().client();
        let candidates =
            RmsNormRoutineCandidate::list(self.routine, &Hardware::from_client(&client));
        let mut cases = Vec::new();
        for block in Block::every() {
            let settings = match block.every_setting {
                true => &candidates[..],
                false => &candidates[..1],
            };
            for run in block.runs.iter().filter(|run| run.runs_on(&client)) {
                for candidate in settings {
                    cases.extend(block.rows.iter().map(|&rows| RmsNormCase {
                        candidate: candidate.clone(),
                        run: *run,
                        rows,
                        microkernel: core::marker::PhantomData,
                    }));
                }
            }
        }
        cases
    }
}

impl<R: RmsNormRoutine<M> + Copy, M: Microkernel> ScopeUnderTest for RmsNorm<R, M> {
    type Case = RmsNormCase<R, M>;

    /// # Panics
    ///
    /// Where the routine refuses a case it may not.
    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let call = case.rows.call(client, &case.run);
        let out = match (case.candidate.launch(&call), self.declines) {
            (Ok(out), _) => out,
            (Err(why), Declines::AsItsMicrokernelStates)
                if why == (Refusal::Declined { why: M::DECLINE }) =>
            {
                return Err(why);
            }
            (Err(why), _) => panic!("{case:?} refused: {why:?}"),
        };
        Ok(vec![Written {
            name: "out",
            tensor: out,
            host: HostDataType::F32,
        }])
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        vec![case.rows.reference(case.run.element, case.run.order)]
    }

    /// `f16` keeps ten bits of each value it reads and writes, `bf16` seven; a gate's `silu` adds
    /// an exponential's rounding to an `f32` result.
    fn tolerance(&self, case: &Self::Case) -> f32 {
        let Run {
            element,
            output_element,
            order,
        } = case.run;
        match output_element {
            F32 if element == F32 && order.is_none() => 1e-5,
            F32 => 1e-4,
            F16 => 4e-3,
            _ => 2e-2,
        }
    }
}

/// `values` as `element` holds them, in rows `pitch` elements apart, starting `start` elements
/// into a buffer that is `POISON` elsewhere.
pub(crate) fn view(
    client: &Client,
    values: &[f32],
    shape: &[usize],
    (pitch, start): (usize, usize),
    element: ElemType,
) -> PlainTensor {
    let dim = shape[shape.len() - 1];
    let rows = values.len() / dim;
    let mut buffer = vec![POISON; start + rows * pitch];
    for row in 0..rows {
        let at = start + row * pitch;
        buffer[at..at + dim].copy_from_slice(&values[row * dim..(row + 1) * dim]);
    }
    let bytes: Vec<u8> = match element {
        F16 => buffer
            .iter()
            .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
            .collect(),
        BF16 => buffer
            .iter()
            .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
            .collect(),
        _ => buffer.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    let handle = client
        .create_from_slice(&bytes)
        .offset_start((start * element.size()) as u64);
    let strides = match shape.len() {
        1 => vec![1],
        _ => vec![pitch, 1],
    };
    PlainTensor::of_handle(TensorHandle::from_metadata(
        handle,
        Metadata::new(shape.to_vec(), strides),
        element,
    ))
}
