//! What every RMS-norm routine computes: each row divided by its root mean square and scaled by a
//! weight row, `out = x / sqrt(mean(x²) + ε) · γ`, at shapes chosen to be awkward: a decode step's
//! model row, a prompt's head rows, a short row, a prime row that binds one element at a time, a
//! row one line past a power of two, a row longer than a plane's registers hold, and a weight with
//! a row for each row.

use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::launch::Refusal;
use cubecl::zspace::{Shape, shape};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_rms_norm::kernels::{RmsNormCall, RmsNormRoutine, RmsNormRoutineCandidate};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};
use half::f16;

pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);

/// The `ε` every case adds to the mean of squares.
pub(crate) const EPSILON: f32 = 1e-6;

/// Which refusals a routine may answer a case with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Declines {
    /// None: the autotune's floor, which a decline would leave with nothing to fall back on.
    Never,
    /// Its microkernel's own, where the device runs none at the case.
    AsItsMicrokernelStates,
}

/// `rows` rows of `dim`, scaled by one weight row, or by a weight row for each row.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Rows {
    rows: usize,
    dim: usize,
    weight: WeightRows,
}

/// How many rows the weight holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WeightRows {
    One,
    EachRow,
}

impl Rows {
    const fn new(rows: usize, dim: usize) -> Self {
        Self {
            rows,
            dim,
            weight: WeightRows::One,
        }
    }

    const fn weight_a_row(self) -> Self {
        Self {
            weight: WeightRows::EachRow,
            ..self
        }
    }

    /// Every case a routine is held to.
    pub(crate) fn awkward() -> [Self; 9] {
        [
            // Decode steps: the model's row, and a head norm's rows.
            Self::new(1, 2560),
            Self::new(16, 128),
            // A prompt's rows of the model and of its heads.
            Self::new(37, 1024),
            Self::new(300, 128),
            // A short row, a prime one that binds one element at a time, and one a line past a
            // power of two.
            Self::new(3, 96),
            Self::new(2, 111),
            Self::new(1, 4100),
            // A row longer than a plane's registers hold, which only the cube and the floor take.
            Self::new(2, 8192),
            // A weight a row, as the query and key heads of a decode step share their norms'
            // weights.
            Self::new(6, 128).weight_a_row(),
        ]
    }

    fn input(&self) -> Vec<f32> {
        (0..self.rows * self.dim)
            .map(|i| Self::wobble(i, 1))
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

    /// The norm of the input's rows scaled by the weight, both as `element` holds them, in `f64`.
    fn reference(&self, element: ElemType) -> HostData {
        let (input, weight) = (self.input(), self.weight());
        let rounded = |value: f32| match element {
            F16 => f16::from_f32(value).to_f32() as f64,
            _ => value as f64,
        };
        let mut out = Vec::with_capacity(input.len());
        for row in 0..self.rows {
            let values: Vec<f64> = input[row * self.dim..(row + 1) * self.dim]
                .iter()
                .map(|&x| rounded(x))
                .collect();
            let mean = values.iter().map(|x| x * x).sum::<f64>() / self.dim as f64;
            let scale = 1.0 / (mean + EPSILON as f64).sqrt();
            let weight_row = match self.weight {
                WeightRows::One => 0,
                WeightRows::EachRow => row,
            };
            for (d, x) in values.iter().enumerate() {
                out.push((x * scale * rounded(weight[weight_row * self.dim + d])) as f32);
            }
        }
        HostData::row_major(HostDataVec::F32(out), shape![self.rows, self.dim])
    }

    /// Values in `[-2, 2)`, not periodic in any extent a case uses.
    fn wobble(i: usize, salt: usize) -> f32 {
        ((i.wrapping_mul(2654435761) + salt * 40503) % 2048) as f32 / 512.0 - 2.0
    }
}

/// A routine at one of its settings, on one case, in one element.
#[derive(Clone)]
pub(crate) struct RmsNormCase<R: RmsNormRoutine<M>, M: Microkernel> {
    candidate: RmsNormRoutineCandidate<R>,
    element: ElemType,
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
            self.element
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

    /// Every setting the routine lists on the test device, in `f32` and `f16`, on every case.
    pub(crate) fn every_case(&self) -> Vec<RmsNormCase<R, M>> {
        let hw = Hardware::from_client(&cubecl::test_device().client());
        RmsNormRoutineCandidate::list(self.routine, &hw)
            .into_iter()
            .flat_map(|candidate| {
                [F32, F16].into_iter().flat_map(move |element| {
                    let candidate = candidate.clone();
                    Rows::awkward().map(move |rows| RmsNormCase {
                        candidate: candidate.clone(),
                        element,
                        rows,
                        microkernel: core::marker::PhantomData,
                    })
                })
            })
            .collect()
    }

    /// `values` uploaded as `element`, row-major in `shape`.
    fn tensor(client: &Client, values: Vec<f32>, shape: Shape, element: ElemType) -> PlainTensor {
        let handle = TestInput::builder(client.clone(), shape)
            .dtype(element)
            .custom(values)
            .generate_without_host_data();
        PlainTensor::of_handle(handle)
    }
}

impl<R: RmsNormRoutine<M> + Copy, M: Microkernel> ScopeUnderTest for RmsNorm<R, M> {
    type Case = RmsNormCase<R, M>;

    /// # Panics
    ///
    /// Where the routine refuses a case it may not.
    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let rows = case.rows;
        let call = RmsNormCall {
            client: client.clone(),
            input: Self::tensor(
                client,
                rows.input(),
                shape![rows.rows, rows.dim],
                case.element,
            ),
            weight: Self::tensor(
                client,
                rows.weight(),
                shape![rows.weight().len()],
                case.element,
            ),
            epsilon: EPSILON,
        };
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
        vec![case.rows.reference(case.element)]
    }

    /// `f16` keeps ten bits of each value it reads and writes.
    fn tolerance(&self, case: &Self::Case) -> f32 {
        match case.element {
            F16 => 4e-3,
            _ => 1e-5,
        }
    }
}
