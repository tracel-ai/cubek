//! What every matmul microkernel computes alone: `out = lhs · rhs`, both row-major.

use crate::launch::Lines;
use crate::launch::matmul::{MatmulElements, MatmulLaunch};
use core::fmt::Debug;
use core::marker::PhantomData;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::shape;
use cubek_microkernel::matmul::MatmulLayout;
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);

/// A product on the host: its inputs, small whole numbers exact in every float element a product
/// here reads and sums in, and what it writes.
pub(crate) struct HostMatmul {
    pub lhs: Vec<f32>,
    pub rhs: Vec<f32>,
    shape: [usize; 3],
}

impl HostMatmul {
    /// The inputs of a product of `m`, `n` and `k`.
    pub(crate) fn new(shape: [usize; 3]) -> Self {
        let [m, n, k] = shape;
        let values = |len: usize, modulo: usize, offset: f32| -> Vec<f32> {
            (0..len).map(|i| (i % modulo) as f32 - offset).collect()
        };
        Self {
            lhs: values(m * k, 7, 3.0),
            rhs: values(k * n, 5, 2.0),
            shape,
        }
    }

    /// The rhs laid in columns: `[n, k]`.
    pub(crate) fn rhs_transposed(&self) -> Vec<f32> {
        let [_, n, k] = self.shape;
        (0..n * k)
            .map(|idx| self.rhs[(idx % k) * n + idx / k])
            .collect()
    }

    /// `lhs · rhs`, `m × n`.
    pub(crate) fn out(&self) -> HostData {
        let [m, n, k] = self.shape;
        let out = (0..m * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                (0..k)
                    .map(|p| self.lhs[i * k + p] * self.rhs[p * n + j])
                    .sum()
            })
            .collect();
        HostData::row_major(HostDataVec::F32(out), shape![m, n])
    }
}

/// A matmul microkernel's settings, and the launch alone they pick on a device.
pub(crate) trait MatmulSettings: Debug + Copy {
    type Microkernel: Microkernel<Settings = Self>;
    type Launch: MatmulLaunch;

    /// # Errors
    ///
    /// Why the device runs no such microkernel.
    fn launch(
        &self,
        hw: &Hardware,
        elements: MatmulElements,
        lines: Lines,
        steps: usize,
    ) -> Result<Self::Launch, Refusal>;
}

/// A product of `elements` at `settings`, `steps` of the microkernel's step of `K` deep, its
/// operands bound as `lines` says.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MatmulCase<S> {
    pub elements: MatmulElements,
    pub settings: S,
    pub lines: Lines,
    pub steps: usize,
}

/// The matmul microkernel `hw` picks at each case's settings.
pub(crate) struct Matmul<S> {
    pub hw: Hardware,
    settings: PhantomData<S>,
}

impl<S: MatmulSettings> Matmul<S> {
    pub(crate) fn new(hw: Hardware) -> Self {
        Self {
            hw,
            settings: PhantomData,
        }
    }

    /// Every setting the device lists, for `f16` and `f32` inputs summed in `f32`, one step of
    /// `K` and three, its operands one value a line and as wide as a routine binds them.
    pub(crate) fn every_setting(&self) -> Vec<MatmulCase<S>> {
        S::Microkernel::settings(&self.hw)
            .into_iter()
            .flat_map(|settings| {
                [F16, F32].into_iter().flat_map(move |input| {
                    let elements = MatmulElements {
                        input,
                        sum: F32,
                        out: F32,
                    };
                    Lines::EVERY.into_iter().flat_map(move |lines| {
                        [1, 3].map(|steps| MatmulCase {
                            elements,
                            settings,
                            lines,
                            steps,
                        })
                    })
                })
            })
            .collect()
    }
}

impl<S: MatmulSettings> ScopeUnderTest for Matmul<S> {
    type Case = MatmulCase<S>;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let launch = case
            .settings
            .launch(&self.hw, case.elements, case.lines, case.steps)?;
        let [m, n, k] = launch.shape();
        let product = HostMatmul::new(launch.shape());
        let elements = launch.elements();
        let input = |values: Vec<f32>, rows, columns, element| {
            TestInput::builder(client.clone(), shape![rows, columns])
                .dtype(element)
                .custom(values)
                .generate_without_host_data()
        };
        let out = input(vec![-1e4; m * n], m, n, elements.out);
        let rhs = match launch.rhs_layout() {
            MatmulLayout::Col => input(product.rhs_transposed(), n, k, elements.input),
            _ => input(product.rhs.clone(), k, n, elements.input),
        };
        launch.launch(
            client,
            input(product.lhs, m, k, elements.input)
                .binding()
                .into_tensor_arg(),
            rhs.binding().into_tensor_arg(),
            out.clone().binding().into_tensor_arg(),
        )?;
        Ok(vec![Written {
            name: "out",
            tensor: out,
            host: HostDataType::F32,
        }])
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        let launch = case
            .settings
            .launch(&self.hw, case.elements, case.lines, case.steps)
            .unwrap_or_else(|why| panic!("{case:?} launched: {why:?}"));
        vec![HostMatmul::new(launch.shape()).out()]
    }

    /// Whole numbers whose products and sums every element here holds exactly.
    fn tolerance(&self, _case: &Self::Case) -> f32 {
        1e-2
    }
}
