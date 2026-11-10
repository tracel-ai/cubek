//! The plane delta microkernel alone: each query of one plane's the dot product of its output
//! with the output's gradient, `delta = Σ_dv out_gradient · out`.

use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::shape;
use cubek_microkernel::attention_backward::delta::plane::{
    PlaneDeltaMicrokernel, PlaneDeltaMicrokernelSettings,
};
use cubek_microkernel::testing::attention_backward::PlaneDeltaLaunch;
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

const F16: ElemType = ElemType::Float(FloatKind::F16);
const F32: ElemType = ElemType::Float(FloatKind::F32);

/// The plane delta microkernel `hw` picks for each case.
struct PlaneDelta {
    hw: Hardware,
}

/// Outputs of `element`, `val_dim` wide, at the microkernel's `settings`.
#[derive(Debug, Clone, Copy)]
struct PlaneDeltaCase {
    element: ElemType,
    val_dim: usize,
    settings: PlaneDeltaMicrokernelSettings,
}

impl PlaneDelta {
    /// # Errors
    ///
    /// Why the device runs no plane delta microkernel at `case`.
    fn launch_at(&self, case: &PlaneDeltaCase) -> Result<PlaneDeltaLaunch, Refusal> {
        PlaneDeltaLaunch::new(&self.hw, case.element, F32, case.val_dim, &case.settings)
    }

    /// Every query's output and its gradient, eighths exact in `f16`.
    fn operands(queries: usize, val_dim: usize) -> [Vec<f32>; 2] {
        let values = |modulo: usize, offset: f32| -> Vec<f32> {
            (0..queries * val_dim)
                .map(|i| ((i * 37) % modulo) as f32 / 8.0 - offset)
                .collect()
        };
        [values(23, 1.375), values(19, 1.125)]
    }
}

impl ScopeUnderTest for PlaneDelta {
    type Case = PlaneDeltaCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let launch = self.launch_at(case)?;
        let (queries, val_dim) = (launch.queries(), case.val_dim);
        let [out, out_gradient] = Self::operands(queries, val_dim);
        let tensor = |values: Vec<f32>, shape, element| {
            TestInput::builder(client.clone(), shape)
                .dtype(element)
                .custom(values)
                .generate_without_host_data()
        };
        let out = tensor(out, shape![1, 1, 1, queries, val_dim], case.element);
        let out_gradient = tensor(
            out_gradient,
            shape![1, 1, 1, queries, val_dim],
            case.element,
        );
        let delta = tensor(vec![-7.0; queries], shape![1, 1, 1, queries], F32);
        launch.launch(
            client,
            out.binding().into_tensor_arg(),
            out_gradient.binding().into_tensor_arg(),
            delta.clone().binding().into_tensor_arg(),
        )?;
        Ok(vec![Written {
            name: "delta",
            tensor: delta,
            host: HostDataType::F32,
        }])
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        let launch = self
            .launch_at(case)
            .unwrap_or_else(|why| panic!("{case:?} launched: {why:?}"));
        let (queries, val_dim) = (launch.queries(), case.val_dim);
        let [out, out_gradient] = Self::operands(queries, val_dim);
        let delta = (0..queries)
            .map(|q| {
                (0..val_dim)
                    .map(|d| out[q * val_dim + d] * out_gradient[q * val_dim + d])
                    .sum()
            })
            .collect();
        vec![HostData::row_major(
            HostDataVec::F32(delta),
            shape![1, 1, 1, queries],
        )]
    }

    /// Sums of products of eighths, exact in `f32`.
    fn tolerance(&self, _case: &Self::Case) -> f32 {
        1e-5
    }
}

/// A plane sums every query's dot product as the host does, at each of its settings, for `f16`
/// and `f32` outputs: a value dim of one line and of many.
#[test]
fn a_plane_sums_every_delta_like_the_host() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let cases: Vec<_> = PlaneDeltaMicrokernel::settings(&hw)
        .into_iter()
        .flat_map(|settings| {
            [F16, F32].into_iter().flat_map(move |element| {
                [1, 96].map(|val_dim| PlaneDeltaCase {
                    element,
                    val_dim,
                    settings,
                })
            })
        })
        .collect();
    PlaneDelta { hw }.check(cases);
}
