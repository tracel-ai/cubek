//! The plane fragment microkernel alone: one plane walks its tile through its own stages.

use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::shape;
use cubek_microkernel::matmul::plane::{
    PlaneFragmentMicrokernel, PlaneFragmentMicrokernelSettings,
};
use cubek_microkernel::testing::matmul::{MatmulElements, PlaneFragmentLaunch};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::{HostData, HostDataType, ScopeUnderTest, TestInput, Written};

use crate::matmul::{F16, F32, HostMatmul};

/// The microkernel `hw` picks for each case.
struct PlaneFragment {
    hw: Hardware,
}

/// The elements of a product, the microkernel's settings and how many stages of `K` it walks.
#[derive(Debug)]
struct PlaneFragmentCase {
    elements: MatmulElements,
    settings: PlaneFragmentMicrokernelSettings,
    steps: usize,
}

impl PlaneFragment {
    /// # Errors
    ///
    /// Why the device runs no plane fragment microkernel at `case`.
    fn launch_at(&self, case: &PlaneFragmentCase) -> Result<PlaneFragmentLaunch, Refusal> {
        PlaneFragmentLaunch::new(&self.hw, case.elements, &case.settings, case.steps)
    }
}

impl ScopeUnderTest for PlaneFragment {
    type Case = PlaneFragmentCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let launch = self.launch_at(case)?;
        let [m, n, k] = launch.shape;
        let product = HostMatmul::new(launch.shape);
        let elements = launch.elements;
        let input = |values: Vec<f32>, rows, columns, element| {
            TestInput::builder(client.clone(), shape![rows, columns])
                .dtype(element)
                .custom(values)
                .generate_without_host_data()
        };
        let out = input(vec![-1e4; m * n], m, n, elements.out);
        launch.launch(
            client,
            input(product.lhs, m, k, elements.input)
                .binding()
                .into_tensor_arg(),
            input(product.rhs, k, n, elements.input)
                .binding()
                .into_tensor_arg(),
            out.clone().binding().into_tensor_arg(),
        )?;
        Ok(vec![Written {
            name: "out",
            tensor: out,
            host: HostDataType::F32,
        }])
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        let launch = self
            .launch_at(case)
            .unwrap_or_else(|why| panic!("{case:?} launched: {why:?}"));
        vec![HostMatmul::new(launch.shape).out()]
    }

    fn tolerance(&self, _case: &Self::Case) -> f32 {
        1e-2
    }
}

/// Every plane fragment microkernel the test device runs, at each of its settings, over one and
/// three stages of `K`, for `f16` and `f32` inputs summed in `f32`.
#[test]
fn plane_fragment_matmul_matches_the_cpu() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let cases: Vec<_> = [F16, F32]
        .into_iter()
        .flat_map(|input| {
            let elements = MatmulElements {
                input,
                sum: F32,
                out: F32,
            };
            PlaneFragmentMicrokernel::settings(&hw)
                .into_iter()
                .flat_map(move |settings| {
                    [1, 3].map(|steps| PlaneFragmentCase {
                        elements,
                        settings,
                        steps,
                    })
                })
        })
        .collect();
    PlaneFragment { hw }.check(cases);
}
