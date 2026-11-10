//! The plane fragment attention microkernel alone: one plane's queries over every key.

use cubecl::ir::{ElemType, UIntKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::shape;
use cubek_microkernel::attention::plane::{
    PlaneFragmentAttentionMicrokernel, PlaneFragmentAttentionMicrokernelSettings,
};
use cubek_microkernel::testing::attention::{AttentionElements, PlaneFragmentAttentionLaunch};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::{HostData, HostDataType, ScopeUnderTest, TestInput, Written};

use crate::attention::HostAttention;
use crate::matmul::{F16, F32};

/// The rows of the query and key, and of the value.
const DIMS: [usize; 2] = [64, 64];

/// The microkernel `hw` picks for each case.
struct PlaneFragmentAttention {
    hw: Hardware,
}

/// The elements of an attention, the microkernel's settings and how many stages of keys it walks.
#[derive(Debug)]
struct PlaneFragmentAttentionCase {
    elements: AttentionElements,
    settings: PlaneFragmentAttentionMicrokernelSettings,
    steps: usize,
}

impl PlaneFragmentAttention {
    /// # Errors
    ///
    /// Why the device runs no plane fragment attention microkernel at `case`.
    fn launch_at(
        &self,
        case: &PlaneFragmentAttentionCase,
    ) -> Result<PlaneFragmentAttentionLaunch, Refusal> {
        PlaneFragmentAttentionLaunch::new(&self.hw, case.elements, DIMS, &case.settings, case.steps)
    }
}

impl ScopeUnderTest for PlaneFragmentAttention {
    type Case = PlaneFragmentAttentionCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let launch = self.launch_at(case)?;
        let [queries, keys] = launch.seq;
        let [head, val] = DIMS;
        let attention = HostAttention::new(launch.seq, DIMS);
        let input = launch.item.elements.query;
        // The query and output over `[B, KV, G, Q, _]`, the keys and values over `[B, KV, S, _]`,
        // each of one batch and one head.
        let tensor = |data: Vec<f32>, shape, element| {
            TestInput::builder(client.clone(), shape)
                .dtype(element)
                .custom(data)
                .generate_without_host_data()
        };
        let out = tensor(
            vec![-1e4; queries * val],
            shape![1, 1, 1, queries, val],
            launch.item.elements.out,
        );
        let logsumexp = tensor(
            vec![-1e4; queries],
            shape![1, 1, 1, queries],
            launch.item.sum,
        );
        let lengths = tensor(vec![keys as f32], shape![1], ElemType::UInt(UIntKind::U32));
        let scale = attention.scale();
        launch.launch(
            client,
            tensor(attention.query, shape![1, 1, 1, queries, head], input).binding(),
            tensor(attention.key, shape![1, 1, keys, head], input).binding(),
            tensor(attention.value, shape![1, 1, keys, val], input).binding(),
            out.clone().binding(),
            Some(logsumexp.clone().binding()),
            lengths.binding().into_tensor_arg(),
            scale,
        )?;
        Ok(vec![
            Written {
                name: "out",
                tensor: out,
                host: HostDataType::F32,
            },
            Written {
                name: "logsumexp",
                tensor: logsumexp,
                host: HostDataType::F32,
            },
        ])
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        let launch = self
            .launch_at(case)
            .unwrap_or_else(|why| panic!("{case:?} launched: {why:?}"));
        HostAttention::new(launch.seq, DIMS).out().into()
    }

    fn tolerance(&self, _case: &Self::Case) -> f32 {
        2e-2
    }
}

/// Every plane fragment attention microkernel the test device runs, at each of its settings, over
/// one and three stages of keys, for `f16` and `f32` inputs summed in `f32`.
#[test]
fn plane_fragment_attention_matches_the_cpu() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let cases: Vec<_> = [F16, F32]
        .into_iter()
        .flat_map(|input| {
            let elements = AttentionElements {
                input,
                sum: F32,
                out: F32,
            };
            PlaneFragmentAttentionMicrokernel::settings(&hw)
                .into_iter()
                .flat_map(move |settings| {
                    [1, 3].map(|steps| PlaneFragmentAttentionCase {
                        elements,
                        settings,
                        steps,
                    })
                })
        })
        .collect();
    PlaneFragmentAttention { hw }.check(cases);
}
