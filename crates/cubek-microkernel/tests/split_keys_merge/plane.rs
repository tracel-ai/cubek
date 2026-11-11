//! The plane split-keys merge alone: one plane a query merges the query's slots.

use crate::launch::split_keys_merge::PlaneMergeLaunch;
use cubecl::ir::ElemType;
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::shape;
use cubek_microkernel::split_keys_merge::SlotShape;
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, ScopeUnderTest, TestInput, Written};

use crate::split_keys_merge::{F16, F32, HostSlots};

/// The plane merge on the test device.
struct PlaneMerge {
    hw: Hardware,
}

/// `queries` queries of `splits` slots `val_dim` wide, merged into an output of `out`.
#[derive(Debug, Clone, Copy)]
struct PlaneMergeCase {
    out: ElemType,
    queries: usize,
    splits: usize,
    val_dim: usize,
}

impl ScopeUnderTest for PlaneMerge {
    type Case = PlaneMergeCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let PlaneMergeCase {
            out: out_element,
            queries,
            splits,
            val_dim,
        } = *case;
        let launch = PlaneMergeLaunch {
            slots: SlotShape {
                units: self.hw.dispatch_plane_size() as usize,
                val_dim,
                splits,
            },
            queries,
        };
        let slots = HostSlots::new(queries, splits, val_dim);
        let tensor = |values: Vec<f32>, shape, element| {
            TestInput::builder(client.clone(), shape)
                .dtype(element)
                .custom(values)
                .generate_without_host_data()
        };
        let out = tensor(
            vec![-7.0; queries * val_dim],
            shape![1, 1, 1, queries, val_dim],
            out_element,
        );
        let logsumexp = tensor(vec![-7.0; queries], shape![1, 1, 1, queries], F32);
        let slot_out = tensor(slots.out, shape![1, 1, 1, queries, splits, val_dim], F32);
        let slot_max = tensor(slots.max, shape![1, 1, 1, queries, splits], F32);
        let slot_sum = tensor(slots.sum, shape![1, 1, 1, queries, splits], F32);
        launch.launch(
            client,
            out.clone().binding().into_tensor_arg(),
            logsumexp.clone().binding().into_tensor_arg(),
            slot_out.binding().into_tensor_arg(),
            slot_max.binding().into_tensor_arg(),
            slot_sum.binding().into_tensor_arg(),
            [F32, out_element],
        );
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
        HostSlots::new(case.queries, case.splits, case.val_dim)
            .merged()
            .into()
    }

    /// The device's `exp` and `ln`; an `f16` output rounds what it writes.
    fn tolerance(&self, case: &Self::Case) -> f32 {
        match case.out {
            F32 => 1e-5,
            _ => 2e-3,
        }
    }
}

/// A plane merges every query's slots as the host does, written in `f16` and `f32`: fewer slots
/// than the plane's units and more, a row narrower than the plane and wider, and queries whose
/// slots saw nothing.
#[test]
fn a_plane_merges_every_query_like_the_host() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let cases: Vec<_> = [F16, F32]
        .into_iter()
        .flat_map(|out| {
            [(3, 4, 8), (2, 48, 128)].map(|(queries, splits, val_dim)| PlaneMergeCase {
                out,
                queries,
                splits,
                val_dim,
            })
        })
        .collect();
    PlaneMerge { hw }.check(cases);
}
