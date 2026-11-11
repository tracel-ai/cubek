//! The hypercube split-keys merge alone: the cubes of a box of queries each count themselves in,
//! and the last to arrive merges the box's slots by one plane.

use crate::launch::split_keys_merge::HypercubeMergeLaunch;
use cubecl::ir::ElemType;
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::shape;
use cubek_microkernel::split_keys_merge::SlotShape;
use cubek_microkernel::split_keys_merge::hypercube::HypercubeMergeShape;
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, ScopeUnderTest, TestInput, Written};

use crate::split_keys_merge::{F16, F32, HostSlots};

/// The hypercube merge on the test device.
struct HypercubeMerge {
    hw: Hardware,
}

/// `queries` queries of `splits` slots `val_dim` wide, in boxes of `box_queries`, each box counted
/// in by `cubes` cubes, merged into an output of `out`.
#[derive(Debug, Clone, Copy)]
struct HypercubeMergeCase {
    out: ElemType,
    queries: usize,
    splits: usize,
    val_dim: usize,
    box_queries: usize,
    cubes: usize,
}

impl ScopeUnderTest for HypercubeMerge {
    type Case = HypercubeMergeCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        if !self.hw.properties.features.device_memory_scope {
            return Err(Refusal::Declined {
                why: "the last arrival reads what other cubes stored within the launch",
            });
        }
        let HypercubeMergeCase {
            out: out_element,
            queries,
            splits,
            val_dim,
            box_queries,
            cubes,
        } = *case;
        let launch = HypercubeMergeLaunch {
            merge: HypercubeMergeShape {
                slots: SlotShape {
                    units: self.hw.dispatch_plane_size() as usize,
                    val_dim,
                    splits,
                },
                cubes,
                seq_q: box_queries,
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

/// The last of a box's cubes merges every query of the box as the host does, written in `f16` and
/// `f32`: one query a box, and boxes of two whose last holds one, among three cubes a box; and a
/// box the one cube that counts in merges alone.
#[test]
fn the_last_cube_of_a_box_merges_it_like_the_host() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let cases: Vec<_> = [F16, F32]
        .into_iter()
        .flat_map(|out| {
            [(3, 4, 8, 1, 3), (3, 48, 128, 2, 3), (2, 4, 8, 1, 1)].map(
                |(queries, splits, val_dim, box_queries, cubes)| HypercubeMergeCase {
                    out,
                    queries,
                    splits,
                    val_dim,
                    box_queries,
                    cubes,
                },
            )
        })
        .collect();
    HypercubeMerge { hw }.check(cases);
}
