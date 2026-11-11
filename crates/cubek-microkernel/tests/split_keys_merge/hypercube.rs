//! The hypercube split-keys merge alone: the cubes of a box of queries each count themselves in,
//! and the last to arrive merges the box's slots by one plane.

use crate::launch::split_keys_merge::{HypercubeMergeLaunch, HypercubeMergeTensors};
use cubecl::ir::ElemType;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::{Shape, shape};
use cubek_microkernel::split_keys_merge::SlotShape;
use cubek_microkernel::split_keys_merge::hypercube::HypercubeMergeShape;
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

use crate::split_keys_merge::{F16, F32, HostSlots};

/// What an output holds before anything writes it: no merge here reaches it.
const POISON: f32 = -7.0;

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
        let slot_shape = shape![1, 1, 1, queries, splits];
        let row_shape = shape![1, 1, 1, queries, splits, val_dim];
        let outputs = || {
            (
                tensor(
                    vec![POISON; queries * val_dim],
                    shape![1, 1, 1, queries, val_dim],
                    out_element,
                ),
                tensor(vec![POISON; queries], shape![1, 1, 1, queries], F32),
            )
        };
        let arrivals = TensorHandle::zeros(client, [launch.boxes()], u32::elem_type_native());
        let elements = [F32, out_element];
        // Twice on the same counters, which a box's last cube must leave at zero, or the second
        // round merges early or never. In each, the slots start as NaN, and every box's cubes but
        // the last write into outputs of their own, which must keep their poison: only the last
        // has every share stored.
        let mut written = Vec::new();
        for _ in 0..2 {
            let (out, logsumexp) = outputs();
            let (early_out, early_logsumexp) = outputs();
            let mut tensors = HypercubeMergeTensors {
                out: early_out.clone(),
                logsumexp: early_logsumexp.clone(),
                source_out: tensor(slots.out.clone(), row_shape.clone(), F32),
                source_max: tensor(slots.max.clone(), slot_shape.clone(), F32),
                source_sum: tensor(slots.sum.clone(), slot_shape.clone(), F32),
                slot_out: tensor(vec![f32::NAN; slots.out.len()], row_shape.clone(), F32),
                slot_max: tensor(vec![f32::NAN; slots.max.len()], slot_shape.clone(), F32),
                slot_sum: tensor(vec![f32::NAN; slots.sum.len()], slot_shape.clone(), F32),
            };
            for cube in 0..cubes - 1 {
                launch.launch_cube(client, &tensors, &arrivals, cube, elements);
            }
            tensors.out = out.clone();
            tensors.logsumexp = logsumexp.clone();
            launch.launch_cube(client, &tensors, &arrivals, cubes - 1, elements);
            written = vec![early_out, early_logsumexp, out, logsumexp];
        }
        let [early_out, early_logsumexp, out, logsumexp] = <[TensorHandle; 4]>::try_from(written)
            .expect("two outputs a round, and their early twins");
        Ok([
            ("out before the last cube", early_out),
            ("logsumexp before the last cube", early_logsumexp),
            ("out", out),
            ("logsumexp", logsumexp),
        ]
        .map(|(name, tensor)| Written {
            name,
            tensor,
            host: HostDataType::F32,
        })
        .into_iter()
        .collect())
    }

    /// The early outputs untouched, then the merge.
    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        let HypercubeMergeCase {
            queries, val_dim, ..
        } = *case;
        let untouched = |shape: Shape| {
            let cells = shape.num_elements();
            HostData::row_major(HostDataVec::F32(vec![POISON; cells]), shape)
        };
        let [out, logsumexp] = HostSlots::new(queries, case.splits, val_dim).merged();
        vec![
            untouched(shape![1, 1, 1, queries, val_dim]),
            untouched(shape![1, 1, 1, queries]),
            out,
            logsumexp,
        ]
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
    let shapes = [
        // A query a box, three cubes a box.
        HypercubeMergeCase {
            out: F32,
            queries: 3,
            splits: 4,
            val_dim: 8,
            box_queries: 1,
            cubes: 3,
        },
        // Boxes of two queries, the last holding one, rows wider than the plane.
        HypercubeMergeCase {
            out: F32,
            queries: 3,
            splits: 48,
            val_dim: 128,
            box_queries: 2,
            cubes: 3,
        },
        // A box one cube counts in and merges alone.
        HypercubeMergeCase {
            out: F32,
            queries: 2,
            splits: 4,
            val_dim: 8,
            box_queries: 1,
            cubes: 1,
        },
    ];
    let cases: Vec<_> = [F16, F32]
        .into_iter()
        .flat_map(|out| shapes.map(|case| HypercubeMergeCase { out, ..case }))
        .collect();
    HypercubeMerge { hw }.check(cases);
}
