//! Products whose inputs read a table, one entry per batch, through the dense routines: each batch
//! against the slab of a stacked rhs its entry names, or over its run of `K`. Every routine that
//! walks within its inputs answers the reference at each setting it prepares; an entry past the
//! stack, or an empty run, answers zeros.

use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::Shape;
use cubek_matmul::kernels::tensor::{PlainTensor, StoredTensor};
use cubek_matmul::kernels::{
    CubeStagedMatmulRoutine, MatmulIndex, MatmulIndexTable, MatmulRoutine,
    PlaneShuffleMatmulRoutine, UnitRegistersMatmulRoutine,
};
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_std::Hardware;
use cubek_std::routine::Routine;
use cubek_test_utils::{HostData, HostDataType, client};

/// `values` uploaded as a dense tensor of `shape`.
fn upload<E: CubeElement + Scalar>(client: &Client, shape: &[usize], values: &[E]) -> PlainTensor {
    let handle = TensorHandle::new_contiguous(
        Shape::from(shape.to_vec()),
        client.create(Bytes::from_elems(values.to_vec())),
        E::elem_type_native(),
    );
    PlainTensor::of_handle(handle)
}

/// The coordinates of the `i`-th element of a dense row-major tensor of `shape`.
fn unravel(mut i: usize, shape: &[usize]) -> Vec<usize> {
    let mut at = vec![0; shape.len()];
    for (dim, &extent) in shape.iter().enumerate().rev() {
        at[dim] = i % extent;
        i /= extent;
    }
    at
}

/// Deterministic values in `[-1, 1)`, different for each `seed`.
fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 7919 + seed * 104_729) % 2003) as f32 / 1001.5) - 1.)
        .collect()
}

/// Every setting of `routine` that prepares on this device, launched on the tensors `inputs`
/// makes and checked against `expected`; at least one prepares.
fn check_every_setting<M, SM, R>(
    routine: R,
    inputs: impl Fn() -> (PlainTensor, PlainTensor, MatmulIndexTable),
    expected: &[f32],
) where
    M: MatmulMicrokernel,
    R: MatmulRoutine<M, SM>,
{
    let client = client();
    let hw = Hardware::from_client(&client);
    let mut ran = 0;
    for settings in routine.settings(&hw) {
        let (lhs, rhs, table) = inputs();
        // `[batches.., rows, n]`: the table's dims, then the activation's rows and the weight's
        // columns.
        let (lhs_shape, rhs_shape) = (lhs.shape(), rhs.shape());
        let shape = [
            table.entries.shape().as_slice(),
            &[
                lhs_shape[lhs_shape.rank() - 2],
                rhs_shape[rhs_shape.rank() - 1],
            ],
        ]
        .concat();
        let (out, bound) =
            PlainTensor::empty_contiguous(&client, Shape::from(shape.clone()), lhs.elem);
        let launched = routine.launch_indexed_into(
            &hw,
            &client,
            lhs,
            StoredTensor::Plain(rhs),
            table,
            bound,
            &settings,
        );
        if launched.is_err() {
            continue;
        }
        ran += 1;
        let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
        for (i, &want) in expected.iter().enumerate() {
            let have = got.get_f32(&unravel(i, &shape));
            assert!(
                (have - want).abs() <= 1e-3 * (1. + want.abs()),
                "{settings:?}: element {i} is {have}, want {want}"
            );
        }
    }
    assert!(ran > 0, "no setting prepared");
}

/// `tokens` rows, each routed to `picks` of `slabs` slabs of `[k, n]`: batches `[tokens, picks]`
/// of one row, each against the slab its route names. The last route names one past the stack.
struct Routed {
    tokens: usize,
    picks: usize,
    slabs: usize,
    k: usize,
    n: usize,
}

impl Routed {
    fn routes(&self) -> Vec<u32> {
        let mut routes: Vec<u32> = (0..self.tokens * self.picks)
            .map(|i| ((i * 3 + 1) % self.slabs) as u32)
            .collect();
        *routes.last_mut().unwrap() = self.slabs as u32;
        routes
    }

    fn inputs(&self) -> (PlainTensor, PlainTensor, MatmulIndexTable) {
        let client = client();
        let Self {
            tokens,
            picks,
            slabs,
            k,
            n,
        } = *self;
        // A route is a batch of one row: the activation's rows are its first batch dim, broadcast
        // along the picks.
        let lhs = upload(&client, &[tokens, 1, 1, k], &values(tokens * k, 1));
        let stack = upload(&client, &[slabs, k, n], &values(slabs * k * n, 2));
        let entries = upload(&client, &[tokens, picks], &self.routes());
        let table = MatmulIndexTable {
            index: MatmulIndex::Slab { slabs },
            entries,
        };
        (lhs, stack, table)
    }

    fn expected(&self) -> Vec<f32> {
        let Self {
            tokens,
            picks,
            slabs,
            k,
            n,
        } = *self;
        let (lhs, stack, routes) = (
            values(tokens * k, 1),
            values(slabs * k * n, 2),
            self.routes(),
        );
        let mut out = vec![0.; tokens * picks * n];
        for t in 0..tokens {
            for p in 0..picks {
                let slab = routes[t * picks + p] as usize;
                if slab >= slabs {
                    continue;
                }
                for j in 0..n {
                    out[(t * picks + p) * n + j] = (0..k)
                        .map(|kk| lhs[t * k + kk] * stack[(slab * k + kk) * n + j])
                        .sum();
                }
            }
        }
        out
    }
}

/// `rows` rows grouped by `ends`, each group's cumulative end: `out[g] = lhs[.., rows of g] ·
/// rhs[rows of g, ..]`, batches `[groups]` over the runs of `K`. The second group is empty.
struct Grouped {
    m: usize,
    n: usize,
    ends: Vec<u32>,
}

impl Grouped {
    fn rows(&self) -> usize {
        *self.ends.last().unwrap() as usize
    }

    fn inputs(&self) -> (PlainTensor, PlainTensor, MatmulIndexTable) {
        let client = client();
        let (m, n, rows) = (self.m, self.n, self.rows());
        let lhs = upload(&client, &[m, rows], &values(m * rows, 3));
        let rhs = upload(&client, &[rows, n], &values(rows * n, 4));
        let entries = upload(&client, &[self.ends.len()], &self.ends);
        let table = MatmulIndexTable {
            index: MatmulIndex::RunOfK,
            entries,
        };
        (lhs, rhs, table)
    }

    fn expected(&self) -> Vec<f32> {
        let (m, n, rows) = (self.m, self.n, self.rows());
        let (lhs, rhs) = (values(m * rows, 3), values(rows * n, 4));
        let mut out = vec![0.; self.ends.len() * m * n];
        let mut start = 0;
        for (g, &end) in self.ends.iter().enumerate() {
            for i in 0..m {
                for j in 0..n {
                    out[(g * m + i) * n + j] = (start..end as usize)
                        .map(|r| lhs[i * rows + r] * rhs[r * n + j])
                        .sum();
                }
            }
            start = end as usize;
        }
        out
    }
}

/// `chunks` chunks of `rows` rows each, sorted by slab and padded to whole chunks: batches
/// `[chunks]`, each against the slab its entry names. The last chunk is padding, its entry one
/// past the stack.
struct Chunked {
    chunks: usize,
    rows: usize,
    slabs: usize,
    k: usize,
    n: usize,
}

impl Chunked {
    fn slab_of_each(&self) -> Vec<u32> {
        let mut slabs: Vec<u32> = (0..self.chunks)
            .map(|c| ((c * 5 + 2) % self.slabs) as u32)
            .collect();
        *slabs.last_mut().unwrap() = self.slabs as u32;
        slabs
    }

    fn inputs(&self) -> (PlainTensor, PlainTensor, MatmulIndexTable) {
        let client = client();
        let Self {
            chunks,
            rows,
            slabs,
            k,
            n,
        } = *self;
        let lhs = upload(&client, &[chunks, rows, k], &values(chunks * rows * k, 5));
        let stack = upload(&client, &[slabs, k, n], &values(slabs * k * n, 6));
        let entries = upload(&client, &[chunks], &self.slab_of_each());
        let table = MatmulIndexTable {
            index: MatmulIndex::Slab { slabs },
            entries,
        };
        (lhs, stack, table)
    }

    fn expected(&self) -> Vec<f32> {
        let Self {
            chunks,
            rows,
            slabs,
            k,
            n,
        } = *self;
        let (lhs, stack) = (values(chunks * rows * k, 5), values(slabs * k * n, 6));
        let mut out = vec![0.; chunks * rows * n];
        for (c, &slab) in self.slab_of_each().iter().enumerate() {
            let slab = slab as usize;
            if slab >= slabs {
                continue;
            }
            for r in c * rows..(c + 1) * rows {
                for j in 0..n {
                    out[r * n + j] = (0..k)
                        .map(|kk| lhs[r * k + kk] * stack[(slab * k + kk) * n + j])
                        .sum();
                }
            }
        }
        out
    }
}

const ROUTED: Routed = Routed {
    tokens: 5,
    picks: 3,
    slabs: 4,
    k: 128,
    n: 96,
};

#[test]
fn a_routed_product_on_the_plane_shuffle_reads_each_route_s_slab() {
    check_every_setting(
        PlaneShuffleMatmulRoutine,
        || ROUTED.inputs(),
        &ROUTED.expected(),
    );
}

#[test]
fn a_routed_product_on_the_unit_registers_reads_each_route_s_slab() {
    check_every_setting(
        UnitRegistersMatmulRoutine,
        || ROUTED.inputs(),
        &ROUTED.expected(),
    );
}

#[test]
fn a_grouped_product_on_the_cube_staged_sums_each_group_s_run() {
    let grouped = Grouped {
        m: 64,
        n: 64,
        ends: vec![37, 37, 100],
    };
    check_every_setting(
        CubeStagedMatmulRoutine,
        || grouped.inputs(),
        &grouped.expected(),
    );
}

#[test]
fn a_chunked_product_on_the_cube_staged_reads_each_chunk_s_slab() {
    let chunked = Chunked {
        chunks: 5,
        rows: 32,
        slabs: 3,
        k: 96,
        n: 80,
    };
    check_every_setting(
        CubeStagedMatmulRoutine,
        || chunked.inputs(),
        &chunked.expected(),
    );
}

#[test]
fn a_chunked_product_on_the_plane_shuffle_reads_each_chunk_s_slab() {
    let chunked = Chunked {
        chunks: 5,
        rows: 4,
        slabs: 3,
        k: 96,
        n: 80,
    };
    check_every_setting(
        PlaneShuffleMatmulRoutine,
        || chunked.inputs(),
        &chunked.expected(),
    );
}
