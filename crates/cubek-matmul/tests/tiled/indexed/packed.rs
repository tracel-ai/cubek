//! A packed stack read through an indexed launch answers the product with its dequantized values:
//! each route, or each group, reads the slab its entry names, words and scales both.

use super::super::packed::{PackedFormat, PackedSlabs, PackedSlabsSpec};
use super::*;
use cubek_matmul::kernels::m_grouped::{MGroupedMatmulTensors, MGroupedSettings};
use cubek_matmul::kernels::stack::SlabAxis;
use cubek_matmul::kernels::tensor::PACKED_STACK_UNREAD;
use cubek_microkernel::matmul::MatmulOperand;
use half::f16;

/// Loose enough for an `f16` output.
const HALF_TOLERANCE: f32 = 1e-2;

/// An activation in `f16`, and what its values read back as.
struct HalfValues {
    halves: Vec<f16>,
    floats: Vec<f32>,
}

impl HalfValues {
    fn new(count: usize, seed: usize) -> Self {
        let halves: Vec<f16> = values(count, seed).into_iter().map(f16::from_f32).collect();
        let floats = halves.iter().map(|half| half.to_f32()).collect();
        Self { halves, floats }
    }

    fn upload(&self, client: &Client, shape: &[usize]) -> PlainTensor {
        upload(client, shape, &self.halves)
    }
}

/// Rows grouped by `sizes`, one group a slab of a packed stack: the second group is empty.
struct MGrouped {
    k: usize,
    n: usize,
    sizes: [usize; 4],
}

impl MGrouped {
    fn rows(&self) -> usize {
        self.sizes.iter().sum()
    }

    fn ends(&self) -> Vec<u32> {
        self.sizes
            .iter()
            .scan(0, |end, size| {
                *end += size;
                Some(*end as u32)
            })
            .collect()
    }

    /// The product of `lhs`, `[rows, k]`, with `stack(group, k, n)` read at each row's group.
    fn expected_from(&self, lhs: &[f32], stack: impl Fn(usize, usize, usize) -> f32) -> Vec<f32> {
        let Self { k, n, .. } = *self;
        let group_ends = self.ends();
        let group_of = |row: usize| group_ends.iter().position(|&end| row < end as usize);
        let mut out = vec![0.; self.rows() * n];
        for row in 0..self.rows() {
            let group = group_of(row).expect("every row is in a group");
            for j in 0..n {
                out[row * n + j] = (0..k)
                    .map(|kk| lhs[row * k + kk] * stack(group, kk, j))
                    .sum();
            }
        }
        out
    }
}

/// A second routed shape, whose `n` is not a multiple of the column tile.
const NARROW: Routed = Routed {
    tokens: 5,
    picks: 3,
    slabs: 4,
    k: 128,
    n: 80,
};

fn a_routed_packed_stack_reads_each_route_s_slab(format: PackedFormat) {
    for routed in [ROUTED, NARROW] {
        let client = client();
        let lhs = HalfValues::new(routed.tokens * routed.k, 1);
        let stack = PackedSlabs::new(
            &client,
            PackedSlabsSpec {
                format,
                operand: MatmulOperand::Rhs,
                slabs: routed.slabs,
                rows: routed.n,
                k: routed.k,
                seed: 2,
            },
        );
        let expected = routed.expected_from(&lhs.floats, |slab, k, j| stack.cell(slab, k, j));
        check_settings(
            PlaneShuffleMatmulRoutine,
            |routine, _| routine.packed_settings(),
            || {
                let entries = upload(&client, &[routed.tokens, routed.picks], &routed.routes());
                let table = MatmulIndexTable {
                    index: MatmulIndex::Slab {
                        slabs: routed.slabs,
                    },
                    entries,
                };
                let lhs = lhs.upload(&client, &[routed.tokens, 1, 1, routed.k]);
                (lhs, stack.stack(), table)
            },
            &expected,
            HALF_TOLERANCE,
        );
    }
}

fn a_grouped_packed_stack_reads_each_group_s_slab(format: PackedFormat) {
    let grouped = MGrouped {
        k: 128,
        n: 96,
        sizes: [5, 0, 11, 2],
    };
    let (rows, k, n) = (grouped.rows(), grouped.k, grouped.n);
    let client = client();
    let hw = Hardware::from_client(&client);
    let lhs = HalfValues::new(rows * k, 3);
    let stack = PackedSlabs::new(
        &client,
        PackedSlabsSpec {
            format,
            operand: MatmulOperand::Rhs,
            slabs: grouped.sizes.len(),
            rows: n,
            k,
            seed: 4,
        },
    );
    let expected = grouped.expected_from(&lhs.floats, |group, k, j| stack.cell(group, k, j));
    let mut walk = Walk::new();
    for settings in MGroupedSettings::plane_shuffle(&hw) {
        let tensors = MGroupedMatmulTensors::new(
            &client,
            lhs.upload(&client, &[rows, k]),
            StoredTensor::Packed(stack.matrix()),
            upload(&client, &[grouped.sizes.len()], &grouped.ends()),
            SlabAxis::Output,
        )
        .expect("a packed stack of whole slabs derives a grouped launch");
        let launched = tensors.launch(&hw, &PlaneShuffleMatmulRoutine, &settings);
        if let Some(out) = walk.ran(launched) {
            let got = HostData::from_tensor_handle(&client, out.unpad(), HostDataType::F32);
            assert_close(&got, &[rows, n], &expected, HALF_TOLERANCE, &settings);
        }
    }
    walk.settle();
}

#[test]
fn a_routed_product_on_the_plane_shuffle_reads_each_route_s_packed_q8_slab() {
    a_routed_packed_stack_reads_each_route_s_slab(PackedFormat::Q8);
}

#[test]
fn a_routed_product_on_the_plane_shuffle_reads_each_route_s_packed_q4_slab() {
    a_routed_packed_stack_reads_each_route_s_slab(PackedFormat::Q4);
}

#[test]
fn a_routed_product_on_the_plane_shuffle_reads_each_route_s_packed_nvfp4_slab() {
    a_routed_packed_stack_reads_each_route_s_slab(PackedFormat::Nvfp4);
}

#[test]
fn a_grouped_product_on_the_plane_shuffle_reads_each_group_s_packed_q8_slab() {
    a_grouped_packed_stack_reads_each_group_s_slab(PackedFormat::Q8);
}

#[test]
fn a_grouped_product_on_the_plane_shuffle_reads_each_group_s_packed_q4_slab() {
    a_grouped_packed_stack_reads_each_group_s_slab(PackedFormat::Q4);
}

#[test]
fn a_grouped_product_on_the_plane_shuffle_reads_each_group_s_packed_nvfp4_slab() {
    a_grouped_packed_stack_reads_each_group_s_slab(PackedFormat::Nvfp4);
}

/// Slabs along `K` that split a scale block, or a word of block scales, are refused rather than
/// read across.
#[test]
fn a_packed_stack_whose_slabs_split_a_scale_block_is_refused() {
    let client = client();
    let along_k = |format, slabs| {
        let matrix = PackedSlabs::new(
            &client,
            PackedSlabsSpec {
                format,
                operand: MatmulOperand::Rhs,
                slabs: 1,
                rows: 8,
                k: 128,
                seed: 5,
            },
        );
        matrix.matrix().stacked(slabs, K).err()
    };
    // 32-value blocks: 4 slabs hold whole blocks, 8 hold half of one.
    assert_eq!(along_k(PackedFormat::Q8, 4), None);
    assert_eq!(along_k(PackedFormat::Q8, 8), Some(PACKED_STACK_UNREAD));
    // NVFP4: four 16-value blocks a word of scales, so 4 slabs split a word.
    assert_eq!(along_k(PackedFormat::Nvfp4, 2), None);
    assert_eq!(along_k(PackedFormat::Nvfp4, 4), Some(PACKED_STACK_UNREAD));
}
