//! A packed stack read through an indexed launch answers the product with its dequantized values:
//! each route, or each group, reads the slab its entry names, words and scales both.

use super::super::packed::{Laid, PackedFormat, PackedSlabs, PackedSlabsSpec};
use super::*;
use cubecl::ir::{ElemType, FloatKind};
use cubek_matmul::kernels::MatmulRoutineProblem;
use cubek_matmul::kernels::indexed::m_grouped::{MGroupedMatmulTensors, MGroupedSettings};
use cubek_matmul::kernels::indexed::stack::SlabAxis;
use cubek_matmul::kernels::tensor::PACKED_STACK_UNREAD;
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};
use cubek_std::layout::tile_order::TileWords;
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

fn a_tuned_routed_packed_stack_reads_each_route_s_slab(format: PackedFormat) {
    let client = client();
    let lhs = HalfValues::new(ROUTED.tokens * ROUTED.k, 1);
    let stack = PackedSlabs::new(
        &client,
        PackedSlabsSpec {
            format,
            operand: MatmulOperand::Rhs,
            slabs: ROUTED.slabs,
            rows: ROUTED.n,
            k: ROUTED.k,
            seed: 2,
        },
    );
    let expected = ROUTED.expected_from(&lhs.floats, |slab, k, j| stack.cell(slab, k, j));
    let tensors = RoutedMatmulTensors::new(
        &client,
        lhs.upload(&client, &[ROUTED.tokens, ROUTED.k]),
        StoredTensor::Packed(stack.matrix()),
        upload(&client, &[ROUTED.tokens, ROUTED.picks], &ROUTED.routes()),
        ROUTED.slabs,
        SlabAxis::Output,
    )
    .expect("a packed stack of whole slabs derives a routed launch");
    let hw = Hardware::from_client(&client);
    let no_matrix_instruction = hw.properties.features.matmul.cmma.iter().next().is_none();
    let out = match launch_tuned(&cubecl::test_device(), tensors, GroupPriority::DEFAULT) {
        Ok(out) => out,
        Err(why) if matches!(why, Refusal::CubePastDevice { .. }) || no_matrix_instruction => {
            TestOutcome::Validated(ValidationResult::Skipped(format!(
                "no floor serves a packed stack on this runtime: {why:?}"
            )))
            .enforce();
            return;
        }
        Err(why) => panic!("a packed routed stack is refused: {why:?}"),
    };
    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    let shape = [ROUTED.tokens, ROUTED.picks, ROUTED.n];
    assert_close(&got, &shape, &expected, HALF_TOLERANCE, &"routed");
}

fn a_grouped_packed_stack_reads_each_group_s_slab<Mk, SM, R>(
    format: PackedFormat,
    laid: Laid,
    routine: R,
    settings: impl Fn(&Hardware) -> Vec<MGroupedSettings<R::Settings>>,
) where
    Mk: MatmulMicrokernel,
    R: MatmulRoutine<Mk, SM>,
{
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
    for settings in settings(&hw) {
        let tensors = MGroupedMatmulTensors::new(
            &client,
            lhs.upload(&client, &[rows, k]),
            StoredTensor::Packed(stack.laid(&client, laid)),
            upload(&client, &[grouped.sizes.len()], &grouped.ends()),
            SlabAxis::Output,
        )
        .expect("a packed stack of whole slabs derives a grouped launch");
        let launched = tensors.launch(&hw, &routine, &settings);
        if let Some(out) = walk.ran(launched) {
            let got = HostData::from_tensor_handle(&client, out.unpad(), HostDataType::F32);
            assert_close(
                &got,
                &[rows, n],
                &expected,
                HALF_TOLERANCE,
                &(laid, settings),
            );
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

/// Whether the test is skipped on `hw` for offering no matrix instruction, which the cube staged
/// routine stages a packed input for: every packed setting declines there.
fn skipped_without_a_matrix_instruction(hw: &Hardware) -> bool {
    let without = hw.properties.features.matmul.cmma.iter().next().is_none();
    if without {
        TestOutcome::Validated(ValidationResult::Skipped(
            "the cube staged routine stages a packed input for a matrix instruction".to_string(),
        ))
        .enforce();
    }
    without
}

fn a_grouped_cube_staged_packed_stack_reads_each_group_s_slab(format: PackedFormat) {
    if skipped_without_a_matrix_instruction(&Hardware::from_client(&client())) {
        return;
    }
    a_grouped_packed_stack_reads_each_group_s_slab(
        format,
        Laid::InRows,
        CubeStagedMatmulRoutine,
        MGroupedSettings::cube_staged_packed,
    );
}

#[test]
fn a_tuned_routed_product_reads_each_route_s_packed_q8_slab() {
    a_tuned_routed_packed_stack_reads_each_route_s_slab(PackedFormat::Q8);
}

#[test]
fn a_tuned_routed_product_reads_each_route_s_packed_q4_slab() {
    a_tuned_routed_packed_stack_reads_each_route_s_slab(PackedFormat::Q4);
}

#[test]
fn a_tuned_routed_product_reads_each_route_s_packed_nvfp4_slab() {
    a_tuned_routed_packed_stack_reads_each_route_s_slab(PackedFormat::Nvfp4);
}

#[test]
fn a_grouped_product_on_the_plane_shuffle_reads_each_group_s_packed_q8_slab() {
    a_grouped_packed_stack_reads_each_group_s_slab(
        PackedFormat::Q8,
        Laid::InRows,
        PlaneShuffleMatmulRoutine,
        |_| MGroupedSettings::plane_shuffle_packed(),
    );
}

#[test]
fn a_grouped_product_on_the_plane_shuffle_reads_each_group_s_packed_q4_slab() {
    a_grouped_packed_stack_reads_each_group_s_slab(
        PackedFormat::Q4,
        Laid::InRows,
        PlaneShuffleMatmulRoutine,
        |_| MGroupedSettings::plane_shuffle_packed(),
    );
}

#[test]
fn a_grouped_product_on_the_plane_shuffle_reads_each_group_s_packed_nvfp4_slab() {
    a_grouped_packed_stack_reads_each_group_s_slab(
        PackedFormat::Nvfp4,
        Laid::InRows,
        PlaneShuffleMatmulRoutine,
        |_| MGroupedSettings::plane_shuffle_packed(),
    );
}

#[test]
fn a_grouped_product_on_the_cube_staged_reads_each_group_s_packed_q8_slab() {
    a_grouped_cube_staged_packed_stack_reads_each_group_s_slab(PackedFormat::Q8);
}

#[test]
fn a_grouped_product_on_the_cube_staged_reads_each_group_s_packed_q4_slab() {
    a_grouped_cube_staged_packed_stack_reads_each_group_s_slab(PackedFormat::Q4);
}

#[test]
fn a_grouped_product_on_the_cube_staged_reads_each_group_s_packed_nvfp4_slab() {
    a_grouped_cube_staged_packed_stack_reads_each_group_s_slab(PackedFormat::Nvfp4);
}

#[test]
fn a_tuned_routed_product_reads_each_route_s_packed_mxfp4_slab() {
    a_tuned_routed_packed_stack_reads_each_route_s_slab(PackedFormat::Mxfp4);
}

#[test]
fn a_grouped_product_on_the_plane_shuffle_reads_each_group_s_packed_mxfp4_slab() {
    a_grouped_packed_stack_reads_each_group_s_slab(
        PackedFormat::Mxfp4,
        Laid::InRows,
        PlaneShuffleMatmulRoutine,
        |_| MGroupedSettings::plane_shuffle_packed(),
    );
}

#[test]
fn a_grouped_product_on_the_cube_staged_reads_each_group_s_packed_mxfp4_slab() {
    a_grouped_cube_staged_packed_stack_reads_each_group_s_slab(PackedFormat::Mxfp4);
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

/// The chunk rows of a cube staged setting over a packed stack are read from a prepare over every
/// grouped row, and the launch prepares again on the chunk rows alone: the cube tile it launches
/// on is the rows the groups were padded to.
#[test]
fn a_cube_staged_launch_over_a_packed_stack_holds_the_rows_its_groups_were_padded_to() {
    let (k, n, sizes) = (128, 96, [100usize, 0, 300, 60]);
    let rows: usize = sizes.iter().sum();
    let ends: Vec<u32> = sizes
        .iter()
        .scan(0, |end, size| {
            *end += size;
            Some(*end as u32)
        })
        .collect();
    let client = client();
    let hw = Hardware::from_client(&client);
    if skipped_without_a_matrix_instruction(&hw) {
        return;
    }
    let lhs = HalfValues::new(rows * k, 3);
    let mut walk = Walk::new();
    for format in [
        PackedFormat::Q8,
        PackedFormat::Q4,
        PackedFormat::Nvfp4,
        PackedFormat::Mxfp4,
    ] {
        let stack = PackedSlabs::new(
            &client,
            PackedSlabsSpec {
                format,
                operand: MatmulOperand::Rhs,
                slabs: sizes.len(),
                rows: n,
                k,
                seed: 4,
            },
        );
        for settings in MGroupedSettings::cube_staged_packed(&hw) {
            let tensors = MGroupedMatmulTensors::new(
                &client,
                lhs.upload(&client, &[rows, k]),
                StoredTensor::Packed(stack.matrix()),
                upload(&client, &[sizes.len()], &ends),
                SlabAxis::Output,
            )
            .expect("a packed stack of whole slabs derives a grouped launch");
            let launched = tensors.launch(&hw, &CubeStagedMatmulRoutine, &settings);
            let Some(padded) = walk.ran(launched) else {
                continue;
            };
            let chunk = padded.layout().align();
            let packing = stack
                .matrix()
                .stacked(sizes.len(), SlabAxis::Output.axis())
                .expect("whole slabs stack")
                .packing();
            let problem = MatmulRoutineProblem::packed(
                chunk,
                n,
                k,
                ElemType::Float(FloatKind::F16),
                MatmulOperands {
                    rhs: Some(packing),
                    ..MatmulOperands::all(None)
                },
            );
            let (blueprint, _) = CubeStagedMatmulRoutine
                .prepare(&hw, &problem, &settings.routine)
                .expect("the launch prepared these chunk rows");
            assert_eq!(
                blueprint.partitioning.rows() as usize,
                chunk,
                "{format:?} {settings}"
            );
        }
    }
    walk.settle();
}

/// The tile orders a stack of routed experts is laid in: row by row in the tile the TMA engine
/// fills a Blackwell stage of, and block by block in one software reads a load at a time. At
/// `k` 128 a strip is four transactions, so the stagger pads it to five: each slab starts past
/// padding.
const TILE_ORDERS: [Laid; 2] = [
    Laid::InTiles(TileWords::RowByRow, (128, 8)),
    Laid::InTiles(TileWords::BlockByBlock, (64, 8)),
];

/// A routed shape whose slab is one strip of eight rows.
const ONE_STRIP: Routed = Routed {
    tokens: 5,
    picks: 3,
    slabs: 4,
    k: 128,
    n: 8,
};

/// A routed product reads each route's slab of an NVFP4 stack laid in tile order on every setting
/// of `routine` that prepares it, the route past the stack summing nothing.
fn a_routed_stack_in_tile_order_reads_each_route_s_slab<Mk, SM, R>(
    routine: R,
    settings: impl Fn(&R, &Hardware) -> Vec<R::Settings>,
) where
    Mk: MatmulMicrokernel,
    R: MatmulRoutine<Mk, SM>,
{
    let client = client();
    let hw = Hardware::from_client(&client);
    let mut walk = Walk::new();
    for laid in TILE_ORDERS {
        for routed in [ROUTED, NARROW, ONE_STRIP] {
            let lhs = HalfValues::new(routed.tokens * routed.k, 1);
            let stack = PackedSlabs::new(
                &client,
                PackedSlabsSpec {
                    format: PackedFormat::Nvfp4,
                    operand: MatmulOperand::Rhs,
                    slabs: routed.slabs,
                    rows: routed.n,
                    k: routed.k,
                    seed: 2,
                },
            );
            let expected = routed.expected_from(&lhs.floats, |slab, k, j| stack.cell(slab, k, j));
            let tensors = RoutedMatmulTensors::new(
                &client,
                lhs.upload(&client, &[routed.tokens, routed.k]),
                StoredTensor::Packed(stack.laid(&client, laid)),
                upload(&client, &[routed.tokens, routed.picks], &routed.routes()),
                routed.slabs,
                SlabAxis::Output,
            )
            .expect("a packed stack in tile order derives a routed launch");
            for settings in settings(&routine, &hw) {
                if let Some(out) = walk.ran(tensors.launch(&hw, &routine, &settings)) {
                    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
                    let shape = [routed.tokens, routed.picks, routed.n];
                    assert_close(&got, &shape, &expected, HALF_TOLERANCE, &(laid, settings));
                }
            }
        }
    }
    walk.settle();
}

#[test]
fn a_routed_product_on_the_plane_shuffle_reads_each_route_s_slab_of_a_stack_in_tile_order() {
    a_routed_stack_in_tile_order_reads_each_route_s_slab(
        PlaneShuffleMatmulRoutine,
        |routine, _| routine.packed_settings(),
    );
}

#[test]
fn a_routed_product_on_the_cube_staged_reads_each_route_s_slab_of_a_stack_in_tile_order() {
    if skipped_without_a_matrix_instruction(&Hardware::from_client(&client())) {
        return;
    }
    a_routed_stack_in_tile_order_reads_each_route_s_slab(CubeStagedMatmulRoutine, |_, hw| {
        MGroupedSettings::cube_staged_packed(hw)
            .into_iter()
            .map(|settings| settings.routine)
            .collect()
    });
}

/// Autotune serves a routed product over a stack in tile order: some floor reads it.
#[test]
fn a_tuned_routed_product_reads_each_route_s_slab_of_a_stack_in_tile_order() {
    let client = client();
    let hw = Hardware::from_client(&client);
    for laid in TILE_ORDERS {
        let lhs = HalfValues::new(ROUTED.tokens * ROUTED.k, 1);
        let stack = PackedSlabs::new(
            &client,
            PackedSlabsSpec {
                format: PackedFormat::Nvfp4,
                operand: MatmulOperand::Rhs,
                slabs: ROUTED.slabs,
                rows: ROUTED.n,
                k: ROUTED.k,
                seed: 2,
            },
        );
        let expected = ROUTED.expected_from(&lhs.floats, |slab, k, j| stack.cell(slab, k, j));
        let tensors = RoutedMatmulTensors::new(
            &client,
            lhs.upload(&client, &[ROUTED.tokens, ROUTED.k]),
            StoredTensor::Packed(stack.laid(&client, laid)),
            upload(&client, &[ROUTED.tokens, ROUTED.picks], &ROUTED.routes()),
            ROUTED.slabs,
            SlabAxis::Output,
        )
        .expect("a packed stack in tile order derives a routed launch");
        let no_matrix_instruction = hw.properties.features.matmul.cmma.iter().next().is_none();
        let out = match launch_tuned(&cubecl::test_device(), tensors, GroupPriority::DEFAULT) {
            Ok(out) => out,
            Err(why) if matches!(why, Refusal::CubePastDevice { .. }) || no_matrix_instruction => {
                TestOutcome::Validated(ValidationResult::Skipped(format!(
                    "no floor serves a packed stack on this runtime: {why:?}"
                )))
                .enforce();
                return;
            }
            Err(why) => panic!("a routed stack in tile order is refused: {laid:?} {why:?}"),
        };
        let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
        let shape = [ROUTED.tokens, ROUTED.picks, ROUTED.n];
        assert_close(&got, &shape, &expected, HALF_TOLERANCE, &laid);
    }
}

#[test]
fn a_grouped_product_on_the_plane_shuffle_reads_each_group_s_slab_of_a_stack_in_tile_order() {
    for laid in TILE_ORDERS {
        a_grouped_packed_stack_reads_each_group_s_slab(
            PackedFormat::Nvfp4,
            laid,
            PlaneShuffleMatmulRoutine,
            |_| MGroupedSettings::plane_shuffle_packed(),
        );
    }
}

#[test]
fn a_grouped_product_on_the_cube_staged_reads_each_group_s_slab_of_a_stack_in_tile_order() {
    if skipped_without_a_matrix_instruction(&Hardware::from_client(&client())) {
        return;
    }
    for laid in TILE_ORDERS {
        a_grouped_packed_stack_reads_each_group_s_slab(
            PackedFormat::Nvfp4,
            laid,
            CubeStagedMatmulRoutine,
            MGroupedSettings::cube_staged_packed,
        );
    }
}

/// A stack in tile order is read slab by slab only where each slab is whole strips of its rows:
/// slabs that split a strip, or lie along `k` across every strip, are refused rather than read
/// across.
#[test]
fn a_stack_in_tile_order_whose_slabs_split_a_strip_is_refused() {
    let client = client();
    for laid in TILE_ORDERS {
        let matrix = PackedSlabs::new(
            &client,
            PackedSlabsSpec {
                format: PackedFormat::Nvfp4,
                operand: MatmulOperand::Rhs,
                slabs: 1,
                rows: 24,
                k: 256,
                seed: 5,
            },
        )
        .laid(&client, laid);
        assert!(matrix.clone().stacked(3, N).is_ok(), "{laid:?}");
        assert_eq!(
            matrix.clone().stacked(2, N).err(),
            Some(PACKED_STACK_UNREAD)
        );
        assert_eq!(matrix.stacked(2, K).err(), Some(PACKED_STACK_UNREAD));
    }
}
