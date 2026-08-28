//! Phase 0 probe: the scales named in `Tiling::over` and staged at a level.
//!
//! Not a design, a question. How far does the operand/staging machinery already carry a third
//! operand that broadcasts, when a level states the count as a *cut* and the residence as a
//! *stage*, rather than the leaf reconciling a binding's width against the walk?

use cubecl::{Runtime, TestRuntime, prelude::*};
use cubek_test_utils::{HostData, HostDataType, TileInput};
use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const KB: Axis = Axis(2);
const KI: Axis = Axis(3);

/// `c = (a ⊗ s) · b`, the scales staged a level above the instruction.
#[cube(launch)]
fn staged_scales_matmul<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    scales: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let scales = scales.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.mm_scaled(&a, &b, &scales, Semiring::SUM_PROD);
}

/// The shape the design wants said out loud: a unit takes 8 blocks of `K` and reads its 8 scales
/// once, then walks them one block at a time.
#[test]
fn scales_are_an_operand_staged_at_a_level() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (m, n, block, blocks) = (4usize, 4usize, 4usize, 8usize);
    let dtype = f32::elem_type_native();

    let mut operands = (
        Operand::new(&[M, KB, KI], dtype),
        Operand::new(&[KB, KI, N], dtype),
        // The scales omit the position inside a block: that omission is the broadcast.
        Operand::new(&[M, KB], dtype),
        Operand::new(&[M, N], dtype),
    );

    let space = Tiling::over(&mut operands, &[(M, m), (N, n), (KB, blocks), (KI, block)])
        // This unit takes 8 blocks, and reads its 8 scales here, once.
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |level, ops| {
            level
                .axis(M, Cut::sequential(m))
                .axis(N, Cut::sequential(n))
                .axis(KB, Cut::sequential(blocks))
                .axis(KI, Cut::sequential(block));
            let _ = &ops;
        })
        // ...and walks them one block at a time below.
        .instruction(Instruction::registers(64), |level, _| {
            level
                .axis(M, Cut::sequential(m))
                .axis(N, Cut::sequential(n))
                .axis(KB, Cut::sequential(1))
                .axis(KI, Cut::sequential(block));
        })
        .build();

    let a = TileInput::builder(&client, space.project(&[M, KB, KI]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[KB, KI, N]))
        .untiled()
        .arange();
    let s = TileInput::builder(&client, space.project(&[M, KB]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    let launcher = space.clone().launcher(&client);
    let a_op = launcher.bind(&operands.0, a.handle().binding()).build();
    let b_op = launcher.bind(&operands.1, b.handle().binding()).build();
    let s_op = launcher.bind(&operands.2, s.handle().binding()).build();
    let c_op = launcher.bind(&operands.3, c.handle().binding()).build();

    staged_scales_matmul::launch::<TestRuntime>(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a_op.arg(),
        b_op.arg(),
        s_op.arg(),
        c_op.arg(),
        space,
        dtype,
    );
    cubecl::future::block_on(client.sync()).unwrap();

    // `arange` over each operand's own box, so the reference is the walk's own indexing.
    let depth = blocks * block;
    let got = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    for row in 0..m {
        for col in 0..n {
            let want: f32 = (0..depth)
                .map(|k| {
                    let a = (row * depth + k) as f32;
                    let b = (k * n + col) as f32;
                    let s = (row * blocks + k / block) as f32;
                    a * s * b
                })
                .sum();
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-2,
                "at ({row}, {col}): got {have}, want {want}"
            );
        }
    }
}
