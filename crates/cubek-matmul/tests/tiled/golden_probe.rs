//! Compiles the CpuGemm kernel on whatever runtime the tests run on, checking nothing: the
//! generated source is what the golden capture reads, and the CPU runtime that would run this
//! routine for real does not link on every machine.

use cubecl::{Runtime, TestRuntime, frontend::Scalar, prelude::*, zspace::shape};
use cubek_matmul::{
    definition::MatmulElems,
    routine::BlueprintStrategy,
    tiled::cpu_gemm::{CpuGemmBlueprint, InstructionShape, PlaneGrid, WithLayout, launch_ref},
};
use cubek_std::InputBinding;
use cubek_test_utils::TestInput;

#[test]
fn cpu_gemm_kernel_compiles() {
    let client = TestRuntime::client(&Default::default());
    let (m, n, k, tile) = (32usize, 32usize, 64usize, 8usize);
    let dtypes = MatmulElems::from_single_dtype(f32::elem_type_native());
    let a = TestInput::builder(client.clone(), shape![1, m, k])
        .uniform(1234, -1., 1.)
        .generate_without_host_data();
    let b = TestInput::builder(client.clone(), shape![1, k, n])
        .uniform(5678, -1., 1.)
        .generate_without_host_data();
    let out = TestInput::builder(client.clone(), shape![1, m, n])
        .zeros()
        .generate_without_host_data();
    launch_ref::<TestRuntime>(
        &client,
        WithLayout::strided_input(InputBinding::Normal(a.binding(), dtypes.lhs_global)).unwrap(),
        WithLayout::strided_input(InputBinding::Normal(b.binding(), dtypes.rhs_global)).unwrap(),
        WithLayout::strided_output(out.binding()).unwrap(),
        &BlueprintStrategy::Forced(CpuGemmBlueprint {
            instruction: InstructionShape {
                m: tile,
                n: tile,
                k: tile,
            },
            planes: PlaneGrid { m: 2, n: 2 },
        }),
        &dtypes,
    )
    .unwrap();
    client.sync();
}
