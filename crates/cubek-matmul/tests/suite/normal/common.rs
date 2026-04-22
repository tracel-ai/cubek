use cubecl::{
    Runtime, TestRuntime,
    client::ComputeClient,
    ir::{AddressType, BarrierLevel, OpaqueType, SemanticType},
    zspace::shape,
};
use cubek_matmul::definition::{MatmulElems, MatmulGlobalElems, MatmulProblem};
use cubek_std::MatrixLayout;

pub(crate) fn client() -> ComputeClient<TestRuntime> {
    TestRuntime::client(&Default::default())
}

pub(crate) fn f16_elems() -> MatmulGlobalElems {
    use cubecl::frontend::CubePrimitive;
    MatmulElems::from_single_dtype(half::f16::as_type_native_unchecked()).as_global_elems()
}

pub(crate) fn f32_elems() -> MatmulGlobalElems {
    use cubecl::frontend::CubePrimitive;
    MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems()
}

pub(crate) fn square(dim: usize, elems: MatmulGlobalElems) -> MatmulProblem {
    rect(dim, dim, dim, elems)
}

pub(crate) fn rect(m: usize, n: usize, k: usize, elems: MatmulGlobalElems) -> MatmulProblem {
    rect_with_layouts(
        m,
        n,
        k,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        elems,
    )
}

pub(crate) fn rect_with_layouts(
    m: usize,
    n: usize,
    k: usize,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    elems: MatmulGlobalElems,
) -> MatmulProblem {
    MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![1],
        shape![1],
        lhs_layout,
        rhs_layout,
        MatrixLayout::RowMajor,
        None,
        None,
        elems,
        AddressType::U32,
    )
}
