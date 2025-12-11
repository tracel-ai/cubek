#[cfg(all(feature = "matmul_tests_simple", feature = "matmul_tests_cyclic"))]
mod simple_cyclic {
    use super::*;
    type Algorithm = cubek_matmul::kernels::layered::simple::SimpleAlgorithm<TMM>;

    include!("precision.rs");
}

#[cfg(all(feature = "matmul_tests_simple", feature = "matmul_tests_strided"))]
mod simple_strided {
    use super::*;
    type Algorithm = cubek_matmul::kernels::layered::simple::SimpleAlgorithm<
        TMM,
        cubek_matmul::components::global::read::sync_full_strided::SyncFullStridedLoading,
        cubek_matmul::components::global::read::sync_full_strided::SyncFullStridedLoading,
    >;

    include!("precision.rs");
}

#[cfg(all(feature = "matmul_tests_simple", feature = "matmul_tests_tilewise"))]
mod simple_tilewise {
    use super::*;
    type Algorithm = cubek_matmul::kernels::layered::simple::SimpleAlgorithm<
        TMM,
        cubek_matmul::components::global::read::sync_full_tilewise::SyncFullTilewiseLoading<
            cubek_matmul::components::stage::RowMajorTilingOrder,
        >,
        cubek_matmul::components::global::read::sync_full_tilewise::SyncFullTilewiseLoading<
            cubek_matmul::components::stage::ColMajorTilingOrder,
        >,
    >;

    include!("precision.rs");
}

#[cfg(all(feature = "matmul_tests_simple", feature = "matmul_tests_barrier"))]
mod simple_barrier_cooperative {
    use super::*;
    type Algorithm = cubek_matmul::kernels::layered::simple::SimpleBarrierAlgorithm<
        TMM,
        cubek_matmul::components::global::read::async_full_cooperative::AsyncFullCooperativeLoading,
    >;

    include!("precision.rs");
}

#[cfg(all(feature = "matmul_tests_simple", feature = "matmul_tests_barrier"))]
mod simple_barrier_cyclic {
    use super::*;
    type Algorithm = cubek_matmul::kernels::layered::simple::SimpleBarrierAlgorithm<
        TMM,
        cubek_matmul::components::global::read::async_full_cyclic::AsyncFullCyclicLoading<
            cubek_matmul::components::stage::ColMajorTilingOrder,
        >,
    >;

    include!("precision.rs");
}

#[cfg(all(feature = "matmul_tests_double", feature = "matmul_tests_cyclic"))]
mod double_buffering_cyclic {
    use super::*;
    type Algorithm =
        cubek_matmul::kernels::layered::double_buffering::CyclicDoubleBufferingAlgorithm<TMM>;

    include!("precision.rs");
}

#[cfg(all(feature = "matmul_tests_double", feature = "matmul_tests_tilewise"))]
mod double_buffering_tilewise {
    use super::*;
    type Algorithm =
        cubek_matmul::kernels::layered::double_buffering::TilewiseDoubleBufferingAlgorithm<TMM>;

    include!("precision.rs");
}

#[cfg(all(feature = "matmul_tests_double", feature = "matmul_tests_hybrid"))]
mod double_buffering_hybrid {
    use super::*;
    type Algorithm =
        cubek_matmul::kernels::layered::double_buffering::HybridDoubleBufferingAlgorithm<TMM>;

    include!("precision.rs");
}

#[cfg(feature = "matmul_tests_ordered")]
mod ordered_double_buffering {
    use super::*;

    type Algorithm =
        cubek_matmul::kernels::layered::ordered_double_buffering::OrderedDoubleBufferingAlgorithm<
            TMM,
        >;

    include!("precision.rs");
}

#[cfg(all(feature = "matmul_tests_double", feature = "matmul_tests_barrier"))]
mod specialized_cyclic {
    use super::*;

    type Algorithm = cubek_matmul::kernels::layered::specialized::SpecializedAlgorithm<
        TMM,
        cubek_matmul::components::global::read::async_partial_cyclic::AsyncPartialCyclicLoading<
            cubek_matmul::components::stage::ColMajorTilingOrder,
        >,
    >;

    include!("precision.rs");
}

#[cfg(all(feature = "matmul_tests_double", feature = "matmul_tests_barrier"))]
mod specialized_strided {
    use super::*;

    type Algorithm = cubek_matmul::kernels::layered::specialized::SpecializedAlgorithm<
        TMM,
        cubek_matmul::components::global::read::async_partial_strided::AsyncPartialStridedLoading,
    >;

    include!("precision.rs");
}
