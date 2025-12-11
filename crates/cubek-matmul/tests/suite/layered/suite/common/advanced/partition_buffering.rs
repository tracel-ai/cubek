#[cfg(not(feature = "matmul_tests_partition_buffering"))]
pub mod no_partition_buffering {
    use super::*;
    use cubek_matmul::components::stage::PartitionBuffering;

    fn partition_buffering() -> PartitionBuffering {
        PartitionBuffering::Single
    }

    include!("../problem/layouts.rs");
}

#[cfg(feature = "matmul_tests_partition_buffering")]
pub mod pb1 {
    use super::*;
    use cubek_matmul::components::stage::PartitionBuffering;

    fn partition_buffering() -> PartitionBuffering {
        PartitionBuffering::Single
    }

    include!("../problem/layouts.rs");
}

#[cfg(feature = "matmul_tests_partition_buffering")]
pub mod pb2 {
    use super::*;
    use cubek_matmul::components::stage::PartitionBuffering;

    fn partition_buffering() -> PartitionBuffering {
        PartitionBuffering::Double
    }

    include!("../problem/layouts.rs");
}
