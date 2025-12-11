mod p1x1x4 {
    use super::*;
    use cubek_matmul::components::{PartitionSize, TilingSchemeBuilder};

    fn partition(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_partition_size(PartitionSize { m: 1, n: 1, k: 4 })
    }

    include!("stage.rs");
}

mod p2x1x4 {
    use super::*;
    use cubek_matmul::components::{PartitionSize, TilingSchemeBuilder};

    fn partition(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_partition_size(PartitionSize { m: 2, n: 1, k: 4 })
    }

    include!("stage.rs");
}
