mod p1x1x1 {
    use super::*;
    use cubek_matmul::components::{PartitionSize, TilingSchemeBuilder};

    fn partition(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_partition_size(PartitionSize { m: 1, n: 1, k: 1 })
    }

    include!("stage.rs");
}

mod p1x2x1 {
    use super::*;
    use cubek_matmul::components::{PartitionSize, TilingSchemeBuilder};

    fn partition(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_partition_size(PartitionSize { m: 1, n: 2, k: 1 })
    }

    include!("stage.rs");
}

mod p1x1x2 {
    use super::*;
    use cubek_matmul::components::{PartitionSize, TilingSchemeBuilder};

    fn partition(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_partition_size(PartitionSize { m: 1, n: 1, k: 2 })
    }

    include!("stage.rs");
}
