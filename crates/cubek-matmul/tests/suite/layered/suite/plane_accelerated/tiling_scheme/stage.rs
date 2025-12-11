mod s1x1x1 {
    use super::*;
    use cubek_matmul::components::{StageSize, TilingSchemeBuilder};

    fn stage(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_stage_size(StageSize { m: 1, n: 1, k: 1 })
    }

    include!("../../common/advanced/specialized.rs");
}

mod s2x2x1 {
    use super::*;
    use cubek_matmul::components::{StageSize, TilingSchemeBuilder};

    fn stage(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_stage_size(StageSize { m: 2, n: 2, k: 1 })
    }

    include!("../../common/advanced/specialized.rs");
}

mod s4x4x1 {
    use super::*;
    use cubek_matmul::components::{StageSize, TilingSchemeBuilder};

    fn stage(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_stage_size(StageSize { m: 4, n: 4, k: 1 })
    }

    include!("../../common/advanced/specialized.rs");
}

mod s8x4x1 {
    use super::*;
    use cubek_matmul::components::{StageSize, TilingSchemeBuilder};

    fn stage(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_stage_size(StageSize { m: 8, n: 4, k: 1 })
    }

    include!("../../common/advanced/specialized.rs");
}

mod s8x8x1 {
    use super::*;
    use cubek_matmul::components::{StageSize, TilingSchemeBuilder};

    fn stage(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_stage_size(StageSize { m: 8, n: 8, k: 1 })
    }

    include!("../../common/advanced/specialized.rs");
}
