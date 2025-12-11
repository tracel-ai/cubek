mod mm {
    use super::*;
    use cubek_matmul::components::global::{LoadSpecializationConfig, SpecializationTensorConfig};

    fn specialization() -> LoadSpecializationConfig {
        LoadSpecializationConfig {
            lhs: SpecializationTensorConfig::MainFlowOnly,
            rhs: SpecializationTensorConfig::MainFlowOnly,
        }
    }

    include!("swizzle.rs");
}

#[cfg(feature = "matmul_tests_specialized")]
mod ml {
    use super::*;
    use cubek_matmul::components::global::{LoadSpecializationConfig, SpecializationTensorConfig};

    fn specialization() -> LoadSpecializationConfig {
        LoadSpecializationConfig {
            lhs: SpecializationTensorConfig::MainFlowOnly,
            rhs: SpecializationTensorConfig::LoadFlowOnly,
        }
    }

    include!("swizzle.rs");
}

#[cfg(feature = "matmul_tests_specialized")]
mod lm {
    use super::*;
    use cubek_matmul::components::global::{LoadSpecializationConfig, SpecializationTensorConfig};

    fn specialization() -> LoadSpecializationConfig {
        LoadSpecializationConfig {
            lhs: SpecializationTensorConfig::LoadFlowOnly,
            rhs: SpecializationTensorConfig::MainFlowOnly,
        }
    }

    include!("swizzle.rs");
}

#[cfg(feature = "matmul_tests_specialized")]
mod ll {
    use super::*;
    use cubek_matmul::components::global::{LoadSpecializationConfig, SpecializationTensorConfig};

    fn specialization() -> LoadSpecializationConfig {
        LoadSpecializationConfig {
            lhs: SpecializationTensorConfig::LoadFlowOnly,
            rhs: SpecializationTensorConfig::LoadFlowOnly,
        }
    }

    include!("swizzle.rs");
}
