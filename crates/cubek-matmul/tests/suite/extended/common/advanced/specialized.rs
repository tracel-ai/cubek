mod mm {
    use super::*;
    use cubek_matmul::components::global::{InputLoadFlow, LoadFlows};

    fn specialization() -> LoadFlows {
        LoadFlows {
            lhs: InputLoadFlow::MainOnly,
            rhs: InputLoadFlow::MainOnly,
        }
    }

    include!("swizzle.rs");
}

#[cfg(feature = "matmul_tests_specialized")]
mod ml {
    use super::*;
    use cubek_matmul::components::global::{InputLoadFlow, LoadFlows};

    fn specialization() -> LoadFlows {
        LoadFlows {
            lhs: InputLoadFlow::MainOnly,
            rhs: InputLoadFlow::LoadOnly,
        }
    }

    include!("swizzle.rs");
}

#[cfg(feature = "matmul_tests_specialized")]
mod lm {
    use super::*;
    use cubek_matmul::components::global::{InputLoadFlow, LoadFlows};

    fn specialization() -> LoadFlows {
        LoadFlows {
            lhs: InputLoadFlow::LoadOnly,
            rhs: InputLoadFlow::MainOnly,
        }
    }

    include!("swizzle.rs");
}

#[cfg(feature = "matmul_tests_specialized")]
mod ll {
    use super::*;
    use cubek_matmul::components::global::{InputLoadFlow, LoadFlows};

    fn specialization() -> LoadFlows {
        LoadFlows {
            lhs: InputLoadFlow::LoadOnly,
            rhs: InputLoadFlow::LoadOnly,
        }
    }

    include!("swizzle.rs");
}
