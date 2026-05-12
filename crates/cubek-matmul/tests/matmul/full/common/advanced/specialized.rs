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
